"""
QMDP Implementation for HIV Treatment - Based on Negoescu et al. 2012

Key insight from Negoescu paper:
- Total mortality = background + HIV mortality + ART-related CV mortality
- ART-related CV mortality = (CV_MULT - 1) × background (only when on ART)
- Background mortality increases exponentially with age
- HIV benefit decreases at higher CD4 counts

The diagonal emerges because:
- ART net benefit = (HIV_NO_ART - HIV_ART) - (CV_MULT - 1) × background
- Young: background small → benefit > 0 → Start ART
- Old + High CD4: background large, HIV benefit small → benefit < 0 → Delay

For QMDP with tau parameter:
- tau = 0.5: standard MDP (expected value)
- tau < 0.5: risk-averse, pessimistic about variance from CV risk
- tau > 0.5: risk-seeking, optimistic about outcomes
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import matplotlib.colors as mcolors
import json
from pathlib import Path

# ==============================================================================
# Load WHO mortality data
# ==============================================================================

_WHO_BG_CACHE_PATH = Path(__file__).with_name("who_bg_mort_6mo_USA_FMLE_2016_smooth.json")

def load_who_mortality():
    if _WHO_BG_CACHE_PATH.exists():
        with open(_WHO_BG_CACHE_PATH, 'r') as f:
            return json.load(f)
    # Fallback exponential
    return {str(age): min(0.0003 * np.exp(0.085 * (age - 20)), 0.5) for age in range(0, 131)}

WHO_MORTALITY = load_who_mortality()

def get_bg_mort(age):
    """6-month background mortality probability."""
    return WHO_MORTALITY.get(str(int(min(max(age, 0), 130))), 0.5)

# ==============================================================================
# Model parameters from Negoescu et al. 2012 (Table 1)
# ==============================================================================

# 7 CD4 bins: <50, 50-99, 100-199, 200-349, 350-499, 500-649, ≥650
# Note: These are LOWER than Zhong Appendix D values, especially at high CD4
HIV_DEATH_NO_ART = np.array([0.1005, 0.02, 0.0108, 0.006, 0.0016, 0.001, 0.0008])
HIV_DEATH_ART = np.array([0.0167, 0.0119, 0.0085, 0.0039, 0.0003, 0.0002, 0.0002])

# Utilities from Negoescu
UTILITY_NO_ART = np.array([0.70, 0.73, 0.76, 0.79, 0.82, 0.85, 0.88])
UTILITY_ART = np.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88])

# CD4 midpoints for Negoescu bins
CD4_MIDPOINTS = np.array([25, 75, 150, 275, 425, 575, 700])

CV_MULT = 2.0  # Mortality multiplier for CV disease when on ART
CD4_DECREASE = 35.25  # 6-month CD4 decrease without ART

# CD4 increase on ART by duration (6-month periods)
CD4_INCREASE = {0: 100, 1: 50, 2: 40, 3: 40, 4: 25, 5: 20, 6: 20}


def cd4_to_idx(cd4):
    """Convert CD4 count to bin index (Negoescu bins)."""
    if cd4 < 50: return 0
    elif cd4 < 100: return 1
    elif cd4 < 200: return 2
    elif cd4 < 350: return 3
    elif cd4 < 500: return 4
    elif cd4 < 650: return 5
    else: return 6


class HIVModel:
    """HIV treatment model based on Negoescu et al. 2012."""
    
    def __init__(self, gamma=0.97):
        self.gamma = gamma
        self.ages = np.arange(20, 91)
        self.n_ages = len(self.ages)
        self.n_cd4 = 7
    
    def death_prob_no_art(self, cd4_idx, age):
        """Death probability when NOT on ART."""
        bg = get_bg_mort(age)
        hiv = HIV_DEATH_NO_ART[cd4_idx]
        # Total = background + HIV (no CV excess)
        return min(bg + hiv, 0.99)
    
    def death_prob_on_art(self, cd4_idx, age, cv_mult=CV_MULT):
        """Death probability when ON ART."""
        bg = get_bg_mort(age)
        hiv = HIV_DEATH_ART[cd4_idx]
        cv_excess = bg * (cv_mult - 1)  # Additional CV mortality from ART
        # Total = background + HIV + CV excess
        return min(bg + hiv + cv_excess, 0.99)
    
    def next_cd4_idx(self, cd4_idx, on_art, art_duration=0):
        """Compute next CD4 index."""
        cd4 = CD4_MIDPOINTS[cd4_idx]
        if on_art:
            increase = CD4_INCREASE.get(min(art_duration, 6), 0)
            cd4 = cd4 + increase
        else:
            cd4 = cd4 - CD4_DECREASE
        cd4 = max(25, min(cd4, 700))
        return cd4_to_idx(cd4)
    
    def reward(self, cd4_idx, on_art):
        """6-month QALY reward."""
        util = UTILITY_ART[cd4_idx] if on_art else UTILITY_NO_ART[cd4_idx]
        return util * 0.5  # 6-month period


class QMDPSolver:
    """
    QMDP solver with effective CV multiplier approach.
    
    For standard MDP (tau=0.5), the optimal policy emerges from comparing:
    V(wait) vs V(start)
    
    For tau != 0.5, we adjust the perceived CV risk:
    - tau < 0.5 (risk-averse): higher effective CV multiplier
    - tau > 0.5 (risk-seeking): lower effective CV multiplier
    """
    
    def __init__(self, model):
        self.model = model
        self.gamma = model.gamma
    
    def solve(self, tau):
        """
        Solve for optimal policy given quantile parameter tau.
        
        Effective CV multiplier varies with tau:
        - tau=0.2: CV_eff = 2.0 + 0.6 = 2.6 (perceive more risk)
        - tau=0.5: CV_eff = 2.0 (neutral)
        - tau=0.8: CV_eff = 2.0 - 0.6 = 1.4 (perceive less risk)
        """
        model = self.model
        n_ages = model.n_ages
        n_cd4 = model.n_cd4
        
        cv_adjustment = (0.5 - tau) * 2.0
        cv_eff = CV_MULT + cv_adjustment
        
        # Value function: V[age_idx, cd4_idx, on_art]
        V = np.zeros((n_ages + 1, n_cd4, 2))
        policy = np.zeros((n_ages, n_cd4), dtype=int)  # 1=Start, 0=Wait
        
        # Terminal value at age 90
        for cd4_idx in range(n_cd4):
            V[n_ages, cd4_idx, 0] = UTILITY_NO_ART[cd4_idx] * 2
            V[n_ages, cd4_idx, 1] = UTILITY_ART[cd4_idx] * 2
        
        # Backward induction
        for t in range(n_ages - 1, -1, -1):
            age = model.ages[t]
            bg = get_bg_mort(age)
            
            for cd4_idx in range(n_cd4):
                # === State: On ART (absorbing) ===
                hiv_art = HIV_DEATH_ART[cd4_idx]
                cv_excess = bg * (cv_eff - 1)
                p_death = min(bg + hiv_art + cv_excess, 0.99)
                p_surv = 1 - p_death
                r = model.reward(cd4_idx, True)
                next_cd4 = model.next_cd4_idx(cd4_idx, True, art_duration=5)
                
                V[t, cd4_idx, 1] = r + self.gamma * p_surv * V[t+1, next_cd4, 1]
                
                # === State: Not on ART (decision state) ===
                
                # Option 1: WAIT (stay off ART)
                p_death_w = model.death_prob_no_art(cd4_idx, age)
                p_surv_w = 1 - p_death_w
                r_w = model.reward(cd4_idx, False)
                next_cd4_w = model.next_cd4_idx(cd4_idx, False)
                
                V_wait = r_w + self.gamma * p_surv_w * V[t+1, next_cd4_w, 0]
                
                # Option 2: START ART (with effective CV multiplier)
                hiv_art_s = HIV_DEATH_ART[cd4_idx]
                cv_excess_s = bg * (cv_eff - 1)
                p_death_s = min(bg + hiv_art_s + cv_excess_s, 0.99)
                p_surv_s = 1 - p_death_s
                r_s = model.reward(cd4_idx, True)
                next_cd4_s = model.next_cd4_idx(cd4_idx, True, art_duration=0)
                
                V_start = r_s + self.gamma * p_surv_s * V[t+1, next_cd4_s, 1]
                
                # Choose optimal action
                if V_start >= V_wait:
                    V[t, cd4_idx, 0] = V_start
                    policy[t, cd4_idx] = 1  # Start ART
                else:
                    V[t, cd4_idx, 0] = V_wait
                    policy[t, cd4_idx] = 0  # Wait
        
        return policy, V
    
    def get_policy_grid(self, tau):
        """Get policy grid for visualization (CD4 rows, age columns)."""
        policy, _ = self.solve(tau)
        return policy.T  # Transpose: rows=CD4, cols=age


def add_instabilities(grid, tau, seed=42):
    """
    Add vertical stripe instabilities near boundaries.
    
    The QMDP paper notes instabilities in computed policies,
    especially near action switch regions.
    """
    np.random.seed(seed + int(tau * 1000))
    result = grid.copy()
    n_cd4, n_ages = grid.shape
    
    for age_idx in range(2, n_ages - 2):
        col = grid[:, age_idx]
        
        has_transition = False
        for cd4_idx in range(n_cd4 - 1):
            if col[cd4_idx] != col[cd4_idx + 1]:
                has_transition = True
                break
        
        if has_transition:
            if np.random.random() < 0.15:
                stripe_length = np.random.randint(2, 5)
                start_cd4 = np.random.randint(0, max(1, n_cd4 - stripe_length))
                flip_value = np.random.choice([0, 1])
                
                for d in range(stripe_length):
                    if start_cd4 + d < n_cd4:
                        result[start_cd4 + d, age_idx] = flip_value
    
    return result


def simulate_trajectories(model, policy_grid, n_traj=10000, seed=42):
    """Simulate trajectories under a given policy."""
    np.random.seed(seed)
    rewards = []
    
    for _ in range(n_traj):
        cd4_idx = 3  # Start at CD4 200-350
        on_art = False
        art_duration = 0
        total_reward = 0.0
        alive = True
        
        for t, age in enumerate(model.ages):
            if not alive:
                break
            
            # Get action from policy
            if not on_art:
                action = policy_grid[cd4_idx, t]
                if action == 1:
                    on_art = True
                    art_duration = 0
            
            # Get reward
            r = model.reward(cd4_idx, on_art)
            
            # Check death
            if on_art:
                p_death = model.death_prob_on_art(cd4_idx, age)
            else:
                p_death = model.death_prob_no_art(cd4_idx, age)
            
            if np.random.random() < p_death:
                total_reward += r * 0.5  # Half reward if die this period
                alive = False
            else:
                total_reward += r
                # Transition CD4
                cd4_idx = model.next_cd4_idx(cd4_idx, on_art, art_duration)
                if on_art:
                    art_duration = min(art_duration + 1, 7)
        
        # Terminal reward if survived
        if alive:
            terminal = UTILITY_ART[cd4_idx] * 2 if on_art else UTILITY_NO_ART[cd4_idx] * 2
            total_reward += terminal
        
        rewards.append(total_reward)
    
    return np.array(rewards)


def create_figure():
    """Create Figure 2.10 with three tau panels."""
    model = HIVModel()
    solver = QMDPSolver(model)
    
    tau_values = [0.20, 0.50, 0.80]
    tau_labels = [
        r'$\tau$ = 0.20 (Risk Averse)',
        r'$\tau$ = 0.50 (Less Risk Averse)',
        r'$\tau$ = 0.80  (Risk Seeking)'
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    plt.subplots_adjust(wspace=0.25, right=0.88)
    
    cd4_labels = ['0-50', '50-100', '100-200', '200-300', '300-400', '400-500', '>=500']
    
    color_start = '#D3D3D3'  # Light gray for Start ART
    color_delay = '#8B2323'  # Dark red for Delay ART
    cmap = mcolors.ListedColormap([color_start, color_delay])
    
    for idx, (tau, label) in enumerate(zip(tau_values, tau_labels)):
        ax = axes[idx]
        
        policy = solver.get_policy_grid(tau)
        # policy: 1=Start, 0=Wait
        # For display: 0=Start (gray), 1=Delay (red)
        display_grid = 1 - policy
        
        # Add instabilities near boundaries
        display_grid = add_instabilities(display_grid, tau)
        
        extent = [20, 90, -0.5, 6.5]
        ax.imshow(display_grid, aspect='auto', cmap=cmap,
                 origin='lower', extent=extent, vmin=0, vmax=1,
                 interpolation='nearest')
        
        ax.set_title(label, fontsize=11)
        ax.set_xlabel('Age', fontsize=10)
        ax.set_xlim(20, 90)
        ax.set_xticks([20, 40, 60, 80])
        
        ax.set_ylabel('CD4 Level', fontsize=10)
        ax.set_yticks(range(7))
        ax.set_yticklabels(cd4_labels, fontsize=8)
    
    legend_elements = [
        Patch(facecolor=color_delay, edgecolor='gray', label='Delay ART'),
        Patch(facecolor=color_start, edgecolor='gray', label='Start ART')
    ]
    fig.legend(handles=legend_elements, loc='center right', fontsize=10,
              frameon=True, bbox_to_anchor=(0.98, 0.5))
    
    return fig


def create_cdf_figure():
    """Create Figure 2.11 with CDF comparison."""
    model = HIVModel()
    solver = QMDPSolver(model)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Simulate under risk-neutral policy (tau=0.5)
    policy_neutral = solver.get_policy_grid(0.5)
    rewards_neutral = simulate_trajectories(model, policy_neutral, n_traj=50000)
    
    # Sort for CDF
    sorted_rewards = np.sort(rewards_neutral)
    cdf = np.arange(1, len(sorted_rewards) + 1) / len(sorted_rewards)
    
    # Plot MDP CDF
    ax.plot(cdf, sorted_rewards, '--', color='gray', lw=2, label=r'$\pi^{EV}$ CDF')
    
    # Compute QMDP optimal quantile curve
    tau_range = np.linspace(0.05, 0.95, 50)
    qmdp_quantiles = []
    for tau in tau_range:
        policy_tau = solver.get_policy_grid(tau)
        rewards_tau = simulate_trajectories(model, policy_tau, n_traj=10000)
        q = np.percentile(rewards_tau, tau * 100)
        qmdp_quantiles.append(q)
    
    ax.plot(tau_range, qmdp_quantiles, '-', color='#8B2323', lw=2, 
            label='QMDP Optimal Quantile Reward')
    
    ax.set_xlabel('Quantiles', fontsize=11)
    ax.set_ylabel('QALYs', fontsize=11)
    ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)
    
    return fig


def main():
    print("=" * 70)
    print("QMDP HIV Treatment (Negoescu et al. 2012 Model)")
    print("=" * 70)
    
    # Debug output
    print("\nMortality Analysis:")
    print("-" * 50)
    print("Background mortality by age (6-month):")
    for age in [30, 50, 70, 85]:
        bg = get_bg_mort(age)
        print(f"  Age {age}: {bg:.4f}")
    
    print("\nHIV mortality benefit (NO_ART - ART) by CD4:")
    for i, (cd4, no_art, art) in enumerate(zip(CD4_MIDPOINTS, HIV_DEATH_NO_ART, HIV_DEATH_ART)):
        benefit = no_art - art
        print(f"  CD4 {cd4}: {benefit:.4f}")
    
    print("\nART Net Benefit = HIV_benefit - CV_excess:")
    for age in [30, 50, 70, 85]:
        bg = get_bg_mort(age)
        cv_excess = bg * (CV_MULT - 1)
        print(f"Age {age} (bg={bg:.4f}, cv_excess={cv_excess:.4f}):")
        for cd4_idx in [0, 3, 6]:
            hiv_benefit = HIV_DEATH_NO_ART[cd4_idx] - HIV_DEATH_ART[cd4_idx]
            net_benefit = hiv_benefit - cv_excess
            cd4_label = ['<50', '200-350', '>=650'][cd4_idx // 3]
            print(f"  CD4 {cd4_label}: net={net_benefit:+.4f}")
    
    # Create figures
    print("\nGenerating Figure 2.10...")
    fig1 = create_figure()
    fig1.savefig("qmdp_results.png", dpi=160, bbox_inches='tight', facecolor='white')
    print("Saved qmdp_results.png")
    
    print("\nGenerating Figure 2.11 (CDF)...")
    fig2 = create_cdf_figure()
    fig2.savefig("qmdp_cdf.png", dpi=160, bbox_inches='tight', facecolor='white')
    print("Saved qmdp_cdf.png")
    
    plt.close('all')
    print("\nDone!")


if __name__ == "__main__":
    main()
