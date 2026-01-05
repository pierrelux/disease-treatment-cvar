"""
QMDP (Quantile MDP) for HIV Treatment Initiation
=================================================

Implements the Quantile MDP approach from Zhong (2020) for the HIV treatment
initiation problem. QMDP maximizes the tau-quantile of cumulative reward.

For a given quantile level tau in (0,1), QMDP finds the policy that maximizes
Q_tau(R), where Q_tau is the tau-quantile of the return distribution.

The algorithm uses threshold-based backward DP:
1. For threshold z, compute V_t(x, r) = P(R >= z | X_t = x, cumulative reward = r)
2. Bellman: V_t(x, r) = max_a sum_{x'} P(x'|x,a) * V_{t+1}(x', r + reward)
3. Terminal: V_T(x, r) = 1 if r >= z, else 0
4. For tau, find z* = max{z : V_0(x_0, 0) >= 1 - tau}

NOTE: The exact parameters from Zhong (2020) Figures 2.10-2.11 are not fully
available (background mortality from reference [139] is not tabulated). This
implementation uses WHO Life Tables 2016 as an approximation, which produces
qualitatively similar but not identical policy patterns.

Reference:
  Li Y, Zhong M, Marecki M, Makar M, Ghavamzadeh M (2022). Quantile Markov
  Decision Processes. AISTATS 2022.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import json

# ==============================================================================
# Import model parameters from static_cvar_hiv.py
# ==============================================================================

# Thesis parameters (Table A.1)
CD4_MIDPOINTS = np.array([25, 75, 150, 250, 350, 450, 750], dtype=float)
DEATH_NO_ART = np.array([0.1618, 0.0692, 0.0549, 0.0428, 0.0348, 0.0295, 0.0186, 0.0], dtype=float)
DEATH_ART = np.array([0.1356, 0.0472, 0.0201, 0.0103, 0.0076, 0.0076, 0.0045, 0.0], dtype=float)
UTILITY_NO_ART = np.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.0], dtype=float) * 0.5
UTILITY_ART = np.array([0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.0], dtype=float) * 0.5
CD4_DECREASE = 35.25
CARDIAC_MULT = 2.0

# Model config
START_AGE, END_AGE = 20, 90
T = (END_AGE - START_AGE) * 2  # 140 six-month periods
N_CD4 = 8  # includes dead state 7
MAX_DT = 8

# Reward discretization
N_BINS = 320
MAX_REWARD = 80.0
BIN_WIDTH = MAX_REWARD / N_BINS
REWARD_GRID = np.arange(N_BINS, dtype=float) * BIN_WIDTH

# ==============================================================================
# Background mortality
# ==============================================================================

_WHO_BG_CACHE_PATH = Path(__file__).with_name("who_bg_mort_6mo_USA_FMLE_2016.json")
_WHO_BG_CACHE: Optional[Dict[int, float]] = None

def _load_who_cache() -> Optional[Dict[int, float]]:
    global _WHO_BG_CACHE
    if _WHO_BG_CACHE is not None:
        return _WHO_BG_CACHE
    if _WHO_BG_CACHE_PATH.exists():
        try:
            _WHO_BG_CACHE = {int(k): float(v) for k, v in json.loads(_WHO_BG_CACHE_PATH.read_text()).items()}
            return _WHO_BG_CACHE
        except Exception:
            pass
    return None

def bg_mort(age: float) -> float:
    a = int(np.floor(age))
    cache = _load_who_cache()
    if cache is not None and a in cache:
        return float(cache[a])
    p = 0.0003 * np.exp(0.085 * (age - 20.0))
    return float(min(p, 0.3))

def combine_mortality(p_hiv: float, p_bg: float) -> float:
    return float(1.0 - (1.0 - p_hiv) * (1.0 - p_bg))

def bg_mort_on_art(p_bg: float) -> float:
    return float(1.0 - (1.0 - p_bg) ** CARDIAC_MULT)

# Precompute background mortality
AGES = np.array([START_AGE + (t // 2) for t in range(T)], dtype=float)
BG_MORT = np.array([bg_mort(a) for a in AGES], dtype=float)

# ==============================================================================
# CD4 transitions
# ==============================================================================

def cd4_increase_by_period(k: int) -> float:
    if k <= 0: return 0.0
    if k == 1: return 100.0
    if k == 2: return 50.0
    if k == 3: return 40.0
    if k == 4: return 40.0
    if k == 5: return 25.0
    if k == 6: return 20.0
    if k == 7: return 20.0
    return 0.0

def cd4_to_level(cd4: float) -> int:
    bins = [0, 50, 100, 200, 300, 400, 500, 1500]
    for i in range(7):
        if bins[i] <= cd4 < bins[i+1]:
            return i
    return 6

NEXT_CD4 = np.zeros((7, 2, MAX_DT + 1), dtype=np.int32)
for cd4 in range(7):
    for on_art in (0, 1):
        for dt in range(MAX_DT + 1):
            curr = CD4_MIDPOINTS[cd4]
            if on_art == 1:
                inc = cd4_increase_by_period(dt)
                new = curr + inc
            else:
                new = max(curr - CD4_DECREASE, 0.0)
            NEXT_CD4[cd4, on_art, dt] = cd4_to_level(new)

def reward_to_bin(R: float) -> int:
    return int(np.clip(np.round(R / BIN_WIDTH), 0, N_BINS - 1))

# ==============================================================================
# Terminal value (risk-neutral tail beyond END_AGE)
# ==============================================================================

TAIL_END_AGE = 110

def compute_terminal_value() -> np.ndarray:
    """Risk-neutral expected future QALYs for ages 90-110."""
    T_tail = (TAIL_END_AGE - END_AGE) * 2
    if T_tail == 0:
        return np.zeros((N_CD4, MAX_DT + 1), dtype=float)
    
    ages_tail = np.array([END_AGE + (t // 2) for t in range(T_tail)], dtype=float)
    bg_tail = np.array([bg_mort(a) for a in ages_tail], dtype=float)
    
    V = np.zeros((N_CD4, MAX_DT + 1), dtype=float)
    
    for t in range(T_tail - 1, -1, -1):
        V_new = np.zeros_like(V)
        p_bg = bg_tail[t]
        
        for cd4 in range(7):
            for dt in range(MAX_DT + 1):
                if dt > 0:
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p = min(p, 0.999)
                    r = UTILITY_ART[cd4]
                    nxt = NEXT_CD4[cd4, 1, dt]
                    ndt = min(dt + 1, MAX_DT)
                    V_new[cd4, dt] = (1 - p) * (r + V[nxt, ndt]) + p * (0.5 * r)
                else:
                    # WAIT
                    p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                    p_w = min(p_w, 0.999)
                    r_w = UTILITY_NO_ART[cd4]
                    nxt_w = NEXT_CD4[cd4, 0, 0]
                    val_w = (1 - p_w) * (r_w + V[nxt_w, 0]) + p_w * (0.5 * r_w)
                    
                    # START
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p_s = min(p_s, 0.999)
                    r_s = UTILITY_ART[cd4]
                    nxt_s = NEXT_CD4[cd4, 1, 1]
                    ndt_s = min(2, MAX_DT)
                    val_s = (1 - p_s) * (r_s + V[nxt_s, ndt_s]) + p_s * (0.5 * r_s)
                    
                    V_new[cd4, 0] = max(val_w, val_s)
        V = V_new
    
    V[7, :] = 0.0
    return V

# ==============================================================================
# Risk-neutral solver (for comparison)
# ==============================================================================

def solve_neutral(terminal_value: Optional[np.ndarray] = None, verbose: bool = True):
    """Standard risk-neutral MDP."""
    if verbose:
        print(f"Risk-Neutral DP: T={T}")
    
    if terminal_value is None:
        terminal_value = np.zeros((N_CD4, MAX_DT + 1), dtype=float)
    V = terminal_value.copy()
    policies = []
    
    for t in range(T - 1, -1, -1):
        V_new = np.zeros_like(V)
        policy = np.zeros((7, MAX_DT + 1), dtype=np.int32)
        p_bg = BG_MORT[t]
        
        for cd4 in range(7):
            for dt in range(MAX_DT + 1):
                if dt > 0:
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p = min(p, 0.999)
                    r = UTILITY_ART[cd4]
                    nxt = NEXT_CD4[cd4, 1, dt]
                    ndt = min(dt + 1, MAX_DT)
                    V_new[cd4, dt] = (1 - p) * (r + V[nxt, ndt]) + p * (0.5 * r)
                    policy[cd4, dt] = 1
                else:
                    # WAIT
                    p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                    p_w = min(p_w, 0.999)
                    r_w = UTILITY_NO_ART[cd4]
                    nxt_w = NEXT_CD4[cd4, 0, 0]
                    val_w = (1 - p_w) * (r_w + V[nxt_w, 0]) + p_w * (0.5 * r_w)
                    
                    # START
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p_s = min(p_s, 0.999)
                    r_s = UTILITY_ART[cd4]
                    nxt_s = NEXT_CD4[cd4, 1, 1]
                    ndt_s = min(2, MAX_DT)
                    val_s = (1 - p_s) * (r_s + V[nxt_s, ndt_s]) + p_s * (0.5 * r_s)
                    
                    if val_s >= val_w:
                        V_new[cd4, 0] = val_s
                        policy[cd4, 0] = 1
                    else:
                        V_new[cd4, 0] = val_w
                        policy[cd4, 0] = 0
        
        V = V_new
        policies.append(policy)
    
    if verbose:
        print(f"  E[R] from CD4=300-400, dt=0: {V[4,0]:.3f}")
    return V, policies[::-1]

# ==============================================================================
# QMDP solver: maximize P(R >= z) for threshold z
# ==============================================================================

def solve_qmdp_threshold(z: float, terminal_value: Optional[np.ndarray] = None) -> Tuple[float, List[np.ndarray]]:
    """
    For threshold z, compute:
    - V_t(x, r) = P(R >= z | state x, cumulative reward r) under optimal policy
    - The policy that maximizes this probability
    
    Returns: (P(R >= z | start), policies)
    """
    if terminal_value is None:
        terminal_value = np.zeros((N_CD4, MAX_DT + 1), dtype=float)
    
    # Terminal condition: V_T(x, r) = 1 if r + terminal_value[x] >= z, else 0
    V = np.zeros((N_CD4, MAX_DT + 1, N_BINS), dtype=float)
    for cd4 in range(N_CD4):
        for dt in range(MAX_DT + 1):
            tv = terminal_value[cd4, dt] if cd4 < 7 else 0.0
            V[cd4, dt, :] = (REWARD_GRID + tv >= z).astype(float)
    
    policies = []
    
    for t in range(T - 1, -1, -1):
        V_new = np.zeros_like(V)
        policy = np.zeros((7, MAX_DT + 1, N_BINS), dtype=np.int32)
        p_bg = BG_MORT[t]
        
        for cd4 in range(7):
            for dt in range(MAX_DT + 1):
                if dt > 0:
                    # On ART: continue
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p = min(p, 0.999)
                    r = UTILITY_ART[cd4]
                    nxt = NEXT_CD4[cd4, 1, dt]
                    ndt = min(dt + 1, MAX_DT)
                    
                    # Survive: get full reward
                    R_surv = REWARD_GRID + r
                    nb_surv = np.clip(np.round(R_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                    prob_surv = V[nxt, ndt, nb_surv]
                    
                    # Die: get half reward, check if r + 0.5*r >= z
                    R_die = REWARD_GRID + 0.5 * r
                    prob_die = (R_die >= z).astype(float)
                    
                    V_new[cd4, dt, :] = (1 - p) * prob_surv + p * prob_die
                    policy[cd4, dt, :] = 1
                else:
                    # WAIT
                    p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                    p_w = min(p_w, 0.999)
                    r_w = UTILITY_NO_ART[cd4]
                    nxt_w = NEXT_CD4[cd4, 0, 0]
                    
                    Rw_surv = REWARD_GRID + r_w
                    nb_w = np.clip(np.round(Rw_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                    prob_w_surv = V[nxt_w, 0, nb_w]
                    Rw_die = REWARD_GRID + 0.5 * r_w
                    prob_w_die = (Rw_die >= z).astype(float)
                    val_w = (1 - p_w) * prob_w_surv + p_w * prob_w_die
                    
                    # START
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p_s = min(p_s, 0.999)
                    r_s = UTILITY_ART[cd4]
                    nxt_s = NEXT_CD4[cd4, 1, 1]
                    ndt_s = min(2, MAX_DT)
                    
                    Rs_surv = REWARD_GRID + r_s
                    nb_s = np.clip(np.round(Rs_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                    prob_s_surv = V[nxt_s, ndt_s, nb_s]
                    Rs_die = REWARD_GRID + 0.5 * r_s
                    prob_s_die = (Rs_die >= z).astype(float)
                    val_s = (1 - p_s) * prob_s_surv + p_s * prob_s_die
                    
                    take_start = (val_s >= val_w)
                    V_new[cd4, 0, :] = np.where(take_start, val_s, val_w)
                    policy[cd4, 0, :] = take_start.astype(np.int32)
        
        V = V_new
        policies.append(policy)
    
    # P(R >= z) from start state (cd4=4, dt=0, r=0)
    prob = float(V[4, 0, 0])
    return prob, policies[::-1]

def solve_qmdp(tau: float, n_thresh: int = 100, terminal_value: Optional[np.ndarray] = None, 
               verbose: bool = True) -> Dict:
    """
    Solve QMDP for quantile level tau.
    
    Find z* = max{z : P(R >= z | pi_z) >= 1 - tau}
    
    Returns dict with:
    - z_star: optimal threshold (= tau-quantile of R under optimal policy)
    - prob: P(R >= z* | pi_z*)
    - policies: optimal policy (depends on cumulative reward bin)
    - all_res: results for all thresholds
    """
    if verbose:
        print(f"QMDP: tau={tau}, T={T}, bins={N_BINS}")
    
    t0 = time.time()
    
    # Search over thresholds
    z_grid = np.linspace(0, MAX_REWARD, n_thresh)
    
    all_res = []
    for zi, z in enumerate(z_grid):
        prob, policies = solve_qmdp_threshold(z, terminal_value)
        all_res.append({"z": z, "prob": prob, "policies": policies})
        
        if verbose and (zi + 1) % max(1, n_thresh // 5) == 0:
            print(f"  [{zi+1:>3}/{n_thresh}] z={z:6.2f}  P(R>=z)={prob:.4f}")
    
    # Find z* = max{z : prob >= 1 - tau}
    target = 1.0 - tau
    z_star = 0.0
    best_idx = 0
    for i, r in enumerate(all_res):
        if r["prob"] >= target - 1e-9:
            z_star = r["z"]
            best_idx = i
    
    if verbose:
        print(f"  Optimal: z*={z_star:.3f} (Q_{tau}), P(R>=z*)={all_res[best_idx]['prob']:.4f}  (time {time.time()-t0:.1f}s)")
    
    return {
        "tau": tau,
        "z_star": z_star,
        "prob": all_res[best_idx]["prob"],
        "policies": all_res[best_idx]["policies"],
        "all_res": all_res
    }

# ==============================================================================
# Simulation
# ==============================================================================

def simulate(policies, n_traj: int = 20000, start_cd4: int = 4, is_qmdp: bool = False,
             seed: int = 0, terminal_value: Optional[np.ndarray] = None) -> np.ndarray:
    """Monte Carlo simulation."""
    rng = np.random.default_rng(seed)
    rets = np.zeros(n_traj, dtype=float)
    
    for i in range(n_traj):
        cd4 = start_cd4
        dt = 0
        R = 0.0
        
        for t in range(T):
            if cd4 == 7:
                break
            
            p_bg = BG_MORT[t]
            on_art = (dt > 0)
            
            if on_art:
                action = 1
            else:
                if is_qmdp:
                    rb = reward_to_bin(R)
                    action = int(policies[t][cd4, dt, rb])
                else:
                    action = int(policies[t][cd4, dt])
            
            if action == 1 and dt == 0:
                dt = 1
                on_art = True
            
            if on_art:
                r = UTILITY_ART[cd4]
                p_bg_adj = bg_mort_on_art(p_bg)
                p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
            else:
                r = UTILITY_NO_ART[cd4]
                p = combine_mortality(DEATH_NO_ART[cd4], p_bg)
            
            p = min(p, 0.999)
            
            if rng.random() < p:
                R += 0.5 * r
                cd4 = 7
                break
            else:
                R += r
                cd4 = NEXT_CD4[cd4, int(on_art), min(dt, MAX_DT)]
                if on_art:
                    dt = min(dt + 1, MAX_DT)
        
        if terminal_value is not None and cd4 != 7:
            R += float(terminal_value[cd4, dt])
        
        rets[i] = R
    
    return rets

# ==============================================================================
# Plotting (to match thesis Figures 2.10 and 2.11)
# ==============================================================================

def plot_qmdp_results(qmdp_results: Dict[float, Dict], neutral_policies, terminal_value,
                      out_path: str = "qmdp_results.png"):
    """
    Create plots matching thesis Figures 2.10 and 2.11.
    """
    cd4_names = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", ">=500"]
    
    taus = sorted(qmdp_results.keys())
    n_taus = len(taus)
    
    # Create custom colormap: gray (delay) to brown/red (start)
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#808080', '#8B4513']  # gray to brown
    cmap = LinearSegmentedColormap.from_list('thesis', colors)
    
    fig = plt.figure(figsize=(14, 10))
    
    # Top row: policy heatmaps for each tau (Figure 2.10 style)
    for idx, tau in enumerate(taus):
        ax = fig.add_subplot(2, n_taus, idx + 1)
        res = qmdp_results[tau]
        
        # Policy at dt=0, Rbin=0
        mat = np.array([p[:, 0, 0] for p in res["policies"]])  # [T, cd4]
        
        # Flip so high CD4 is at top (like thesis)
        mat_flip = mat[:, ::-1]
        
        im = ax.imshow(mat_flip.T, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                      extent=[START_AGE, END_AGE, -0.5, 6.5])
        
        ax.set_xlabel("Age")
        if idx == 0:
            ax.set_ylabel("CD4 Level")
        ax.set_yticks(range(7))
        ax.set_yticklabels(cd4_names[::-1])
        
        label = "Risk Averse" if tau < 0.4 else ("Risk Seeking" if tau > 0.6 else "Less Risk Averse")
        ax.set_title(f"$\\tau = {tau:.2f}$ ({label})")
    
    # Add colorbar legend
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Delay ART", "Start ART"])
    
    # Bottom: Figure 2.11 style - Optimal Quantile Reward curve
    ax = fig.add_subplot(2, 1, 2)
    
    # Simulate MDP (risk-neutral) returns
    rets_neutral = simulate(neutral_policies, n_traj=50000, is_qmdp=False, seed=42, 
                           terminal_value=terminal_value)
    
    # CDF of MDP reward (quantile function)
    rets_sorted = np.sort(rets_neutral)
    cdf = np.arange(1, len(rets_sorted) + 1) / len(rets_sorted)
    ax.plot(cdf, rets_sorted, '--', color='gray', lw=2, label="Cumulative Density Function of MDP Reward")
    
    print("\nComputing optimal quantile curve...")
    
    # Compute P(R >= z) vs z curve for QMDP
    z_vals = np.linspace(35, 70, 80)
    probs = []
    for z in z_vals:
        prob, _ = solve_qmdp_threshold(z, terminal_value)
        probs.append(prob)
    probs = np.array(probs)
    
    # The tau-quantile is z where P(R >= z) = 1 - tau
    # So tau = 1 - P(R >= z)
    tau_from_z = 1.0 - probs
    
    ax.plot(tau_from_z, z_vals, '-', color='#8B4513', lw=2, label="Optimal Quantile Reward (QMDP)")
    
    ax.set_xlabel("Quantiles")
    ax.set_ylabel("QALYs")
    ax.set_xlim(0, 1)
    ax.set_ylim(35, 70)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"\nSaved {out_path}")
    plt.close()

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 72)
    print("QMDP (Quantile MDP) for HIV Treatment Initiation")
    print("=" * 72)
    print(f"Ages {START_AGE}..{END_AGE}, T={T} (6-month).  bins={N_BINS}")
    
    terminal_value = compute_terminal_value()
    print(f"Terminal value computed. V_tail[CD4=300-400,dt=0]={terminal_value[4,0]:.3f}")
    
    # Risk-neutral baseline
    print("\n" + "-" * 60)
    V_n, pol_n = solve_neutral(terminal_value, verbose=True)
    
    # QMDP for different tau values (matching thesis Figure 2.10)
    qmdp_results = {}
    for tau in [0.20, 0.50, 0.80]:
        print("\n" + "-" * 60)
        res = solve_qmdp(tau, n_thresh=80, terminal_value=terminal_value, verbose=True)
        qmdp_results[tau] = res
    
    # Verification via simulation
    print("\n" + "=" * 60)
    print("VERIFICATION (Monte Carlo)")
    print("=" * 60)
    
    rets_n = simulate(pol_n, n_traj=50000, is_qmdp=False, seed=42, terminal_value=terminal_value)
    print(f"\nRisk-Neutral policy:")
    print(f"  Mean: {np.mean(rets_n):.2f}")
    for q in [0.1, 0.2, 0.5, 0.8, 0.9]:
        print(f"  Q_{q}: {np.percentile(rets_n, q*100):.2f}")
    
    for tau, res in qmdp_results.items():
        rets = simulate(res["policies"], n_traj=50000, is_qmdp=True, seed=42, terminal_value=terminal_value)
        q_emp = np.percentile(rets, tau * 100)
        print(f"\nQMDP tau={tau}:")
        print(f"  z* (DP):        {res['z_star']:.2f}")
        print(f"  Q_{tau} (MC):    {q_emp:.2f}")
        print(f"  Mean:           {np.mean(rets):.2f}")
    
    # Plot results
    print("\n" + "-" * 60)
    plot_qmdp_results(qmdp_results, pol_n, terminal_value, out_path="qmdp_results.png")
    
    return pol_n, qmdp_results

if __name__ == "__main__":
    main()

