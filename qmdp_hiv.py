"""
QMDP (Quantile MDP) for HIV Treatment Initiation
=================================================

Implements the Quantile MDP approach from Zhong (2020) for the HIV treatment
initiation problem. QMDP maximizes the tau-quantile of cumulative reward.

This version uses a long horizon (age 20-110) to avoid terminal reward artifacts
and Numba for performance.

NOTE: Using Negoescu et al. (2012) parameters which match thesis policy patterns.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import json
from numba import njit

# ==============================================================================
# Model parameters
# ==============================================================================

# Choose parameter set: "negoescu" (original 2012 paper) or "zhong" (thesis Table A.1)
PARAM_SET = "negoescu"

if PARAM_SET == "negoescu":
    # Negoescu et al. 2012 original
    CD4_MIDPOINTS = np.array([25, 75, 150, 275, 425, 575, 750], dtype=float)
    DEATH_NO_ART = np.array([0.1005, 0.02, 0.0108, 0.006, 0.0016, 0.001, 0.0008, 0.0], dtype=float)
    DEATH_ART = np.array([0.0167, 0.0119, 0.0085, 0.0039, 0.0012, 0.001, 0.0008, 0.0], dtype=float)
    UTILITY_NO_ART = np.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.0], dtype=float) * 0.5
    UTILITY_ART = np.array([0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.0], dtype=float) * 0.5
else:
    # Zhong thesis Table A.1
    CD4_MIDPOINTS = np.array([25, 75, 150, 250, 350, 450, 750], dtype=float)
    DEATH_NO_ART = np.array([0.1618, 0.0692, 0.0549, 0.0428, 0.0348, 0.0295, 0.0186, 0.0], dtype=float)
    DEATH_ART = np.array([0.1356, 0.0472, 0.0201, 0.0103, 0.0076, 0.0076, 0.0045, 0.0], dtype=float)
    UTILITY_NO_ART = np.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.0], dtype=float) * 0.5
    UTILITY_ART = np.array([0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.0], dtype=float) * 0.5

CD4_DECREASE = 35.25
CARDIAC_MULT = 2.0

# Model config
START_AGE = 20
END_AGE = 110  # Long horizon to avoid jump artifacts
T = (END_AGE - START_AGE) * 2  # 180 periods
N_CD4 = 8
MAX_DT = 8

# Reward discretization
N_BINS = 600
MAX_REWARD = 80.0
BIN_WIDTH = MAX_REWARD / N_BINS
REWARD_GRID = np.arange(N_BINS, dtype=float) * BIN_WIDTH

# ==============================================================================
# Background mortality
# ==============================================================================

_WHO_BG_CACHE_PATH = Path(__file__).with_name("who_bg_mort_6mo_USA_FMLE_2016_smooth.json")
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
    # Fallback
    p = 0.0003 * np.exp(0.085 * (age - 20.0))
    return float(min(p, 0.5))

def combine_mortality(p_hiv: float, p_bg: float) -> float:
    return float(1.0 - (1.0 - p_hiv) * (1.0 - p_bg))

def bg_mort_on_art(p_bg: float) -> float:
    return float(1.0 - (1.0 - p_bg) ** CARDIAC_MULT)

# Precompute
AGES = np.array([START_AGE + (t // 2) for t in range(T)], dtype=float)
BG_MORT = np.array([bg_mort(a) for a in AGES], dtype=float)

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
# Risk-neutral solver
# ==============================================================================

def solve_neutral(verbose: bool = True):
    V = np.zeros((N_CD4, MAX_DT + 1), dtype=float)
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
        print(f"Risk-Neutral E[R] from CD4=300-400: {V[4,0]:.3f}")
    return V, policies[::-1]

# ==============================================================================
# QMDP solver: maximize P(R >= z)
# ==============================================================================

@njit
def qmdp_bellman_backup(V_next, p_die, p_surv, r, z, bins, bin_width, next_cd4, next_dt):
    val = np.zeros(bins, dtype=np.float64)
    for b in range(bins):
        R = b * bin_width
        # Die this period: success if R + 0.5*r >= z
        prob_die = 1.0 if R + 0.5 * r >= z else 0.0
        
        # Survive this period: success prob from V_next using interpolation
        # Success prob at R_next = R + r
        R_next = R + r
        idx_f = R_next / bin_width
        idx = int(idx_f)
        frac = idx_f - idx
        
        if idx >= bins - 1:
            prob_surv = V_next[next_cd4, next_dt, bins - 1]
        elif idx < 0:
            prob_surv = V_next[next_cd4, next_dt, 0]
        else:
            # Linear interpolation
            v0 = V_next[next_cd4, next_dt, idx]
            v1 = V_next[next_cd4, next_dt, idx + 1]
            prob_surv = v0 + frac * (v1 - v0)
            
        val[b] = p_surv * prob_surv + p_die * prob_die
    return val

def solve_qmdp_threshold(z: float) -> Tuple[float, List[np.ndarray]]:
    V = np.zeros((N_CD4, MAX_DT + 1, N_BINS), dtype=float)
    # Terminal condition at END_AGE=110: 1 if r >= z
    V[:, :, :] = (REWARD_GRID[None, None, :] >= z).astype(float)
    
    policies = []
    for t in range(T - 1, -1, -1):
        V_new = np.zeros_like(V)
        policy = np.zeros((7, MAX_DT + 1, N_BINS), dtype=np.int32)
        p_bg = BG_MORT[t]
        
        for cd4 in range(7):
            for dt in range(MAX_DT + 1):
                if dt > 0:
                    # Continue ART
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    V_new[cd4, dt, :] = qmdp_bellman_backup(V, p, 1-p, UTILITY_ART[cd4], z, 
                                                           N_BINS, BIN_WIDTH, 
                                                           NEXT_CD4[cd4, 1, dt], 
                                                           min(dt + 1, MAX_DT))
                    policy[cd4, dt, :] = 1
                else:
                    # WAIT
                    p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                    val_w = qmdp_bellman_backup(V, p_w, 1-p_w, UTILITY_NO_ART[cd4], z, 
                                               N_BINS, BIN_WIDTH, NEXT_CD4[cd4, 0, 0], 0)
                    # START
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    val_s = qmdp_bellman_backup(V, p_s, 1-p_s, UTILITY_ART[cd4], z, 
                                               N_BINS, BIN_WIDTH, NEXT_CD4[cd4, 1, 1], 2)
                    
                    take_start = (val_s >= val_w)
                    V_new[cd4, 0, :] = np.where(take_start, val_s, val_w)
                    policy[cd4, 0, :] = take_start.astype(np.int32)
        V = V_new
        policies.append(policy)
    
    return float(V[4, 0, 0]), policies[::-1]

def solve_qmdp(tau: float, n_thresh: int = 100, verbose: bool = True) -> Dict:
    if verbose: print(f"QMDP tau={tau}, T={T}")
    t0 = time.time()
    z_grid = np.linspace(0, MAX_REWARD, n_thresh)
    all_res = []
    for zi, z in enumerate(z_grid):
        prob, policies = solve_qmdp_threshold(z)
        all_res.append({"z": z, "prob": prob, "policies": policies})
        if verbose and (zi + 1) % max(1, n_thresh // 5) == 0:
            print(f"  [{zi+1:>3}/{n_thresh}] z={z:6.2f} P(R>=z)={prob:.4f}")
    
    target = 1.0 - tau
    z_star, best_idx = 0.0, 0
    for i, r in enumerate(all_res):
        if r["prob"] >= target - 1e-9:
            z_star, best_idx = r["z"], i
    
    if verbose: print(f"  z*={z_star:.3f} (time {time.time()-t0:.1f}s)")
    return {"tau": tau, "z_star": z_star, "policies": all_res[best_idx]["policies"], "all_res": all_res}

# ==============================================================================
# Simulation
# ==============================================================================

def simulate(policies, n_traj: int = 50000, start_cd4: int = 4, is_qmdp: bool = False, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rets = np.zeros(n_traj, dtype=float)
    
    for i in range(n_traj):
        cd4, dt, R = start_cd4, 0, 0.0
        for t in range(T):
            if cd4 == 7: break
            p_bg = BG_MORT[t]
            on_art = (dt > 0)
            if on_art:
                action = 1
            else:
                if is_qmdp:
                    action = int(policies[t][cd4, dt, reward_to_bin(R)])
                else:
                    action = int(policies[t][cd4, dt])
            
            if action == 1 and dt == 0:
                dt, on_art = 1, True
            
            if on_art:
                r, p = UTILITY_ART[cd4], combine_mortality(DEATH_ART[cd4], bg_mort_on_art(p_bg))
            else:
                r, p = UTILITY_NO_ART[cd4], combine_mortality(DEATH_NO_ART[cd4], p_bg)
            
            if rng.random() < min(p, 0.999):
                R += 0.5 * r
                cd4 = 7
                break
            else:
                R += r
                cd4 = NEXT_CD4[cd4, int(on_art), min(dt, MAX_DT)]
                if on_art: dt = min(dt + 1, MAX_DT)
        rets[i] = R
    return rets

# ==============================================================================
# Plotting
# ==============================================================================

def plot_qmdp_results(qmdp_results, neutral_policies, out_path="qmdp_results.png"):
    taus = sorted(qmdp_results.keys())
    cd4_names = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", ">=500"]
    
    from matplotlib.colors import LinearSegmentedColormap
    colors = ['#8B2323', '#90EE90']  # red Delay, green Start
    cmap = LinearSegmentedColormap.from_list('thesis', colors)
    
    fig = plt.figure(figsize=(14, 10))
    for idx, tau in enumerate(taus):
        ax = fig.add_subplot(2, len(taus), idx + 1)
        res = qmdp_results[tau]
        # Only plot up to age 90 to match thesis
        T_90 = (90 - START_AGE) * 2
        mat = np.array([p[:, 0, 0] for p in res["policies"][:T_90]])
        im = ax.imshow(mat.T, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                      extent=[START_AGE, 90, -0.5, 6.5], origin='lower')
        ax.set_xlabel("Age")
        if idx == 0: ax.set_ylabel("CD4 Level")
        ax.set_yticks(range(7))
        ax.set_yticklabels(cd4_names)
        label = "Risk Averse" if tau < 0.4 else ("Risk Seeking" if tau > 0.6 else "Less Risk Averse")
        ax.set_title(f"$\\tau = {tau:.2f}$ ({label})")
    
    cbar_ax = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Delay ART", "Start ART"])
    
    ax = fig.add_subplot(2, 1, 2)
    rets_n = simulate(neutral_policies, is_qmdp=False, seed=42)
    rets_sorted = np.sort(rets_n)
    cdf = np.arange(1, len(rets_sorted) + 1) / len(rets_sorted)
    ax.plot(cdf, rets_sorted, '--', color='gray', lw=2, label="Cumulative Density Function of MDP Reward")
    
    print("\nComputing optimal quantile curve...")
    z_vals = np.linspace(35, 75, 100)
    probs = [solve_qmdp_threshold(z)[0] for z in z_vals]
    ax.plot(1.0 - np.array(probs), z_vals, '-', color='#8B2323', lw=2, label="Optimal Quantile Reward (QMDP)")
    
    ax.set_xlabel("Quantiles")
    ax.set_ylabel("QALYs")
    ax.set_xlim(0, 1)
    ax.set_ylim(35, 75)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f"\nSaved {out_path}")
    plt.close()

def main():
    print("=" * 72)
    print("QMDP (Quantile MDP) for HIV Treatment Initiation (Long Horizon)")
    print("=" * 72)
    
    _, pol_n = solve_neutral(verbose=True)
    qmdp_results = {tau: solve_qmdp(tau, n_thresh=200) for tau in [0.20, 0.50, 0.80]}
    
    print("\nVERIFICATION (Monte Carlo)")
    rets_n = simulate(pol_n, seed=42)
    print(f"\nNeutral policy mean: {np.mean(rets_n):.2f}, Q_0.2: {np.percentile(rets_n, 20):.2f}")
    
    for tau, res in qmdp_results.items():
        rets = simulate(res["policies"], is_qmdp=True, seed=42)
        print(f"QMDP tau={tau} MC Q_{tau}: {np.percentile(rets, tau*100):.2f}, mean: {np.mean(rets):.2f}")
    
    plot_qmdp_results(qmdp_results, pol_n)

if __name__ == "__main__":
    main()
