"""
Static CVaR for HIV Treatment Initiation (Bauerle 2011 approach)
================================================================

Implements static (pre-commitment) lower-tail CVaR optimization for the HIV treatment
initiation MDP from Zhong (2020), Appendix A.4, Table A.1.

The CVaR objective uses the Rockafellar-Uryasev representation:

    CVaR_alpha(R) = max_s { s + (1/alpha) * E[min(R - s, 0)] }

where R is the cumulative discounted reward (QALYs). This is solved via an augmented-state
dynamic program where the state includes (CD4 level, ART duration, cumulative reward bin).

Model specifics from the thesis:
  - Death during a period yields half the period's reward (uniform death time assumption)
  - CD4 increase on ART depends on treatment duration (Table A.1)
  - Background mortality uses WHO Life Tables 2016 for US females (cached locally)
  - Combined mortality: p = 1 - (1 - p_hiv)(1 - p_bg)

Reference:
  Bauerle N, Ott J (2011). Markov Decision Processes with Average-Value-at-Risk criteria.
  Mathematical Methods of Operations Research, 74(3):361-379.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Optional, Dict
from pathlib import Path
import json

# ==============================================================================
# THESIS PARAMETERS (Table A.1)
# ==============================================================================

# Alive CD4 level midpoints (and an absorbing "dead" state index 7)
CD4_MIDPOINTS = np.array([25, 75, 150, 250, 350, 450, 750], dtype=float)

# 6-month death probability WITHOUT ART (plus 0 for absorbing dead)
DEATH_NO_ART = np.array([0.1618, 0.0692, 0.0549, 0.0428, 0.0348, 0.0295, 0.0186, 0.0], dtype=float)

# 6-month death probability WITH ART (plus 0 for absorbing dead)
DEATH_ART = np.array([0.1356, 0.0472, 0.0201, 0.0103, 0.0076, 0.0076, 0.0045, 0.0], dtype=float)

# Utility per YEAR; thesis multiplies by 0.5 to get 6‑month utility
UTILITY_NO_ART = np.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.0], dtype=float) * 0.5
UTILITY_ART    = np.array([0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.0], dtype=float) * 0.5

# CD4 decrease without ART (per 6 months)
CD4_DECREASE = 35.25

# Cardiac risk multiplier when on ART
CARDIAC_MULT = 2.0

# ==============================================================================
# Background mortality — WHO Life Tables 2016 (cached) with fallback
# ==============================================================================

_WHO_BG_CACHE_PATH = Path(__file__).with_name("who_bg_mort_6mo_USA_FMLE_2016_smooth.json")
_WHO_BG_CACHE: Optional[Dict[int, float]] = None

def _load_who_cache() -> Optional[Dict[int, float]]:
    """Load WHO mortality cache from disk if available."""
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
    """
    General-population mortality per 6 months.
    Uses WHO Life Tables 2016 (US Female) if cached, else fallback curve.
    """
    a = int(np.floor(age))
    cache = _load_who_cache()
    if cache is not None and a in cache:
        return float(cache[a])
    
    # Fallback: mild increasing curve, clipped for numerical safety
    p = 0.0003 * np.exp(0.085 * (age - 20.0))
    return float(min(p, 0.3))

def combine_mortality(p_hiv: float, p_bg: float) -> float:
    """Independence-style combination: 1 - (1-p_hiv)(1-p_bg)."""
    return float(1.0 - (1.0 - p_hiv) * (1.0 - p_bg))

def bg_mort_on_art(p_bg: float) -> float:
    """
    "Double cardiac death rate" on ART.

    If CARDIAC_MULT is interpreted as hazard multiplier, a reasonable probability transform is:
        p' = 1 - (1 - p)^{CARDIAC_MULT}
    This avoids p' > 1 when p is not tiny.
    """
    return float(1.0 - (1.0 - p_bg) ** CARDIAC_MULT)

# ==============================================================================
# ART duration -> CD4 increase (discrete 6‑month periods)
# ==============================================================================

def cd4_increase_by_period(k: int) -> float:
    """
    k = 1,2,... denotes the *current* 6‑month period index on ART (1 = first period on ART).
    Table A.1 (months) maps to periods as:
      period 1: +100
      period 2: +50
      period 3: +40
      period 4: +40
      period 5: +25
      period 6: +20
      period 7: +20
      period >=8: +0
    """
    if k <= 0:
        return 0.0
    if k == 1: return 100.0
    if k == 2: return 50.0
    if k == 3: return 40.0
    if k == 4: return 40.0
    if k == 5: return 25.0
    if k == 6: return 20.0
    if k == 7: return 20.0
    return 0.0

# ==============================================================================
# Model Config
# ==============================================================================

START_AGE, END_AGE = 20, 90  # thesis uses 20..90
T = (END_AGE - START_AGE) * 2  # 6-month steps

N_CD4 = 8  # includes dead state 7

# dt state is ART period index k in {0,1,2,...,MAX_DT}; dt=0 means not on ART
# After k>=8, cd4_increase is 0, so we can safely cap dt at 8 (dt=8 means ">=8").
MAX_DT = 8

# Reward discretization (augmented state for static CVaR DP)
N_BINS = 320
MAX_REWARD = 80.0  # must exceed plausible total QALYs incl. terminal tail (thesis reaches ~67)
BIN_WIDTH = MAX_REWARD / N_BINS
REWARD_GRID = np.arange(N_BINS, dtype=float) * BIN_WIDTH  # consistent with BIN_WIDTH

# Precompute age-dependent background mortality
AGES = np.array([START_AGE + (t // 2) for t in range(T)], dtype=float)
BG_MORT = np.array([bg_mort(a) for a in AGES], dtype=float)

# CD4 level bins used in the thesis example
def cd4_to_level(cd4: float) -> int:
    bins = [0, 50, 100, 200, 300, 400, 500, 1500]
    for i in range(7):
        if bins[i] <= cd4 < bins[i+1]:
            return i
    return 6  # cap at >500

# Deterministic CD4 transition: NEXT_CD4[cd4_level, on_art(0/1), dt]
NEXT_CD4 = np.zeros((7, 2, MAX_DT + 1), dtype=np.int32)
for cd4 in range(7):
    for on_art in (0, 1):
        for dt in range(MAX_DT + 1):
            curr = CD4_MIDPOINTS[cd4]
            if on_art == 1:
                # dt is ART period index (1..MAX_DT, with MAX_DT meaning ">=MAX_DT")
                k = dt
                inc = cd4_increase_by_period(k)
                new = curr + inc
            else:
                new = max(curr - CD4_DECREASE, 0.0)
            NEXT_CD4[cd4, on_art, dt] = cd4_to_level(new)

def reward_to_bin(R: float) -> int:
    """Map reward to bin using ROUND (not floor) to avoid systematic underestimation."""
    return int(np.clip(np.round(R / BIN_WIDTH), 0, N_BINS - 1))
# ==============================================================================
# Terminal reward beyond END_AGE (thesis uses a terminal reward R^E)
# ------------------------------------------------------------------------------
# Zhong (2020) assigns a terminal reward for patients who reach the terminal age,
# equal to expected remaining lifetime QALYs, estimated by cohort simulation.
#
# We approximate this in a model-consistent way by solving a *risk-neutral* tail DP
# from END_AGE to TAIL_END_AGE, then using its value function as a deterministic
# terminal reward at END_AGE.
# ------------------------------------------------------------------------------

TAIL_END_AGE = 110  # far enough that survival beyond is negligible

def compute_terminal_value_risk_neutral(age0: int = END_AGE, age1: int = TAIL_END_AGE) -> np.ndarray:
    """Return V_tail[cd4, dt] = E[future QALYs | alive at age0, state (cd4,dt)], over [age0, age1]."""
    assert age1 >= age0
    T_tail = (age1 - age0) * 2  # 6-month steps
    if T_tail == 0:
        return np.zeros((N_CD4, MAX_DT + 1), dtype=float)

    ages_tail = np.array([age0 + (t // 2) for t in range(T_tail)], dtype=float)
    bg_tail = np.array([bg_mort(a) for a in ages_tail], dtype=float)

    V = np.zeros((N_CD4, MAX_DT + 1), dtype=float)  # terminal at age1 is 0

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

    # Dead state has 0 continuation value
    V[7, :] = 0.0
    return V


# ==============================================================================
# Solvers
# ==============================================================================

def solve_neutral(verbose: bool = True, terminal_value: Optional[np.ndarray] = None):
    """
    Risk-neutral MDP:
      action=0: WAIT (if dt=0)
      action=1: START/CONTINUE ART
    For dt>0 we force action=1 (continue ART).
    Death-in-period halves the current period reward.
    """
    if verbose:
        print(f"Risk-Neutral DP: ages {START_AGE}..{END_AGE}, T={T}, dt_max={MAX_DT}")
    # V[cd4, dt], where dt in 0..MAX_DT
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
                    # On ART (period index dt)
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p = min(p, 0.999)

                    r = UTILITY_ART[cd4]
                    nxt = NEXT_CD4[cd4, 1, dt]
                    ndt = min(dt + 1, MAX_DT)

                    # If die, get half reward and terminate; else get full reward + future
                    V_new[cd4, dt] = (1 - p) * (r + V[nxt, ndt]) + p * (0.5 * r)
                    policy[cd4, dt] = 1
                else:
                    # Not on ART: choose WAIT vs START

                    # WAIT this period
                    p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                    p_w = min(p_w, 0.999)
                    r_w = UTILITY_NO_ART[cd4]
                    nxt_w = NEXT_CD4[cd4, 0, 0]
                    val_w = (1 - p_w) * (r_w + V[nxt_w, 0]) + p_w * (0.5 * r_w)

                    # START ART this period => ART period index k=1 now; next dt becomes 2 if survive
                    p_bg_adj = bg_mort_on_art(p_bg)
                    p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                    p_s = min(p_s, 0.999)
                    r_s = UTILITY_ART[cd4]

                    nxt_s = NEXT_CD4[cd4, 1, 1]  # apply the first-period CD4 jump
                    ndt_s = min(2, MAX_DT)       # next period index

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
        print(f"  E[R] from start CD4=300-400 (idx=4), dt=0: {V[4,0]:.3f}")
    return V, policies[::-1]

def solve_cvar(alpha: float = 0.1, n_thresh: int = 50, start_cd4: int = 4, verbose: bool = True, terminal_value: Optional[np.ndarray] = None):
    """
    Static lower-tail CVaR via Rockafellar–Uryasev:
      maximize over s:  s + (1/α) E[(R - s)^-]
    Uses an augmented-state DP over (cd4, dt, reward_bin).

    IMPORTANT: The optimal policy *can depend on reward_bin* (pre-commitment CVaR).
    """
    if verbose:
        print(f"Static CVaR DP: α={alpha}, T={T}, bins={N_BINS}, dt_max={MAX_DT}")

    # Search s over a conservative range; widen if needed
    s_grid = np.linspace(0.0, MAX_REWARD, n_thresh)

    best = {"cvar": -np.inf, "s_star": 0.0, "policies": None, "all_res": []}
    t0 = time.time()

    # For vectorization over reward bins
    R_grid = REWARD_GRID  # shape [N_BINS]

    for si, s in enumerate(s_grid):
        # Terminal: V_T = (R - s)^- = min(R - s, 0)
        if terminal_value is None:
            terminal_value = np.zeros((N_CD4, MAX_DT + 1), dtype=float)
        # Terminal: include deterministic terminal reward (tail value) for survivors at END_AGE
        V = np.zeros((N_CD4, MAX_DT + 1, N_BINS), dtype=float)
        V[:, :, :] = np.minimum(R_grid[None, None, :] + terminal_value[:, :, None] - s, 0.0)

        policies = []

        for t in range(T - 1, -1, -1):
            V_new = np.zeros_like(V)
            policy = np.zeros((7, MAX_DT + 1, N_BINS), dtype=np.int32)
            p_bg = BG_MORT[t]

            for cd4 in range(7):
                for dt in range(MAX_DT + 1):

                    if dt > 0:
                        # On ART (period index dt)
                        p_bg_adj = bg_mort_on_art(p_bg)
                        p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                        p = min(p, 0.999)

                        r = UTILITY_ART[cd4]
                        nxt = NEXT_CD4[cd4, 1, dt]
                        ndt = min(dt + 1, MAX_DT)

                        # Survive: full reward, transition (use ROUND binning)
                        R_surv = R_grid + r
                        nb_surv = np.clip(np.round(R_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                        cont = V[nxt, ndt, nb_surv]

                        # Die: half reward, terminate now
                        die_term = np.minimum(R_grid + 0.5 * r - s, 0.0)

                        V_new[cd4, dt, :] = (1 - p) * cont + p * die_term
                        policy[cd4, dt, :] = 1

                    else:
                        # Not on ART: choose WAIT vs START, elementwise over reward bins

                        # WAIT
                        p_w = combine_mortality(DEATH_NO_ART[cd4], p_bg)
                        p_w = min(p_w, 0.999)
                        r_w = UTILITY_NO_ART[cd4]
                        nxt_w = NEXT_CD4[cd4, 0, 0]

                        Rw_surv = R_grid + r_w
                        nb_w = np.clip(np.round(Rw_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                        cont_w = V[nxt_w, 0, nb_w]
                        die_w = np.minimum(R_grid + 0.5 * r_w - s, 0.0)
                        val_w = (1 - p_w) * cont_w + p_w * die_w

                        # START (ART k=1 this period; next dt becomes 2 if survive)
                        p_bg_adj = bg_mort_on_art(p_bg)
                        p_s = combine_mortality(DEATH_ART[cd4], p_bg_adj)
                        p_s = min(p_s, 0.999)
                        r_s = UTILITY_ART[cd4]
                        nxt_s = NEXT_CD4[cd4, 1, 1]
                        ndt_s = min(2, MAX_DT)

                        Rs_surv = R_grid + r_s
                        nb_s = np.clip(np.round(Rs_surv / BIN_WIDTH).astype(np.int32), 0, N_BINS - 1)
                        cont_s = V[nxt_s, ndt_s, nb_s]
                        die_s = np.minimum(R_grid + 0.5 * r_s - s, 0.0)
                        val_s = (1 - p_s) * cont_s + p_s * die_s

                        take_start = (val_s >= val_w)
                        V_new[cd4, 0, :] = np.where(take_start, val_s, val_w)
                        policy[cd4, 0, :] = take_start.astype(np.int32)

            V = V_new
            policies.append(policy)

        shortfall = float(V[start_cd4, 0, 0])
        cvar_val = float(s + (1.0 / alpha) * shortfall)

        best["all_res"].append({"s": float(s), "cvar": cvar_val, "shortfall": shortfall})

        if cvar_val > best["cvar"]:
            best["cvar"] = cvar_val
            best["s_star"] = float(s)
            best["policies"] = policies[::-1]

        if verbose and (si + 1) % max(1, n_thresh // 5) == 0:
            print(f"  [{si+1:>3}/{n_thresh}] s={s:6.2f}  CVaR={cvar_val:8.4f}")

    if verbose:
        print(f"  Optimal: CVaR={best['cvar']:.4f} at s*={best['s_star']:.3f}  (time {time.time()-t0:.1f}s)")

    return best["cvar"], best


# ==============================================================================
# Simulation
# ==============================================================================

def simulate(policies, n_traj: int = 20000, start_cd4: int = 4, is_cvar: bool = False, seed: int = 0, terminal_value: Optional[np.ndarray] = None):
    """
    Monte Carlo simulation under a (time-dependent) policy.

    For CVaR policies: policies[t] has shape [cd4, dt, reward_bin].
    For risk-neutral:  policies[t] has shape [cd4, dt].

    Death-in-period gives half reward and terminates.
    """
    rng = np.random.default_rng(seed)
    rets = np.zeros(n_traj, dtype=float)

    for i in range(n_traj):
        cd4 = start_cd4
        dt = 0  # ART period index (0 = off ART)
        R = 0.0

        for t in range(T):
            if cd4 == 7:
                break

            p_bg = BG_MORT[t]
            on_art = (dt > 0)

            # choose action
            if on_art:
                action = 1
            else:
                if is_cvar:
                    rb = reward_to_bin(R)
                    action = int(policies[t][cd4, dt, rb])
                else:
                    action = int(policies[t][cd4, dt])

            # apply action (start ART at beginning of period)
            if action == 1 and dt == 0:
                dt = 1
                on_art = True

            # reward + death prob for this period
            if on_art:
                r = UTILITY_ART[cd4]
                p_bg_adj = bg_mort_on_art(p_bg)
                p = combine_mortality(DEATH_ART[cd4], p_bg_adj)
            else:
                r = UTILITY_NO_ART[cd4]
                p = combine_mortality(DEATH_NO_ART[cd4], p_bg)

            p = min(p, 0.999)

            # sample death time within period (uniform -> half reward on death)
            if rng.random() < p:
                R += 0.5 * r
                cd4 = 7
                break
            else:
                R += r
                cd4 = NEXT_CD4[cd4, int(on_art), min(dt, MAX_DT)]
                if on_art:
                    dt = min(dt + 1, MAX_DT)

        # Add deterministic terminal reward if the patient reaches END_AGE alive.
        if terminal_value is not None and cd4 != 7:
            R += float(terminal_value[cd4, dt])

        rets[i] = R

    return rets

# ==============================================================================
# Verification helpers
# ==============================================================================

def var_cvar_lower(rets: np.ndarray, alpha: float):
    """Empirical lower-tail VaR and CVaR of rewards."""
    try:
        var = np.quantile(rets, alpha, method="lower")
    except TypeError:
        # older NumPy
        var = np.quantile(rets, alpha, interpolation="lower")
    tail = rets[rets <= var + 1e-12]
    cvar = float(tail.mean()) if len(tail) else float(var)
    return float(var), cvar

def ru_obj(rets: np.ndarray, alpha: float, s: float):
    return float(s + (1.0 / alpha) * np.mean(np.minimum(rets - s, 0.0)))

def verify(result, alpha: float, start_cd4: int, n_mc: int = 30000, seed: int = 1, terminal_value: Optional[np.ndarray] = None):
    print("\n" + "=" * 60)
    print(f"VERIFICATION  α={alpha}")
    print("=" * 60)

    s_star = float(result["s_star"])

    # DP values at s*
    row = min(result["all_res"], key=lambda d: abs(d["s"] - s_star))
    shortfall_dp = float(row["shortfall"])
    cvar_dp = float(row["cvar"])

    # Monte Carlo under the DP-optimal CVaR policy
    rets = simulate(result["policies"], n_traj=n_mc, start_cd4=start_cd4, is_cvar=True, seed=seed, terminal_value=terminal_value)

    var_emp, cvar_emp = var_cvar_lower(rets, alpha)
    shortfall_mc = float(np.mean(np.minimum(rets - s_star, 0.0)))
    cvar_ru = ru_obj(rets, alpha, s_star)

    print("\n1) s* ≈ VaR_α ?")
    print(f"   s* (DP)    : {s_star:.4f}")
    print(f"   VaR_α (MC) : {var_emp:.4f}")
    print(f"   |diff|     : {abs(s_star - var_emp):.4f}")

    print("\n2) DP shortfall ≈ MC shortfall ?   (E[(R-s*)^-])")
    print(f"   DP         : {shortfall_dp:.6f}")
    print(f"   MC         : {shortfall_mc:.6f}")
    print(f"   |diff|     : {abs(shortfall_dp - shortfall_mc):.6f}")

    print("\n3) CVaR values consistent ?")
    print(f"   CVaR (DP)        : {cvar_dp:.4f}")
    print(f"   CVaR (RU @ s*)   : {cvar_ru:.4f}")
    print(f"   Empirical tail   : {cvar_emp:.4f}")

    return {
        "s_star": s_star,
        "var_emp": var_emp,
        "cvar_dp": cvar_dp,
        "cvar_emp": cvar_emp,
        "rets": rets,
    }

# ==============================================================================
# Plotting
# ==============================================================================

def plot_results(policies_neutral, cvar_results, verifications, out_path="cvar_thesis_results.png", terminal_value: Optional[np.ndarray] = None):
    cd4_names = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", ">500"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1: policy heatmaps (slice dt=0, reward_bin=0 for CVaR)
    ax = axes[0, 0]
    mat = np.array([p[:, 0] for p in policies_neutral])  # [T, cd4]
    ax.imshow(mat.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
              extent=[START_AGE, END_AGE, 6.5, -0.5])
    ax.set_xlabel("Age"); ax.set_ylabel("CD4")
    ax.set_yticks(range(7)); ax.set_yticklabels(cd4_names)
    ax.set_title("Risk-Neutral (dt=0 slice)")

    for idx, (alpha, res) in enumerate(list(cvar_results.items())[:2]):
        ax = axes[0, 1 + idx]
        mat = np.array([p[:, 0, 0] for p in res["policies"]])  # dt=0, Rbin=0
        ax.imshow(mat.T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1,
                  extent=[START_AGE, END_AGE, 6.5, -0.5])
        ax.set_xlabel("Age"); ax.set_ylabel("CD4")
        ax.set_yticks(range(7)); ax.set_yticklabels(cd4_names)
        ax.set_title(f"Static CVaR α={alpha}\n(dt=0,Rbin=0, s*={res['s_star']:.2f})")

    # Row 2: CVaR vs s
    ax = axes[1, 0]
    for alpha, res in cvar_results.items():
        ss = [r["s"] for r in res["all_res"]]
        cvs = [r["cvar"] for r in res["all_res"]]
        ax.plot(ss, cvs, "o-", label=f"α={alpha}", markersize=3)
        ax.axvline(res["s_star"], ls="--", alpha=0.5)
    ax.set_xlabel("Threshold s"); ax.set_ylabel("RU objective value")
    ax.set_title("Rockafellar–Uryasev objective vs s"); ax.legend(); ax.grid(alpha=0.3)

    # Reward CDFs
    ax = axes[1, 1]
    rets_n = simulate(policies_neutral, n_traj=20000, is_cvar=False, seed=123, terminal_value=terminal_value)
    sr = np.sort(rets_n)
    ax.plot(sr, np.arange(1, len(sr) + 1) / len(sr), "k-", label="Neutral", lw=2)

    for alpha, ver in verifications.items():
        sr = np.sort(ver["rets"])
        ax.plot(sr, np.arange(1, len(sr) + 1) / len(sr), label=f"α={alpha}", lw=1.5)
        ax.axvline(ver["var_emp"], ls=":", alpha=0.5)

    ax.set_xlabel("Total QALYs"); ax.set_ylabel("CDF")
    ax.set_title("Return CDFs (MC)"); ax.legend(); ax.grid(alpha=0.3)

    # Summary stats bars
    ax = axes[1, 2]
    labels = ["Neutral"] + [f"α={a}" for a in verifications.keys()]
    all_rets = [rets_n] + [v["rets"] for v in verifications.values()]

    means = [float(np.mean(r)) for r in all_rets]
    var10 = [float(np.percentile(r, 10)) for r in all_rets]
    cvar10 = [var_cvar_lower(r, 0.1)[1] for r in all_rets]

    x = np.arange(len(labels))
    ax.bar(x - 0.2, means, 0.2, label="Mean")
    ax.bar(x,        var10, 0.2, label="VaR₀.₁")
    ax.bar(x + 0.2, cvar10, 0.2, label="CVaR₀.₁")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("QALYs"); ax.set_title("Stats"); ax.legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    print(f"\nSaved {out_path}")
    plt.close()

# ==============================================================================
# Main
# ==============================================================================

def main():
    print("=" * 72)
    print("HIV Treatment — Static (Pre-commitment) CVaR on THESIS MODEL (Table A.1)")
    print("=" * 72)
    print(f"Ages {START_AGE}..{END_AGE}, T={T} (6‑month).  dt_max={MAX_DT}.  bins={N_BINS}, MAX_REWARD={MAX_REWARD}")
    print("NOTE: bg_mort(age) is a placeholder unless you replace it with the thesis’s [139] table.\n")

    start_cd4 = 4  # 300–400

    terminal_value = compute_terminal_value_risk_neutral(age0=END_AGE, age1=TAIL_END_AGE)
    print(f"Terminal tail computed over ages {END_AGE}..{TAIL_END_AGE}. Example V_tail[CD4=300-400,dt=0]={terminal_value[4,0]:.3f}")

    print("-" * 60)
    V_n, pol_n = solve_neutral(verbose=True, terminal_value=terminal_value)

    cvar_results = {}
    for alpha in [0.1, 0.2]:
        print("\n" + "-" * 60)
        cvar, res = solve_cvar(alpha=alpha, n_thresh=60, start_cd4=start_cd4, verbose=True, terminal_value=terminal_value)
        res["cvar"] = cvar
        cvar_results[alpha] = res

    print("\n" + "-" * 60)
    print("Monotonicity check: CVaR^- should (typically) increase with α")
    for a in sorted(cvar_results.keys()):
        print(f"  CVaR^-_{a} = {cvar_results[a]['cvar']:.4f}")

    verifications = {}
    for alpha, res in cvar_results.items():
        verifications[alpha] = verify(res, alpha, start_cd4, n_mc=40000, seed=10 + int(alpha * 100), terminal_value=terminal_value)

    print("\n" + "-" * 60)
    plot_results(pol_n, cvar_results, verifications, out_path="cvar_thesis_results.png", terminal_value=terminal_value)

    return pol_n, cvar_results, verifications

if __name__ == "__main__":
    main()
