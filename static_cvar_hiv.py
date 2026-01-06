"""
Static CVaR for HIV Treatment Initiation (Bauerle 2011 approach) - JAX Version
==============================================================================

Implements static (pre-commitment) lower-tail CVaR optimization for the HIV treatment
initiation MDP using JAX for high-performance vectorization.
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
from typing import Optional, Dict
from pathlib import Path
import json

# ==============================================================================
# Model parameters (Zhong Table A.1)
# ==============================================================================

CD4_MIDPOINTS = jnp.array([25, 75, 150, 250, 350, 450, 750], dtype=jnp.float32)
DEATH_NO_ART = jnp.array([0.1618, 0.0692, 0.0549, 0.0428, 0.0348, 0.0295, 0.0186, 0.0], dtype=jnp.float32)
DEATH_ART = jnp.array([0.1356, 0.0472, 0.0201, 0.0103, 0.0076, 0.0076, 0.0045, 0.0], dtype=jnp.float32)
UTILITY_NO_ART = jnp.array([0.82, 0.83, 0.84, 0.85, 0.86, 0.87, 0.88, 0.0], dtype=jnp.float32) * 0.5
UTILITY_ART = jnp.array([0.72, 0.75, 0.78, 0.81, 0.84, 0.87, 0.90, 0.0], dtype=jnp.float32) * 0.5

CD4_DECREASE = 35.25
CARDIAC_MULT = 2.0

START_AGE, END_AGE = 20, 90
T = (END_AGE - START_AGE) * 2
TAIL_END_AGE = 110
T_FULL = (TAIL_END_AGE - START_AGE) * 2

N_CD4 = 8
MAX_DT = 8

N_BINS = 320
MAX_REWARD = 80.0
BIN_WIDTH = MAX_REWARD / N_BINS
REWARD_GRID = jnp.arange(N_BINS, dtype=jnp.float32) * BIN_WIDTH

# ==============================================================================
# Background mortality
# ==============================================================================

_WHO_BG_CACHE_PATH = Path(__file__).with_name("who_bg_mort_6mo_USA_FMLE_2016_smooth.json")

def load_bg_mort():
    if _WHO_BG_CACHE_PATH.exists():
        with open(_WHO_BG_CACHE_PATH) as f:
            cache = json.load(f)
        return jnp.array([cache.get(str(int(START_AGE + t/2)), 0.001) for t in range(T_FULL)], dtype=jnp.float32)
    ages = jnp.arange(T_FULL) / 2.0 + START_AGE
    return jnp.minimum(0.0003 * jnp.exp(0.085 * (ages - 20.0)), 0.5)

BG_MORT = load_bg_mort()

# ==============================================================================
# CD4 Transitions
# ==============================================================================

def get_cd4_increase(dt):
    return jnp.where(dt == 1, 100.0,
           jnp.where(dt == 2, 50.0,
           jnp.where((dt == 3) | (dt == 4), 40.0,
           jnp.where(dt == 5, 25.0,
           jnp.where((dt == 6) | (dt == 7), 20.0, 0.0)))))

def get_next_cd4_idx(cd4_idx, on_art, dt):
    curr_val = CD4_MIDPOINTS[cd4_idx]
    inc = jnp.where(on_art, get_cd4_increase(dt), -CD4_DECREASE)
    new_val = jnp.maximum(curr_val + inc, 0.0)
    bins = jnp.array([0, 50, 100, 200, 300, 400, 500, 1500])
    idx = jnp.searchsorted(bins, new_val, side='right') - 1
    return jnp.clip(idx, 0, 6)

CD4_INDICES = jnp.arange(7)
ON_ART_FLAGS = jnp.array([0, 1])
DT_INDICES = jnp.arange(MAX_DT + 1)

@jit
def compute_transition_table():
    def single_trans(c, oa, d): return get_next_cd4_idx(c, oa, d)
    return jnp.transpose(vmap(vmap(vmap(single_trans, (0, None, None)), (None, 0, None)), (None, None, 0))(CD4_INDICES, ON_ART_FLAGS, DT_INDICES), (2, 1, 0))

NEXT_CD4_TABLE = compute_transition_table()

# ==============================================================================
# Solvers
# ==============================================================================

@jit
def cvar_step(V_next, p_bg, s):
    p_bg_art = 1.0 - (1.0 - p_bg)**CARDIAC_MULT
    
    def get_val(cd4, dt, on_art):
        p_hiv = jnp.where(on_art, DEATH_ART[cd4], DEATH_NO_ART[cd4])
        p_base = jnp.where(on_art, p_bg_art, p_bg)
        p = jnp.clip(1.0 - (1.0 - p_hiv) * (1.0 - p_base), 0.0, 0.999)
        r = jnp.where(on_art, UTILITY_ART[cd4], UTILITY_NO_ART[cd4])
        
        nxt_c = NEXT_CD4_TABLE[cd4, jnp.where(on_art, 1, 0), dt]
        ndt = jnp.where(on_art, jnp.minimum(dt + 1, MAX_DT), 0)
        
        R_surv = REWARD_GRID + r
        coords = jnp.array([(R_surv / BIN_WIDTH)])
        val_surv = jax.scipy.ndimage.map_coordinates(V_next[nxt_c, ndt], coords, order=1, mode='nearest')
        val_die = jnp.minimum(REWARD_GRID + 0.5 * r - s, 0.0)
        return (1 - p) * val_surv + p * val_die

    v_on = vmap(vmap(get_val, (0, None, None)), (None, 0, None))(CD4_INDICES, jnp.arange(1, MAX_DT + 1), True)
    vw, vs = vmap(get_val, (0, None, None))(CD4_INDICES, 0, False), vmap(get_val, (0, None, None))(CD4_INDICES, 0, True)
    
    take_start = (vs >= vw)
    V_new = jnp.zeros((N_CD4, MAX_DT + 1, N_BINS))
    V_new = V_new.at[:7, 1:].set(v_on.transpose(1, 0, 2))
    V_new = V_new.at[:7, 0].set(jnp.where(take_start, vs, vw))
    
    policy = jnp.zeros((7, MAX_DT + 1, N_BINS), dtype=jnp.int32)
    policy = policy.at[:, 1:].set(1)
    policy = policy.at[:, 0].set(take_start.astype(jnp.int32))
    return V_new, policy

@jit
def solve_cvar_for_threshold(s, terminal_value):
    V_term = jnp.minimum(REWARD_GRID[None, None, :] + terminal_value[:, :, None] - s, 0.0)
    def scan_fn(carry, p_bg): 
        Vn, pol = cvar_step(carry, p_bg, s)
        return Vn, pol
    final_V, policies = jax.lax.scan(scan_fn, V_term, BG_MORT[:T][::-1])
    return final_V[4, 0, 0], policies[::-1]

@jit
def neutral_step(V, p_bg):
    p_bg_art = 1.0 - (1.0 - p_bg)**CARDIAC_MULT
    def get_val(cd4, dt, on_art):
        p_hiv = jnp.where(on_art, DEATH_ART[cd4], DEATH_NO_ART[cd4])
        p_base = jnp.where(on_art, p_bg_art, p_bg)
        p = jnp.clip(1.0 - (1.0 - p_hiv) * (1.0 - p_base), 0.0, 0.999)
        r = jnp.where(on_art, UTILITY_ART[cd4], UTILITY_NO_ART[cd4])
        nxt_c = NEXT_CD4_TABLE[cd4, jnp.where(on_art, 1, 0), dt]
        ndt = jnp.where(on_art, jnp.minimum(dt + 1, MAX_DT), 0)
        return (1 - p) * (r + V[nxt_c, ndt]) + p * 0.5 * r
    v_on = vmap(vmap(get_val, (0, None, None)), (None, 0, None))(CD4_INDICES, jnp.arange(1, MAX_DT+1), True)
    vw, vs = vmap(get_val, (0, None, None))(CD4_INDICES, 0, False), vmap(get_val, (0, None, None))(CD4_INDICES, 0, True)
    V_new = jnp.zeros((N_CD4, MAX_DT+1))
    V_new = V_new.at[:7, 1:].set(v_on.transpose(1, 0)); V_new = V_new.at[:7, 0].set(jnp.maximum(vw, vs))
    pol = jnp.ones((7, MAX_DT+1), dtype=jnp.int32); pol = pol.at[:, 0].set((vs >= vw).astype(jnp.int32))
    return V_new, pol

@jit
def solve_neutral(tv):
    def scan_fn(V, p_bg): return neutral_step(V, p_bg)
    final_V, pols = jax.lax.scan(scan_fn, tv, BG_MORT[:T][::-1])
    return pols[::-1]

@jit
def compute_terminal_value():
    p_bg = BG_MORT[T:]
    V = jnp.zeros((N_CD4, MAX_DT + 1))
    def scan_fn(V, pb): 
        Vn, _ = neutral_step(V, pb)
        return Vn, None
    final_V, _ = jax.lax.scan(scan_fn, V, p_bg[::-1])
    return final_V

@partial(jit, static_argnums=(2, 3))
def simulate_jax(policies, key, n_traj=40000, is_cvar=True, tv=None):
    def body_fn(carry, t):
        cd4, dt, R, alive, key = carry
        key, subkey = jax.random.split(key)
        on_art = (dt > 0)
        p_bg = BG_MORT[t]
        def get_action():
            if is_cvar:
                rb = jnp.clip(jnp.round(R / BIN_WIDTH).astype(jnp.int32), 0, N_BINS - 1)
                return policies[t, cd4, dt, rb]
            else: return policies[t, cd4, dt]
        action = jax.lax.cond(alive & (cd4 < 7), get_action, lambda: 1)
        eff_oa = on_art | (action == 1); c_dt = jnp.where(on_art, dt, jnp.where(action == 1, 1, 0))
        p = jnp.clip(1.0 - (1.0 - jnp.where(eff_oa, DEATH_ART[cd4], DEATH_NO_ART[cd4])) * (1.0 - jnp.where(eff_oa, 1.0 - (1.0 - p_bg)**CARDIAC_MULT, p_bg)), 0.0, 0.999)
        u = jnp.where(eff_oa, UTILITY_ART[cd4], UTILITY_NO_ART[cd4])
        die = jax.random.uniform(subkey) < p
        new_R = R + jnp.where(alive & (cd4 < 7), jnp.where(die, 0.5 * u, u), 0.0)
        new_alive = alive & ~die
        new_c = jnp.where(new_alive, NEXT_CD4_TABLE[cd4, jnp.where(eff_oa, 1, 0), c_dt], 7)
        new_d = jnp.where(eff_oa, jnp.minimum(c_dt + 1, MAX_DT), 0)
        return (new_c, new_d, new_R, new_alive, key), None
    init_cd4, init_dt, init_R, init_alive = jnp.full((n_traj,), 4, dtype=jnp.int32), jnp.zeros((n_traj,), dtype=jnp.int32), jnp.zeros((n_traj,), dtype=jnp.float32), jnp.ones((n_traj,), dtype=jnp.bool_)
    keys = jax.random.split(key, n_traj)
    def single_traj(c, d, r, a, k):
        (fc, fd, fr, fa, fk), _ = jax.lax.scan(body_fn, (c, d, r, a, k), jnp.arange(T))
        return fr + jax.lax.cond(fa & (fc < 7), lambda: tv[fc, fd], lambda: 0.0)
    return vmap(single_traj)(init_cd4, init_dt, init_R, init_alive, keys)

def main():
    print("Static CVaR HIV - JAX Version")
    tv = compute_terminal_value()
    neutral_pols = solve_neutral(tv)
    
    cvar_results = {}
    for alpha in [0.1, 0.2]:
        print(f"Solving alpha={alpha}...")
        s_grid = jnp.linspace(0, 80, 60)
        shortfalls, all_pols = vmap(solve_cvar_for_threshold, (0, None))(s_grid, tv)
        cvars = s_grid + (1.0/alpha) * shortfalls
        best_idx = jnp.argmax(cvars)
        cvar_results[alpha] = {"cvar": float(cvars[best_idx]), "s_star": float(s_grid[best_idx]), "policies": all_pols[best_idx], "all_res": [{"s": float(s), "cvar": float(c)} for s, c in zip(s_grid, cvars)]}

    # Plotting
    cd4_names = ["0-50", "50-100", "100-200", "200-300", "300-400", "400-500", ">500"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Neutral
    axes[0,0].imshow(np.array(neutral_pols)[:,:,0].T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, extent=[START_AGE, END_AGE, 6.5, -0.5])
    axes[0,0].set_title("Risk-Neutral"); axes[0,0].set_yticks(range(7)); axes[0,0].set_yticklabels(cd4_names)
    
    for idx, alpha in enumerate([0.1, 0.2]):
        axes[0, 1+idx].imshow(np.array(cvar_results[alpha]["policies"])[:,:,0,0].T, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, extent=[START_AGE, END_AGE, 6.5, -0.5])
        axes[0, 1+idx].set_title(f"CVaR alpha={alpha}"); axes[0,1+idx].set_yticks(range(7)); axes[0,1+idx].set_yticklabels(cd4_names)
    
    # Row 2
    for alpha, res in cvar_results.items(): axes[1,0].plot([r["s"] for r in res["all_res"]], [r["cvar"] for r in res["all_res"]], label=f"a={alpha}")
    axes[1,0].set_title("RU Objective"); axes[1,0].legend()
    
    key = jax.random.PRNGKey(42)
    rets_n = simulate_jax(neutral_pols, key, is_cvar=False, tv=tv)
    sr = np.sort(np.array(rets_n)); axes[1,1].plot(sr, np.arange(1, len(sr)+1)/len(sr), "k-", label="Neutral")
    
    for alpha, res in cvar_results.items():
        rets = simulate_jax(res["policies"], key, is_cvar=True, tv=tv)
        sr = np.sort(np.array(rets)); axes[1,1].plot(sr, np.arange(1, len(sr)+1)/len(sr), label=f"a={alpha}")
    axes[1,1].set_title("Return CDFs"); axes[1,1].legend()
    
    plt.tight_layout(); plt.savefig("cvar_thesis_results.png"); print("Saved cvar_thesis_results.png")

if __name__ == "__main__": main()
