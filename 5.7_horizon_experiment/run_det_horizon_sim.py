"""
Deterministic-only horizon-sensitivity simulation
==================================================

Pure deterministic rolling-horizon experiment used as the OPENER for
the "Sensitivity to horizon length" thesis subsubsection.  The full
SP-flavoured experiment (DET + 30M + 60M with the 27-scenario tree)
lives in `run_horizon_study.py`; this script is the simpler scaffold
that comes BEFORE it in the thesis to introduce the long-term effect
of horizon length without uncertainty getting in the way.

Setup
-----
* Real horizon            : 120 months (12 years)
* Plan-builder            : `ip.SalmonFarmingMILP` solved each roll
                            under the all-normal expected scenario
                            (the EV plan).  No PH, no scenario tree.
* Planning horizon        : 30 months  vs  60 months
* Commit window           : 12 months   →   10 rolls per (variant × path)
* Realised temperature    : a panel of named paths — 4 archetypes
                            (normal / warm / cold / mixed) plus
                            `n_extra` random uniform-{bad,normal,good}
                            paths under a fixed seed for reproducibility.

Both variants assume all-normal weather inside the planning model,
so the only thing varying is the terminal-value boundary (30 mo vs
60 mo away from the current commit).  This isolates the *pure
horizon effect* on operational decisions.

Output
------
    horizon_runs_det_only/<variant>/<path_name>/result.pkl
        config, rolls (per-roll diagnostics), real_state_log
        (decisions + fwd_log + units_df_before/after), units_df_final.

    Use `plot_det_horizon_sim.py` to produce the cumulative-NPV figure.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RH = os.path.normpath(os.path.join(HERE, "..", "rolling_horizon_experiment"))
for p in (RH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import ip                                                  # noqa: E402
from forward_sim_long import advance_state_long            # noqa: E402


# ---------------------------------------------------------------------------
# Experiment constants
# ---------------------------------------------------------------------------
REAL_HORIZON      = 120
COMMIT            = 12
N_ROLLS           = REAL_HORIZON // COMMIT      # 10
PLAN_HORIZONS     = (30, 60)
LABELS            = ("bad", "normal", "good")
BLOCK_MONTHS      = 12   # one realised label per 12 months


# ---------------------------------------------------------------------------
# Instance loader (the regional fleet from rolling_horizon_experiment/)
# ---------------------------------------------------------------------------

def _load_instance():
    spec = importlib.util.spec_from_file_location(
        "rh_inst", os.path.join(RH, "instance.py"))
    rh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rh)
    if isinstance(rh.loc_mab, dict):
        loc_mab = dict(rh.loc_mab)
    else:
        loc_mab = dict(zip(rh.loc_mab.iloc[:, 0], rh.loc_mab.iloc[:, 1]))
    return {
        "units_df": rh.units_df.copy(),
        "loc_mab":  loc_mab,
        "reg_mab":  float(rh.regional_mab),
        "temp_map": {"normal": rh.temps_normal_12,
                     "bad":    rh.temps_bad_12,
                     "good":   rh.temps_good_12},
        "S_normal": float(rh.S_normal_v),
        "S_bad":    float(rh.S_bad_v),
    }


# ---------------------------------------------------------------------------
# Realised path panel
# ---------------------------------------------------------------------------

ARCHETYPES: List[Tuple[str, List[str]]] = [
    # 10 twelve-month blocks → 120 months
    ("normal", ["normal"] * 10),
    ("warm",   ["normal", "good", "normal", "good", "normal",
                "good",   "normal", "good", "normal", "good"]),
    ("cold",   ["normal", "bad", "normal", "bad", "normal",
                "bad",    "normal", "bad", "normal", "bad"]),
    ("mixed",  ["normal", "good", "bad", "normal", "good",
                "bad",    "normal", "good", "bad", "normal"]),
]


def sample_paths(n_extra: int = 4, seed: int = 0) -> List[Dict]:
    """4 archetypes + n_extra random paths under fixed seed."""
    paths = [{"name": n, "blocks": list(b)} for n, b in ARCHETYPES]
    rng = np.random.default_rng(seed)
    for i in range(n_extra):
        blocks = [str(rng.choice(LABELS)) for _ in range(N_ROLLS)]
        paths.append({"name": f"random_{i:02d}", "blocks": blocks})
    return paths


def _path_to_monthly(path_blocks: List[str]) -> List[str]:
    return [lbl for lbl in path_blocks for _ in range(BLOCK_MONTHS)]


def build_realised(
    path_blocks: List[str], start_calendar: int, inst: Dict
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-month realised temps + survival rates over the 120-month
    real horizon.  Applies the 'good after good → S_bad' rule at
    block-level (same convention as `run_horizon_study.build_full_realized`)."""
    monthly = _path_to_monthly(path_blocks)
    temps   = np.zeros(REAL_HORIZON, dtype=float)
    S       = np.full(REAL_HORIZON, inst["S_normal"], dtype=float)

    cal0 = start_calendar % 12
    prev_block_lbl: str | None = None
    for i, lbl in enumerate(monthly):
        if i % BLOCK_MONTHS == 0:
            S_block = (inst["S_bad"]
                       if (lbl == "good" and prev_block_lbl == "good")
                       else inst["S_normal"])
            prev_block_lbl = lbl
        temps[i] = inst["temp_map"][lbl][(cal0 + i) % 12]
        S[i]     = S_block
    return temps, S


# ---------------------------------------------------------------------------
# Deterministic plan builder + decision extractor
# ---------------------------------------------------------------------------

def _solve_det_milp(
    units_df: pd.DataFrame, loc_mab: Dict, regional_mab: float,
    horizon: int, cal0: int, inst: Dict, mip_gap: float = 0.03,
    time_limit: float | None = None,
):
    n = int(horizon)
    temps = np.array(
        [float(inst["temp_map"]["normal"][(cal0 + t) % 12]) for t in range(n)],
        dtype=float,
    )
    S = np.full(n, inst["S_normal"], dtype=float)
    milp = ip.SalmonFarmingMILP(
        units_df=units_df, temps_t=temps, survival_rates=S,
        loc_mab=loc_mab, regional_mab=regional_mab,
        horizon_months=n, scenario_name="det_normal",
    )
    milp.solve(time_limit=time_limit, mip_gap=mip_gap)
    return milp


def _extract_decisions(milp, n_implement: int) -> Dict:
    n = int(n_implement)
    z, q, h_e, h = {}, {}, {}, {}
    for (u, t), var in milp.variables.get("s", {}).items():
        if int(t) < n:
            z[(u, int(t))] = int(round(float(var.X)))
    for (u, t), var in milp.variables.get("q", {}).items():
        if int(t) < n:
            q[(u, int(t))] = float(var.X)
    for (u, t), var in milp.variables.get("h_exist", {}).items():
        if int(t) < n:
            h_e[(u, int(t))] = int(round(float(var.X)))
    for (u, ss, t), var in milp.variables.get("h", {}).items():
        if int(t) < n:
            h[(u, int(ss), int(t))] = int(round(float(var.X)))
    return {"z": z, "q": q, "h_exist": h_e, "h": h}


# ---------------------------------------------------------------------------
# Per-path runner
# ---------------------------------------------------------------------------

def run_one_path(
    horizon: int, path_spec: Dict, inst: Dict,
    start_calendar: int = 0, mip_gap: float = 0.03,
    time_limit: float | None = None,
) -> Dict:
    real_temps, real_S = build_realised(path_spec["blocks"], start_calendar, inst)
    units_df = inst["units_df"].copy()
    cal      = start_calendar % 12

    rolls: List[Dict]            = []
    real_state_log: List[Dict]   = []

    for k in range(N_ROLLS):
        t0 = k * COMMIT
        t1 = min(t0 + COMMIT, REAL_HORIZON)

        t_solve_0 = time.time()
        milp = _solve_det_milp(units_df, inst["loc_mab"], inst["reg_mab"],
                               horizon, cal, inst,
                               mip_gap=mip_gap, time_limit=time_limit)
        t_solve_1 = time.time()
        decisions = _extract_decisions(milp, COMMIT)

        # Realised forward sim over the commit window.
        temps_w = real_temps[t0:t1]
        S_w     = real_S[t0:t1]
        if len(temps_w) < COMMIT:
            pad = COMMIT - len(temps_w)
            temps_w = np.concatenate([temps_w, np.full(pad, inst["temp_map"]["normal"][0])])
            S_w     = np.concatenate([S_w,     np.full(pad, inst["S_normal"])])
        new_units_df, fwd_log = advance_state_long(
            units_df, decisions, temps_w, S_w
        )
        for ev in fwd_log:
            ev["t_real_offset"] = t0

        rolls.append({
            "roll_idx":     k,
            "t_start":      t0,
            "horizon":      horizon,
            "wallclock_s":  t_solve_1 - t_solve_0,
            "objval":       float(milp.model.ObjVal)
                            if milp.model.SolCount > 0 else float("nan"),
            "calendar_month": cal,
        })
        real_state_log.append({
            "solve_idx":      k,
            "t_start":        t0,
            "fwd_log":        fwd_log,
            "decisions":      decisions,
            "realised_temps": list(map(float, temps_w)),
            "realised_S":     list(map(float, S_w)),
            "units_df_before": units_df.copy(),
            "units_df_after":  new_units_df.copy(),
        })
        units_df = new_units_df
        cal = (cal + COMMIT) % 12

    return {
        "config": {
            "variant":              f"DET-{horizon}M",
            "horizon_months":       horizon,
            "commit_months":        COMMIT,
            "n_solves":             N_ROLLS,           # for compat with cashflow_table
            "n_rolls":              N_ROLLS,
            "real_horizon_months":  REAL_HORIZON,
            "path_name":            path_spec["name"],
            "path_blocks":          path_spec["blocks"],
            "start_calendar_month": start_calendar,
            "mip_gap":              mip_gap,
        },
        "rolls":            rolls,
        "real_state_log":   real_state_log,
        "units_df_final":   units_df,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--save_dir", default=os.path.join(HERE, "horizon_runs_det_only"),
                   help="Output directory for per-(variant,path) result.pkl trees.")
    p.add_argument("--n_extra", type=int, default=4,
                   help="Number of random paths to add on top of the 4 archetypes.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed for the random paths.")
    p.add_argument("--mip_gap", type=float, default=0.03)
    p.add_argument("--time_limit", type=float, default=120.0,
                   help="Per-MILP Gurobi TimeLimit (seconds).  Default 120 s — "
                        "DET MILPs are usually solved in single-digit seconds, "
                        "but this caps the worst case.  Pass 0 for no cap.")
    p.add_argument("--start_calendar_month", type=int, default=0)
    p.add_argument("--variants", default="30,60",
                   help="Comma-separated planning horizons to run (default 30,60).")
    args = p.parse_args()

    inst    = _load_instance()
    paths   = sample_paths(n_extra=args.n_extra, seed=args.seed)
    horizons = [int(h.strip()) for h in args.variants.split(",")]

    print("=" * 74)
    print("DET-only 120-month horizon-sensitivity simulation")
    print("=" * 74)
    print(f"  variants (horizons)   : {horizons}")
    print(f"  paths                 : {[p_['name'] for p_ in paths]}")
    print(f"  rolls per (var, path) : {N_ROLLS}")
    print(f"  commit months         : {COMMIT}")
    print(f"  real horizon          : {REAL_HORIZON} months")
    print(f"  save_dir              : {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)
    overall_t0 = time.time()
    tl = None if args.time_limit and args.time_limit <= 0 else float(args.time_limit)

    for horizon in horizons:
        for path_spec in paths:
            print(f"\n--- DET-{horizon}M  path={path_spec['name']} ---", flush=True)
            t_path_0 = time.time()
            result = run_one_path(horizon, path_spec, inst,
                                  start_calendar=args.start_calendar_month,
                                  mip_gap=args.mip_gap, time_limit=tl)
            t_path_1 = time.time()
            wall = sum(r["wallclock_s"] for r in result["rolls"])
            print(f"  rolls={N_ROLLS}  total solve wallclock={wall:.1f}s  "
                  f"path wallclock={t_path_1 - t_path_0:.1f}s")

            out_dir = os.path.join(args.save_dir, f"det_{horizon}m", path_spec["name"])
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, "result.pkl"), "wb") as fh:
                pickle.dump(result, fh)

    print(f"\nTotal wallclock: {time.time() - overall_t0:.1f}s")
    print(f"Results in:      {args.save_dir}")
    print(f"\nNext: python plot_det_horizon_sim.py")


if __name__ == "__main__":
    main()
