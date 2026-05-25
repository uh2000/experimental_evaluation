"""
Rolling-horizon driver for the salmon-farming SP experiment.

Each roll solves a 30-month, 27-scenario stochastic programme (3 stages,
3-month deterministic prefix), implements the first 3 months of decisions
under a *realised* prefix temperature, then shifts the horizon 3 months
forward and re-solves.

Usage
-----
    from instance import units_df, loc_mab, regional_mab, temp_map, S_normal_v, S_bad_v
    from rolling_horizon import run_rolling_horizon, REALITY_PATHS

    result = run_rolling_horizon(
        units_df0=units_df,
        loc_mab=loc_mab,
        regional_mab=regional_mab,
        reality_path=REALITY_PATHS["all_normal"],
        save_dir="rh_runs/all_normal",
        ph_kwargs=dict(K=200, mip_gap=0.02),
    )

The returned dict (also written to disk under `save_dir/`) contains:

    {
        "roll_logs": [ {roll_idx, prefix_label, calendar_month,
                        units_df_before, decisions, log, ph_obj,
                        scenario_plans}, ... ],
        "config": {...},
    }

`scenario_plans[roll_idx]` is a dataframe of all 27 planned trajectories
(z, h, h_exist consensus values per node + scenario-specific stage-3
decisions) suitable for the "fan-chart" visualisation.

Reality paths
-------------
Five hand-picked 12-step sequences cover the qualitative regimes:
    all_normal           N N N N N N N N N N N N
    warming              N N N G G N N G G N G G
    cooling              N N B B N N B B N B B B
    oscillating          N B G N B G N B G N B G
    stress_then_recover  B B B N N N G G G N N N
"""
from __future__ import annotations

import json
import os
import pickle
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# These imports are local to avoid pulling Gurobi for unit tests
# (sp.py imports gurobipy at module load; rolling_horizon does it lazily).


# ---------------------------------------------------------------------------
# Reality paths
# ---------------------------------------------------------------------------

REALITY_PATHS: Dict[str, List[str]] = {
    "all_normal":          ["normal"] * 12,
    "warming":             ["normal", "normal", "normal", "good", "good", "normal",
                            "normal", "good", "good", "normal", "good", "good"],
    "cooling":             ["normal", "normal", "bad", "bad", "normal", "normal",
                            "bad", "bad", "normal", "bad", "bad", "bad"],
    "oscillating":         ["normal", "bad", "good", "normal", "bad", "good",
                            "normal", "bad", "good", "normal", "bad", "good"],
    "stress_then_recover": ["bad", "bad", "bad", "normal", "normal", "normal",
                            "good", "good", "good", "normal", "normal", "normal"],
}


# ---------------------------------------------------------------------------
# One-roll PH solve helper
# ---------------------------------------------------------------------------

def _solve_one_roll(
    *,
    units_df: pd.DataFrame,
    loc_mab: dict,
    regional_mab: float,
    horizon_months: int,
    prefix_months: List[int],
    stage_slices: List[List[int]],
    prefix_label: str,
    start_calendar_month: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    ph_kwargs: Optional[dict] = None,
):
    """Build & solve a 30-month, 27-scenario SP. Returns the solved
    `AugmentedLagrangianDecomposition` instance."""
    # Lazy import — pulls in gurobipy.
    from sp import AugmentedLagrangianDecomposition

    ph_kwargs = dict(ph_kwargs or {})

    ald = AugmentedLagrangianDecomposition(
        units_df=units_df,
        loc_mab=loc_mab,
        regional_mab=regional_mab,
        T=horizon_months,
        temps_normal=temp_map["normal"],
        temps_bad=temp_map["bad"],
        temps_good=temp_map["good"],
        S_normal=S_normal,
        S_bad=S_bad,
        prefix_months=prefix_months,
        prefix_label=prefix_label,
        stage_slices=stage_slices,
        start_calendar_month=start_calendar_month,
        **ph_kwargs,
    )
    ald.build()
    ald.solve()
    return ald


# ---------------------------------------------------------------------------
# Scenario-plan extraction (for fan-chart visualisations)
# ---------------------------------------------------------------------------

def _extract_scenario_plans(ald) -> pd.DataFrame:
    """
    Build a long-form dataframe of every active stocking / harvest in every
    scenario's solved plan.  One row per active decision.  Columns:
        scenario_id, scenario_name, label_tuple, action_kind, unit, month,
        cohort_start (for harvests), qty, weight_g
    """
    rows = []
    for s, milp in enumerate(ald.milp_objects):
        sname = ald.scenario_names[s]
        lt    = list(ald._label_tuples[s])
        if milp.model.SolCount == 0:
            continue
        z_vars  = milp.variables.get("z", {})
        q_vars  = milp.variables.get("q", {})
        h_vars  = milp.variables.get("h", {})
        hx_vars = milp.variables.get("h_exist", {})

        for (u, ss), zvar in z_vars.items():
            if float(zvar.X) < 0.5:
                continue
            qv = float(q_vars[u, ss].X) if (u, ss) in q_vars else 0.0
            rows.append({
                "scenario_id": s, "scenario_name": sname, "label_tuple": tuple(lt),
                "action": "stock", "unit": u, "month": int(ss),
                "cohort_start": None, "qty": qv, "weight_g": None,
            })
        for (u, ss, th), hvar in h_vars.items():
            if float(hvar.X) < 0.5:
                continue
            wt = float(milp.wpf[int(ss)][int(th)])
            rows.append({
                "scenario_id": s, "scenario_name": sname, "label_tuple": tuple(lt),
                "action": "harvest", "unit": u, "month": int(th),
                "cohort_start": int(ss), "qty": None, "weight_g": wt,
            })
        for (u, th), hvar in hx_vars.items():
            if float(hvar.X) < 0.5:
                continue
            wt = float(milp.w_exist[u][int(th)])
            rows.append({
                "scenario_id": s, "scenario_name": sname, "label_tuple": tuple(lt),
                "action": "harvest_existing", "unit": u, "month": int(th),
                "cohort_start": "existing", "qty": None, "weight_g": wt,
            })
    return pd.DataFrame(rows)


def _extract_scenario_biomass(ald) -> pd.DataFrame:
    """Per-scenario standing biomass (tonnes) per month — for fan charts."""
    n = ald.n_scenarios
    T = ald.T
    rows = []
    for s, milp in enumerate(ald.milp_objects):
        if milp.model.SolCount == 0:
            continue
        b = np.zeros(T)
        # New cohorts
        for u in milp.U:
            for ss in milp.Tset:
                for t in milp.Tset:
                    if (u, ss, t) in milp.A_harv_set:
                        harvested_through_t = sum(
                            milp.variables["x"][u, ss, tau].X
                            for tau in milp.H_by_us[(u, ss)] if tau <= t
                        )
                        b[t] += float(milp.bpf[ss][t]) * (
                            milp.variables["q"][u, ss].X - harvested_through_t
                        )
        # Existing cohorts
        for u in milp.U_exist:
            for t in milp.Tset:
                e_val = milp.variables["e_alive"][u, t].X
                h_val = milp.variables["h_exist"][u, t].X
                b[t] += milp.b_exist[u][t] * (e_val - h_val)
        for t in range(T):
            rows.append({
                "scenario_id": s,
                "scenario_name": ald.scenario_names[s],
                "month": t,
                "biomass_t": b[t] / 1e3,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_rolling_horizon(
    *,
    units_df0: pd.DataFrame,
    loc_mab: dict,
    regional_mab: float,
    reality_path: List[str],
    horizon_months: int = 30,
    prefix_months: Optional[List[int]] = None,
    stage_slices: Optional[List[List[int]]] = None,
    months_per_roll: int = 3,
    start_calendar_month: int = 0,
    temp_map: Optional[Dict[str, np.ndarray]] = None,
    S_normal: Optional[float] = None,
    S_bad: Optional[float] = None,
    ph_kwargs: Optional[dict] = None,
    save_dir: Optional[str] = None,
    save_full_plans: bool = True,
) -> Dict:
    """
    Run a complete rolling-horizon experiment.

    Parameters
    ----------
    units_df0 : pd.DataFrame
        Initial unit roster.
    reality_path : list[str]
        Sequence of realised prefix labels (length = n_rolls).  Each entry
        is the temperature regime that actually unfolds during the roll's
        first 3 months.
    horizon_months : int
        Length of each roll's planning horizon (default 30).
    prefix_months, stage_slices : default rolling defaults from `instance.py`.
    months_per_roll : int
        How many months each roll commits to (must equal len(prefix_months)).
    start_calendar_month : int
        Calendar month corresponding to month 0 of roll 0.
    temp_map, S_normal, S_bad : temperature & survival profiles.
    ph_kwargs : dict
        Forwarded to `AugmentedLagrangianDecomposition.__init__`.  Use
        e.g. `{"K": 200, "mip_gap": 0.02}` to bound the work per roll.
    save_dir : str
        If set, every roll's outputs (consensus plan + scenario-plan dump
        + log) are pickled under `save_dir/roll_{k:02d}/`.
    save_full_plans : bool
        If True, dump the full 27-scenario plan dataframes per roll.
        Skipping these (False) keeps disk usage modest for long sweeps.

    Returns
    -------
    dict
        {
            "roll_logs": [ ... per-roll dicts ... ],
            "config":    { ... },
        }
    """
    # ── Defaults from `instance.py` ──────────────────────────────────────
    from instance import (
        prefix_months as _DEFAULT_PREFIX,
        stage_slices  as _DEFAULT_STAGES,
        temp_map      as _DEFAULT_TMAP,
        S_normal_v    as _DEFAULT_SN,
        S_bad_v       as _DEFAULT_SB,
    )

    prefix_months = list(prefix_months) if prefix_months is not None else list(_DEFAULT_PREFIX)
    stage_slices  = ([list(sl) for sl in stage_slices]
                     if stage_slices is not None else [list(sl) for sl in _DEFAULT_STAGES])
    temp_map      = temp_map or _DEFAULT_TMAP
    S_normal      = _DEFAULT_SN if S_normal is None else S_normal
    S_bad         = _DEFAULT_SB if S_bad    is None else S_bad
    ph_kwargs     = dict(ph_kwargs or {})

    n_rolls = len(reality_path)
    assert months_per_roll == len(prefix_months), (
        f"months_per_roll ({months_per_roll}) must equal len(prefix_months) "
        f"({len(prefix_months)})"
    )

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # Imports that should NOT happen until we actually run a roll
    # (sp.py loads gurobipy).
    from forward_sim import (
        extract_implemented_decisions,
        realised_temps_and_S,
        advance_state,
    )

    roll_logs: List[Dict] = []

    units_df_curr = units_df0.copy()
    cal_offset    = start_calendar_month % 12
    prev_label    = None

    overall_start = time.time()

    for k, prefix_label in enumerate(reality_path):
        print(f"\n{'='*72}\nRoll {k+1}/{n_rolls}  realised='{prefix_label}'  "
              f"calendar_month={cal_offset}  prev='{prev_label}'\n{'='*72}")
        roll_start = time.time()

        # 1) Solve the SP for this roll
        ald = _solve_one_roll(
            units_df=units_df_curr,
            loc_mab=loc_mab,
            regional_mab=regional_mab,
            horizon_months=horizon_months,
            prefix_months=prefix_months,
            stage_slices=stage_slices,
            prefix_label=prefix_label,
            start_calendar_month=cal_offset,
            temp_map=temp_map,
            S_normal=S_normal,
            S_bad=S_bad,
            ph_kwargs=ph_kwargs,
        )

        # 2) Extract consensus first-3-month decisions
        decisions = extract_implemented_decisions(ald, n_implement=months_per_roll)

        # 3) Realised temperatures / survival for the prefix
        real_temps, real_S = realised_temps_and_S(
            prefix_label, months_per_roll,
            start_calendar_month=cal_offset,
            temp_map=temp_map,
            S_normal=S_normal, S_bad=S_bad,
            prev_label=prev_label,
        )

        # 4) Advance state
        units_df_after, fwd_log = advance_state(
            units_df_curr, decisions, real_temps, real_S,
        )

        # 5) Optional dumps
        scenario_plans = scenario_biomass = None
        if save_full_plans:
            try:
                scenario_plans   = _extract_scenario_plans(ald)
                scenario_biomass = _extract_scenario_biomass(ald)
            except Exception as exc:
                print(f"  [roll {k}] WARNING: plan extraction failed: {exc}")

        roll_log = {
            "roll_idx": k,
            "prefix_label": prefix_label,
            "calendar_month": cal_offset,
            "prev_label": prev_label,
            "units_df_before": units_df_curr.copy(),
            "units_df_after":  units_df_after.copy(),
            "decisions":  decisions,
            "fwd_log":    fwd_log,
            "ph_eval_obj": float(getattr(ald, "eval_obj", 0.0)),
            "ph_n_iters":  int(getattr(ald, "n_iters", 0)),
            "ph_total_time": float(getattr(ald, "total_time", 0.0)),
            "scenario_plans":   scenario_plans,
            "scenario_biomass": scenario_biomass,
            "realised_temps": real_temps.tolist(),
            "realised_S":     real_S.tolist(),
            "wallclock_s":   time.time() - roll_start,
        }
        roll_logs.append(roll_log)

        if save_dir is not None:
            roll_dir = os.path.join(save_dir, f"roll_{k:02d}")
            os.makedirs(roll_dir, exist_ok=True)
            # Lightweight JSON summary
            summary = {
                "roll_idx": k,
                "prefix_label": prefix_label,
                "calendar_month": cal_offset,
                "prev_label": prev_label,
                "ph_eval_obj":   roll_log["ph_eval_obj"],
                "ph_n_iters":    roll_log["ph_n_iters"],
                "ph_total_time": roll_log["ph_total_time"],
                "wallclock_s":   roll_log["wallclock_s"],
                "realised_temps": roll_log["realised_temps"],
                "realised_S":     roll_log["realised_S"],
            }
            with open(os.path.join(roll_dir, "summary.json"), "w") as fh:
                json.dump(summary, fh, indent=2)
            # Heavier pickle dump
            with open(os.path.join(roll_dir, "roll.pkl"), "wb") as fh:
                pickle.dump(roll_log, fh)

        # 6) Update state for next roll
        units_df_curr = units_df_after
        cal_offset    = (cal_offset + months_per_roll) % 12
        prev_label    = prefix_label

    config = {
        "horizon_months":       horizon_months,
        "prefix_months":        prefix_months,
        "stage_slices":         stage_slices,
        "months_per_roll":      months_per_roll,
        "start_calendar_month": start_calendar_month,
        "n_rolls":              n_rolls,
        "reality_path":         list(reality_path),
        "ph_kwargs":            ph_kwargs,
        "total_wallclock_s":    time.time() - overall_start,
    }
    result = {"roll_logs": roll_logs, "config": config}

    if save_dir is not None:
        with open(os.path.join(save_dir, "config.json"), "w") as fh:
            json.dump(config, fh, indent=2, default=str)
        with open(os.path.join(save_dir, "result.pkl"), "wb") as fh:
            pickle.dump(result, fh)

    return result


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--path", choices=list(REALITY_PATHS) + ["all"],
                   default="all_normal",
                   help="Which reality path to run (or 'all').")
    p.add_argument("--n-rolls", type=int, default=12,
                   help="Cap the path length (for smoke tests).")
    p.add_argument("--save-dir", default="rh_runs",
                   help="Directory to write per-path outputs into.")
    p.add_argument("--K", type=int, default=200, help="PH max iterations.")
    p.add_argument("--mip-gap", type=float, default=0.02)
    p.add_argument("--start-calendar-month", type=int, default=0)
    p.add_argument("--no-full-plans", action="store_true",
                   help="Skip per-scenario plan/biomass dumps (faster).")
    args = p.parse_args()

    from instance import units_df, loc_mab, regional_mab

    paths = (list(REALITY_PATHS) if args.path == "all" else [args.path])

    for name in paths:
        path = REALITY_PATHS[name][:args.n_rolls]
        print(f"\n>>> Running reality path: {name}  (n_rolls={len(path)})")
        save_dir = os.path.join(args.save_dir, name)
        result = run_rolling_horizon(
            units_df0=units_df,
            loc_mab=loc_mab,
            regional_mab=regional_mab,
            reality_path=path,
            start_calendar_month=args.start_calendar_month,
            ph_kwargs=dict(K=args.K, mip_gap=args.mip_gap),
            save_dir=save_dir,
            save_full_plans=not args.no_full_plans,
        )
        print(f"<<< Done {name}: total wallclock "
              f"{result['config']['total_wallclock_s']:.1f}s, "
              f"{len(result['roll_logs'])} rolls")
