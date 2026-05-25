"""
Rolling-horizon experiment v2: 30M vs 45M vs 60M planning horizons
compared over a 120-month real-time evaluation period.

Design
------
All three variants use the same 15-month stage length and share the
non-anticipativity tree structure from sp_rh.AugmentedLagrangianDecomposition:

  Variant  Horizon  Stages  Scenarios  Commit  Solves  Solve starts
  -------  -------  ------  ---------  ------  ------  ------------
  30M        30 mo     2       9 (3²)    15 mo     8    0,15,...,105
  45M        45 mo     3      27 (3³)    30 mo     4    0,30,60,90
  60M        60 mo     4      81 (3⁴)    45 mo     3    0,45,90

Rolling schedule over the 120-month real horizon:
  30M — 8 solves × 15-month commits = 120 months exactly.
        Last solve optimises months 105–135; only months 105–119 count.
  45M — 4 solves × 30-month commits = 120 months exactly.
        Last solve optimises months 90–135; only months 90–119 count.
  60M — 2 solves × 45-month commits (months 0–89) + 1 final solve from
        t=90 optimising months 90–150, but only months 90–119 implemented.

The final solve of every variant extends beyond month 120, eliminating
end-of-horizon bias for the evaluation window.

Trajectories (5, each defined as 10 consecutive 15-month blocks)
-----------------------------------------------------------------
  normal:              all "normal"
  warm:                all "good"
  cold:                all "bad"
  oscillating:         alternating "good","bad" (starts warm)
  stress_then_recover: "bad" for first 4 blocks (months 0–59),
                       "normal" for blocks 5–9 (months 60–149)

Feasibility
-----------
A solve is infeasible if any scenario in the SP tree exceeds
`slack_threshold_kg` total slack.  A trajectory is excluded from the
cross-trajectory weighted average for a given variant if it is
infeasible in any of its solves.  Feasible trajectories are weighted
equally (1 / n_feasible).  Infeasible trajectories are reported but
not zeroed out — they are simply omitted from the average.

Outputs
-------
  runs_v2/<variant>/<trajectory>/result.pkl
      {
        "config":         dict,
        "solves":         [per-solve dict, ...],
        "real_state_log": [per-commit dict, ...],
        "units_df_final": pd.DataFrame,   # farm state at month 120
      }
  runs_v2/<variant>/<trajectory>/summary.json  — lightweight summary
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RH   = os.path.normpath(os.path.join(HERE, "..", "rolling_horizon_experiment"))
for _p in (RH, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Use sp_rh for all three variants — it supports arbitrary n_stages via
# the `stage_slices` parameter.
import sp_rh  # noqa: E402

from extract_scenario import (   # noqa: E402
    find_scenario_idx_by_labels,
    find_best_match_scenario,
    extract_scenario_decisions,
    feasibility_weighted_expected_obj,
    scenario_slack_summary,
)
from forward_sim_long import advance_state_long  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REAL_HORIZON  = 120   # months we evaluate profit over
STAGE_LEN     = 15    # months per stage (all three variants)

# 10 blocks × 15 months covers the furthest opt window (60M last solve → 150)
TRAJECTORY_BLOCKS: Dict[str, List[str]] = {
    "normal":              ["normal"] * 10,
    "warm":                ["good"]   * 10,
    "cold":                ["bad"]    * 10,
    "oscillating":         ["good", "bad"] * 5,
    "stress_then_recover": ["bad"] * 4 + ["normal"] * 6,
}

VARIANT_CFG: Dict[str, Dict] = {
    "30M": {"T": 30, "commit": 15, "n_stages": 2},
    "45M": {"T": 45, "commit": 30, "n_stages": 3},
    "60M": {"T": 60, "commit": 45, "n_stages": 4},
    # 30-month horizon, 4 stages (7+8+7+8 = 30), 81 scenarios (3^4).
    # Commit = first two stages = 7+8 = 15 months → same rolling cadence as 30M.
    # Interesting comparison: same horizon as 30M but models 4× more uncertainty
    # branches — does richer scenario tree beat the longer 60M horizon?
    "30M_81": {
        "T":      30,
        "commit": 15,
        "n_stages": 4,
        "stage_slices": [
            list(range(0,  7)),   # stage 0: months 0-6   } macro-block 0
            list(range(7,  15)),  # stage 1: months 7-14  }   (= 15-month block)
            list(range(15, 22)),  # stage 2: months 15-21 } macro-block 1
            list(range(22, 30)),  # stage 3: months 22-29 }   (= 15-month block)
        ],
        # Pairs of stages share one 15-month biological block, so S_bad only
        # fires when crossing from a 'good' macro-block into another 'good'
        # macro-block — i.e. all four stage labels must be 'good'.
        "macro_block_len": 2,
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_stage_slices(n_stages: int, stage_len: int = STAGE_LEN) -> List[List[int]]:
    """Return [range(0,15), range(15,30), ...] for `n_stages` stages."""
    return [list(range(k * stage_len, (k + 1) * stage_len)) for k in range(n_stages)]


def solve_starts(commit: int, real_horizon: int = REAL_HORIZON) -> List[int]:
    """Return t_start values: [0, commit, 2*commit, ...] up to real_horizon-1."""
    return list(range(0, real_horizon, commit))


def build_realized(
    blocks: List[str],
    n_months: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    start_cal: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-month realized temperatures and survival rates for `n_months`
    months using `blocks` (one label per 15-month block).

    Applies the 'good after good → S_bad' rule per 15-month block.
    """
    monthly_temps = np.zeros(n_months)
    monthly_S     = np.zeros(n_months)
    for t in range(n_months):
        bi      = t // STAGE_LEN
        lbl     = blocks[bi] if bi < len(blocks) else "normal"
        bi_prev = bi - 1
        prev_lbl = (blocks[bi_prev] if 0 <= bi_prev < len(blocks) else None)
        S_block = S_bad if (lbl == "good" and prev_lbl == "good") else S_normal
        monthly_S[t]     = S_block
        monthly_temps[t] = temp_map[lbl][(start_cal + t) % 12]
    return monthly_temps, monthly_S


def derive_stage_labels(
    blocks: List[str],
    t_start: int,
    n_stages: int,
    stage_len: int = STAGE_LEN,
    stage_offsets: Optional[List[int]] = None,
) -> Tuple[str, ...]:
    """
    Return the scenario label tuple for a solve starting at real month
    `t_start`.

    For equal-length stages (default), stage k starts at t_start + k*stage_len
    and its temperature block is determined by integer division by stage_len.

    For unequal stages (30M_81), pass `stage_offsets` = the start month of
    each stage within the solve window (e.g. [0, 7, 15, 22]).  The trajectory
    block for each stage is still derived using the global STAGE_LEN = 15.
    """
    offsets = (stage_offsets if stage_offsets is not None
               else [k * stage_len for k in range(n_stages)])
    labels = []
    for off in offsets:
        real_start = t_start + off
        bi  = real_start // STAGE_LEN          # always use 15-month blocks
        lbl = blocks[bi] if bi < len(blocks) else "normal"
        labels.append(lbl)
    return tuple(labels)


def _state_signature(units_df: pd.DataFrame) -> int:
    arr = (
        units_df[["count", "avg_weight_g"]]
        .fillna(-1.0)
        .round(6)
        .to_numpy()
    )
    return hash(arr.tobytes())


# ---------------------------------------------------------------------------
# Single-trajectory runner
# ---------------------------------------------------------------------------

def run_trajectory(
    *,
    variant: str,
    trajectory: str,
    blocks: List[str],
    units_df0: pd.DataFrame,
    loc_mab: dict,
    regional_mab: float,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    start_cal: int = 0,
    ph_kwargs: Optional[Dict] = None,
    slack_threshold_kg: float = 100.0,
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Run one (variant × trajectory) combination through all solves and
    return the result dict.
    """
    cfg          = VARIANT_CFG[variant]
    T            = cfg["T"]
    commit       = cfg["commit"]
    n_stages     = cfg["n_stages"]
    # Use custom stage_slices if provided (e.g. 30M_81), else equal 15-month stages
    stage_slices    = cfg.get("stage_slices") or make_stage_slices(n_stages)
    stage_offsets   = [sl[0] for sl in stage_slices]   # start month of each stage
    macro_block_len = cfg.get("macro_block_len", 1)
    tstarts       = solve_starts(commit)
    ph_kw         = dict(ph_kwargs or {})

    # Pre-build realized temps / S for the full span (up to 150 months).
    max_months = REAL_HORIZON + T   # enough for any solve window
    real_temps, real_S = build_realized(
        blocks, max_months, temp_map, S_normal, S_bad, start_cal
    )

    units_df = units_df0.copy()
    cal      = start_cal % 12
    solves: List[Dict]   = []
    fwd_logs: List[Dict] = []

    overall_t0 = time.time()

    for solve_k, t_start in enumerate(tstarts):
        n_impl = min(commit, REAL_HORIZON - t_start)   # truncate last solve at 120
        stage_labels = derive_stage_labels(blocks, t_start, n_stages,
                                           stage_offsets=stage_offsets)

        print(
            f"\n{'='*72}\n[{variant}/{trajectory}] solve {solve_k+1}/{len(tstarts)}"
            f"  t_start={t_start}  n_impl={n_impl}"
            f"  labels={stage_labels}\n{'='*72}",
            flush=True,
        )

        t_solve0 = time.time()
        ald = sp_rh.AugmentedLagrangianDecomposition(
            units_df=units_df,
            loc_mab=loc_mab,
            regional_mab=regional_mab,
            T=T,
            stage_slices=stage_slices,
            start_calendar_month=cal,
            temps_normal=temp_map["normal"],
            temps_bad=temp_map["bad"],
            temps_good=temp_map["good"],
            S_normal=S_normal,
            S_bad=S_bad,
            macro_block_len=macro_block_len,
            **ph_kw,
        )
        ald.build()
        ald.solve()
        t_solve1 = time.time()

        # Feasibility summary over the full SP tree
        feas_summary = feasibility_weighted_expected_obj(
            ald, slack_threshold_kg=slack_threshold_kg
        )

        # Match scenario to realized labels
        try:
            s_idx   = find_scenario_idx_by_labels(ald, stage_labels)
            matched = "exact"
        except KeyError:
            s_idx   = find_best_match_scenario(ald, stage_labels)
            matched = "best"
        s_name = ald.scenario_names[s_idx]
        slack  = scenario_slack_summary(ald, s_idx)

        # Extract decisions for the implement window
        try:
            decisions = extract_scenario_decisions(ald, s_idx, n_implement=n_impl)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}; using empty decisions")
            decisions = {"z": {}, "q": {}, "h_exist": {}, "h": {}}

        # Forward-simulate the implement window under realized temps
        temps_win = real_temps[t_start : t_start + n_impl]
        S_win     = real_S    [t_start : t_start + n_impl]

        new_units_df, fwd_log = advance_state_long(
            units_df, decisions, temps_win, S_win
        )
        for ev in fwd_log:
            ev["t_real_offset"] = t_start

        # Record
        solve_rec = {
            "solve_idx":             solve_k,
            "t_start":               t_start,
            "horizon_months":        T,
            "commit_months":         commit,
            "n_implement":           n_impl,
            "stage_labels":          list(stage_labels),
            "matched_scenario":      s_name,
            "matched_scenario_idx":  int(s_idx),
            "matched_kind":          matched,
            "ph_iters":              int(getattr(ald, "n_iters", 0)),
            "ph_eval_obj":           float(getattr(ald, "eval_obj", 0.0)),
            "ph_total_time":         float(getattr(ald, "total_time", 0.0)),
            "wallclock_s":           t_solve1 - t_solve0,
            "matched_scenario_slack_kg": slack,
            **{f"feas_{k_}": v for k_, v in feas_summary.items()},
        }
        fwd_rec = {
            "solve_idx":       solve_k,
            "t_start":         t_start,
            "n_implement":     n_impl,
            "fwd_log":         fwd_log,
            "decisions":       decisions,
            "realised_temps":  list(map(float, temps_win)),
            "realised_S":      list(map(float, S_win)),
            "units_df_before": units_df.copy(),
            "units_df_after":  new_units_df.copy(),
        }
        solves.append(solve_rec)
        fwd_logs.append(fwd_rec)

        units_df = new_units_df
        cal      = (cal + n_impl) % 12

        # Incremental save
        if save_dir is not None:
            _save(
                save_dir=save_dir, variant=variant, trajectory=trajectory,
                solves=solves, fwd_logs=fwd_logs,
                units_df_final=units_df,
                cfg=cfg, ph_kw=ph_kw, start_cal=start_cal,
                slack_threshold_kg=slack_threshold_kg,
                total_wall=time.time() - overall_t0,
                partial=True,
            )

        del ald   # free Gurobi models

    overall_t1 = time.time()

    result = _save(
        save_dir=save_dir, variant=variant, trajectory=trajectory,
        solves=solves, fwd_logs=fwd_logs,
        units_df_final=units_df,
        cfg=cfg, ph_kw=ph_kw, start_cal=start_cal,
        slack_threshold_kg=slack_threshold_kg,
        total_wall=overall_t1 - overall_t0,
        partial=False,
    )
    return result


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(
    *,
    save_dir: Optional[str],
    variant: str,
    trajectory: str,
    solves: List[Dict],
    fwd_logs: List[Dict],
    units_df_final: pd.DataFrame,
    cfg: Dict,
    ph_kw: Dict,
    start_cal: int,
    slack_threshold_kg: float,
    total_wall: float,
    partial: bool,
) -> Dict:
    result = {
        "config": {
            "variant":              variant,
            "trajectory":           trajectory,
            "horizon_months":       cfg["T"],
            "commit_months":        cfg["commit"],
            "n_stages":             cfg["n_stages"],
            "real_horizon_months":  REAL_HORIZON,
            "start_calendar_month": start_cal,
            "ph_kwargs":            ph_kw,
            "slack_threshold_kg":   slack_threshold_kg,
            "total_wallclock_s":    total_wall,
            "partial":              partial,
            "n_solves_completed":   len(solves),
        },
        "solves":          list(solves),
        "real_state_log":  list(fwd_logs),
        "units_df_final":  units_df_final,
    }

    if save_dir is None:
        return result

    out_dir = os.path.join(save_dir, variant.lower(), trajectory)
    os.makedirs(out_dir, exist_ok=True)

    pkl_path = os.path.join(out_dir, "result.pkl")
    tmp_path = pkl_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        pickle.dump(result, fh)
    os.replace(tmp_path, pkl_path)

    # Lightweight JSON summary (omit large slack arrays)
    light = {
        "config": result["config"],
        "solves": [
            {k: v for k, v in s.items() if k != "matched_scenario_slack_kg"}
            for s in solves
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(light, fh, indent=2, default=str)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--variants", default="30M,60M",
        help='Comma-separated subset of "30M","45M","60M","30M_81" (default 30M,60M).',
    )
    p.add_argument(
        "--trajectories",
        default=",".join(TRAJECTORY_BLOCKS.keys()),
        help="Comma-separated trajectory names (default all five).",
    )
    p.add_argument(
        "--save-dir",
        default=os.path.join(HERE, "runs_v2"),
        help="Output directory (default ./runs_v2/).",
    )
    p.add_argument(
        "--K", type=int, default=200,
        help="PH max iterations (default 200).",
    )
    p.add_argument(
        "--mip-gap", type=float, default=0.03,
        help="Subproblem MIP gap (default 0.03).",
    )
    p.add_argument(
        "--subproblem-time-limit", type=float, default=600.0,
        help="Per-subproblem Gurobi time limit in seconds (0 = unlimited).",
    )
    p.add_argument(
        "--start-calendar-month", type=int, default=0,
        help="Month-of-year (0=Jan) for the first solve (default 0).",
    )
    p.add_argument(
        "--slack-threshold-kg", type=float, default=100.0,
        help="Slack threshold (kg) for feasibility flag (default 100).",
    )
    args = p.parse_args()

    variants     = [v.strip() for v in args.variants.split(",")     if v.strip()]
    trajectories = [t.strip() for t in args.trajectories.split(",") if t.strip()]

    for v in variants:
        if v not in VARIANT_CFG:
            raise SystemExit(f"Unknown variant {v!r}. Choose from {list(VARIANT_CFG)}.")
    for traj in trajectories:
        if traj not in TRAJECTORY_BLOCKS:
            raise SystemExit(
                f"Unknown trajectory {traj!r}. Choose from {list(TRAJECTORY_BLOCKS)}."
            )

    ph_kwargs: Dict = {"K": args.K, "mip_gap": args.mip_gap}
    if args.subproblem_time_limit and args.subproblem_time_limit > 0:
        ph_kwargs["subproblem_time_limit"] = float(args.subproblem_time_limit)

    # Load instance from this folder (instance.py has the reduced 10-cage setup)
    import importlib.util
    _spec = importlib.util.spec_from_file_location(
        "local_instance", os.path.join(HERE, "instance.py")
    )
    inst = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(inst)

    temp_map = {
        "normal": inst.temps_normal_12,
        "bad":    inst.temps_bad_12,
        "good":   inst.temps_good_12,
    }
    S_normal = float(inst.S_normal_v)
    S_bad    = float(inst.S_bad_v)

    print("=" * 76)
    print("ROLLING-HORIZON EXPERIMENT v2 — 30M vs 45M vs 60M")
    print("=" * 76)
    print(f"  variants          : {variants}")
    print(f"  trajectories      : {trajectories}")
    print(f"  ph_kwargs         : {ph_kwargs}")
    print(f"  start_cal_month   : {args.start_calendar_month}")
    print(f"  slack_threshold   : {args.slack_threshold_kg} kg")
    print(f"  save_dir          : {args.save_dir}")
    print(f"  real_horizon      : {REAL_HORIZON} months")
    print()

    os.makedirs(args.save_dir, exist_ok=True)
    overall_t0 = time.time()

    for variant in variants:
        for trajectory in trajectories:
            print(f"\n{'#'*76}")
            print(f"# VARIANT={variant}  TRAJECTORY={trajectory}")
            print(f"{'#'*76}", flush=True)

            out_dir  = os.path.join(args.save_dir, variant.lower(), trajectory)
            pkl_path = os.path.join(out_dir, "result.pkl")
            if os.path.exists(pkl_path):
                with open(pkl_path, "rb") as fh:
                    existing = pickle.load(fh)
                if not existing.get("config", {}).get("partial", True):
                    print(f"  [SKIP] already completed — delete {pkl_path} to re-run.")
                    continue

            run_trajectory(
                variant=variant,
                trajectory=trajectory,
                blocks=TRAJECTORY_BLOCKS[trajectory],
                units_df0=inst.units_df,
                loc_mab=inst.loc_mab,
                regional_mab=inst.regional_mab,
                temp_map=temp_map,
                S_normal=S_normal,
                S_bad=S_bad,
                start_cal=args.start_calendar_month,
                ph_kwargs=ph_kwargs,
                slack_threshold_kg=args.slack_threshold_kg,
                save_dir=args.save_dir,
            )

    total_wall = time.time() - overall_t0
    print(f"\n{'='*76}\nDONE. Total wall: {total_wall:.1f}s\n{'='*76}")


if __name__ == "__main__":
    main()
