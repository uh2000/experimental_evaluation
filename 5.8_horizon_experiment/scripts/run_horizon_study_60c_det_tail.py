"""
Open-loop horizon-study driver: 30-month vs 60-month planning horizon
under three deterministic in-sample temperature realizations over a
60-month real-time horizon.

Design (from the user's specification)
--------------------------------------
Both variants commit half their planning horizon, then re-solve.  This
makes the comparison "apples to apples": each variant uses the same
50% rolling-horizon ratio.

* 60-month variant
    horizon_months   = 60
    commit_months    = 30
    n_solves         = 2     (real months 0-29, then 30-59)
    SP structure: 3-month prefix + 3 stages of 9 months + 30-month
        deterministic-normal tail (handled by `sp_60m`).

* 30-month variant
    horizon_months   = 30
    commit_months    = 15
    n_solves         = 4     (real months 0-14, 15-29, 30-44, 45-59)
    SP structure: 3-month prefix + 3 stages of 9 months (no tail).

In-sample realizations (3 paths, each defined by labels at the
30-month "regime" granularity)
----------------------------------------------------------------
    "normal":      regime_0 = (N, N, N, N), regime_1 = (N, N, N, N)
    "mixed":       regime_0 = (N, N, N, N), regime_1 = (N, G, N, B)
    "oscillating": regime_0 = (N, G, N, B), regime_1 = (N, G, N, B)

Each regime is decomposed as (prefix=3 mo, s1=9 mo, s2=9 mo, s3=9 mo);
the realized monthly label sequence is piecewise-constant on those
blocks.  None of the realized paths puts two "good" blocks back-to-back,
so the SP's "good after good" → S_bad rule never triggers in the
realized survival sequence.  Earlier versions of the experiment used
all-good "warm" / "hot" paths, but the SP at the second 30-month
commit (prefix=good) ran for ~16 hours without converging — those paths
are intractable for the 60M variant under PH defaults.

`normal` and `mixed` share regime_0, so the first 60M solve and the
first two 30M solves are cached across them.

For each solve we
    1. build & solve the SP under the current `units_df` and the
       realized prefix label;
    2. find the scenario in the tree whose stage labels match the
       realized in-sample stage labels (by majority-vote when the
       commit window straddles a regime boundary);
    3. extract that scenario's commit-window decisions
       (`extract_scenario.extract_scenario_decisions`);
    4. forward-simulate state under the realized monthly temps + S;
    5. record per-solve infeasibility statistics
       (`extract_scenario.feasibility_weighted_expected_obj`).

Outputs
-------
    horizon_runs/<variant>/<path>/result.pkl
        {
          "config":         {variant, horizon_months, commit_months, n_solves, ...},
          "solves":         [ per-solve dict ... ],
          "real_state_log": [ per-commit fwd_log ... ],
          "units_df_final": pd.DataFrame,
        }
    horizon_runs/<variant>/<path>/summary.json    — light summary

Each per-solve dict contains
    solve_idx, t_start, prefix_label, stage_labels, matched_scenario_idx,
    matched_scenario_name, ph_iters, ph_eval_obj, total_time,
    n_feas, n_total, infeas_rate, E_obj_feas_only, E_obj_naive,
    max_total_slack_kg.
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
# Path setup — make rolling_horizon_experiment/ + this folder importable.
# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
RH = os.path.normpath(os.path.join(HERE, "..", "rolling_horizon_experiment"))
for p in (RH, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

# Local copy of sp.py + the 60M subclass.  Done identically to
# run_60m_experiment.py — patch sys.modules["sp"] before importing
# anything that does `from sp import ...`.
import sp_rh                                              # noqa: E402
import sp_60m                                             # noqa: E402


# ---------------------------------------------------------------------------
# In-sample realized paths
# ---------------------------------------------------------------------------

# Each path defines two 30-month regimes.  Each regime is a 4-tuple of
# block labels: (prefix=3 mo, s1=9 mo, s2=9 mo, s3=9 mo).
REALIZED_PATHS: Dict[str, Dict[str, Tuple[str, str, str, str]]] = {
    "normal": {
        "regime_0": ("normal", "normal", "normal", "normal"),
        "regime_1": ("normal", "normal", "normal", "normal"),
    },
    "mixed": {
        # Mostly normal, with one warm block (months 33–41) and one cold
        # block (months 51–59).  Single-G surrounded by N → no "good after
        # good" trigger.
        "regime_0": ("normal", "normal", "normal", "normal"),
        "regime_1": ("normal", "good",   "normal", "bad"),
    },
    "oscillating": {
        # Alternating N → G → N → B every 9 months over the full horizon.
        "regime_0": ("normal", "good",   "normal", "bad"),
        "regime_1": ("normal", "good",   "normal", "bad"),
    },
}

REAL_HORIZON_MONTHS = 60   # the real-time horizon we care about


# ---------------------------------------------------------------------------
# Realized-path helpers
# ---------------------------------------------------------------------------

def _realized_blocks(path_spec: Dict) -> List[Tuple[str, int]]:
    """Return [(label, n_months), ...] for full 60 months of real time."""
    return [
        (path_spec["regime_0"][0], 3),
        (path_spec["regime_0"][1], 9),
        (path_spec["regime_0"][2], 9),
        (path_spec["regime_0"][3], 9),
        (path_spec["regime_1"][0], 3),
        (path_spec["regime_1"][1], 9),
        (path_spec["regime_1"][2], 9),
        (path_spec["regime_1"][3], 9),
    ]


def realized_monthly_labels(path_spec: Dict) -> List[str]:
    """Return a list of 60 monthly labels."""
    out: List[str] = []
    for lbl, n in _realized_blocks(path_spec):
        out.extend([lbl] * n)
    return out


def build_full_realized(
    path_spec: Dict,
    start_calendar_month: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build the realized monthly temperatures and survival rates over the
    60-month real horizon.  Applies the SP's "good after good" rule:
    a "good" block immediately following another "good" block has S_bad.
    """
    monthly_temps: List[float] = []
    monthly_S: List[float] = []
    cal = start_calendar_month % 12
    prev_lbl: Optional[str] = None
    for lbl, n in _realized_blocks(path_spec):
        S_block = S_bad if (lbl == "good" and prev_lbl == "good") else S_normal
        for m in range(n):
            monthly_temps.append(float(temp_map[lbl][(cal + m) % 12]))
            monthly_S.append(float(S_block))
        cal = (cal + n) % 12
        prev_lbl = lbl
    return np.array(monthly_temps), np.array(monthly_S)


def derive_solve_labels(
    monthly_labels: Sequence[str],
    start_t: int,
    prefix_months: Sequence[int],
    stage_slices: Sequence[Sequence[int]],
    ext_label: str = "normal",
) -> Tuple[str, Tuple[str, ...]]:
    """
    Given a real-time monthly-label sequence and a solve's start month,
    derive (prefix_label, (stage1_label, stage2_label, stage3_label))
    by taking the dominant label in each phase's real-time window.

    Months past `len(monthly_labels)` (e.g. 30M solve 4 reaches into
    real month 74) use `ext_label`.
    """
    n_real = len(monthly_labels)

    def get(t: int) -> str:
        return monthly_labels[t] if 0 <= t < n_real else ext_label

    def dominant(lo: int, hi: int) -> str:
        cnt = Counter(get(t) for t in range(lo, hi))
        # Counter.most_common preserves insertion order on ties; iterate
        # keys explicitly to make tie-breaking reproducible.
        if not cnt:
            return ext_label
        max_n = max(cnt.values())
        for t in range(lo, hi):
            if cnt[get(t)] == max_n:
                return get(t)
        return ext_label

    prefix_lo = start_t + prefix_months[0]
    prefix_hi = start_t + prefix_months[-1] + 1
    prefix_label = dominant(prefix_lo, prefix_hi)

    stage_labels: List[str] = []
    for sl in stage_slices:
        real_lo = start_t + sl[0]
        real_hi = start_t + sl[-1] + 1
        stage_labels.append(dominant(real_lo, real_hi))

    return prefix_label, tuple(stage_labels)


# ---------------------------------------------------------------------------
# State signature for cross-path solve caching
# ---------------------------------------------------------------------------

def _state_signature(units_df: pd.DataFrame) -> int:
    """Hash a units_df by (count, avg_weight_g) tuples — enough to identify
    SP-equivalent states.  Two states with the same hash will produce the
    same SP build."""
    arr = (
        units_df[["count", "avg_weight_g"]]
        .fillna(-1.0)
        .round(6)
        .to_numpy()
    )
    return hash(arr.tobytes())


# ---------------------------------------------------------------------------
# Variant configs
# ---------------------------------------------------------------------------

# These mirror the production rolling-horizon defaults exactly.
DEFAULT_PREFIX_MONTHS: List[int] = [0, 1, 2]
DEFAULT_STAGE_SLICES: List[List[int]] = [
    list(range(3, 12)),
    list(range(12, 21)),
    list(range(21, 30)),
]


def _variant_kwargs(variant: str) -> Dict:
    if variant == "30M":
        return {
            "horizon_months": 30,
            "commit_months":  15,
            "n_solves":       4,
            "kind":           "sp",
            "use_60m_class":  False,
        }
    elif variant == "60M":
        return {
            "horizon_months": 60,
            "commit_months":  30,
            "n_solves":       2,
            "kind":           "sp",
            "use_60m_class":  True,
        }
    elif variant == "DET":
        # Deterministic baseline: a single MILP per commit using all-normal
        # weather (no scenario tree).  Same horizon + commit + #solves as
        # the 30M SP variant so per-solve compute is directly comparable.
        # Used to compute the value of the stochastic solution (VSS) under
        # the in-sample realizations.
        return {
            "horizon_months": 30,
            "commit_months":  15,
            "n_solves":       4,
            "kind":           "det",
            "use_60m_class":  False,
        }
    raise ValueError(f"Unknown variant {variant!r}")


# ---------------------------------------------------------------------------
# SP build
# ---------------------------------------------------------------------------

def _solve_one(
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
    ph_kwargs: Dict,
    sp_class,
):
    """Build, solve, and return the BinaryProgressiveHedging (or
    its 60M subclass)."""
    ald = sp_class(
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


def _solve_one_det(
    *,
    units_df: pd.DataFrame,
    loc_mab: dict,
    regional_mab: float,
    horizon_months: int,
    start_calendar_month: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    mip_gap: float = 0.02,
    time_limit: Optional[float] = None,
):
    """
    Build + solve a deterministic single-scenario `SalmonFarmingMILP`
    over `horizon_months` months under all-normal weather (the model's
    expected scenario).  Returns the solved MILP.
    """
    import ip
    n = int(horizon_months)
    cal0 = start_calendar_month % 12
    temps = np.array(
        [float(temp_map["normal"][(cal0 + t) % 12]) for t in range(n)],
        dtype=float,
    )
    S = np.full(n, float(S_normal), dtype=float)
    milp = ip.SalmonFarmingMILP(
        units_df=units_df,
        temps_t=temps,
        survival_rates=S,
        loc_mab=loc_mab,
        regional_mab=regional_mab,
        horizon_months=n,
        scenario_name="det_normal",
    )
    milp.solve(time_limit=time_limit, mip_gap=mip_gap)
    return milp


# ---------------------------------------------------------------------------
# Per-path result save (used both for incremental + final saves)
# ---------------------------------------------------------------------------

def _save_path_result(
    *,
    save_dir: str,
    variant: str,
    path: str,
    state_p: Dict,
    ph_kwargs: Dict,
    horizon_months: int,
    commit_months: int,
    n_solves: int,
    start_calendar_month: int,
    slack_threshold_kg: float,
    wallclock_s_so_far: float,
    partial: bool,
) -> Dict:
    """Write one path's result.pkl + summary.json. Used for both the
    incremental per-solve save and the final post-loop save."""
    result = {
        "config": {
            "variant":              variant,
            "horizon_months":       horizon_months,
            "commit_months":        commit_months,
            "n_solves":             n_solves,
            "ph_kwargs":            ph_kwargs,
            "start_calendar_month": start_calendar_month,
            "real_horizon_months":  REAL_HORIZON_MONTHS,
            "path":                 path,
            "path_spec":            REALIZED_PATHS[path],
            "slack_threshold_kg":   slack_threshold_kg,
            "total_wallclock_s":    wallclock_s_so_far,
            "partial":              partial,
            "n_solves_completed":   len(state_p["solves"]),
        },
        "solves":          list(state_p["solves"]),
        "real_state_log":  list(state_p["fwd_logs"]),
        "units_df_final":  state_p["units_df"],
    }

    out_dir = os.path.join(save_dir, variant.lower(), path)
    os.makedirs(out_dir, exist_ok=True)
    pkl_path = os.path.join(out_dir, "result.pkl")
    tmp_path = pkl_path + ".tmp"
    with open(tmp_path, "wb") as fh:
        pickle.dump(result, fh)
    os.replace(tmp_path, pkl_path)

    light = {
        "config": result["config"],
        "solves": [
            {k_: v for k_, v in s.items()
             if k_ not in ("matched_scenario_slack_kg",)}
            for s in result["solves"]
        ],
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(light, fh, indent=2, default=str)
    return result


# ---------------------------------------------------------------------------
# Per-variant runner
# ---------------------------------------------------------------------------

def run_variant(
    *,
    variant: str,
    paths: List[str],
    units_df0: pd.DataFrame,
    loc_mab: dict,
    regional_mab: float,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    start_calendar_month: int,
    ph_kwargs: Optional[Dict] = None,
    save_dir: Optional[str] = None,
    slack_threshold_kg: float = 100.0,
) -> Dict[str, Dict]:
    """
    Run one variant (30M or 60M) for every path in `paths`.  SP solves
    are cached across paths whenever (state, prefix_label, calendar
    month) coincide — implementing the open-loop "share the first solve"
    optimization the user asked for.

    Returns a dict path_name -> result.
    """
    from extract_scenario import (
        find_scenario_idx_by_labels,
        find_best_match_scenario,
        extract_scenario_decisions,
        feasibility_weighted_expected_obj,
        scenario_slack_summary,
        extract_det_decisions,
        det_slack_summary,
        det_feasibility_summary,
    )
    # Use the long-commit forward simulator — the base
    # `forward_sim.advance_state` is hard-coded for 3-month commits and
    # silently drops new-cohort harvests, which CAN happen in 15- and
    # 30-month commit windows.
    from forward_sim_long import advance_state_long as advance_state

    cfg = _variant_kwargs(variant)
    horizon_months  = cfg["horizon_months"]
    commit_months   = cfg["commit_months"]
    n_solves        = cfg["n_solves"]
    kind            = cfg["kind"]   # "sp" or "det"
    sp_class        = (sp_60m.BinaryProgressiveHedging60M
                       if variant == "60M"
                       else sp_rh.BinaryProgressiveHedging)
    ph_kwargs       = dict(ph_kwargs or {})

    # Pull the MIP gap separately for the deterministic variant, since
    # it doesn't use the PH kwargs.
    det_mip_gap = float(ph_kwargs.get("mip_gap", 0.03))

    # Per-path state
    state: Dict[str, Dict] = {
        p: {
            "units_df": units_df0.copy(),
            "cal":      start_calendar_month % 12,
            "solves":   [],
            "fwd_logs": [],
        } for p in paths
    }

    # Pre-compute realized monthly temps + S + label sequences per path
    real_temps_full: Dict[str, np.ndarray] = {}
    real_S_full:     Dict[str, np.ndarray] = {}
    monthly_lbls:    Dict[str, List[str]]  = {}
    for p in paths:
        spec = REALIZED_PATHS[p]
        mt, ms = build_full_realized(spec, start_calendar_month, temp_map,
                                     S_normal, S_bad)
        real_temps_full[p] = mt
        real_S_full[p]     = ms
        monthly_lbls[p]    = realized_monthly_labels(spec)

    overall_t0 = time.time()

    for k in range(n_solves):
        t_start = k * commit_months
        print(f"\n{'='*72}\n[{variant}] solve {k+1}/{n_solves}  "
              f"t_start={t_start}\n{'='*72}", flush=True)

        # Per-path SP signature for THIS solve
        per_path: Dict[str, Dict] = {}
        for p in paths:
            prefix_label, stage_labels = derive_solve_labels(
                monthly_lbls[p], t_start,
                DEFAULT_PREFIX_MONTHS, DEFAULT_STAGE_SLICES,
            )
            per_path[p] = {
                "prefix_label": prefix_label,
                "stage_labels": stage_labels,
                "state_sig":    _state_signature(state[p]["units_df"]),
                "cal":          state[p]["cal"],
            }

        # Group paths whose SP is identical (same state, prefix_label, cal)
        groups: Dict[Tuple[int, str, int], List[str]] = {}
        for p in paths:
            info = per_path[p]
            key = (info["state_sig"], info["prefix_label"], info["cal"])
            groups.setdefault(key, []).append(p)

        for grp_key, grp_paths in groups.items():
            ref_p = grp_paths[0]
            info = per_path[ref_p]
            print(f"  group: paths={grp_paths}  prefix={info['prefix_label']}  "
                  f"cal={info['cal']}  kind={kind}", flush=True)

            t_solve_0 = time.time()
            if kind == "sp":
                ald = _solve_one(
                    units_df=state[ref_p]["units_df"],
                    loc_mab=loc_mab,
                    regional_mab=regional_mab,
                    horizon_months=horizon_months,
                    prefix_months=DEFAULT_PREFIX_MONTHS,
                    stage_slices=DEFAULT_STAGE_SLICES,
                    prefix_label=info["prefix_label"],
                    start_calendar_month=info["cal"],
                    temp_map=temp_map,
                    S_normal=S_normal,
                    S_bad=S_bad,
                    ph_kwargs=ph_kwargs,
                    sp_class=sp_class,
                )
                feas_summary = feasibility_weighted_expected_obj(
                    ald, slack_threshold_kg=slack_threshold_kg
                )
            else:   # kind == "det"
                ald = None    # not used in det branch
                milp = _solve_one_det(
                    units_df=state[ref_p]["units_df"],
                    loc_mab=loc_mab,
                    regional_mab=regional_mab,
                    horizon_months=horizon_months,
                    start_calendar_month=info["cal"],
                    temp_map=temp_map,
                    S_normal=S_normal,
                    mip_gap=det_mip_gap,
                )
                feas_summary = det_feasibility_summary(
                    milp, slack_threshold_kg=slack_threshold_kg
                )
            t_solve_1 = time.time()

            # For each path in the group, extract decisions + advance state
            for p in grp_paths:
                stage_labels = per_path[p]["stage_labels"]

                if kind == "sp":
                    try:
                        s_idx = find_scenario_idx_by_labels(ald, stage_labels)
                        matched = "exact"
                    except KeyError:
                        s_idx = find_best_match_scenario(ald, stage_labels)
                        matched = "best"
                    s_name = ald.scenario_names[s_idx]
                    slack  = scenario_slack_summary(ald, s_idx)
                    try:
                        decisions = extract_scenario_decisions(
                            ald, s_idx, n_implement=commit_months
                        )
                    except RuntimeError as exc:
                        print(f"    [{p}] WARNING: {exc}; using empty decisions")
                        decisions = {"z": {}, "q": {}, "h_exist": {}, "h": {}}
                    ph_iters     = int(getattr(ald, "n_iters", 0))
                    ph_eval_obj  = float(getattr(ald, "eval_obj", 0.0))
                    ph_total_t   = float(getattr(ald, "total_time", 0.0))
                else:
                    s_idx   = -1
                    matched = "det"
                    s_name  = "det_normal"
                    slack   = det_slack_summary(milp)
                    try:
                        decisions = extract_det_decisions(
                            milp, n_implement=commit_months
                        )
                    except RuntimeError as exc:
                        print(f"    [{p}] WARNING: {exc}; using empty decisions")
                        decisions = {"z": {}, "q": {}, "h_exist": {}, "h": {}}
                    ph_iters    = 0
                    ph_eval_obj = (float(milp.model.ObjVal)
                                   if milp.model.SolCount > 0 else float("nan"))
                    ph_total_t  = float(milp.model.Runtime)

                # Real-time slice for forward-sim.  For solve k starting
                # at t_start, the commit window is real months
                # [t_start, t_start + commit_months).
                t_lo, t_hi = t_start, t_start + commit_months
                if t_hi > REAL_HORIZON_MONTHS:
                    t_hi = REAL_HORIZON_MONTHS
                temps_window = real_temps_full[p][t_lo:t_hi]
                S_window     = real_S_full[p][t_lo:t_hi]

                # Pad to the full commit window if it extends past 60.
                # (Won't happen for our 30M / 60M settings, but be safe.)
                if len(temps_window) < commit_months:
                    pad = commit_months - len(temps_window)
                    pad_t = np.full(pad, temp_map["normal"][0])
                    pad_S = np.full(pad, S_normal)
                    temps_window = np.concatenate([temps_window, pad_t])
                    S_window     = np.concatenate([S_window,     pad_S])

                new_units_df, fwd_log = advance_state(
                    state[p]["units_df"], decisions, temps_window, S_window,
                )

                # Mark fwd_log with real time month offset
                for ev in fwd_log:
                    ev["t_real_offset"] = t_start

                # Per-solve summary
                solve_record = {
                    "solve_idx":          k,
                    "t_start":            t_start,
                    "horizon_months":     horizon_months,
                    "commit_months":      commit_months,
                    "prefix_label":       info["prefix_label"],
                    "stage_labels":       list(stage_labels),
                    "matched_scenario":   s_name,
                    "matched_scenario_idx": int(s_idx),
                    "matched_kind":       matched,
                    "ph_iters":           ph_iters,
                    "ph_eval_obj":        ph_eval_obj,
                    "ph_total_time":      ph_total_t,
                    "wallclock_s":        t_solve_1 - t_solve_0,
                    "matched_scenario_slack_kg":  slack,
                    **{f"feas_{k_}": v for k_, v in feas_summary.items()},
                }
                state[p]["solves"].append(solve_record)
                state[p]["fwd_logs"].append({
                    "solve_idx": k,
                    "t_start":   t_start,
                    "fwd_log":   fwd_log,
                    "decisions": decisions,
                    "realised_temps": list(map(float, temps_window)),
                    "realised_S":     list(map(float, S_window)),
                    "units_df_before": state[p]["units_df"].copy(),
                    "units_df_after":  new_units_df.copy(),
                })
                state[p]["units_df"] = new_units_df
                state[p]["cal"]      = (state[p]["cal"] + commit_months) % 12

                # ── Incremental save: write per-path partial results.
                # If a later (path, solve) crashes / is killed, the
                # already-completed solves are preserved on disk.
                if save_dir is not None:
                    _save_path_result(
                        save_dir=save_dir, variant=variant, path=p,
                        state_p=state[p],
                        ph_kwargs=ph_kwargs,
                        horizon_months=horizon_months,
                        commit_months=commit_months,
                        n_solves=n_solves,
                        start_calendar_month=start_calendar_month,
                        slack_threshold_kg=slack_threshold_kg,
                        wallclock_s_so_far=time.time() - overall_t0,
                        partial=True,
                    )

            # Drop the model after we're done with all paths in the group.
            del ald
            if kind == "det":
                del milp

    overall_t1 = time.time()

    # ── Final save: re-write each path with partial=False ───────────
    results: Dict[str, Dict] = {}
    for p in paths:
        if save_dir is not None:
            results[p] = _save_path_result(
                save_dir=save_dir, variant=variant, path=p,
                state_p=state[p],
                ph_kwargs=ph_kwargs,
                horizon_months=horizon_months,
                commit_months=commit_months,
                n_solves=n_solves,
                start_calendar_month=start_calendar_month,
                slack_threshold_kg=slack_threshold_kg,
                wallclock_s_so_far=overall_t1 - overall_t0,
                partial=False,
            )
        else:
            results[p] = {
                "config": {
                    "variant":              variant,
                    "horizon_months":       horizon_months,
                    "commit_months":        commit_months,
                    "n_solves":             n_solves,
                    "ph_kwargs":            ph_kwargs,
                    "start_calendar_month": start_calendar_month,
                    "real_horizon_months":  REAL_HORIZON_MONTHS,
                    "path":                 p,
                    "path_spec":            REALIZED_PATHS[p],
                    "slack_threshold_kg":   slack_threshold_kg,
                    "total_wallclock_s":    overall_t1 - overall_t0,
                    "partial":              False,
                    "n_solves_completed":   len(state[p]["solves"]),
                },
                "solves":          list(state[p]["solves"]),
                "real_state_log":  list(state[p]["fwd_logs"]),
                "units_df_final":  state[p]["units_df"],
            }

    return results


# ---------------------------------------------------------------------------
# Top-level CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants", default="DET,30M,60M",
                   help='Comma-separated subset of "30M","60M","DET" '
                        '(default all three).  DET is the deterministic '
                        'baseline used to compute VSS.')
    p.add_argument("--paths", default="normal,mixed,oscillating",
                   help="Comma-separated subset of "
                        f"{list(REALIZED_PATHS)}.")
    p.add_argument("--save-dir", default=os.path.join(HERE, "horizon_runs"))
    p.add_argument("--K", type=int, default=200,
                   help="PH max iterations (matches 30M production).")
    p.add_argument("--mip-gap", type=float, default=0.03,
                   help="Sub-MILP MIP gap.  Looser than the 0.02 used in "
                        "the rolling-horizon production run — speeds up "
                        "subproblem MIP solves at a small optimality cost.")
    p.add_argument("--subproblem-time-limit", type=float, default=600.0,
                   help="Per-subproblem Gurobi TimeLimit in seconds.  "
                        "Caps the wallclock pathology where one "
                        "penalty-deformed scenario MIP hangs the entire "
                        "PH iteration.  Default 600s (10 min).  Pass 0 to "
                        "disable the cap (original behaviour).")
    p.add_argument("--start-calendar-month", type=int, default=0)
    p.add_argument("--slack-threshold-kg", type=float, default=100.0,
                   help="Total-slack threshold (kg) above which a scenario "
                        "is flagged infeasible.  Default 100 kg ≈ "
                        "0.0003%% of regional MAB.")
    args = p.parse_args()

    # Load instance via the same trick as run_60m_experiment.py — sp_rh
    # has been imported but we still need rh_instance for units_df, etc.
    import importlib.util
    _inst_path = os.path.join(RH, "instance.py")
    _spec = importlib.util.spec_from_file_location("rh_instance", _inst_path)
    rh_instance = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(rh_instance)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    paths    = [p_.strip() for p_ in args.paths.split(",") if p_.strip()]
    for v in variants:
        if v not in ("30M", "60M", "DET"):
            raise SystemExit(f"Unknown variant {v}")
    for path in paths:
        if path not in REALIZED_PATHS:
            raise SystemExit(f"Unknown path {path}")

    ph_kwargs = dict(K=args.K, mip_gap=args.mip_gap)
    if args.subproblem_time_limit and args.subproblem_time_limit > 0:
        ph_kwargs["subproblem_time_limit"] = float(args.subproblem_time_limit)

    print("=" * 76)
    print("OPEN-LOOP HORIZON STUDY")
    print("=" * 76)
    print(f"  variants            : {variants}")
    print(f"  paths               : {paths}")
    print(f"  ph_kwargs           : {ph_kwargs}")
    print(f"  start_calendar_month: {args.start_calendar_month}")
    print(f"  slack_threshold_kg  : {args.slack_threshold_kg}")
    print(f"  save_dir            : {args.save_dir}")

    os.makedirs(args.save_dir, exist_ok=True)

    overall_t0 = time.time()
    for v in variants:
        print(f"\n{'#'*76}\n# RUNNING VARIANT: {v}\n{'#'*76}", flush=True)
        run_variant(
            variant=v,
            paths=paths,
            units_df0=rh_instance.units_df,
            loc_mab=rh_instance.loc_mab,
            regional_mab=rh_instance.regional_mab,
            temp_map={
                "normal": rh_instance.temps_normal_12,
                "bad":    rh_instance.temps_bad_12,
                "good":   rh_instance.temps_good_12,
            },
            S_normal=rh_instance.S_normal_v,
            S_bad=rh_instance.S_bad_v,
            start_calendar_month=args.start_calendar_month,
            ph_kwargs=ph_kwargs,
            save_dir=args.save_dir,
            slack_threshold_kg=args.slack_threshold_kg,
        )

    overall_t1 = time.time()
    print(f"\n{'='*76}\nTOTAL: {overall_t1 - overall_t0:.1f}s\n{'='*76}")


if __name__ == "__main__":
    main()
