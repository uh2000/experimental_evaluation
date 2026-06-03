"""
Per-scenario decision extraction from a solved BinaryProgressiveHedging.

The forward-sim helper `extract_implemented_decisions` in
`rolling_horizon_experiment/forward_sim.py` only handles the "@s0" consensus
node — i.e. it is valid for the 3-month deterministic prefix only.  In the
horizon-study experiment we commit half of the planning horizon (15 or 30
months), so we need to read decisions for a *specific scenario* in the tree
(the one whose stage labels match an in-sample realization).

Each scenario in the SP has its own SalmonFarmingMILP instance held in
`ald.milp_objects[s]`.  After solve, that scenario's individual MILP carries
the per-scenario decision values in `milp.variables`:

    milp.variables = {
        "s":       Var dict keyed (u, t)        — z[u,t] in 0/1
        "q":       Var dict keyed (u, t)        — q[u,t] continuous
        "h":       Var dict keyed (u, ss, th)   — h[u,ss,th] in 0/1
        "h_exist": Var dict keyed (u, t)        — h_exist[u,t] in 0/1
        ...
    }

This module exposes:

    find_scenario_idx_by_labels(ald, label_tuple)
        Map a stage-label tuple (e.g. ("normal","normal","normal")) to the
        scenario index in the tree.  Supports a fall-back "best match" when
        the tuple doesn't appear (e.g. when the realization lands on labels
        outside {bad, normal, good}).

    extract_scenario_decisions(ald, scenario_idx, n_implement)
        Pull (z, q, h, h_exist) decisions for a specific scenario over the
        first `n_implement` months.  Returns the same structure as
        `extract_implemented_decisions` so it can drop into
        `forward_sim.advance_state` unchanged.

    scenario_slack_summary(ald, scenario_idx)
        Sum the biomass slack variables for a single scenario.  Used to
        flag infeasible scenarios in the analyser.

    feasibility_mask(ald, slack_threshold_kg=100.0)
        Per-scenario boolean array: True if total slack <= threshold.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Scenario lookup
# ---------------------------------------------------------------------------

def find_scenario_idx_by_labels(
    ald, label_tuple: Sequence[str]
) -> int:
    """
    Return the scenario index whose stage-label tuple equals `label_tuple`.

    `ald._label_tuples[s]` is a tuple like ("normal","good","good") for a
    3-stage tree.  Raises KeyError if no exact match is found.
    """
    target = tuple(label_tuple)
    for s, lt in enumerate(ald._label_tuples):
        if tuple(lt) == target:
            return s
    raise KeyError(
        f"No scenario in tree matches labels {target}. "
        f"Available: {sorted({tuple(lt) for lt in ald._label_tuples})}"
    )


def find_best_match_scenario(
    ald, label_tuple: Sequence[str]
) -> int:
    """
    Return the scenario index whose label tuple agrees with `label_tuple`
    in the most positions.  Ties are broken by earlier-stage agreement
    (so disagreement late in the horizon is preferred over disagreement
    early — same fish, smaller mismatch).
    """
    target = tuple(label_tuple)
    best_idx, best_score = None, (-1, -1)
    for s, lt in enumerate(ald._label_tuples):
        agree = sum(1 for a, b in zip(lt, target) if a == b)
        # Tie-break: weight earlier stages higher.
        early_weight = sum(
            (len(target) - i) for i, (a, b) in enumerate(zip(lt, target)) if a == b
        )
        score = (agree, early_weight)
        if score > best_score:
            best_score = score
            best_idx = s
    return best_idx


# ---------------------------------------------------------------------------
# Per-scenario decision extraction
# ---------------------------------------------------------------------------

def extract_scenario_decisions(
    ald,
    scenario_idx: int,
    n_implement: int,
) -> Dict:
    """
    Extract the decision values for scenario `scenario_idx` over months
    [0, n_implement).  Returns a dict with the same shape as
    `forward_sim.extract_implemented_decisions`:

        {
            "z":       {(u, m): 0/1},
            "q":       {(u, m): qty},
            "h_exist": {(u, m): 0/1},
            "h":       {(u, ss, t): 0/1},
        }

    Notes
    -----
    * Reads directly from the scenario's solved MILP via
      `ald.milp_objects[s].variables[...]`.  This bypasses the consensus
      machinery — the values come out exactly as the per-scenario MILP set
      them at the last PH iteration.
    * For new-cohort harvest decisions h[u, ss, t], we include the cohort
      only if its harvest month `t` is < n_implement, irrespective of the
      cohort's stocking month `ss`.  (A cohort stocked late in the
      previous SP could in principle harvest within the new commit
      window, but in this single-SP context the cohort starts at ss in
      this same horizon, so ss < n_implement covers everything.)
    * Existing-cohort harvest h_exist[u, t]: included if t < n_implement.
    * Continuous q[u, t] is read straight from the MILP variable, NOT
      from `ald.q_robust_vals` (which stores the "robust" PH-consensus
      value across scenarios).  For the prefix and any stage with one
      node, the per-scenario q equals the robust q; for fully scenario-
      dependent stages, the per-scenario value is what the implemented
      plan actually uses.
    """
    if scenario_idx is None or not (0 <= scenario_idx < ald.n_scenarios):
        raise ValueError(f"Invalid scenario_idx: {scenario_idx}")

    milp = ald.milp_objects[scenario_idx]
    if milp.model.SolCount == 0:
        raise RuntimeError(
            f"Scenario {scenario_idx} ({ald.scenario_names[scenario_idx]}) "
            "has no solution — cannot extract decisions."
        )

    n = int(n_implement)
    z_dec: Dict[Tuple[str, int], int] = {}
    q_dec: Dict[Tuple[str, int], float] = {}
    h_exist_dec: Dict[Tuple[str, int], int] = {}
    h_dec: Dict[Tuple[str, int, int], int] = {}

    # Stocking binaries (z, named "s" in the MILP variables dict)
    z_vars = milp.variables.get("s", {}) or milp.variables.get("z", {})
    for (u, t), var in z_vars.items():
        if int(t) < n:
            z_dec[(u, int(t))] = int(round(float(var.X)))

    # Stocking quantities
    q_vars = milp.variables.get("q", {})
    for (u, t), var in q_vars.items():
        if int(t) < n:
            q_dec[(u, int(t))] = float(var.X)

    # Existing-cohort harvest
    he_vars = milp.variables.get("h_exist", {})
    for (u, t), var in he_vars.items():
        if int(t) < n:
            h_exist_dec[(u, int(t))] = int(round(float(var.X)))

    # New-cohort harvest
    h_vars = milp.variables.get("h", {})
    for (u, ss, t), var in h_vars.items():
        if int(t) < n:
            h_dec[(u, int(ss), int(t))] = int(round(float(var.X)))

    return {
        "z":       z_dec,
        "q":       q_dec,
        "h_exist": h_exist_dec,
        "h":       h_dec,
    }


# ---------------------------------------------------------------------------
# Feasibility / slack diagnostics
# ---------------------------------------------------------------------------

def scenario_slack_summary(ald, scenario_idx: int) -> Dict[str, float]:
    """
    Sum the slack variables for one scenario.

    Slack vars are NOT stored in `milp.variables` — they live on the
    Gurobi model under the names "slack_density[...]", "slack_mab_loc[...]",
    "slack_mab_reg[...]".  We pull them by name prefix from `model.getVars()`.

    Returns
    -------
    dict with keys:
        density_kg, loc_mab_kg, reg_mab_kg, total_kg
    """
    milp = ald.milp_objects[scenario_idx]
    if milp.model.SolCount == 0:
        return {"density_kg": np.nan, "loc_mab_kg": np.nan,
                "reg_mab_kg": np.nan, "total_kg": np.nan}

    s_dens = 0.0
    s_loc  = 0.0
    s_reg  = 0.0
    for v in milp.model.getVars():
        nm = v.VarName
        if nm.startswith("slack_density["):
            s_dens += float(v.X)
        elif nm.startswith("slack_mab_loc["):
            s_loc += float(v.X)
        elif nm.startswith("slack_mab_reg["):
            s_reg += float(v.X)
    total = s_dens + s_loc + s_reg
    return {
        "density_kg": s_dens,
        "loc_mab_kg": s_loc,
        "reg_mab_kg": s_reg,
        "total_kg":   total,
    }


def feasibility_mask(
    ald,
    slack_threshold_kg: float = 100.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per-scenario feasibility (slack ≤ threshold).

    Parameters
    ----------
    slack_threshold_kg : float
        Total slack (density + loc-MAB + regional-MAB) above which a
        scenario is flagged infeasible.  Default 100 kg = ~0.0003 % of
        regional MAB; effectively any nonzero slack.

    Returns
    -------
    mask : np.ndarray of bool, shape (n_scenarios,)
    total_slack : np.ndarray of float, shape (n_scenarios,)
    """
    n = ald.n_scenarios
    mask = np.zeros(n, dtype=bool)
    totals = np.zeros(n, dtype=float)
    for s in range(n):
        summ = scenario_slack_summary(ald, s)
        totals[s] = summ["total_kg"]
        mask[s]   = (summ["total_kg"] <= slack_threshold_kg)
    return mask, totals


# ---------------------------------------------------------------------------
# Scenario-level realised objective
# ---------------------------------------------------------------------------

def scenario_objectives(ald) -> np.ndarray:
    """
    Per-scenario objective value (penalty-stripped, i.e. the original
    salmon-farming objective without PH penalties).  Returns NaN for
    scenarios without a solution.

    The original objective is cached on `ald.original_objectives` at
    build time.  We evaluate it at the current scenario solution.

    Returns
    -------
    np.ndarray, shape (n_scenarios,)
    """
    n = ald.n_scenarios
    obj = np.full(n, np.nan, dtype=float)
    for s in range(n):
        milp = ald.milp_objects[s]
        if milp.model.SolCount == 0:
            continue
        try:
            obj[s] = float(ald.original_objectives[s].getValue())
        except Exception:
            obj[s] = float(milp.model.ObjVal)
    return obj


def feasibility_weighted_expected_obj(
    ald,
    slack_threshold_kg: float = 100.0,
) -> Dict[str, float]:
    """
    Compute the expected scenario objective, weighted only over feasible
    scenarios (slack ≤ threshold) with renormalised probabilities.

    Returns a dict:
        n_feas, n_total, infeas_rate,
        E_obj_feas_only, E_obj_naive (= sum p_s * obj_s, no renorm),
        max_total_slack_kg
    """
    mask, totals = feasibility_mask(ald, slack_threshold_kg)
    obj = scenario_objectives(ald)
    probs = np.asarray(ald.probabilities, dtype=float)

    n_total = int(mask.size)
    n_feas  = int(mask.sum())

    if n_feas == 0:
        return {
            "n_feas":             0,
            "n_total":             n_total,
            "infeas_rate":         1.0,
            "E_obj_feas_only":     float("nan"),
            "E_obj_naive":         float("nan"),
            "max_total_slack_kg":  float(np.nanmax(totals)) if totals.size else 0.0,
        }

    # Renormalised feasibility-weighted expectation
    p_feas = probs[mask] / probs[mask].sum()
    E_feas = float((p_feas * obj[mask]).sum())

    # "Naive" — sum p_s * obj_s, treating infeasible scenarios with their
    # solved obj (which may include the BIOMASS_PENALTY for slack).
    E_naive = float(np.nansum(probs * np.where(np.isnan(obj), 0.0, obj)))

    return {
        "n_feas":             n_feas,
        "n_total":             n_total,
        "infeas_rate":         1.0 - n_feas / n_total,
        "E_obj_feas_only":     E_feas,
        "E_obj_naive":         E_naive,
        "max_total_slack_kg":  float(totals.max()),
    }


# ---------------------------------------------------------------------------
# Deterministic single-MILP variant
# ---------------------------------------------------------------------------

def extract_det_decisions(
    milp,
    n_implement: int,
) -> Dict:
    """
    Decision extractor for a deterministic single-scenario `SalmonFarmingMILP`.
    Returns the same shape as `extract_scenario_decisions`.
    """
    if milp.model.SolCount == 0:
        raise RuntimeError(
            "Deterministic MILP has no solution — cannot extract decisions."
        )
    n = int(n_implement)
    z_dec, q_dec, h_exist_dec, h_dec = {}, {}, {}, {}

    z_vars = milp.variables.get("s", {}) or milp.variables.get("z", {})
    for (u, t), var in z_vars.items():
        if int(t) < n:
            z_dec[(u, int(t))] = int(round(float(var.X)))

    for (u, t), var in milp.variables.get("q", {}).items():
        if int(t) < n:
            q_dec[(u, int(t))] = float(var.X)

    for (u, t), var in milp.variables.get("h_exist", {}).items():
        if int(t) < n:
            h_exist_dec[(u, int(t))] = int(round(float(var.X)))

    for (u, ss, t), var in milp.variables.get("h", {}).items():
        if int(t) < n:
            h_dec[(u, int(ss), int(t))] = int(round(float(var.X)))

    return {
        "z":       z_dec,
        "q":       q_dec,
        "h_exist": h_exist_dec,
        "h":       h_dec,
    }


def det_slack_summary(milp) -> Dict[str, float]:
    """
    Slack diagnostics for a deterministic MILP.  Same return shape as
    `scenario_slack_summary`.
    """
    if milp.model.SolCount == 0:
        return {"density_kg": np.nan, "loc_mab_kg": np.nan,
                "reg_mab_kg": np.nan, "total_kg": np.nan}

    s_dens = s_loc = s_reg = 0.0
    for v in milp.model.getVars():
        nm = v.VarName
        if nm.startswith("slack_density["):
            s_dens += float(v.X)
        elif nm.startswith("slack_mab_loc["):
            s_loc += float(v.X)
        elif nm.startswith("slack_mab_reg["):
            s_reg += float(v.X)
    return {
        "density_kg": s_dens,
        "loc_mab_kg": s_loc,
        "reg_mab_kg": s_reg,
        "total_kg":   s_dens + s_loc + s_reg,
    }


def det_feasibility_summary(
    milp,
    slack_threshold_kg: float = 100.0,
) -> Dict[str, float]:
    """
    Match the shape of `feasibility_weighted_expected_obj` so the same
    `feas_*` keys can be written into the per-solve record by the
    horizon-study driver.  For a deterministic plan there is only one
    realization, so 'feasibility-weighted' is just 'feasible y/n'.
    """
    summ = det_slack_summary(milp)
    is_feas = bool(summ["total_kg"] <= slack_threshold_kg)
    obj = float(milp.model.ObjVal) if milp.model.SolCount > 0 else float("nan")
    return {
        "n_feas":             1 if is_feas else 0,
        "n_total":             1,
        "infeas_rate":         0.0 if is_feas else 1.0,
        "E_obj_feas_only":     obj if is_feas else float("nan"),
        "E_obj_naive":         obj,
        "max_total_slack_kg":  float(summ["total_kg"]),
    }
