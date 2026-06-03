"""
Forward simulator for the rolling-horizon salmon-farming experiment.

Each roll covers 30 months: months 0-2 are deterministically implemented
(prefix), months 3-29 are scenario-tree-stochastic (3 stages × 9 months,
27 scenarios).  After solving the SP, this module:

1. Extracts the consensus first-3-month decisions from a solved
   `BinaryProgressiveHedging` (everything tagged "@s0").
2. Advances each unit's state through those 3 months using the realised
   temperature sequence (one of the {bad, normal, good} labels), respecting
   harvest decisions.
3. Returns a fresh `units_df` for the next roll plus structured logs of
   what was actually implemented.

Notes
-----
* Stocking quantities (`q`) are continuous and live outside the binary
  consensus dict.  After solve, they are stored in `ald.q_robust_vals`
  keyed by `(node_key, u, ss)`; for the prefix and first stochastic stage
  the node is "s0" so all 27 scenarios share the same q value.  We pull
  them directly from there.
* New cohorts stocked in months 0-2 cannot reach harvest weight within
  the 3-month window (smolt at 115 g, well-boat minimum 2000 g) so we
  do not check `h[u, ss, t]` for ss in [0, 2] and t in [0, 2].
* The new "existing" cohorts in the next roll's `units_df` are populated
  from (a) original existing cohorts that survived the 3 months without
  being harvested, and (b) cohorts stocked in months 0-2.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Growth / survival helpers (mirror SalmonFarmingMILP._next_weight exactly)
# ---------------------------------------------------------------------------

def _next_weight(
    w_g: float,
    temp: float,
    *,
    alpha: float = 0.011,
    beta: float = 0.646,
    temp_opt: float = 13.6,
    T_max: float = 23.9,
    days_per_month: int = 30,
) -> float:
    """Single-month weight transition (matches `SalmonFarmingMILP._next_weight`)."""
    if temp <= temp_opt:
        temp_eff = float(temp)
    else:
        temp_eff = temp_opt * (T_max - temp) / (T_max - temp_opt)
    return w_g + days_per_month * alpha * temp_eff * (w_g ** beta)


def _advance_weight(w0_g: float, monthly_temps: np.ndarray, **kwargs) -> float:
    w = float(w0_g)
    for t in monthly_temps:
        w = _next_weight(w, float(t), **kwargs)
    return w


def _advance_count(n0: float, monthly_S: np.ndarray) -> float:
    n = float(n0)
    for s in monthly_S:
        n = n * float(s)
    return n


# ---------------------------------------------------------------------------
# Decision extraction
# ---------------------------------------------------------------------------

def extract_implemented_decisions(
    ald,
    *,
    n_implement: int = 3,
) -> Dict:
    """
    Pull the consensus stage-0 decisions (z, h_exist, h, q) from a solved
    `BinaryProgressiveHedging` and return them as a tidy dict keyed
    by month, unit, etc.

    Parameters
    ----------
    ald : BinaryProgressiveHedging
        A *solved* SP — must have `bin_consensus`, `q_robust_vals`,
        `name_to_idx`, `all_na_names_ordered` populated (call `solve()` first).
    n_implement : int
        Number of months to actually implement (default 3 = the deterministic
        prefix plus / the first stage's blind commit, all sharing "s0").

    Returns
    -------
    dict with keys:
        "z"        : {(u, m): 0/1}        — newly stocked units this month
        "q"        : {(u, m): qty}        — fish stocked at (u, m) (semicont)
        "h_exist"  : {(u, m): 0/1}        — existing cohort harvested at m
        "h"        : {(u, ss, t): 0/1}    — new cohort (u, ss) harvested at t
                                              (only ss, t < n_implement)
    """
    n = n_implement

    z_dec = {}
    q_dec = {}
    h_exist_dec = {}
    h_dec = {}

    # The first n_implement months all sit in NA group "s0".
    for vi, qname in enumerate(ald.all_na_names_ordered):
        node = qname.split("@")[1]
        if node != "s0":
            continue
        vname = qname.split("@")[0]
        head, _, rest = vname.partition("[")
        if not rest:
            continue
        inside = rest[:-1]   # strip trailing "]"

        if head == "z":
            u, m_str = inside.rsplit(",", 1)
            m = int(m_str)
            if m < n:
                z_dec[(u, m)] = int(round(ald.bin_consensus.get(vi, 0.0)))
        elif head == "h_exist":
            u, m_str = inside.rsplit(",", 1)
            m = int(m_str)
            if m < n:
                h_exist_dec[(u, m)] = int(round(ald.bin_consensus.get(vi, 0.0)))
        elif head == "h":
            u, ss_str, th_str = inside.rsplit(",", 2)
            ss = int(ss_str); th = int(th_str)
            if th < n:
                h_dec[(u, ss, th)] = int(round(ald.bin_consensus.get(vi, 0.0)))

    # Robust q is keyed by (node_key, u, ss); for the implemented months
    # the node is always "s0".
    for (nk, u, ss), v in ald.q_robust_vals.items():
        if nk != "s0" or ss >= n:
            continue
        q_dec[(u, ss)] = float(v)

    return {
        "z":       z_dec,
        "q":       q_dec,
        "h_exist": h_exist_dec,
        "h":       h_dec,
    }


# ---------------------------------------------------------------------------
# Realised temperature / survival builders
# ---------------------------------------------------------------------------

def realised_temps_and_S(
    label: str,
    n_months: int,
    *,
    start_calendar_month: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    prev_label: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the `n_months` realised monthly temperatures and survival rates
    for a single block labelled `label`.

    `start_calendar_month` is the calendar offset that horizon-relative
    month 0 maps to (mod 12).  `prev_label` is the label of the immediately
    preceding block — only matters for the "good after good" high-mortality
    rule used by the SP scenarios.
    """
    profile = temp_map[label]
    temps = np.array([profile[(start_calendar_month + m) % 12]
                      for m in range(n_months)])
    S_block = S_bad if (label == "good" and prev_label == "good") else S_normal
    S = np.full(n_months, S_block)
    return temps, S


# ---------------------------------------------------------------------------
# Advance the unit roster forward
# ---------------------------------------------------------------------------

def advance_state(
    units_df: pd.DataFrame,
    decisions: Dict,
    realised_temps: np.ndarray,
    realised_S: np.ndarray,
    *,
    smolt_weight_g: float = 115.0,
    growth_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Advance every unit forward by `len(realised_temps)` months under the
    realised temperatures and survival rates, applying the implemented
    decisions.  Returns the new `units_df` ready for the next roll plus a
    log of what happened to each unit.

    Convention
    ----------
    Decision keys (`u`) use the *internal* unit_id form
    "Location X :: Unit Y" (matching `SalmonFarmingMILP.unit_id`).  The
    input `units_df` uses ("location", "unit") columns ("Loc1", "Unit 1");
    we map back and forth on the fly.
    """
    growth_kwargs = growth_kwargs or {}
    n_implement   = len(realised_temps)

    # Map between units_df rows and unit_id strings
    def _unit_id(loc_short: str, unit_name: str) -> str:
        # "Loc1" -> "Location 1"; trailing digits are extracted.
        digits = "".join(ch for ch in loc_short if ch.isdigit())
        return f"Location {digits} :: {unit_name}"

    new_rows = []
    log: List[Dict] = []

    for _, row in units_df.iterrows():
        loc       = row["location"]
        unit_name = row["unit"]
        vol       = row["volume_m3"]
        uid       = _unit_id(loc, unit_name)

        n0       = row["count"]
        w0_g     = row["avg_weight_g"]
        is_empty = pd.isna(n0) or pd.isna(w0_g)

        entry = {
            "location": loc,
            "unit": unit_name,
            "volume_m3": vol,
        }

        # Did anything get harvested at this unit during the 3 months?
        h_exist_months = [m for m in range(n_implement)
                          if decisions["h_exist"].get((uid, m), 0) == 1]
        # Did we stock a new cohort at this unit during the 3 months?
        z_months       = [m for m in range(n_implement)
                          if decisions["z"].get((uid, m), 0) == 1]

        if not is_empty:
            # Existing cohort path
            if h_exist_months:
                # Harvested at month m_h.  Unit becomes empty after that.
                # We do NOT support a re-stock in the same 3-month window
                # (the MILP's one-cohort + fallow rules forbid it).
                m_h = h_exist_months[0]
                entry["count"]        = float("nan")
                entry["avg_weight_g"] = float("nan")
                log.append({
                    "uid": uid, "loc": loc, "unit": unit_name,
                    "event": "harvest_existing", "month": m_h,
                    "count_before": float(n0), "weight_g_before": float(w0_g),
                })
            else:
                # Survived all n_implement months → carry forward.
                w_new = _advance_weight(w0_g, realised_temps, **growth_kwargs)
                n_new = _advance_count(n0, realised_S)
                entry["count"]        = float(n_new)
                entry["avg_weight_g"] = float(w_new)
                log.append({
                    "uid": uid, "loc": loc, "unit": unit_name,
                    "event": "carry_existing",
                    "count_before": float(n0), "weight_g_before": float(w0_g),
                    "count_after":  float(n_new), "weight_g_after": float(w_new),
                })
        else:
            # Empty unit at start of roll
            if z_months:
                m_s = z_months[0]
                q   = decisions["q"].get((uid, m_s), 0.0)
                # Cohort starts at month m_s with 115 g, q fish.
                # It cannot reach harvest weight within the 3-month window,
                # so we just advance to month n_implement.
                temps_after = realised_temps[m_s:]
                S_after     = realised_S[m_s:]
                w_new = _advance_weight(smolt_weight_g, temps_after, **growth_kwargs)
                n_new = _advance_count(q, S_after)
                entry["count"]        = float(n_new)
                entry["avg_weight_g"] = float(w_new)
                log.append({
                    "uid": uid, "loc": loc, "unit": unit_name,
                    "event": "stocked_new", "month": m_s,
                    "q": float(q),
                    "count_after": float(n_new), "weight_g_after": float(w_new),
                })
            else:
                # Stays empty
                entry["count"]        = float("nan")
                entry["avg_weight_g"] = float("nan")
                log.append({
                    "uid": uid, "loc": loc, "unit": unit_name,
                    "event": "stay_empty",
                })

        new_rows.append(entry)

    # Preserve original column order (unit, count, avg_weight_g, volume_m3, location)
    new_df = pd.DataFrame(new_rows, columns=[
        "unit", "count", "avg_weight_g", "volume_m3", "location"
    ])
    return new_df, log


# ---------------------------------------------------------------------------
# Convenience: one-step roll
# ---------------------------------------------------------------------------

def roll_forward_one_step(
    ald,
    realised_label: str,
    *,
    units_df: pd.DataFrame,
    start_calendar_month: int,
    temp_map: Dict[str, np.ndarray],
    S_normal: float,
    S_bad: float,
    n_implement: int = 3,
    prev_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
    """
    End-to-end: extract the implemented decisions from `ald`, build the
    realised temperature/survival vectors for `realised_label`, advance the
    unit roster, and return (new_units_df, decisions, log).
    """
    decisions = extract_implemented_decisions(ald, n_implement=n_implement)
    temps, S  = realised_temps_and_S(
        realised_label, n_implement,
        start_calendar_month=start_calendar_month,
        temp_map=temp_map,
        S_normal=S_normal, S_bad=S_bad,
        prev_label=prev_label,
    )
    new_units_df, log = advance_state(units_df, decisions, temps, S)
    return new_units_df, decisions, log
