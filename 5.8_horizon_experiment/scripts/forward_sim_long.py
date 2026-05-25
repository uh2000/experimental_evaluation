"""
Forward simulator for arbitrary commit-window lengths (15 or 30 months).

The base `rolling_horizon_experiment/forward_sim.py` is hard-coded for a
3-month commit window where new smolts cannot reach harvest weight, so it
does not process new-cohort harvest decisions (`h[u, ss, t]`) at all.
For 15- and 30-month commits in this open-loop horizon study, new
cohorts CAN reach harvest weight, so we must apply those decisions or
the resulting state at the next re-solve will be wrong.

This module exposes:

    advance_state_long(units_df, decisions, realised_temps, realised_S,
                       smolt_weight_g=115.0, growth_kwargs=None)
        Drop-in replacement for `advance_state` that processes h_exist,
        z/q stocking, AND h[u,ss,t] new-cohort harvests month-by-month.
        Returns (new_units_df, fwd_log).

    realised_biomass_timeline_long(result)
        Walk every commit window and reconstruct a per-cohort,
        per-month dataframe with `event` ∈ {"alive", "harvested"} and
        `kind` ∈ {"existing", "new"}.  Compatible shape with
        `rh_viz.realised_biomass_timeline` for downstream cashflow code.

Conventions
-----------
* `decisions["z"][(uid, m)]`, `decisions["q"][(uid, m)]` — stocking at
  unit `uid` at month `m` (within the current commit window, 0-indexed).
* `decisions["h_exist"][(uid, m)]` — harvest existing cohort at month m.
* `decisions["h"][(uid, ss, t)]` — harvest the new cohort stocked at
  month `ss` of THIS commit window, at month `t` (0 ≤ ss ≤ t < n_implement).
* `uid` follows `SalmonFarmingMILP.unit_id`: "Location N :: Unit M".
* Within one commit window we may stock several different cohorts at
  different units; if a unit is stocked twice (illegal under the MILP)
  the second stocking overwrites the first in the local roster.
* If the simulator harvests an existing cohort, the unit becomes empty
  AND can be re-stocked later in the same commit (the SP enforces a
  fallow window so this only fires when the SP allows it).
"""
from __future__ import annotations

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
    if temp <= temp_opt:
        temp_eff = float(temp)
    else:
        temp_eff = temp_opt * (T_max - temp) / (T_max - temp_opt)
    return w_g + days_per_month * alpha * temp_eff * (w_g ** beta)


# ---------------------------------------------------------------------------
# Long-commit advance_state
# ---------------------------------------------------------------------------

def _unit_id(loc_short: str, unit_name: str) -> str:
    digits = "".join(ch for ch in loc_short if ch.isdigit())
    return f"Location {digits} :: {unit_name}"


def advance_state_long(
    units_df: pd.DataFrame,
    decisions: Dict,
    realised_temps: np.ndarray,
    realised_S: np.ndarray,
    *,
    smolt_weight_g: float = 115.0,
    growth_kwargs: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Advance every unit forward by `len(realised_temps)` months under
    realised temps and survival, applying stocking, existing-cohort
    harvest, AND new-cohort harvest within the commit window.

    The returned `new_units_df` contains the post-state at the END of
    the commit window: each unit is either alive (latest cohort) or
    empty (NaN count/weight).
    """
    growth_kwargs = growth_kwargs or {}
    n_implement = len(realised_temps)

    # Per-unit state: each unit has at most one active cohort at a time.
    # Cohort: dict(kind="existing"|"new", count, weight_g, ss=stocking month
    # within commit window — only meaningful for "new").
    state: Dict[str, Optional[Dict]] = {}
    fixed_cols = {}    # per-unit static info: location, unit, volume_m3
    for _, row in units_df.iterrows():
        loc       = row["location"]
        unit_name = row["unit"]
        vol       = row["volume_m3"]
        uid       = _unit_id(loc, unit_name)
        fixed_cols[uid] = {"location": loc, "unit": unit_name, "volume_m3": vol}

        n0   = row["count"]
        w0_g = row["avg_weight_g"]
        if pd.isna(n0) or pd.isna(w0_g):
            state[uid] = None
        else:
            state[uid] = {
                "kind": "existing",
                "count": float(n0),
                "weight_g": float(w0_g),
                "ss": -1,
            }

    log: List[Dict] = []

    # Walk month-by-month.
    for i in range(n_implement):
        # Step 1: apply stocking decisions (z[uid, i]=1, q[uid, i])
        for uid in list(state.keys()):
            if decisions["z"].get((uid, i), 0) != 1:
                continue
            q = float(decisions["q"].get((uid, i), 0.0))
            # Overwrite existing cohort entirely (SP rules forbid stocking
            # over a non-empty unit, so we trust the decisions).
            state[uid] = {
                "kind": "new",
                "count": q,
                "weight_g": float(smolt_weight_g),
                "ss": i,
            }
            log.append({
                "uid": uid, **{k: fixed_cols[uid][k] for k in ("location", "unit")},
                "loc": fixed_cols[uid]["location"],
                "event": "stocked_new", "month": i,
                "q": q,
                "weight_g": float(smolt_weight_g),
            })

        # Step 2: apply harvest decisions at month i
        # 2a: existing-cohort harvest
        for uid in list(state.keys()):
            if decisions["h_exist"].get((uid, i), 0) != 1:
                continue
            cur = state[uid]
            if cur is None or cur["kind"] != "existing":
                continue
            log.append({
                "uid": uid, **{k: fixed_cols[uid][k] for k in ("location", "unit")},
                "loc": fixed_cols[uid]["location"],
                "event": "harvest_existing", "month": i,
                "count_before": cur["count"],
                "weight_g_before": cur["weight_g"],
            })
            state[uid] = None

        # 2b: new-cohort harvest (h[uid, ss, t]=1 with t==i)
        for (uid, ss, t), val in decisions["h"].items():
            if val != 1 or t != i:
                continue
            cur = state.get(uid)
            if cur is None or cur["kind"] != "new" or cur["ss"] != ss:
                # Harvesting a cohort we no longer have (or never had) —
                # log a warning but continue.  This shouldn't happen if
                # the SP plan is consistent.
                log.append({
                    "uid": uid,
                    "loc": fixed_cols[uid]["location"],
                    "unit": fixed_cols[uid]["unit"],
                    "event": "harvest_new_INCONSISTENT",
                    "month": i, "ss": int(ss),
                    "note": "no matching new cohort in roster",
                })
                continue
            log.append({
                "uid": uid, **{k: fixed_cols[uid][k] for k in ("location", "unit")},
                "loc": fixed_cols[uid]["location"],
                "event": "harvest_new", "month": i, "ss": int(ss),
                "count_before": cur["count"],
                "weight_g_before": cur["weight_g"],
            })
            state[uid] = None

        # Step 3: advance one month (grow + survival decay)
        t_temp = float(realised_temps[i])
        s_surv = float(realised_S[i])
        for uid, cur in state.items():
            if cur is None:
                continue
            cur["weight_g"] = _next_weight(cur["weight_g"], t_temp, **growth_kwargs)
            cur["count"]    = cur["count"] * s_surv

    # Build the new units_df
    new_rows = []
    for uid, cur in state.items():
        info = fixed_cols[uid]
        if cur is None:
            new_rows.append({
                "unit": info["unit"], "count": float("nan"),
                "avg_weight_g": float("nan"),
                "volume_m3": info["volume_m3"], "location": info["location"],
            })
        else:
            new_rows.append({
                "unit": info["unit"], "count": float(cur["count"]),
                "avg_weight_g": float(cur["weight_g"]),
                "volume_m3": info["volume_m3"], "location": info["location"],
            })

    new_df = pd.DataFrame(new_rows, columns=[
        "unit", "count", "avg_weight_g", "volume_m3", "location"
    ])
    return new_df, log


# ---------------------------------------------------------------------------
# Per-month per-cohort timeline
# ---------------------------------------------------------------------------

def realised_biomass_timeline_long(result: Dict) -> pd.DataFrame:
    """
    Reconstruct a per-month, per-cohort biomass timeline across the
    full real horizon.

    Returns a long DataFrame:
        month, location, unit, count, weight_g, biomass_kg,
        event ("alive" | "harvested" | "stocked"),
        kind  ("existing" | "new"),
        cohort_id (unique str),
        solve_idx (int)

    Compatible with `rh_viz.realised_biomass_timeline` consumers in shape
    (the columns the existing `realized_profit.py` reads — month,
    biomass_kg, event, weight_g — are present), but extended with kind,
    cohort_id, solve_idx for richer downstream analysis.
    """
    rows = []
    smolt_w = 115.0

    # Per-unit state across the full real horizon
    state: Dict[str, Optional[Dict]] = {}
    fixed_cols: Dict[str, Dict] = {}

    for entry in result["real_state_log"]:
        k         = int(entry["solve_idx"])
        t_start   = int(entry["t_start"])
        temps     = np.asarray(entry["realised_temps"], dtype=float)
        S         = np.asarray(entry["realised_S"], dtype=float)
        decisions = entry["decisions"]
        df_before = entry["units_df_before"]
        n_implement = len(temps)

        # Initialise fixed_cols + state from df_before on the first solve.
        if k == 0:
            for _, row in df_before.iterrows():
                uid = _unit_id(row["location"], row["unit"])
                fixed_cols[uid] = {
                    "location": row["location"],
                    "unit":     row["unit"],
                    "volume_m3": row["volume_m3"],
                }
                n0 = row["count"]; w0 = row["avg_weight_g"]
                if pd.isna(n0) or pd.isna(w0):
                    state[uid] = None
                else:
                    state[uid] = {
                        "kind": "existing", "count": float(n0),
                        "weight_g": float(w0), "ss_real": -1,
                        "cohort_id": f"existing__{uid}",
                    }

        for i in range(n_implement):
            month_real = t_start + i

            # 1. Stock at start of month i
            for uid in list(state.keys()):
                if decisions["z"].get((uid, i), 0) != 1:
                    continue
                q = float(decisions["q"].get((uid, i), 0.0))
                cohort_id = f"new__{uid}__solve{k}__m{i}"
                state[uid] = {
                    "kind": "new", "count": q, "weight_g": float(smolt_w),
                    "ss_real": month_real, "ss_in_solve": i,
                    "cohort_id": cohort_id,
                }
                rows.append({
                    "month": month_real,
                    "location": fixed_cols[uid]["location"],
                    "unit":     fixed_cols[uid]["unit"],
                    "count":    q,
                    "weight_g": float(smolt_w),
                    "biomass_kg": q * float(smolt_w) / 1000.0,
                    "event":    "stocked",
                    "kind":     "new",
                    "cohort_id": cohort_id,
                    "solve_idx": k,
                })

            # 2. Snapshot every alive cohort (start-of-month, post-stock)
            for uid, cur in state.items():
                if cur is None:
                    continue
                rows.append({
                    "month": month_real,
                    "location": fixed_cols[uid]["location"],
                    "unit":     fixed_cols[uid]["unit"],
                    "count":    cur["count"],
                    "weight_g": cur["weight_g"],
                    "biomass_kg": cur["count"] * cur["weight_g"] / 1000.0,
                    "event":    "alive",
                    "kind":     cur["kind"],
                    "cohort_id": cur["cohort_id"],
                    "solve_idx": k,
                })

            # 3. Harvest decisions
            # 3a: existing
            for uid in list(state.keys()):
                if decisions["h_exist"].get((uid, i), 0) != 1:
                    continue
                cur = state[uid]
                if cur is None or cur["kind"] != "existing":
                    continue
                rows.append({
                    "month": month_real,
                    "location": fixed_cols[uid]["location"],
                    "unit":     fixed_cols[uid]["unit"],
                    "count":    cur["count"],
                    "weight_g": cur["weight_g"],
                    "biomass_kg": cur["count"] * cur["weight_g"] / 1000.0,
                    "event":    "harvested",
                    "kind":     "existing",
                    "cohort_id": cur["cohort_id"],
                    "solve_idx": k,
                })
                state[uid] = None

            # 3b: new-cohort harvest (uses the in-solve ss index)
            for (uid, ss, t), val in decisions["h"].items():
                if val != 1 or t != i:
                    continue
                cur = state.get(uid)
                if cur is None or cur["kind"] != "new" or cur.get("ss_in_solve") != ss:
                    continue
                rows.append({
                    "month": month_real,
                    "location": fixed_cols[uid]["location"],
                    "unit":     fixed_cols[uid]["unit"],
                    "count":    cur["count"],
                    "weight_g": cur["weight_g"],
                    "biomass_kg": cur["count"] * cur["weight_g"] / 1000.0,
                    "event":    "harvested",
                    "kind":     "new",
                    "cohort_id": cur["cohort_id"],
                    "solve_idx": k,
                })
                state[uid] = None

            # 4. Advance one month
            t_temp = float(temps[i])
            s_surv = float(S[i])
            for uid, cur in state.items():
                if cur is None:
                    continue
                cur["weight_g"] = _next_weight(cur["weight_g"], t_temp)
                cur["count"]    = cur["count"] * s_surv

    return pd.DataFrame(rows)
