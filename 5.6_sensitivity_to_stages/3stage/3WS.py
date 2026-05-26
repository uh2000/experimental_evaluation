"""
WS — Wait-and-See solution.

For each of the 27 scenarios, solves the full IP independently (perfect information
on that scenario's temperatures and survival rates), then takes the probability-
weighted average of the optimal objectives.  This is an upper bound on RP.

WS >= RP >= EEV  (standard stochastic programming hierarchy)
VSS = RP - EEV
EVPI = WS - RP

Run:
    python WS.py
"""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
import time
import numpy as np

from IP import SalmonFarmingMILP
from instance import (units_df, loc_mab, regional_mab, T,
                      temps_bad_12, temps_normal_12, temps_good_12,
                      S_normal_v, S_bad_v)

MIP_GAP = 0.02


def _build_scenarios_3stage():
    """27 scenarios (3³) with stage boundaries 0-19, 20-39, 40-59 matching 3SP.py."""
    labels   = ["bad", "normal", "good"]
    temp_map = {"bad": temps_bad_12, "normal": temps_normal_12, "good": temps_good_12}
    slices   = [list(range(0, 20)), list(range(20, 40)), list(range(40, T))]

    def _tile(arr, n):
        return np.tile(arr, (n // 12) + 1)[:n]

    scenarios = []
    for l1 in labels:
        for l2 in labels:
            for l3 in labels:
                name     = f"s1_{l1}__s2_{l2}__s3_{l3}"
                temps_sc = np.zeros(T)
                S_sc     = np.full(T, S_normal_v)
                seq      = [l1, l2, l3]
                for i, (sl, months) in enumerate(zip(seq, slices)):
                    prev = seq[i - 1] if i > 0 else None
                    surv = S_bad_v if (sl == "good" and prev == "good") else S_normal_v
                    for mm in months:
                        S_sc[mm] = surv
                    temps_sc[months] = _tile(temp_map[sl], len(months))
                scenarios.append((name, temps_sc, S_sc, 1.0 / 27))
    return scenarios


def run_ws():
    scenarios = _build_scenarios_3stage()
    print(f"Building {len(scenarios)} independent scenario models for WS...")

    n_feas  = 0
    ws_obj  = 0.0

    for sc_idx, (sc_name, temps_sc, S_sc, prob) in enumerate(scenarios):
        sc_milp = SalmonFarmingMILP(
            units_df=units_df, temps_t=temps_sc, survival_rates=S_sc,
            horizon_months=T, loc_mab=loc_mab, regional_mab=regional_mab,
            scenario_name=sc_name,
        )
        sc_model = sc_milp.model
        sc_model.Params.OutputFlag = 0
        sc_model.Params.MIPGap     = MIP_GAP
        sc_model.optimize()

        if sc_model.SolCount > 0:
            n_feas  += 1
            ws_obj  += prob * sc_model.ObjVal
            status_str = f"obj={sc_model.ObjVal:>14,.0f}"
        else:
            status_str = "INFEASIBLE"

        print(f"  sc={sc_idx:2d}  {sc_name[:40]:<40}  {status_str}")

    print("\n" + "-" * 70)
    print(f"Feasibility : {n_feas}/{len(scenarios)} feasible")
    print(f"WS (wait-and-see, perfect information): {ws_obj:,.2f}")
    print("=" * 70)
    return ws_obj


if __name__ == "__main__":
    t0 = time.time()
    print("=" * 70)
    print("WS: solving each scenario independently (wait-and-see)")
    print("=" * 70)

    ws = run_ws()

    print(f"Total time: {time.time() - t0:.1f}s")
