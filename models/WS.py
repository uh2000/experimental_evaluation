"""
WS — Wait-and-See solution.

For each of the 81 scenarios, solves the full IP independently (perfect information
on that scenario's temperatures and survival rates), then takes the probability-
weighted average of the optimal objectives.  This is an upper bound on RP.

WS >= RP >= EEV  (standard stochastic programming hierarchy)
VSS = RP - EEV
EVPI = WS - RP

Run:
    python WS.py
"""
import time

from IP import SalmonFarmingMILP
from instance import units_df, loc_mab, regional_mab, T, build_scenarios

MIP_GAP = 0.02


def run_ws():
    scenarios = build_scenarios()
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
    print(f"Feasibility : {n_feas}/81 feasible")
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
