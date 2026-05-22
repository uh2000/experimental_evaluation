"""
EEV — Expected value of the EV solution.

Solves the deterministic IP with expected parameters, extracts the first-stage
decisions (z, h, h_exist, q) for stage 0 (months 0-29), fixes them in each
of the 9 scenario models, and re-solves to compute E[obj | EV decisions].

Run:
    python 2EEV.py
"""
import time
import numpy as np

from IP import SalmonFarmingMILP
from instance import (units_df, loc_mab, regional_mab, T,
                      temps_bad_12, temps_normal_12, temps_good_12,
                      S_normal_v, S_bad_v)

MIP_GAP = 0.02

# ── Stage boundary for non-anticipative decisions ─────────────────────────────
NA_MONTHS = set(range(0, 30))   # stages 0 (fix n-1=1 stage, leave stage 1 free)


def _build_scenarios_2stage():
    """9 scenarios (3²) with stage boundaries 0-29, 30-59 matching 2SP.py."""
    labels     = ["bad", "normal", "good"]
    temp_map   = {"bad": temps_bad_12, "normal": temps_normal_12, "good": temps_good_12}
    slices     = [list(range(0, 30)), list(range(30, T))]

    def _tile(arr, n):
        return np.tile(arr, (n // 12) + 1)[:n]

    scenarios = []
    for l1 in labels:
        for l2 in labels:
            name     = f"s1_{l1}__s2_{l2}"
            temps_sc = np.zeros(T)
            S_sc     = np.full(T, S_normal_v)
            seq      = [l1, l2]
            for i, (sl, months) in enumerate(zip(seq, slices)):
                prev = seq[i - 1] if i > 0 else None
                surv = S_bad_v if (sl == "good" and prev == "good") else S_normal_v
                for mm in months:
                    S_sc[mm] = surv
                temps_sc[months] = _tile(temp_map[sl], len(months))
            scenarios.append((name, temps_sc, S_sc, 1.0 / 9))
    return scenarios


def solve_ev_model():
    """Solve the deterministic IP with expected parameters and return the model."""
    temps_exp_12 = np.array([5, 5, 5, 6, 9, 12, 14, 16.5, 15.5, 13, 10, 7.5])
    temps_t = np.tile(temps_exp_12, (T // 12) + 1)[:T]
    S_exp   = (2.0 * (1.0 - 0.0002) ** 30 + (1.0 - 0.001) ** 30) / 3.0
    S_t     = np.full(T, S_exp)

    print("Building EV model (expected parameters)...")
    m = SalmonFarmingMILP(
        units_df=units_df, temps_t=temps_t, survival_rates=S_t,
        horizon_months=T, loc_mab=loc_mab, regional_mab=regional_mab,
        density_limit=25.0, verbose=False,
    )
    m.model.Params.OutputFlag = 1
    m.model.Params.MIPGap     = MIP_GAP
    m.model.optimize()
    if m.model.SolCount == 0:
        raise RuntimeError("EV model infeasible or no solution found.")
    print(f"EV obj: {m.model.ObjVal:,.2f}  (gap {m.model.MIPGap*100:.2f}%)")
    return m


def extract_ev_decisions(m):
    """Extract first-stage decisions from a solved EV model."""
    ev_stk = {
        (u, t): round(m.variables["z"][u, t].X)
        for u in m.U for t in m.Tset if t in NA_MONTHS
    }
    ev_harv = {
        (u, s, t): round(m.variables["h"][u, s, t].X)
        for u in m.U for s in m.Tset
        for t in m.H_by_us.get((u, s), [])
        if t in NA_MONTHS
    }
    ev_hexist = {
        (u, t): round(m.variables["h_exist"][u, t].X)
        for u in m.U_exist for t in m.Tset if t in NA_MONTHS
    }
    # q must be fixed to preserve non-anticipativity — leaving it free lets the
    # EEV re-solve pick scenario-optimal quantities, inflating EEV above RP.
    ev_q = {
        (u, s): m.variables["q"][u, s].X
        for u in m.U for s in m.Tset
        if s in NA_MONTHS and round(m.variables["z"][u, s].X) == 1
    }
    print(f"EV decisions fixed: "
          f"{sum(ev_stk.values())} z=1, "
          f"{sum(ev_harv.values())} h=1, "
          f"{sum(ev_hexist.values())} h_exist=1, "
          f"{len(ev_q)} q values")
    return ev_stk, ev_harv, ev_hexist, ev_q


def run_eev(ev_stk, ev_harv, ev_hexist, ev_q, ev_obj):
    scenarios = _build_scenarios_2stage()
    print(f"\nBuilding {len(scenarios)} scenario models for EEV evaluation...")

    n_feas, n_inf  = 0, 0
    eev_obj        = 0.0
    feas_prob_sum  = 0.0

    def _build_and_fix(sc_name, temps_sc, S_sc):
        milp = SalmonFarmingMILP(
            units_df=units_df, temps_t=temps_sc, survival_rates=S_sc,
            horizon_months=T, loc_mab=loc_mab, regional_mab=regional_mab,
            scenario_name=sc_name,
            disable_economic_presolve=True,   # keep harvest windows open
        )
        model = milp.model
        model.update()
        for (u, t), val in ev_stk.items():
            v = model.getVarByName(f"z[{u},{t}]")
            if v is not None:
                v.LB = val; v.UB = val
        for (u, s, t), val in ev_harv.items():
            v = model.getVarByName(f"h[{u},{s},{t}]")
            if v is not None:
                v.LB = val; v.UB = val
        for (u, t), val in ev_hexist.items():
            v = model.getVarByName(f"h_exist[{u},{t}]")
            if v is not None:
                v.LB = val; v.UB = val
        for (u, s), val in ev_q.items():
            v = model.getVarByName(f"q[{u},{s}]")
            if v is not None:
                v.LB = val; v.UB = val
        model.Params.OutputFlag = 0
        model.Params.MIPGap     = MIP_GAP
        model.update()
        model.optimize()
        return model

    for sc_idx, (sc_name, temps_sc, S_sc, prob) in enumerate(scenarios):
        sc_model = _build_and_fix(sc_name, temps_sc, S_sc)

        if sc_model.SolCount > 0:
            n_feas        += 1
            feas_prob_sum += prob
            eev_obj       += prob * sc_model.ObjVal
            status_str = f"obj={sc_model.ObjVal:>14,.0f}"
        else:
            n_inf += 1
            status_str = "INFEASIBLE"
            sc_model.computeIIS()
            iis_constrs = [c.ConstrName for c in sc_model.getConstrs() if c.IISConstr]
            iis_bounds  = [v.VarName   for v in sc_model.getVars()    if v.IISLB or v.IISUB]
            print(f"  sc={sc_idx:2d}  {sc_name[:40]:<40}  {status_str}")
            print(f"         IIS constraints ({len(iis_constrs)}): {iis_constrs[:10]}"
                  + (" ..." if len(iis_constrs) > 10 else ""))
            print(f"         IIS var bounds  ({len(iis_bounds)}):  {iis_bounds[:10]}"
                  + (" ..." if len(iis_bounds) > 10 else ""))
            continue

        print(f"  sc={sc_idx:2d}  {sc_name[:40]:<40}  {status_str}")

    eev_normalized = eev_obj / feas_prob_sum if feas_prob_sum > 0 else 0.0

    print("\n" + "-" * 70)
    print(f"Feasibility : {n_feas}/{len(scenarios)} feasible" +
          (f"  [{n_inf} INFEASIBLE]" if n_inf else "  [ALL OK]"))
    print(f"EV  (deterministic obj, expected params): {ev_obj:,.2f}")
    print(f"EEV (EV decisions, feasible scenarios)  : {eev_normalized:,.2f}  "
          f"(prob mass = {feas_prob_sum:.4f})")
    print("=" * 70)
    return eev_normalized


if __name__ == "__main__":
    t0  = time.time()
    m   = solve_ev_model()
    ev_obj = m.model.ObjVal

    print("\n" + "=" * 70)
    print("EEV: applying EV decisions (stage 0) to all 9 scenarios")
    print("=" * 70)

    t_eev = time.time()
    ev_stk, ev_harv, ev_hexist, ev_q = extract_ev_decisions(m)
    eev = run_eev(ev_stk, ev_harv, ev_hexist, ev_q, ev_obj)

    print(f"EEV computation time: {time.time() - t_eev:.1f}s")
    print(f"Total time:           {time.time() - t0:.1f}s")
