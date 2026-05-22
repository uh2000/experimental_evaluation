"""
Salmon Farming MILP Model (SalmonFarmingMILP)

Mixed-integer linear program for multi-location salmon farming production planning.
Used as the core solver by SP.py (stochastic programming), DE.py (deterministic
equivalent), EEV.py (expected value of the EV solution), and WS.py (wait-and-see).

Key modelling choices:
- Precomputed growth trajectories (TGC model) to keep the MILP fully linear.
- McCormick linearisation of binary × semicontinuous headcount products.
- Weight-class pricing: harvest revenue depends on per-fish weight at harvest time.
- Soft MAB and density constraints with penalty variables to guarantee feasibility.
- Terminal value (discounted) incentivises keeping fish alive at the horizon end.
- Optional emergency slaughter variables for EEV post-processing feasibility recovery.
"""

import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB, quicksum
from typing import Dict, Optional, Tuple
from collections import defaultdict
import time
import matplotlib.pyplot as plt


# ──────────────────────────────────────────────────────────────────────────────
# Default salmon index prices (NOK/kg) by weight class — risk-adjusted
# Each entry: (lower_bound_kg, upper_bound_kg, price_nok_per_kg)
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_PRICES: Dict[str, Tuple[float, float, float]] = {
    "1-2":  (1.0,  2.0,  39.72),
    "2-3":  (2.0,  3.0,  52.66),
    "3-4":  (3.0,  4.0,  60.73),
    "4-5":  (4.0,  5.0,  63.33),
    "5-6":  (5.0,  6.0,  64.55),
    "6-7":  (6.0,  7.0,  64.14),
    "7-8":  (7.0,  8.0,  62.85),
    "8-9":  (8.0,  9.0,  61.32),
    "9+":   (9.0, np.inf, 58.80),
}


class SalmonFarmingMILP:
    """
    Simplified MILP model for salmon farming.
    Single location, single price, short horizon.
    """

    def __init__(
        self,
        units_df: pd.DataFrame,
        temps_t: np.ndarray,
        survival_rates: np.ndarray,
        loc_mab: Dict[str, float],
        regional_mab: float,
        density_limit: float = 25.0,
        smolt_weight_g: float = 115.0,
        horizon_months: int = 60,
        scenario_name: str = "base",
        weight_class_prices: Optional[Dict[str, Tuple[float, float, float]]] = None,
        smolt_cost_per_head: float = 10.0,
        annual_discount_rate: float = 0.1,
        alpha: float = 0.011,
        beta: float = 0.646,
        temp_opt: float = 13.6,
        T_max: float = 23.9,
        days_per_month: int = 30,
        misurv_q_exist_life: int = 0,
        max_remain_months: int = 12,
        maturation_t: int = 24,
        cage_fallow_duration: int = 2,
        loc_fallow_start: Optional[Dict[str, int]] = None,
        loc_fallow_duration: int = 2,
        loc_fallow_cycle: int = 24,
        well_boat_min_weight_g: float = 2000.0,
        terminal_value_per_kg: float = 60.0,
        feed_cost_per_kg_month: float = 1.0,
        biomass_penalty: float = 1_000.0,
        verbose: bool = False,
        disable_economic_presolve: bool = False,
    ):
        self.scenario_name = scenario_name
        self.verbose = verbose
        self.disable_economic_presolve = disable_economic_presolve
        self.T = horizon_months
        self.temps_t = temps_t
        self.S_t = survival_rates

        # Model parameters
        self.density_limit = density_limit
        self.smolt_weight_g = smolt_weight_g
        self.alpha = alpha
        self.beta = beta
        self.temp_opt = temp_opt
        self.T_max = T_max
        self.days_per_month = days_per_month
        self.L_MIN_EXIST = misurv_q_exist_life
        self.L_MAX_EXIST = min(max_remain_months, horizon_months - 1)
        self.maturation_t = maturation_t
        self.cage_fallow_duration = cage_fallow_duration
        self.loc_fallow_start = loc_fallow_start if loc_fallow_start is not None else {
            "Location 1": 12, "Location 2": 13, "Location 3": 14,
            "Location 4": 15, "Location 5": 16, "Location 6": 18,
            "Location 7": 12, "Location 8": 13, "Location 9": 14,
            "Location 10": 15, "Location 11": 16, "Location 12": 18,
        }
        self.loc_fallow_duration = loc_fallow_duration
        self.loc_fallow_cycle = loc_fallow_cycle
        self.well_boat_min_weight_g = well_boat_min_weight_g
        self.terminal_value_per_kg = terminal_value_per_kg
        self.feed_cost_per_kg_month = feed_cost_per_kg_month
        self.biomass_penalty = biomass_penalty

        # Weight-class pricing: sorted list of (lower_kg, upper_kg, price_nok_per_kg)
        _wcp = weight_class_prices if weight_class_prices is not None else DEFAULT_PRICES
        self._price_classes = sorted(_wcp.values(), key=lambda x: x[0])
        self.smolt_cost_per_head = smolt_cost_per_head

        # Discount factors
        monthly_rate = (1 + annual_discount_rate) ** (1 / 12) - 1
        self.df = [(1 + monthly_rate) ** (-t) for t in range(self.T)]

        # Process units data
        self.units_df = units_df.copy()
        # Ensure location names are consistent (e.g., 'Location 1', not 'Loc1')
        n_locs = self.units_df["location"].str.extract(r"Loc(\d+)")[0].dropna().astype(int).max()
        self.units_df["location"] = self.units_df["location"].replace({
            f"Loc{i+1}": f"Location {i+1}" for i in range(n_locs)
        })
        self.units_df["unit_id"] = (
            self.units_df["location"] + " :: " + self.units_df["unit"]
        )

        exist_mask = self.units_df["count"].notna() & self.units_df["avg_weight_g"].notna()
        self.U_exist = self.units_df.loc[exist_mask, "unit_id"].tolist()
        self.U_empty = self.units_df.loc[~exist_mask, "unit_id"].tolist()
        self.U = self.U_exist + self.U_empty

        # Lookup maps
        self.vol = dict(zip(self.units_df["unit_id"], self.units_df["volume_m3"]))
        self.loc_of = dict(zip(self.units_df["unit_id"], self.units_df["location"]))

        
        self.loc_mab = loc_mab
        self.regional_mab = regional_mab

        # Initial states
        self.N0 = dict(zip(
            self.units_df.loc[exist_mask, "unit_id"],
            self.units_df.loc[exist_mask, "count"],
        ))
        self.W0 = dict(zip(
            self.units_df.loc[exist_mask, "unit_id"],
            self.units_df.loc[exist_mask, "avg_weight_g"],
        ))

        self.L_list = sorted(set(self.loc_of.values()))

        self._precompute_trajectories()
        self.model = None
        self.variables = {}
        self._build_model()

        if self.verbose:
            print(f"Built model '{scenario_name}': {len(self.U)} units, {self.T} months")

    # ------------------------------------------------------------------
    def _next_weight(self, w_g: float, t: int) -> float:
        if self.temps_t[t] <= self.temp_opt:
            temp_effective = float(self.temps_t[t]) 
        else: 
            temp_effective = self.temp_opt * (self.T_max - self.temps_t[t]) / (self.T_max - self.temp_opt)
        return w_g + self.days_per_month * self.alpha * temp_effective * (w_g ** self.beta)

    def _precompute_trajectories(self):
        self.wpf_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}
        self.surv_q_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}
        self.biomass_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}

        for u in self.U_exist:
            weight_g = self.W0[u]
            self.wpf_exist[u][0] = weight_g / 1000.0
            self.surv_q_exist[u][0] = self.N0[u]
            self.biomass_exist[u][0] = self.surv_q_exist[u][0] * self.wpf_exist[u][0]
            for t in range(self.T):
                weight_g = self._next_weight(weight_g, t)
                self.wpf_exist[u][t + 1] = weight_g / 1000.0
                self.surv_q_exist[u][t + 1] = self.surv_q_exist[u][t] * self.S_t[t]
                self.biomass_exist[u][t + 1] = (
                    self.surv_q_exist[u][t + 1] * self.wpf_exist[u][t + 1]
                )

        # sbpf[s][t] = (weight_at_t_kg) × surv(s→t): survival-adjusted biomass per fish
        # STOCKED at s, at time t.  It is NOT biomass per fish alive at t — the
        # survival factor is embedded so that sbpf[s][t] × q[u,s] gives total
        # expected standing biomass directly.  x[u,s,t] = h[u,s,t] × q[u,s]
        # is therefore a "stocked-head-count equivalent": when the full cohort is
        # harvested (h=1, x=q), sbpf[s][t] × (q − x) = 0 for all t > tau,
        # which correctly zeroes out the standing biomass.  The survival factor
        # compounding beyond the harvest month cancels in full — no inconsistency.
        self.sbpf = {s: np.zeros(self.T) for s in range(self.T)}
        self.wpf = {s: np.zeros(self.T) for s in range(self.T)}

        for s in range(self.T):
            weight = self.smolt_weight_g
            surv = 1.0
            for t in range(s, self.T):
                if t > s:
                    weight = self._next_weight(weight, t - 1)
                    surv *= self.S_t[t - 1]
                self.wpf[s][t] = weight / 1000.0
                self.sbpf[s][t] = (weight / 1000.0) * surv

    # ------------------------------------------------------------------
    def _price_for_kg(self, weight_kg: float) -> float:
        """Look up the harvest price (NOK/kg) for the given per-fish weight in kg."""
        for lb, ub, price in self._price_classes:
            if lb <= weight_kg < ub:
                return price
        # Below the smallest class — use the lowest class price
        return self._price_classes[0][2]

    # ------------------------------------------------------------------
    def _build_model(self):
        model = gb.Model(f"Salmon_{self.scenario_name}")
        model.Params.OutputFlag = 0  # suppress Gurobi console output

        Tset = list(range(self.T))  # index set of all planning months [0, T)

        Q_MAX = 1_000_000  # global upper bound on fish stocked per cohort, fallback

        cap_kg_per_unit = {u: self.density_limit * self.vol[u] for u in self.U}

        # Compute earliest harvest month per stocking time: first t where fish reach
        # well_boat_min_weight_g.  Well-boats and slaughterhouses cannot process salmon
        # below this weight — so harvest is physically impossible before this month.
        earliest_harvest_t = {}
        for s in Tset:
            earliest_harvest_t[s] = self.T  # sentinel: never reaches weight in horizon
            for t in range(s, min(s + self.maturation_t + 1, self.T)):
                if self.wpf[s][t] >= self.well_boat_min_weight_g / 1000.0:
                    earliest_harvest_t[s] = t
                    break

        # A = full alive range: (u,s,t) where cohort stocked at s is alive at t.
        # maturation_t caps the maximum time fish can remain in water.
        A = [
            (u, s, t)
            for u in self.U
            for s in Tset
            for t in Tset
            if s <= t <= s + self.maturation_t
        ]
        A_set = set(A)

        # H = harvest-eligible subset: fish must reach well_boat_min_weight_g first.
        H = [
            (u, s, t)
            for u in self.U
            for s in Tset
            for t in Tset
            if earliest_harvest_t[s] <= t <= s + self.maturation_t
        ]
        H_set = set(H)

        H_by_us = {(u, s): [] for u in self.U for s in Tset}
        for u, s, t in H:
            H_by_us[(u, s)].append(t)

        # Reverse lookup indices — built once from the lists, eliminating O(|Tset|)
        # set-membership checks inside every constraint loop.
        #
        #  A_by_ut[(u,t)]  → [s, ...]  all stocking times alive at (u,t)
        #  H_by_ut[(u,t)]  → [s, ...]  cohorts with valid harvest at (u,t)
        #  prev_x[(u,t)]   → [(s,τ),…] (s,τ) pairs where τ<t — used in density/MAB
        A_by_ut = defaultdict(list)
        H_by_ut = defaultdict(list)
        prev_x  = defaultdict(list)   # (u,t) -> [(s, tau)]

        for u, s, t in A:
            A_by_ut[(u, t)].append(s)
        for u, s, t in H:
            H_by_ut[(u, t)].append(s)
        # prev_x[(u,t)]: for each harvest-window cohort active at t, the (s,τ)
        # pairs whose harvest variable x[u,s,τ] reduces standing biomass at t.
        for u, s, t in A:
            for tau in H_by_us[(u, s)]:
                if tau < t:
                    prev_x[(u, t)].append((s, tau))

        # === VARIABLES ===
        z = model.addVars(self.U, Tset, vtype=GRB.BINARY, name="z")                        # stocking timing: 1 if cohort stocked in cage u at month s
        q = model.addVars(self.U, Tset, lb=1.0, ub=Q_MAX, vtype=GRB.SEMICONT, name="q")  # stocking headcount: 0 or [1, Q_MAX] (SEMICONT tightens LP relaxation)
        h = model.addVars(H, vtype=GRB.BINARY, name="h")                                  # harvest timing: 1 if cohort (u,s) harvested at month t
        x = model.addVars(H, lb=0.0, vtype=GRB.CONTINUOUS, name="x")                     # auxiliary: harvested headcount, McCormick linearisation of h*q
        a = model.addVars(A, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a")             # auxiliary: alive status of new cohort (u,s) at month t; always 0/1 at feasibility
        a_exist = model.addVars(self.U_exist, Tset, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a_exist")  # auxiliary: alive status of existing cohort u at month t
        h_exist = model.addVars(self.U_exist, Tset, vtype=GRB.BINARY, name="h_exist")     # harvest timing for existing cohorts

        # Tightened bounds
        per_unit_qmax = {}
        for u in self.U:
            for s in Tset:
                # Fish must be harvested within maturation_t months of stocking
                horizon_end = min(s + self.maturation_t, self.T - 1)

                # Capacity bound: max fish that fit given peak biomass per fish
                max_sbpf = max(
                    (float(self.sbpf[s][t]) for t in range(s, horizon_end + 1)),
                    default=1e-9,
                )
                max_sbpf = max(max_sbpf, 1e-9)
                cap_bound = max(int(np.floor(cap_kg_per_unit[u] / max_sbpf)), 0)

                # Price-threshold: if the best possible revenue (at max class price or
                # terminal value) cannot recover the discounted stocking cost, skip
                # stocking entirely for this (u, s).  Using the max class price is an
                # upper bound on revenue, so this never cuts an optimal solution.
                # Only harvest-eligible months (≥ well-boat weight) contribute revenue.
                harvest_start = earliest_harvest_t[s]
                _max_class_price = max(p for _, _, p in self._price_classes)
                best_harvest = max(
                    (float(self.sbpf[s][t]) * _max_class_price * self.df[t]
                     for t in range(harvest_start, horizon_end + 1)),
                    default=0.0,
                )
                # Terminal value only applies if the cohort can still be alive at T_end
                best_terminal = (
                    float(self.sbpf[s][self.T - 1]) * self.terminal_value_per_kg
                    if horizon_end >= self.T - 1 else 0.0
                )
                best_revenue = max(best_harvest, best_terminal)

                if best_revenue < self.smolt_cost_per_head * self.df[s]:
                    per_unit_qmax[(u, s)] = 0
                else:
                    per_unit_qmax[(u, s)] = cap_bound

        for u in self.U:
            for s in Tset:
                q[u, s].UB = per_unit_qmax[(u, s)]   # tighter per-cohort UB on the variable itself
                if per_unit_qmax[(u, s)] == 0 and not self.disable_economic_presolve:
                    # No fish can ever be stocked here — fix z=0 so Gurobi presolve
                    # removes this binary and all associated h/a variables.
                    z[u, s].UB = 0
                else:
                    model.addConstr(q[u, s] <= per_unit_qmax[(u, s)] * z[u, s])

        # === CONSTRAINTS ===

        # Harvest at most once per cohort (harvest is optional — unharvested fish
        # carry terminal value at T_end, so the model decides when/whether to harvest)
        for u in self.U:
            for s in Tset:
                allowed_ts = H_by_us[(u, s)]
                if len(allowed_ts) == 0:
                    if not self.disable_economic_presolve:
                        model.addConstr(z[u, s] == 0)
                    # else: no harvest window in this scenario but stocking is still
                    # allowed (EEV evaluation); fish contribute terminal value only.
                else:
                    model.addConstr(
                        quicksum(h[u, s, t] for t in allowed_ts) <= z[u, s]
                    )

        # x[u,s,t] = h[u,s,t] * q[u,s]: McCormick envelope for binary × continuous.
        # Single-binary linearisation gives the convex hull — tightest possible LP relaxation.
        # Cohort biomass at time t is then: bpf[s][t] * (q[u,s] − Σ_{τ<t} x[u,s,τ])
        # which is fully linear with no big-M slack anywhere.
        # addConstrs() batches all three bounds in single API calls (faster than a loop).
        model.addConstrs(
            (x[u, s, t] <= q[u, s] for u, s, t in H), name="x_ub_q"
        )
        model.addConstrs(
            (x[u, s, t] <= per_unit_qmax[(u, s)] * h[u, s, t] for u, s, t in H),
            name="x_ub_h",
        )
        model.addConstrs(
            (x[u, s, t] >= q[u, s] - per_unit_qmax[(u, s)] * (1 - h[u, s, t])
             for u, s, t in H),
            name="x_lb",
        )

        # a propagation — alive status for new cohorts.
        # At stocking time the cohort is alive iff it was stocked.
        # Each subsequent period it loses aliveness if harvested the period before.
        # CONTINUOUS [0,1]: always binary-valued at any feasible integer solution.
        model.addConstrs(
            (a[u, s, s] == z[u, s]
             for u in self.U for s in Tset),
            name="a_init",
        )
        # Propagation splits on whether t-1 is harvest-eligible.
        # If t-1 ∈ H: fish could have been harvested, so subtract h term.
        # If t-1 ∉ H: fish below well-boat weight — alive status just carries forward.
        model.addConstrs(
            (a[u, s, t] == a[u, s, t - 1] - h[u, s, t - 1]
             for u, s, t in A if t > s and (u, s, t - 1) in H_set),
            name="a_prop_h",
        )
        model.addConstrs(
            (a[u, s, t] == a[u, s, t - 1]
             for u, s, t in A if t > s and (u, s, t - 1) not in H_set),
            name="a_prop_alive",
        )

        # Existing cohort harvest (at most once — harvest is optional)
        allowed_harvest_months_exist = set(range(self.L_MIN_EXIST, self.L_MAX_EXIST + 1))
        for u in self.U_exist:
            model.addConstr(quicksum(h_exist[u, t] for t in allowed_harvest_months_exist) <= 1)
            for t in Tset:
                if t not in allowed_harvest_months_exist:
                    model.addConstr(h_exist[u, t] == 0)

        # Auxiliary: realized harvest biomass for existing cohorts.
        # Keeps biomass_exist (up to ~500 t) out of the objective as a binary coefficient;
        # the large constant moves to the constraint matrix instead.
        H_exist_biom = model.addVars(
            self.U_exist, sorted(allowed_harvest_months_exist),
            lb=0.0, vtype=GRB.CONTINUOUS, name="H_exist_biom"
        )
        for u in self.U_exist:
            for t in allowed_harvest_months_exist:
                biomass_t = float(self.biomass_exist[u][t])
                H_exist_biom[u, t].UB = biomass_t
                model.addConstr(H_exist_biom[u, t] == biomass_t * h_exist[u, t])

        # Existing cohort alive tracking — incremental formulation (3 vars per constraint
        # instead of growing cumulative sums of up to T terms).
        for u in self.U_exist:
            model.addConstr(a_exist[u, 0] == 1)
            for t in range(1, self.T):
                model.addConstr(a_exist[u, t] == a_exist[u, t - 1] - h_exist[u, t - 1])

        # One cohort per unit at a time.
        # Uses a (propagated above) instead of a nested cumulative h sum,
        # reducing each constraint from O(T·L) terms to O(L_MAX) terms.
        U_exist_set = set(self.U_exist)
        for u in self.U:
            for t in Tset:
                alive_h = quicksum(a[u, s, t] for s in A_by_ut[(u, t)])
                if u in U_exist_set:
                    model.addConstr(alive_h + a_exist[u, t] <= 1)
                else:
                    model.addConstr(alive_h <= 1)

        # Fallow period: unit cannot be restocked for 2 months after any harvest.
        # H_by_ut replaces the inner `for s in Tset if (u,s,tau) in H_set` scan.
        for u in self.U:
            for t in Tset:
                harvest_terms = []
                for offset in range(1, self.cage_fallow_duration + 1):
                    tau = t - offset
                    if tau < 0:
                        continue
                    harvest_terms.extend(h[u, s, tau] for s in H_by_ut[(u, tau)])
                    if u in U_exist_set and tau in allowed_harvest_months_exist:
                        harvest_terms.append(h_exist[u, tau])
                if harvest_terms:
                    model.addConstr(z[u, t] + quicksum(harvest_terms) <= 1)

        # Location-level mandatory fallow — each location must be completely empty
        # (zero biomass in every cage) for LOC_FALLOW_DURATION consecutive months
        # starting at loc_fallow_start[ell], repeating every LOC_FALLOW_CYCLE months.
        #
        # Two sub-constraints per (unit, fallow month):
        #   1. New cohorts: a[u,s,t] == 0  (cohort must have been harvested before
        #      the fallow; a propagation forces h at t-1 if cohort is alive)
        #   2. Existing cohorts: a_exist[u,t] == 0  (must have been harvested before fallow)

        for ell, fallow_t0 in self.loc_fallow_start.items():
            units_in_loc = [u for u in self.U if self.loc_of[u] == ell]
            fallow_months = []
            t = fallow_t0
            while t < self.T:
                for offset in range(self.loc_fallow_duration):
                    if t + offset < self.T:
                        fallow_months.append(t + offset)
                t += self.loc_fallow_cycle

            for u in units_in_loc:
                for t in fallow_months:
                    if u in U_exist_set:
                        model.addConstr(a_exist[u, t] == 0,
                                    name=f"loc_fallowpf_exist[{u},{t}]")
                    for s in A_by_ut[(u, t)]:
                        model.addConstr(a[u, s, t] == 0,
                                    name=f"loc_fallow_h[{u},{s},{t}]")

        # Soft biomass constraints — slack variables allow MAB/density to be violated
        # at a high penalty, guaranteeing feasibility for any fixed binary pattern
        # while strongly discouraging violations in the optimal solution.
        slack_density = model.addVars(self.U, Tset, lb=0.0, name="slack_density")
        slack_mab_loc = model.addVars(self.L_list, Tset, lb=0.0, name="slack_mab_loc")
        slack_mab_reg = model.addVars(Tset, lb=0.0, name="slack_mab_reg")

        # Density constraints — reverse indices replace O(|Tset|) membership checks.
        # A_by_ut gives active stocking times; prev_x gives (s,τ) pairs to subtract.
        for u in self.U:
            cap_kg = self.density_limit * self.vol[u]
            for t in Tset:
                biom_new = (
                    quicksum(float(self.sbpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                    - quicksum(float(self.sbpf[s][t]) * x[u, s, tau]
                               for s, tau in prev_x[(u, t)])
                )
                biom_old = (
                    float(self.biomass_exist[u][t]) * a_exist[u, t]
                    if u in U_exist_set
                    else 0.0
                )
                model.addConstr(biom_new + biom_old <= cap_kg + slack_density[u, t])

        # Location MAB constraints
        for ell in self.L_list:
            units_in_loc = [u for u in self.U if self.loc_of[u] == ell]
            for t in Tset:
                biom_loc = quicksum(
                    quicksum(float(self.sbpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                    - quicksum(float(self.sbpf[s][t]) * x[u, s, tau]
                               for s, tau in prev_x[(u, t)])
                    + (float(self.biomass_exist[u][t]) * a_exist[u, t] if u in U_exist_set else 0.0)
                    for u in units_in_loc
                )
                model.addConstr(biom_loc <= self.loc_mab[ell] + slack_mab_loc[ell, t])

        # Regional MAB constraint
        for t in Tset:
            biom_reg = quicksum(
                quicksum(float(self.sbpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                - quicksum(float(self.sbpf[s][t]) * x[u, s, tau]
                           for s, tau in prev_x[(u, t)])
                + (float(self.biomass_exist[u][t]) * a_exist[u, t] if u in U_exist_set else 0.0)
                for u in self.U
            )
            model.addConstr(biom_reg <= self.regional_mab + slack_mab_reg[t])

        T_end = self.T - 1

        # end_biomass_min constraint removed: the terminal_val_new / terminal_val_exist
        # terms in the objective already incentivise keeping fish alive at T_end
        # (60 NOK/kg book value), so explicit liquidation prevention is unnecessary
        # and causes large penalties in post-processing when scenarios follow different
        # restocking plans.

        # === OBJECTIVE (weight-class pricing) ===
        # Price depends on per-fish weight at harvest time, looked up from the
        # price-class table.  wpf[s][t] and wpf_exist[u][t] give per-fish weight
        # in kg (actual individual weight, NOT mortality-adjusted).
        # terminal_value_per_kg is undiscounted (going-concern book value at T_end);
        # see terminal_coeff below for the rationale.

        def _price_new(s, t):
            return self._price_for_kg(self.wpf[s][t])

        def _price_exist(u, t):
            return self._price_for_kg(self.wpf_exist[u][t])

        rev_new = quicksum(
            float(self.sbpf[s][t]) * x[u, s, t] * _price_new(s, t) * self.df[t]
            for (u, s, t) in H
        )
        rev_exist = quicksum(
            H_exist_biom[u, t] * _price_exist(u, t) * self.df[t]
            for u in self.U_exist
            for t in allowed_harvest_months_exist
        )
        cost_smolt = quicksum(
            q[u, s] * self.smolt_cost_per_head * self.df[s]
            for u in self.U
            for s in Tset
        )

        # Terminal value is discounted by df[T_end] for consistency with harvest revenue,
        # which is also discounted. Both are on the same NPV basis.
        terminal_coeff = self.terminal_value_per_kg * self.df[T_end]

        terminal_val_new = gb.LinExpr()
        for u in self.U:
            for s in A_by_ut[(u, T_end)]:
                sbpf_end = float(self.sbpf[s][T_end])
                terminal_val_new.add(q[u, s], sbpf_end * terminal_coeff)
                for tau in H_by_us[(u, s)]:
                    if tau <= T_end:
                        terminal_val_new.add(x[u, s, tau], -sbpf_end * terminal_coeff)

        # Terminal value for existing cohorts: if not harvested by T_end, the
        # remaining in-water stock earns terminal_value_per_kg.
        # b_end * a_exist[u,T_end] = b_end - sum_tau (b_end/biomass_exist[u,tau]) * H_exist_biom[u,tau]
        # This keeps large b_end out of the objective as a binary coefficient.
        terminal_val_exist = gb.LinExpr()
        for u in self.U_exist:
            b_end = float(self.biomass_exist[u][T_end])
            terminal_val_exist.addConstant(b_end * terminal_coeff)
            for tau in sorted(allowed_harvest_months_exist):
                b_tau = float(self.biomass_exist[u][tau])
                ratio = b_end / b_tau if b_tau > 1e-9 else 0.0
                terminal_val_exist.add(H_exist_biom[u, tau], -ratio * terminal_coeff)

        feed_cost = quicksum(
            (
                quicksum(float(self.sbpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                - quicksum(float(self.sbpf[s][t]) * x[u, s, tau] for s, tau in prev_x[(u, t)])
                + (float(self.biomass_exist[u][t]) * a_exist[u, t] if u in U_exist_set else 0.0)
            ) * self.feed_cost_per_kg_month * self.df[t]
            for u in self.U
            for t in Tset
        )
        penalty_biomass = self.biomass_penalty * (
            slack_density.sum() + slack_mab_loc.sum() + slack_mab_reg.sum()
        )
        objective = rev_new + rev_exist - cost_smolt - feed_cost + terminal_val_new + terminal_val_exist - penalty_biomass
        model.setObjective(objective, GRB.MAXIMIZE)

        self.model = model
        self.variables = {
            "z": z,
            "q": q,
            "h": h,
            "x": x,
            "a": a,
            "a_exist": a_exist,
            "h_exist": h_exist,
            "H_exist_biom": H_exist_biom,
        }
        self.A_set = A_set
        self.H_set = H_set
        self.H_by_us = H_by_us
        self.H_by_ut = H_by_ut
        self.A_by_ut = A_by_ut
        self.Tset = Tset
        self.earliest_harvest_t = earliest_harvest_t

    # ------------------------------------------------------------------
    def solve(self, time_limit: Optional[float] = None, mip_gap: Optional[float] = None):
        if time_limit is not None:
            self.model.Params.TimeLimit = time_limit
        if mip_gap is not None:
            self.model.Params.MIPGap = mip_gap
        self.model.optimize()
        if self.verbose:
            if self.model.Status == GRB.OPTIMAL:
                print(f"Optimal: Obj = {self.model.ObjVal:.2f}")
            elif self.model.Status == GRB.TIME_LIMIT:
                print(f"Time limit: Best obj = {self.model.ObjBound:.2f}")
            else:
                print(f"Status: {self.model.Status}")

    def get_objective_value(self) -> float:
        if self.model.Status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return self.model.ObjVal
        return float("-inf")


    # ------------------------------------------------------------------
    def plot_biomass(self, output_file: str = None) -> "plt.Figure":
        """Plot total biomass per unit per month after solving.

        Creates one subplot per location showing stacked unit biomass, plus a
        final subplot with the regional total.  MAB limits are drawn as dashed
        red lines on every panel.

        Parameters
        ----------
        output_file : str, optional
            File path to save the figure (e.g. ``"biomass.png"``).  If *None*
            the figure is returned but not saved.

        Returns
        -------
        matplotlib.figure.Figure
        """
        if self.model.SolCount == 0:
            raise RuntimeError(
                "No solution available. Call solve() (or model.optimize()) before plot_biomass()."
            )

        months = list(range(self.T))

        # ------------------------------------------------------------------
        # Extract biomass per unit per month [kg → tonnes]
        # ------------------------------------------------------------------
        unit_biom = {}
        for u in self.U:
            b = np.zeros(self.T)
            # New cohorts: net standing stock = bpf * (q − cumulative harvests up to and
            # including t), so fish harvested at t are excluded from the end-of-period chart.
            for s in self.Tset:
                for t in self.Tset:
                    if (u, s, t) in self.A_set:
                        harvested_through_t = sum(
                            self.variables["x"][u, s, tau].X
                            for tau in self.H_by_us[(u, s)] if tau <= t
                        )
                        b[t] += float(self.sbpf[s][t]) * (
                            self.variables["q"][u, s].X - harvested_through_t
                        )
            # Existing cohort: same logic — subtract if harvested this month
            if u in self.U_exist:
                for t in self.Tset:
                    e_val = self.variables["a_exist"][u, t].X
                    h_val = self.variables["h_exist"][u, t].X
                    b[t] += self.biomass_exist[u][t] * (e_val - h_val)
            unit_biom[u] = b / 1e3  # kg → tonnes

        # ------------------------------------------------------------------
        # Build figure: one subplot per location + one regional subplot
        # ------------------------------------------------------------------
        n_rows = len(self.L_list) + 1
        fig, axes = plt.subplots(n_rows, 1, figsize=(13, 3.2 * n_rows), sharex=True)
        if n_rows == 1:
            axes = [axes]

        cmap = plt.get_cmap("tab10")

        for i, loc in enumerate(self.L_list):
            ax = axes[i]
            units_in_loc = [u for u in self.U if self.loc_of[u] == loc]
            stack = np.vstack([unit_biom[u] for u in units_in_loc])  # (n_units, T)

            # Short label: the part after " :: "
            labels = [
                u.split(" :: ")[-1] if " :: " in u else u for u in units_in_loc
            ]

            ax.stackplot(
                months, stack,
                labels=labels,
                colors=[cmap(j % 10) for j in range(len(units_in_loc))],
                alpha=0.75,
            )

            if loc in self.loc_mab:
                mab_t = self.loc_mab[loc] / 1e3
                ax.axhline(
                    mab_t, color="red", ls="--", lw=1.5,
                    label=f"Location MAB  {mab_t:,.0f} t",
                )

            ax.set_ylabel("Biomass (t)")
            ax.set_title(loc, fontsize=10)
            ax.legend(loc="upper left", fontsize=7, ncol=4, framealpha=0.6)
            ax.grid(True, alpha=0.25)

        # Regional total panel
        ax = axes[-1]
        regional_biom = sum(unit_biom[u] for u in self.U)
        ax.fill_between(months, regional_biom, alpha=0.55, color="steelblue")
        ax.plot(months, regional_biom, color="steelblue", lw=1.8, label="Regional total")
        reg_mab_t = self.regional_mab / 1e3
        ax.axhline(
            reg_mab_t, color="red", ls="--", lw=1.5,
            label=f"Regional MAB  {reg_mab_t:,.0f} t",
        )
        ax.set_ylabel("Biomass (t)")
        ax.set_xlabel("Month")
        ax.set_title("Regional Total", fontsize=10)
        ax.legend(loc="upper left", fontsize=8, framealpha=0.6)
        ax.grid(True, alpha=0.25)

        fig.suptitle(
            f"Biomass over time — {self.scenario_name}",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout()

        if output_file is not None:
            plt.savefig(output_file, dpi=150)
            print(f"Biomass plot saved to {output_file}")

        return fig


# ======================================================================
if __name__ == "__main__":
    from instance import units_df, loc_mab, regional_mab, T, S_normal_v, S_bad_v, temps_normal_12

    # Expected values of the uncertain parameters
    # E[temp] = temps_normal (distributions are symmetric around normal)
    # E[survival] = (2·S_normal + S_bad) / 3  (good temp → higher mortality = S_bad)
    temps_t = np.tile(temps_normal_12, (T // 12) + 1)[:T]
    S_exp   = (2.0 * S_normal_v + S_bad_v) / 3.0
    S_t     = np.full(T, S_exp)


    print("Building large model...")
    t0 = time.time()
    m = SalmonFarmingMILP(
        units_df=units_df,
        temps_t=temps_t,
        survival_rates=S_t,
        horizon_months=T,
        loc_mab=loc_mab,
        regional_mab=regional_mab,
        density_limit=25.0,
        verbose=True,
    )
    t1 = time.time()
    print(f"Build time: {t1 - t0:.2f}s")

    print("Solving...")
    m.model.Params.OutputFlag = 1
    m.model.Params.MIPGap = 0.02
    m.model.optimize()
    t2 = time.time()
    print(f"Solve time: {t2 - t1:.2f}s")
    print(f"Total time: {t2 - t0:.2f}s")
    if m.model.SolCount > 0:
        print(f"Status: {m.model.Status}, Obj: {m.model.ObjVal:.2f}")

        t_last = m.T - 1
        biom_last = sum(
            float(m.sbpf[s][t_last]) * (
                m.variables["q"][u, s].X
                - sum(m.variables["x"][u, s, tau].X
                      for tau in m.H_by_us[(u, s)] if tau <= t_last)
            )
            for u in m.U for s in m.Tset
            if (u, s, t_last) in m.A_set
        ) + sum(
            m.biomass_exist[u][t_last] * (
                m.variables["a_exist"][u, t_last].X
                - m.variables["h_exist"][u, t_last].X
            )
            for u in m.U_exist
        )
        print(f"Total biomass at last period (month {t_last}): {biom_last/1e3:,.1f} tonnes")
        print(f"Initial biomass at t=0: "
              f"{sum(m.biomass_exist[u][0] for u in m.U_exist)/1e3:,.1f} tonnes")

        # m.plot_biomass("stats/monthly_biomass_plot.png")

