"""
Simplified Salmon Farming MILP Model

Stripped-down version of salmon_milp.py for faster PHA testing:
- Single location
- Single price (no premium/base split)
- 20-month horizon
- Fewer units

Designed to work with Progressive Hedging Algorithm (PHA.py).
"""

import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB, quicksum
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


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
        min_exist_life: int = 0,
        max_remain_months: int = 12,
        maturation_t: int = 24,
        well_boat_min_weight_g: float = 2000.0,
        terminal_value_per_kg: float = 60.0,
        feed_cost_per_kg_month: float = 1.0,
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
        self.L_MIN_EXIST = min_exist_life
        self.L_MAX_REMAIN = min(max_remain_months, horizon_months - 1)
        self.maturation_t = maturation_t
        self.well_boat_min_weight_g = well_boat_min_weight_g
        self.terminal_value_per_kg = terminal_value_per_kg
        self.feed_cost_per_kg_month = feed_cost_per_kg_month

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
        self.units_df["location"] = self.units_df["location"].replace({
            f"Loc{i+1}": f"Location {i+1}" for i in range(6)
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

        if isinstance(loc_mab, pd.DataFrame):
            self.loc_mab = dict(zip(loc_mab.iloc[:, 0], loc_mab.iloc[:, 1]))
        else:
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
        self.w_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}
        self.n_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}
        self.b_exist = {u: np.zeros(self.T + 1) for u in self.U_exist}

        for u in self.U_exist:
            self.w_exist[u][0] = self.W0[u]
            self.n_exist[u][0] = self.N0[u]
            self.b_exist[u][0] = self.n_exist[u][0] * self.w_exist[u][0] / 1000.0
            for t in range(self.T):
                self.w_exist[u][t + 1] = self._next_weight(self.w_exist[u][t], t)
                self.n_exist[u][t + 1] = self.n_exist[u][t] * self.S_t[t]
                self.b_exist[u][t + 1] = (
                    self.n_exist[u][t + 1] * self.w_exist[u][t + 1] / 1000.0
                )

        # bpf[s][t] = (weight_at_t / 1000) × surv(s→t): EXPECTED biomass per fish
        # STOCKED at s, at time t.  It is NOT biomass per fish alive at t — the
        # survival factor is embedded so that bpf[s][t] × q[u,s] gives total
        # expected standing biomass directly.  x[u,s,t] = harv[u,s,t] × q[u,s]
        # is therefore a "stocked-head-count equivalent": when the full cohort is
        # harvested (harv=1, x=q), bpf[s][t] × (q − x) = 0 for all t > tau,
        # which correctly zeroes out the standing biomass.  The survival factor
        # compounding beyond the harvest month cancels in full — no inconsistency.
        self.bpf = {s: np.zeros(self.T) for s in range(self.T)}
        self.wpf = {s: np.zeros(self.T) for s in range(self.T)}

        for s in range(self.T):
            w = self.smolt_weight_g
            surv = 1.0
            for t in range(s, self.T):
                if t == s:
                    w = self.smolt_weight_g
                else:
                    w = self._next_weight(w, t - 1)
                if t > s:
                    surv *= self.S_t[t - 1]
                self.wpf[s][t] = w
                self.bpf[s][t] = (w / 1000.0) * surv

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
        m = gb.Model(f"Salmon_{self.scenario_name}")
        m.Params.OutputFlag = 0

        Tset = list(range(self.T))
        Q_MAX = 1_000_000

        cap_kg_per_unit = {u: self.density_limit * self.vol[u] for u in self.U}

        # Compute earliest harvest month per stocking time: first t where fish reach
        # well_boat_min_weight_g.  Well-boats and slaughterhouses cannot process salmon
        # below this weight — so harvest is physically impossible before this month.
        earliest_harvest_t = {}
        for s in Tset:
            earliest_harvest_t[s] = self.T  # sentinel: never reaches weight in horizon
            for t in range(s, min(s + self.maturation_t + 1, self.T)):
                if self.wpf[s][t] >= self.well_boat_min_weight_g:
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

        # A_harv = A: alive tracking covers the full range; H ⊆ A is the harvest-eligible subset.
        A_harv     = A
        A_harv_set = A_set

        # Reverse lookup indices — built once from the lists, eliminating O(|Tset|)
        # set-membership checks inside every constraint loop.
        #
        #  A_by_ut[(u,t)]      → [s, ...]  all stocking times active at (u,t)
        #  A_harv_by_ut[(u,t)] → [s, ...]  all alive cohorts at (u,t) (= A_by_ut)
        #  H_by_ut[(u,t)]      → [s, ...]  cohorts with valid harvest at (u,t)
        #  prev_x[(u,t)]       → [(s,τ),…] (s,τ) pairs where τ<t and (u,s,t)∈A_harv
        #                         — used in density/MAB to subtract cumulative harvests
        A_by_ut      = defaultdict(list)
        A_harv_by_ut = defaultdict(list)
        H_by_ut      = defaultdict(list)
        prev_x       = defaultdict(list)   # (u,t) -> [(s, tau)]

        for u, s, t in A:
            A_by_ut[(u, t)].append(s)
        for u, s, t in A_harv:
            A_harv_by_ut[(u, t)].append(s)
        for u, s, t in H:
            H_by_ut[(u, t)].append(s)
        # prev_x[(u,t)]: for each harvest-window cohort active at t, the (s,τ)
        # pairs whose harvest variable x[u,s,τ] reduces standing biomass at t.
        for u, s, t in A_harv:
            for tau in H_by_us[(u, s)]:
                if tau < t:
                    prev_x[(u, t)].append((s, tau))

        # === VARIABLES ===
        stk = m.addVars(self.U, Tset, vtype=GRB.BINARY, name="z")
        # SEMICONT: q is either exactly 0 (no cohort stocked) or in [1, UB] (cohort exists).
        # Gurobi exploits this to tighten the LP relaxation — no fractional near-zero values.
        q = m.addVars(self.U, Tset, lb=1.0, ub=Q_MAX, vtype=GRB.SEMICONT, name="q")
        harv = m.addVars(H, vtype=GRB.BINARY, name="h")
        # x[u,s,t] = h[u,s,t] * q[u,s]: harvested quantity from cohort (u,s) at time t.
        # McCormick with a single binary h gives the convex hull — no big-M slack.
        x = m.addVars(H, lb=0.0, vtype=GRB.CONTINUOUS, name="x")
        # a_new[u,s,t]: alive status of new cohort (u,s) at time t (harvest window only).
        # Declared CONTINUOUS [0,1] — always integral at a feasible solution, same as e_alive.
        # Propagated by equality constraints below; used in one-cohort constraint.
        a_new = m.addVars(A_harv, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="a_new")
        # e_alive is fully determined by h_exist via equality; CONTINUOUS avoids
        # unnecessary binary branching (it is always integral at a feasible solution).
        e_alive = m.addVars(self.U_exist, Tset, lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="e_alive")
        h_exist = m.addVars(self.U_exist, Tset, vtype=GRB.BINARY, name="h_exist")

        # Tightened bounds
        per_unit_qmax = {}
        for u in self.U:
            for s in Tset:
                # Fish must be harvested within maturation_t months of stocking
                horizon_end = min(s + self.maturation_t, self.T - 1)

                # Capacity bound: max fish that fit given peak biomass per fish
                max_bpf = max(
                    (float(self.bpf[s][t]) for t in range(s, horizon_end + 1)),
                    default=1e-9,
                )
                max_bpf = max(max_bpf, 1e-9)
                cap_bound = max(int(np.floor(cap_kg_per_unit[u] / max_bpf)), 0)

                # Price-threshold: if the best possible revenue (at max class price or
                # terminal value) cannot recover the discounted stocking cost, skip
                # stocking entirely for this (u, s).  Using the max class price is an
                # upper bound on revenue, so this never cuts an optimal solution.
                # Only harvest-eligible months (≥ well-boat weight) contribute revenue.
                harvest_start = earliest_harvest_t[s]
                _max_class_price = max(p for _, _, p in self._price_classes)
                best_harvest = max(
                    (float(self.bpf[s][t]) * _max_class_price * self.df[t]
                     for t in range(harvest_start, horizon_end + 1)),
                    default=0.0,
                )
                # Terminal value only applies if the cohort can still be alive at T_end
                best_terminal = (
                    float(self.bpf[s][self.T - 1]) * self.terminal_value_per_kg
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
                    # No fish can ever be stocked here — fix stk=0 so Gurobi presolve
                    # removes this binary and all associated harv/a_new variables.
                    stk[u, s].UB = 0
                else:
                    m.addConstr(q[u, s] <= per_unit_qmax[(u, s)] * stk[u, s])

        # === CONSTRAINTS ===

        # Harvest at most once per cohort (harvest is optional — unharvested fish
        # carry terminal value at T_end, so the model decides when/whether to harvest)
        for u in self.U:
            for s in Tset:
                allowed_ts = H_by_us[(u, s)]
                if len(allowed_ts) == 0:
                    if not self.disable_economic_presolve:
                        m.addConstr(stk[u, s] == 0)
                    # else: no harvest window in this scenario but stocking is still
                    # allowed (EEV evaluation); fish contribute terminal value only.
                else:
                    m.addConstr(
                        quicksum(harv[u, s, t] for t in allowed_ts) <= stk[u, s]
                    )

        # x[u,s,t] = h[u,s,t] * q[u,s]: McCormick envelope for binary × continuous.
        # Single-binary linearisation gives the convex hull — tightest possible LP relaxation.
        # Cohort biomass at time t is then: bpf[s][t] * (q[u,s] − Σ_{τ<t} x[u,s,τ])
        # which is fully linear with no big-M slack anywhere.
        # addConstrs() batches all three bounds in single API calls (faster than a loop).
        m.addConstrs(
            (x[u, s, t] <= q[u, s] for u, s, t in H), name="x_ub_q"
        )
        m.addConstrs(
            (x[u, s, t] <= per_unit_qmax[(u, s)] * harv[u, s, t] for u, s, t in H),
            name="x_ub_h",
        )
        m.addConstrs(
            (x[u, s, t] >= q[u, s] - per_unit_qmax[(u, s)] * (1 - harv[u, s, t])
             for u, s, t in H),
            name="x_lb",
        )

        # a_new propagation — alive status for new cohorts.
        # At stocking time the cohort is alive iff it was stocked.
        # Each subsequent period it loses aliveness if harvested the period before.
        # CONTINUOUS [0,1]: always binary-valued at any feasible integer solution.
        m.addConstrs(
            (a_new[u, s, s] == stk[u, s]
             for u in self.U for s in Tset),
            name="a_new_init",
        )
        # Emergency early-slaughter binaries (EEV evaluation only).
        # Allows fish to be removed before reaching well-boat weight, at a heavy
        # penalty, so that fallow constraints never render a scenario infeasible.
        if self.disable_economic_presolve:
            h_early = m.addVars(A_harv, vtype=GRB.BINARY, name="h_early")
        else:
            h_early = None

        # Propagation splits on whether t-1 is harvest-eligible.
        # If t-1 ∈ H: fish could have been harvested, so subtract harv term.
        # If t-1 ∉ H: fish below well-boat weight — alive status just carries forward
        #             (or early-slaughtered if h_early is available).
        if h_early is None:
            m.addConstrs(
                (a_new[u, s, t] == a_new[u, s, t - 1] - harv[u, s, t - 1]
                 for u, s, t in A_harv if t > s and (u, s, t - 1) in H_set),
                name="a_new_prop_harv",
            )
            m.addConstrs(
                (a_new[u, s, t] == a_new[u, s, t - 1]
                 for u, s, t in A_harv if t > s and (u, s, t - 1) not in H_set),
                name="a_new_prop_alive",
            )
        else:
            m.addConstrs(
                (a_new[u, s, t] == a_new[u, s, t - 1] - harv[u, s, t - 1] - h_early[u, s, t - 1]
                 for u, s, t in A_harv if t > s and (u, s, t - 1) in H_set),
                name="a_new_prop_harv",
            )
            m.addConstrs(
                (a_new[u, s, t] == a_new[u, s, t - 1] - h_early[u, s, t - 1]
                 for u, s, t in A_harv if t > s and (u, s, t - 1) not in H_set),
                name="a_new_prop_alive",
            )

        # Existing cohort harvest (at most once — harvest is optional)
        allowed_et = set(range(self.L_MIN_EXIST, self.L_MAX_REMAIN + 1))
        for u in self.U_exist:
            m.addConstr(quicksum(h_exist[u, t] for t in allowed_et) <= 1)
            for t in Tset:
                if t not in allowed_et:
                    m.addConstr(h_exist[u, t] == 0)

        # Auxiliary: realized harvest biomass for existing cohorts.
        # Keeps b_exist (up to ~500 t) out of the objective as a binary coefficient;
        # the large constant moves to the constraint matrix instead.
        H_exist_biom = m.addVars(
            self.U_exist, sorted(allowed_et),
            lb=0.0, vtype=GRB.CONTINUOUS, name="H_exist_biom"
        )
        for u in self.U_exist:
            for t in allowed_et:
                b_t = float(self.b_exist[u][t])
                H_exist_biom[u, t].UB = b_t
                m.addConstr(H_exist_biom[u, t] == b_t * h_exist[u, t])

        # Existing cohort alive tracking — incremental formulation (3 vars per constraint
        # instead of growing cumulative sums of up to T terms).
        for u in self.U_exist:
            m.addConstr(e_alive[u, 0] == 1)
            for t in range(1, self.T):
                m.addConstr(e_alive[u, t] == e_alive[u, t - 1] - h_exist[u, t - 1])

        # One cohort per unit at a time.
        # Uses a_new (propagated above) instead of a nested cumulative h sum,
        # reducing each constraint from O(T·L) terms to O(L_MAX) terms.
        U_exist_set = set(self.U_exist)
        for u in self.U:
            for t in Tset:
                alive_harv = quicksum(a_new[u, s, t] for s in A_harv_by_ut[(u, t)])
                if u in U_exist_set:
                    m.addConstr(alive_harv + e_alive[u, t] <= 1)
                else:
                    m.addConstr(alive_harv <= 1)

        # Fallow period: unit cannot be restocked for 2 months after any harvest.
        # H_by_ut replaces the inner `for s in Tset if (u,s,tau) in H_set` scan.
        FALLOW_MONTHS = 2
        for u in self.U:
            for t in Tset:
                harvest_terms = []
                for offset in range(1, FALLOW_MONTHS + 1):
                    tau = t - offset
                    if tau < 0:
                        continue
                    harvest_terms.extend(harv[u, s, tau] for s in H_by_ut[(u, tau)])
                    if u in U_exist_set and tau in allowed_et:
                        harvest_terms.append(h_exist[u, tau])
                if harvest_terms:
                    m.addConstr(stk[u, t] + quicksum(harvest_terms) <= 1)

        # Location-level mandatory fallow — each location must be completely empty
        # (zero biomass in every cage) for LOC_FALLOW_DURATION consecutive months
        # starting at loc_fallow_start[ell], repeating every LOC_FALLOW_CYCLE months.
        #
        # Two sub-constraints per (unit, fallow month):
        #   1. New cohorts: a_new[u,s,t] == 0  (cohort must have been harvested before
        #      the fallow; a_new propagation forces harv at t-1 if cohort is alive)
        #   2. Existing cohorts: e_alive[u,t] == 0  (must have been harvested before fallow)
        loc_fallow_start = {
            "Location 1": 12,
            "Location 2": 13,
            "Location 3": 14,
            "Location 4": 15,
            "Location 5": 16,
            "Location 6": 18,
        }
        LOC_FALLOW_DURATION = 2
        LOC_FALLOW_CYCLE    = 24

        for ell, fallow_t0 in loc_fallow_start.items():
            units_in_loc = [u for u in self.U if self.loc_of[u] == ell]
            # Collect all fallow months in the planning horizon
            fallow_months = []
            t = fallow_t0
            while t < self.T:
                for offset in range(LOC_FALLOW_DURATION):
                    if t + offset < self.T:
                        fallow_months.append(t + offset)
                t += LOC_FALLOW_CYCLE

            for u in units_in_loc:
                for t in fallow_months:
                    if u in U_exist_set:
                        m.addConstr(e_alive[u, t] == 0,
                                    name=f"loc_fallow_exist[{u},{t}]")
                    for s in A_harv_by_ut[(u, t)]:
                        m.addConstr(a_new[u, s, t] == 0,
                                    name=f"loc_fallow_harv[{u},{s},{t}]")

        # Soft biomass constraints — slack variables allow MAB/density to be violated
        # at a high penalty, guaranteeing feasibility for any fixed binary pattern
        # while strongly discouraging violations in the optimal solution.
        BIOMASS_PENALTY = 1_000  # NOK per kg of violation
        slack_density = m.addVars(self.U, Tset, lb=0.0, name="slack_density")
        slack_mab_loc = m.addVars(self.L_list, Tset, lb=0.0, name="slack_mab_loc")
        slack_mab_reg = m.addVars(Tset, lb=0.0, name="slack_mab_reg")

        # Density constraints — reverse indices replace O(|Tset|) membership checks.
        # A_by_ut gives active stocking times; prev_x gives (s,τ) pairs to subtract.
        for u in self.U:
            cap_kg = self.density_limit * self.vol[u]
            for t in Tset:
                biom_new = (
                    quicksum(float(self.bpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                    - quicksum(float(self.bpf[s][t]) * x[u, s, tau]
                               for s, tau in prev_x[(u, t)])
                )
                biom_old = (
                    float(self.b_exist[u][t]) * e_alive[u, t]
                    if u in U_exist_set
                    else 0.0
                )
                m.addConstr(biom_new + biom_old <= cap_kg + slack_density[u, t])

        # Location MAB constraints
        for ell in self.L_list:
            units_in_loc = [u for u in self.U if self.loc_of[u] == ell]
            for t in Tset:
                biom_loc = quicksum(
                    quicksum(float(self.bpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                    - quicksum(float(self.bpf[s][t]) * x[u, s, tau]
                               for s, tau in prev_x[(u, t)])
                    + (float(self.b_exist[u][t]) * e_alive[u, t] if u in U_exist_set else 0.0)
                    for u in units_in_loc
                )
                m.addConstr(biom_loc <= self.loc_mab[ell] + slack_mab_loc[ell, t])

        # Regional MAB constraint
        for t in Tset:
            biom_reg = quicksum(
                quicksum(float(self.bpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                - quicksum(float(self.bpf[s][t]) * x[u, s, tau]
                           for s, tau in prev_x[(u, t)])
                + (float(self.b_exist[u][t]) * e_alive[u, t] if u in U_exist_set else 0.0)
                for u in self.U
            )
            m.addConstr(biom_reg <= self.regional_mab + slack_mab_reg[t])

        T_end = self.T - 1

        # end_biomass_min constraint removed: the terminal_val_new / terminal_val_exist
        # terms in the objective already incentivise keeping fish alive at T_end
        # (60 NOK/kg book value), so explicit liquidation prevention is unnecessary
        # and causes large penalties in post-processing when scenarios follow different
        # restocking plans.

        # === OBJECTIVE (weight-class pricing) ===
        # Price depends on per-fish weight at harvest time, looked up from the
        # price-class table.  wpf[s][t] and w_exist[u][t] give per-fish weight
        # in grams (actual individual weight, NOT mortality-adjusted).
        # terminal_value_per_kg is undiscounted (going-concern book value at T_end);
        # see terminal_coeff below for the rationale.

        def _price_new(s, t):
            return self._price_for_kg(self.wpf[s][t] / 1000.0)

        def _price_exist(u, t):
            return self._price_for_kg(self.w_exist[u][t] / 1000.0)

        rev_new = quicksum(
            float(self.bpf[s][t]) * x[u, s, t] * _price_new(s, t) * self.df[t]
            for (u, s, t) in H
        )
        rev_exist = quicksum(
            H_exist_biom[u, t] * _price_exist(u, t) * self.df[t]
            for u in self.U_exist
            for t in allowed_et
        )
        cost_smolt = quicksum(
            q[u, s] * self.smolt_cost_per_head * self.df[s]
            for u in self.U
            for s in Tset
        )

        # Terminal value uses the UNDISCOUNTED going-concern convention: the in-water
        # asset is booked at its liquidation price at T_end without further NPV
        # adjustment.  Discounting the terminal coefficient by df[T_end] would create
        # an incentive to harvest early even for sub-premium fish — e.g. with a 10%
        # annual rate over T=60 months, 60 × df[59] ≈ 37.5 NOK/kg which is already
        # below the undiscounted non-premium price of 50 NOK/kg, breaking the intended
        # holding incentive.  By leaving terminal_coeff undiscounted, the comparison
        #   bpf[s][T_end] × 60  vs  bpf[s][t_h] × 50 × df[t_h]
        # is always satisfied as long as bpf grows over the remaining horizon, which
        # holds because growth outpaces discounted survival loss for reasonable temps.
        terminal_coeff = self.terminal_value_per_kg

        terminal_val_new = gb.LinExpr()
        for u in self.U:
            for s in A_harv_by_ut[(u, T_end)]:
                bpf_end = float(self.bpf[s][T_end])
                terminal_val_new.add(q[u, s], bpf_end * terminal_coeff)
                for tau in H_by_us[(u, s)]:
                    if tau <= T_end:
                        terminal_val_new.add(x[u, s, tau], -bpf_end * terminal_coeff)

        # Terminal value for existing cohorts: if not harvested by T_end, the
        # remaining in-water stock earns terminal_value_per_kg.
        # b_end * e_alive[u,T_end] = b_end - sum_tau (b_end/b_exist[u,tau]) * H_exist_biom[u,tau]
        # This keeps large b_end out of the objective as a binary coefficient.
        terminal_val_exist = gb.LinExpr()
        for u in self.U_exist:
            b_end = float(self.b_exist[u][T_end])
            terminal_val_exist.addConstant(b_end * terminal_coeff)
            for tau in sorted(allowed_et):
                b_tau = float(self.b_exist[u][tau])
                ratio = b_end / b_tau if b_tau > 1e-9 else 0.0
                terminal_val_exist.add(H_exist_biom[u, tau], -ratio * terminal_coeff)

        feed_cost = quicksum(
            (
                quicksum(float(self.bpf[s][t]) * q[u, s] for s in A_by_ut[(u, t)])
                - quicksum(float(self.bpf[s][t]) * x[u, s, tau] for s, tau in prev_x[(u, t)])
                + (float(self.b_exist[u][t]) * e_alive[u, t] if u in U_exist_set else 0.0)
            ) * self.feed_cost_per_kg_month * self.df[t]
            for u in self.U
            for t in Tset
        )
        penalty_biomass = BIOMASS_PENALTY * (
            slack_density.sum() + slack_mab_loc.sum() + slack_mab_reg.sum()
        )
        # Early-slaughter penalty: 10× biomass penalty so it is used only as last resort.
        EARLY_SLAUGHTER_PENALTY = 10 * BIOMASS_PENALTY
        penalty_early = (
            EARLY_SLAUGHTER_PENALTY * h_early.sum()
            if h_early is not None else 0.0
        )
        objective = rev_new + rev_exist - cost_smolt - feed_cost + terminal_val_new + terminal_val_exist - penalty_biomass - penalty_early
        m.setObjective(objective, GRB.MAXIMIZE)

        self.model = m
        self.variables = {
            "s": stk,
            "q": q,
            "h": harv,
            "x": x,
            "a_new": a_new,
            "e_alive": e_alive,
            "h_exist": h_exist,
            "H_exist_biom": H_exist_biom,
        }
        self.A_set = A_set
        self.A_harv_set = A_harv_set
        self.H_set = H_set
        self.H_by_us = H_by_us
        self.H_by_ut = H_by_ut
        self.A_by_ut = A_by_ut
        self.A_harv_by_ut = A_harv_by_ut
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

    def get_non_anticipative_var_names(
        self, stage_months: List[int], stage_name: str
    ) -> List[str]:
        var_names = []
        stage_set = set(stage_months)

        # Stocking decisions: s and q
        for u in self.U:
            for s in stage_months:
                var_names.append(f"z[{u},{s}]")
                var_names.append(f"q[{u},{s}]")

        # Harvest timing for new cohorts: h[u,s,t] where t is in stage
        for u in self.U:
            for s in self.Tset:
                for t in self.H_by_us.get((u, s), []):
                    if t in stage_set:
                        var_names.append(f"h[{u},{s},{t}]")

        # Harvest timing for existing cohorts: h_exist[u,t] where t is in stage
        for u in self.U_exist:
            for t in stage_months:
                var_names.append(f"h_exist[{u},{t}]")

        return var_names

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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

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
                    if (u, s, t) in self.A_harv_set:
                        harvested_through_t = sum(
                            self.variables["x"][u, s, tau].X
                            for tau in self.H_by_us[(u, s)] if tau <= t
                        )
                        b[t] += float(self.bpf[s][t]) * (
                            self.variables["q"][u, s].X - harvested_through_t
                        )
            # Existing cohort: same logic — subtract if harvested this month
            if u in self.U_exist:
                for t in self.Tset:
                    e_val = self.variables["e_alive"][u, t].X
                    h_val = self.variables["h_exist"][u, t].X
                    b[t] += self.b_exist[u][t] * (e_val - h_val)
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
    import time

    COMPUTE_EEV = False  # Set to True to run the 81-scenario EEV evaluation

    # --- Same instance as SP.py / DE.py: 6 locations (Loc1-6), 5 units each = 30 units ---
    units_rows = [
    {"unit": "Unit 1",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 3",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 4",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 6",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc1"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    # {"unit": "Unit 8",  "count": 187_000.0, "avg_weight_g": 550.0, "volume_m3": 35_000, "location": "Loc1"},
    # {"unit": "Unit 9",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc1"},
    # {"unit": "Unit 10", "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc1"},


    # {"unit": "Unit 1",  "count": 145_000.0, "avg_weight_g": 5700.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 2",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 3",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 4",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 5",  "count": 184_000.0, "avg_weight_g": 890.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 6",  "count": 173_000.0, "avg_weight_g": 1250.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 7",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 8",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},


    # {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},
    # {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc3"},


    # {"unit": "Unit 1",  "count": 152_000.0, "avg_weight_g": 4800.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 2",  "count": 160_000.0, "avg_weight_g": 5100.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},


    # {"unit": "Unit 1",  "count": 148_000.0, "avg_weight_g": 5500.0, "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},


    # {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc6"},
    ]
    units_df = pd.DataFrame(units_rows)

    loc_mab = {
        "Location 1": 4_000_000,
        # "Location 2": 4_300_000,
        # "Location 3": 5_500_000,
        # "Location 4": 5_600_000,
        # "Location 5": 2_600_000,
        # "Location 6": 2_500_000,
    }
    regional_mab = 35_000_000/6




    T = 60

    # Expected values of the uncertain parameters from SP.py
    # (bad/normal/good each with prob 1/3 across 4 stages of 15 months)
    # E[temp] = (bad + normal + good) / 3 = temps_normal (distributions are symmetric)
    temps_exp_12 = np.array([5, 5, 5, 6, 9, 12, 14, 16.5, 15.5, 13, 10, 7.5])
    temps_t = np.tile(temps_exp_12, (T // 12) + 1)[:T]

    # E[survival] = (2·S_normal + S_bad) / 3  (good temp → higher mortality = S_bad)
    S_exp = (2.0 * (1.0 - 0.0002) ** 30 + (1.0 - 0.001) ** 30) / 3.0
    S_t   = np.full(T, S_exp)


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
            float(m.bpf[s][t_last]) * (
                m.variables["q"][u, s].X
                - sum(m.variables["x"][u, s, tau].X
                      for tau in m.H_by_us[(u, s)] if tau <= t_last)
            )
            for u in m.U for s in m.Tset
            if (u, s, t_last) in m.A_harv_set
        ) + sum(
            m.b_exist[u][t_last] * (
                m.variables["e_alive"][u, t_last].X
                - m.variables["h_exist"][u, t_last].X
            )
            for u in m.U_exist
        )
        print(f"Total biomass at last period (month {t_last}): {biom_last/1e3:,.1f} tonnes")
        print(f"Initial biomass at t=0: "
              f"{sum(m.b_exist[u][0] for u in m.U_exist)/1e3:,.1f} tonnes")

        m.plot_biomass("monthly_biomass_plot.png")

    # ======================================================================
    # EEV: Fix EV binary decisions (stages 0-2) → evaluate across 81 scenarios
    # ======================================================================
    if not COMPUTE_EEV:
        print("EEV skipped (set COMPUTE_EEV = True to run)")
    else:
        print("\n" + "=" * 70)
        print("EEV: applying EV binary decisions (stages 0-2) to all 81 scenarios")
        print("=" * 70)

        eev_start = time.time()

        # Stages 0-2 = months 0-44 (the non-anticipative stages in SP.py/DE.py)
        na_months = set(range(0, 45))

        # Extract stocking decisions from the EV solution (round to 0/1).
        # Only stk (z) is fixed — harvest timing (h, h_exist) and quantities (q)
        # are recourse decisions that must adapt to each scenario's growth model.
        ev_stk = {(u, t): round(m.variables["s"][u, t].X)
                      for u in m.U for t in m.Tset if t in na_months}

        print(f"EV stocking decisions fixed: "
              f"{sum(v for v in ev_stk.values())} s=1 "
              f"(harvest timing left free as recourse)")

        # Build 81 scenario models (same structure as SP.py)
        labels          = ["bad", "normal", "good"]
        temps_bad_12    = np.array([3, 3, 3, 4,  7, 10, 12, 14.5, 13.5, 11,  8, 5.5])
        temps_normal_12 = np.array([5, 5, 5, 6,  9, 12, 14, 16.5, 15.5, 13, 10, 7.5])
        temps_good_12   = np.array([7, 7, 7, 8, 11, 14, 16, 18.5, 17.5, 15, 12, 9.5])
        S_normal_v = (1.0 - 0.0002) ** 30
        S_bad_v    = (1.0 - 0.001)  ** 30
        temp_map   = {"bad": temps_bad_12, "normal": temps_normal_12, "good": temps_good_12}
        # High mortality only when this AND preceding stage are both "good"
        stage_slices = [list(range(0, 15)), list(range(15, 30)),
                        list(range(30, 45)), list(range(45, T))]

        def _tile(arr, n):
            return np.tile(arr, (n // 12) + 1)[:n]

        eev_milps, eev_probs, eev_names = [], [], []
        for t1 in labels:
            for t2 in labels:
                for t3 in labels:
                    for t4 in labels:
                        name = f"s1_{t1}__s2_{t2}__s3_{t3}__s4_{t4}"
                        temps_sc = np.zeros(T)
                        S_sc     = np.full(T, S_normal_v)
                        stage_seq = [t1, t2, t3, t4]
                        for i, (sl, months) in enumerate(zip(stage_seq, stage_slices)):
                            prev = stage_seq[i - 1] if i > 0 else None
                            surv = S_bad_v if (sl == "good" and prev == "good") else S_normal_v
                            for mm in months:
                                S_sc[mm] = surv
                            temps_sc[months] = _tile(temp_map[sl], len(months))
                        eev_milps.append(SalmonFarmingMILP(
                            units_df=units_df, temps_t=temps_sc, survival_rates=S_sc,
                            loc_mab=loc_mab, regional_mab=regional_mab,
                            horizon_months=T, scenario_name=name,
                        ))
                        eev_probs.append(1.0 / 81)
                        eev_names.append(name)

        print(f"Built {len(eev_milps)} scenario models for EEV evaluation")

        # Fix EV binary decisions and re-solve each scenario
        n_eev_feas, n_eev_inf, eev_obj = 0, 0, 0.0
        eev_infeasible = []

        for sc_idx, (sc_milp, prob, sc_name) in enumerate(zip(eev_milps, eev_probs, eev_names)):
            sc_model = sc_milp.model
            sc_model.update()   # build name index before getVarByName
            for (u, t), val in ev_stk.items():
                v = sc_model.getVarByName(f"z[{u},{t}]")
                if v is not None:
                    v.LB = val; v.UB = val
            sc_model.Params.OutputFlag = 0
            sc_model.Params.MIPGap     = 0.001
            sc_model.update()
            sc_model.optimize()
            if sc_model.SolCount > 0:
                n_eev_feas += 1
                eev_obj    += prob * sc_model.ObjVal
                status_str  = f"obj={sc_model.ObjVal:>14,.0f}"
            else:
                n_eev_inf  += 1
                eev_infeasible.append(sc_idx)
                status_str  = "INFEASIBLE"
            print(f"  sc={sc_idx:2d}  {sc_name[:40]:<40}  {status_str}")

        eev_time = time.time() - eev_start
        ev_obj   = m.model.ObjVal

        print("\n" + "-" * 70)
        print(f"Feasibility : {n_eev_feas}/81 feasible"
              + (f"  [INFEASIBLE: {eev_infeasible}]" if eev_infeasible else "  [ALL OK]"))
        print(f"EV  (deterministic obj with expected params): {ev_obj:,.2f}")
        print(f"EEV (EV decisions evaluated over 81 scen)  : {eev_obj:,.2f}")
        print(f"EEV computation time: {eev_time:.1f}s")
        print("=" * 70)