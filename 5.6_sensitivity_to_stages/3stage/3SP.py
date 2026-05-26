"""
Augmented Lagrangian Decomposition (ALD) for the 3-stage, 27-scenario salmon
farming MILP defined in IP.py.

The algorithm is a binary PH followed by an extensive-form SLP:
  1. Relax non-anticipativity constraints (NACs) and solve 27 subproblems in parallel.
  2. Penalise binary decision variables (z, h, h_exist) that deviate from the
     node consensus x̄, updating Lagrange multipliers each iteration.
  3. Once binary variables reach consensus, fix them and solve for the continuous
     stocking headcount q in the full extensive form — a pure SLP, tractable to solve directly.

Non-anticipativity is enforced per node of the scenario tree, not globally.
Binary variables are shared only among scenarios that pass through the same node:

  Stage 0  (@s0)                — shared across all 27 scenarios
  Stage 1  (@s1_{t1})           — shared within 9-scenario stage-1 branches
  Stage 2  (@s2_{t1}_{t2})      — shared within 3-scenario stage-1+2 branches

q is recourse and intentionally excluded from the NA set — it adapts to each
scenario's growth model after binary decisions are fixed.

Run:
    python sp.py
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'models'))
from collections import defaultdict
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from time import time

from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D as _L2D
from matplotlib.patches import Patch as _MPatch
import numpy as np
import pandas as pd
import gurobipy as gb
from gurobipy import GRB

from IP import SalmonFarmingMILP
from instance import units_df, loc_mab, regional_mab


class AugmentedLagrangianDecomposition:
    """
    Solves the 27-scenario salmon farming MILP via Augmented Lagrangian
    Decomposition. Call solve() to run the algorithm and plot() to visualise
    the resulting biomass plan.
    """

    BINARY_PREFIXES = {"z", "h", "h_exist"}

    def __init__(
        self,
        units_df: pd.DataFrame,
        loc_mab: dict,
        regional_mab: float,
        *,
        T: int = 60,
        K: int = 1_000,
        epsilon_bin: float = 0.01,
        rho_bin: float = 50.0,
        penalty_fraction: float = 0.1,
        rho_increase: float = 1.3,
        rho_max_bin: float = 1e6,
        rho_max_obj_fraction: float = 10.0,
        stall_window: int = 3,
        slam_stall_window: int = 6,
        slam_stall_tol: float = 0.01,
        min_iters_before_slam: int = 10,
        slam_max_dev_threshold: float = 0.5,
        tail_fix_window: int = 5,
        tail_fix_max_disagree: int = 40,
        fix_threshold: int = 3,
        mip_gap: float = 0.00,
        temps_normal: np.ndarray = None,
        temps_bad: np.ndarray = None,
        temps_good: np.ndarray = None,
        S_normal: float = None,
        S_bad: float = None,
        show_subproblem_incumbents: bool = False,
        show_subproblem_mip_progress: bool = False,
        max_parallel_workers: int = None,
    ):
        self.units_df         = units_df
        self.loc_mab          = loc_mab
        self.regional_mab     = regional_mab
        self.T                = T
        self.K                = K
        self.epsilon_bin      = epsilon_bin
        self.rho_bin          = rho_bin
        self.penalty_fraction = penalty_fraction
        self.rho_increase     = rho_increase
        self.rho_max_bin      = rho_max_bin
        self.rho_max_obj_fraction = rho_max_obj_fraction
        self.stall_window          = stall_window
        self.slam_stall_window      = slam_stall_window
        self.slam_stall_tol         = slam_stall_tol
        self.min_iters_before_slam  = min_iters_before_slam
        self.slam_max_dev_threshold = slam_max_dev_threshold
        self.tail_fix_window        = tail_fix_window
        self.tail_fix_max_disagree  = tail_fix_max_disagree
        self.fix_threshold         = fix_threshold
        self.mip_gap          = mip_gap
        self.show_subproblem_incumbents = show_subproblem_incumbents
        self.show_subproblem_mip_progress = show_subproblem_mip_progress

        # Stage boundaries — 3 evenly spaced 20-month stages
        self.stage0_end    = 20
        self.stage1_end    = 40
        self.stage2_end    = T
        self.stage0_months = list(range(0, 20))
        self.stage1_months = list(range(20, 40))
        self.stage2_months = list(range(40, T))

        days = 30

        self.temps_normal = (temps_normal if temps_normal is not None
                             else np.array([5, 5, 5, 6, 9, 12, 14, 16.5, 15.5, 13, 10, 7.5]))

        self.temps_bad    = (temps_bad    if temps_bad    is not None
                             else np.array([3, 3, 3, 4,  7, 10, 12, 14.5, 13.5, 11,  8, 5.5]))

        self.temps_good   = (temps_good   if temps_good   is not None
                             else np.array([7, 7, 7, 8, 11, 14, 16, 18.5, 17.5, 15, 12, 9.5]))

        self.S_normal     = S_normal if S_normal is not None else (1.0 - 0.0002) ** days
        self.S_bad        = S_bad   if S_bad   is not None else (1.0 - 0.001)  ** days
        self.labels       = ["bad", "normal", "good"]

        # Populated by build()
        self.scenarios:                  list       = []
        self.milp_objects:               list       = []
        self.scenario_names:             list       = []
        self.bundle_id:                  list       = []
        self.n_scenarios:                int        = 0
        self.probs:                      np.ndarray = None
        self.all_na_names_ordered:       list       = []  # qualified names (with @node)
        self.name_to_idx:                dict       = {}
        self.V:                          int        = 0
        self.is_binary:                  np.ndarray = None
        self.n_binary_na:                int        = 0
        self.scenario_vidx:              list       = []
        self.var_cache_flat:             list       = []
        self.na_var_names_per_scenario:  list       = []
        self.original_objectives:        list       = []
        self.sol_matrix:                 np.ndarray = None
        self.multipliers:                np.ndarray = None
        self.fixed:                      np.ndarray = None
        self.fix_val:                    np.ndarray = None
        self.unanimous_count:            np.ndarray = None
        self.xbar_binary_history:        dict       = {}
        self.x_bar:                      np.ndarray = None
        self.obj_lin:                    list       = []
        # Tree-aware additions
        self._xbar_W:                    np.ndarray = None  # weight matrix (n_scen, V)
        self._participating:             np.ndarray = None  # bool mask (n_scen, V)
        self._node_scens:                dict       = {}    # node_key -> list[int]
        self._vi_to_node:                list       = []    # vi -> node_key
        self._vi_occurrences:            list       = []    # vi -> list[(scenario_idx, local_var_idx)]

        # Results — populated by solve()
        self.converged:            bool  = False
        self.convergence_history:  list  = []
        self.objective_history:    list  = []
        self.n_fixed_history:      list  = []
        self.avg_dev_bin_history:  list  = []
        self.total_time:           float = 0.0
        self.n_iters:              int   = 0
        self.max_dev_bin:          float = 0.0
        self.avg_dev_bin:          float = 0.0

        # Post-processing results — populated by solve()
        self.eval_obj:             float = 0.0
        self.n_feasible:           int   = 0
        self.n_infeasible:         int   = 0
        self.infeasible_scenarios: list  = []
        self.bin_consensus:        dict  = {}
        self._stock_na:            dict  = {}
        self._harv_na:             dict  = {}
        self._hexist_na:           dict  = {}
        self.q_robust_vals:        dict  = {}
        self.max_parallel_workers  = max_parallel_workers


    # =========================================================================
    # Static helpers
    # =========================================================================

    @staticmethod
    def _tile(monthly_12: np.ndarray, n: int) -> np.ndarray:
        return np.tile(monthly_12, (n // 12) + 1)[:n]

    @staticmethod
    def _stage_surv(label: str, prev_label: str, S_normal: float, S_bad: float) -> float:
        """High mortality only when this AND the preceding stage are both 'good'."""
        return S_bad if (label == "good" and prev_label == "good") else S_normal
        
    @classmethod
    def _is_binary_name(cls, vname: str) -> bool:
        return vname.split("[")[0] in cls.BINARY_PREFIXES

    # =========================================================================
    # build() — build scenario models and variable index
    # =========================================================================

    def build(self):
        """Build all 27 scenario models and the NA variable indexing structures."""
        T       = self.T
        labels  = self.labels
        n_total = len(labels) ** 3   # 27
        temp_map = {
            "bad":    self.temps_bad,
            "normal": self.temps_normal,
            "good":   self.temps_good,
        }

        scenarios, milp_objects, probabilities, scenario_names, bundle_id = [], [], [], [], []

        for l1 in labels:
            for l2 in labels:
                for l3 in labels:
                    name    = f"s1_{l1}__s2_{l2}__s3_{l3}"
                    temps_t = np.zeros(T)
                    S_t     = np.full(T, self.S_normal)
                    stage_seq = [
                        (l1, self.stage0_months),
                        (l2, self.stage1_months),
                        (l3, self.stage2_months),
                    ]
                    for i, (sl, months) in enumerate(stage_seq):
                        prev = stage_seq[i - 1][0] if i > 0 else None
                        surv = self._stage_surv(sl, prev, self.S_normal, self.S_bad)
                        for m in months:
                            S_t[m] = surv
                        temps_t[months] = self._tile(temp_map[sl], len(months))

                    milp = SalmonFarmingMILP(
                        units_df=self.units_df,
                        temps_t=temps_t,
                        survival_rates=S_t,
                        loc_mab=self.loc_mab,
                        regional_mab=self.regional_mab,
                        horizon_months=T,
                        scenario_name=name,
                    )
                    scenarios.append(milp.model)
                    milp_objects.append(milp)
                    probabilities.append(1.0 / n_total)
                    scenario_names.append(name)
                    bundle_id.append(labels.index(l1))

        self.scenarios      = scenarios
        self.milp_objects   = milp_objects
        self.scenario_names = scenario_names
        self.bundle_id      = bundle_id
        self.n_scenarios    = len(scenarios)
        self.probs          = np.array(probabilities, dtype=np.float64)

        print(f"Built {self.n_scenarios} scenarios")

        # Build flat variable index and per-scenario caches
        self._build_variable_index()

        # Parallel setup
        _default = os.cpu_count() or 4
        self._n_workers_actual = min(
            self.n_scenarios,
            self.max_parallel_workers if self.max_parallel_workers is not None else _default,
        )
        self._progress_lock    = threading.Lock()
        self._progress_counter = [0]

        # Cache original (penalty-free) objectives before any setObjective calls
        self.original_objectives = [scenarios[sc_idx].getObjective() for sc_idx in range(self.n_scenarios)]

    def _build_variable_index(self):
        """
        Build NA variable name → global index mapping with correct non-anticipativity.

        Each variable is qualified as "varname@node" where node encodes only the
        information available at that decision stage:
          - Stage-0 vars: "@s0"            (all 27 scenarios share one x̄)
          - Stage-1 vars: "@s1_{l1}"       (9 scenarios per node, 3 nodes)
          - Stage-2 vars: "@s2_{l1}_{l2}"  (3 scenarios per node, 9 nodes)
        """
        labels      = self.labels
        n_scenarios = self.n_scenarios
        n_labels          = len(labels)  # 3

        na_var_names_per_scenario: list[list[str]] = []  # qualified names per scenario

        for sc_idx, milp in enumerate(self.milp_objects):
            l1 = labels[sc_idx // (n_labels ** 2)]
            l2 = labels[(sc_idx // n_labels) % n_labels]

            node_var_pairs: list[tuple[str, str]] = []

            def _add_stage(node, months, _milp=milp):
                mset = set(months)
                for u in _milp.U:
                    for s in months:
                        node_var_pairs.append((node, f"z[{u},{s}]"))
                for u in _milp.U:
                    for s in _milp.Tset:
                        for t in _milp.H_by_us.get((u, s), []):
                            if t in mset:
                                node_var_pairs.append((node, f"h[{u},{s},{t}]"))
                for u in _milp.U_exist:
                    for t in months:
                        node_var_pairs.append((node, f"h_exist[{u},{t}]"))

            # Node labels encode only information available at decision time:
            #   s0              — blind commit, l1 unknown          (1 node,  27 scenarios)
            #   s1_{l1}         — l1 observed, l2 unknown           (3 nodes,  9 scenarios each)
            #   s2_{l1}_{l2}    — l2 observed, l3 unknown           (9 nodes,  3 scenarios each)
            _add_stage("s0",                    self.stage0_months)
            _add_stage(f"s1_{l1}",              self.stage1_months)
            _add_stage(f"s2_{l1}_{l2}",         self.stage2_months)

            # Qualified names: "varname@node"
            qnames = [f"{vn}@{node}" for (node, vn) in node_var_pairs]
            na_var_names_per_scenario.append(qnames)

        # Global qualified name → index (dedup preserves first-seen order)
        all_na_names_ordered: list[str] = []
        name_to_idx: dict[str, int] = {}
        for sc_idx in range(n_scenarios):
            for qn in na_var_names_per_scenario[sc_idx]:
                if qn not in name_to_idx:
                    name_to_idx[qn] = len(all_na_names_ordered)
                    all_na_names_ordered.append(qn)

        V = len(all_na_names_ordered)
        # Strip @node to get the actual variable name for is_binary check
        is_binary = np.array(
            [self._is_binary_name(qn.split("@")[0]) for qn in all_na_names_ordered],
            dtype=bool
        )
        n_binary_na = int(is_binary.sum())
        print(f"NA variables: {V} total ({n_binary_na} binary)")

        # Per-scenario index arrays and Gurobi variable caches
        scenario_vidx:  list[np.ndarray] = []
        var_cache_flat: list[list]       = []
        vi_occurrences: list[list[tuple[int, int]]] = [[] for _ in range(V)]
        for sc_idx in range(n_scenarios):
            model = self.scenarios[sc_idx]
            model.update()
            idxs, vars_s = [], []
            for qn in na_var_names_per_scenario[sc_idx]:
                vi = name_to_idx[qn]
                vn = qn.split("@")[0]  # actual Gurobi variable name (no @node)
                local_i = len(idxs)
                idxs.append(vi)
                vars_s.append(model.getVarByName(vn))
                vi_occurrences[vi].append((sc_idx, local_i))
            scenario_vidx.append(np.array(idxs, dtype=np.int32))
            var_cache_flat.append(vars_s)

        print("Variable cache built (tree-aware)")

        # ── Node → scenario list ──────────────────────────────────────────────
        _node_scens: dict[str, list[int]] = defaultdict(list)
        for sc_idx in range(n_scenarios):
            l1 = labels[sc_idx // (n_labels ** 2)]
            l2 = labels[(sc_idx // n_labels) % n_labels]
            _node_scens["s0"].append(sc_idx)
            _node_scens[f"s1_{l1}"].append(sc_idx)
            _node_scens[f"s2_{l1}_{l2}"].append(sc_idx)
        _node_scens = dict(_node_scens)

        # vi → node name
        _vi_to_node: list[str] = [qn.split("@")[1] for qn in all_na_names_ordered]

        # ── Node-aware weight matrix W[s, vi] ────────────────────────────────
        # W[s, vi] = conditional probability of scenario s within variable vi's node.
        # All scenarios are equally probable (1/27), so conditional prob = 1/|node|.
        W = np.zeros((n_scenarios, V), dtype=np.float64)
        for vi in range(V):
            node  = _vi_to_node[vi]
            scens = _node_scens[node]
            w     = 1.0 / len(scens)
            for sc_idx in scens:
                W[sc_idx, vi] = w

        # ── Participating mask _participating[s, vi] ─────────────────────────
        # True iff scenario s has variable vi in its NA set (i.e. is in vi's node).
        # Needed to avoid spurious deviations and multiplier updates for scenarios
        # that do not participate in a given node variable.
        _participating = np.zeros((n_scenarios, V), dtype=bool)
        for sc_idx in range(n_scenarios):
            for vi in scenario_vidx[sc_idx]:
                _participating[sc_idx, vi] = True

        self.na_var_names_per_scenario = na_var_names_per_scenario
        self.all_na_names_ordered      = all_na_names_ordered
        self.name_to_idx               = name_to_idx
        self.V                         = V
        self.is_binary                 = is_binary
        self.n_binary_na               = n_binary_na

        self.scenario_vidx             = scenario_vidx
        self.var_cache_flat            = var_cache_flat
        self._node_scens               = _node_scens
        self._vi_to_node               = _vi_to_node
        self._vi_occurrences           = vi_occurrences
        self._xbar_W                   = W
        self._participating            = _participating

        # State arrays
        self.sol_matrix      = np.zeros((n_scenarios, V), dtype=np.float64)
        self.multipliers     = np.zeros((n_scenarios, V), dtype=np.float64)
        self.fixed           = np.zeros(V, dtype=bool)
        self.fix_val         = np.zeros(V, dtype=np.float64)
        self.unanimous_count = np.zeros(V, dtype=np.int32)
        self.xbar_binary_history = {v: [] for v in range(V) if is_binary[v]}

    # =========================================================================
    # Augmented objective infrastructure
    # =========================================================================

    def _compute_xbar(self) -> np.ndarray:
        """Node-aware x̄: weighted average within each variable's node."""
        return (self._xbar_W * self.sol_matrix).sum(axis=0)

    def _build_augmented_objectives(self):
        """Initialise obj_lin and set each subproblem objective from the original."""
        self.obj_lin = [
            np.zeros(len(self.scenario_vidx[sc_idx]), dtype=np.float64)
            for sc_idx in range(self.n_scenarios)
        ]
        for sc_idx in range(self.n_scenarios):
            model = self.scenarios[sc_idx]
            orig  = self.original_objectives[sc_idx]
            # ALD augmented objective is orig - λᵀx - ρ/2||x-x̄||². The quadratic
            # penalty ρ/2||x-x̄||² over binaries collapses to a linear term (since
            # x²=x for binary x), so QuadExpr is used as the container but no
            # explicit quadratic terms are added — _update_obj_linear handles the
            # linearised penalty.
            quad  = gb.QuadExpr()
            quad += orig
            model.setObjective(quad, GRB.MAXIMIZE)
            model.update()

    def _update_obj_linear(self, sc_idx: int, x_bar_vec: np.ndarray, lam_matrix: np.ndarray,
                           rho_b: float):
        """
        Update linear penalty coefficients for binary NA variables in scenario sc_idx.

        Binary: new_coeff = -(λ + (ρ/2)(1 - 2x̄))
        """
        vidx      = self.scenario_vidx[sc_idx]
        vars_flat = self.var_cache_flat[sc_idx]
        old       = self.obj_lin[sc_idx]
        model     = self.scenarios[sc_idx]
        fixed     = self.fixed

        changed_vars   = []
        changed_coeffs = []

        for local_i, (vi, var) in enumerate(zip(vidx, vars_flat)):
            if var is None or fixed[vi] or not self.is_binary[vi]:
                continue
            xb  = x_bar_vec[vi]
            lam = lam_matrix[sc_idx, vi]
            new_coeff = -(lam + (rho_b / 2.0) * (1.0 - 2.0 * xb))
            if abs(new_coeff - old[local_i]) > 1e-12:
                old[local_i] = new_coeff
                changed_vars.append(var)
                changed_coeffs.append(new_coeff)

        if changed_vars:
            for var, coeff in zip(changed_vars, changed_coeffs):
                var.Obj = coeff
            model.update()

    # =========================================================================
    # Gurobi progress callback
    # =========================================================================

    def _make_callback(self, sc_idx: int, phase_label: str):
        last_print = [0.0]
        lock       = self._progress_lock

        def cb(model, where):
            if where == GRB.Callback.MIPSOL:
                obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                t   = model.cbGet(GRB.Callback.RUNTIME)
                with lock:
                    print(f"  [{phase_label} sc={sc_idx:2d}] new incumbent {obj:,.0f}  ({t:.1f}s)",
                          flush=True)
            elif where == GRB.Callback.MIP:
                t = model.cbGet(GRB.Callback.RUNTIME)
                if t - last_print[0] >= 30:
                    last_print[0] = t
                    obj  = model.cbGet(GRB.Callback.MIP_OBJBST)
                    bnd  = model.cbGet(GRB.Callback.MIP_OBJBND)
                    nods = model.cbGet(GRB.Callback.MIP_NODCNT)
                    gap  = abs(obj - bnd) / max(abs(obj), 1e-10) * 100
                    with lock:
                        print(f"  [{phase_label} sc={sc_idx:2d}] {t:.0f}s | obj {obj:,.0f} | "
                              f"bnd {bnd:,.0f} | gap {gap:.2f}% | nodes {nods:.0f}", flush=True)
        return cb

    # =========================================================================
    # Deviation helper
    # =========================================================================

    @staticmethod
    def _devs(sol_mat, x_bar_vec, is_binary, participating_mask=None):
        diff = np.abs(sol_mat - x_bar_vec[np.newaxis, :])
        if participating_mask is not None:
            diff = diff.copy()
            diff[~participating_mask] = 0.0
        bin_diff    = diff[:, is_binary]
        max_dev_bin = bin_diff.max()  if bin_diff.size > 0 else 0.0
        avg_dev_bin = bin_diff.mean() if bin_diff.size > 0 else 0.0
        return max_dev_bin, avg_dev_bin

    # =========================================================================
    # solve() — initial solve + PH loop + post-processing
    # =========================================================================

    def solve(self):
        """Run initial solve → PH main loop → post-processing."""
        n_scenarios     = self.n_scenarios
        V               = self.V
        K               = self.K
        probs           = self.probs
        is_binary       = self.is_binary
        n_binary_na     = self.n_binary_na
        executor        = ThreadPoolExecutor(max_workers=self._n_workers_actual)

        # ------------------------------------------------------------------
        # Step 0: Initial solve (original objectives, no penalty)
        # ------------------------------------------------------------------
        total_ph_start = time()

        threads_per_sub = max(1, (os.cpu_count() or 4) // self._n_workers_actual)

        def _init_solve(sc_idx):
            model = self.scenarios[sc_idx]
            model.Params.MIPGap     = self.mip_gap
            model.Params.OutputFlag = 0
            model.Params.Threads    = threads_per_sub
            model.optimize()
            row = np.zeros(V, dtype=np.float64)
            if model.SolCount > 0:
                vidx      = self.scenario_vidx[sc_idx]
                vars_flat = self.var_cache_flat[sc_idx]
                for _, (vi, var) in enumerate(zip(vidx, vars_flat)):
                    if var is not None:
                        row[vi] = var.X
            obj_val = self.original_objectives[sc_idx].getValue() if model.SolCount > 0 else 0.0
            return sc_idx, row, obj_val, model.SolCount > 0

        futures_0   = {executor.submit(_init_solve, sc_idx): sc_idx for sc_idx in range(n_scenarios)}
        init_results_dict = {}
        with tqdm(total=n_scenarios, desc="Step 0 (initial solve)", unit="sc") as bar:
            for fut in as_completed(futures_0):
                sc_idx, row, obj_val, ok = fut.result()
                init_results_dict[sc_idx] = (row, obj_val)
                bar.update(1)
                if not ok:
                    tqdm.write(f"  [Step0 sc={sc_idx}] WARNING: no solution found")

        init_obj = 0.0
        for sc_idx in range(n_scenarios):
            row, obj_val = init_results_dict[sc_idx]
            self.sol_matrix[sc_idx] = row
            init_obj += probs[sc_idx] * obj_val
        tqdm.write(f"Step 0 done — E[obj]: {init_obj:,.0f}")


        # ------------------------------------------------------------------
        # Auto-calibrate rho (node-aware x̄ from the start)
        # ------------------------------------------------------------------
        x_bar      = self._compute_xbar()
        self.x_bar = x_bar
        diff       = self.sol_matrix - x_bar[np.newaxis, :]
        diff[~self._participating] = 0.0  # exclude non-participating (sc_idx, vi) pairs
        bin_sq = (diff ** 2)[:, is_binary]

        sum_sq_bin = bin_sq.sum()
        if sum_sq_bin > 1e-10:
            self.rho_bin = max(2.0 * self.penalty_fraction * abs(init_obj) / (sum_sq_bin / n_scenarios), 10.0)
            self.rho_bin = min(self.rho_bin, 0.05 * abs(init_obj))
        self.rho_max_bin         = self.rho_max_obj_fraction * abs(init_obj)
        self._rho_bin_calibrated = self.rho_bin
        print(f"Auto-calibrated rho_bin={self.rho_bin:.4g}, rho_max_bin={self.rho_max_bin:.4g}")

        # ------------------------------------------------------------------
        # Build augmented objectives (quadratic part uses calibrated rho)
        # ------------------------------------------------------------------
        self._build_augmented_objectives()

        for sc_idx in range(n_scenarios):
            self._update_obj_linear(sc_idx, x_bar, self.multipliers, self.rho_bin)

        # ------------------------------------------------------------------
        # History initialisation
        # ------------------------------------------------------------------
        max_dev_bin0, avg_dev_bin0 = self._devs(
            self.sol_matrix, x_bar, is_binary, self._participating)

        self.convergence_history = [max_dev_bin0]
        self.objective_history   = [init_obj]
        self.n_fixed_history     = [0]
        self.avg_dev_bin_history = [avg_dev_bin0]
        self.disagree_history    = []

        print(f"\nStep 0 | Obj: {init_obj:12.2f} | MaxDev: {max_dev_bin0:.2e} | "
              f"AvgBin: {avg_dev_bin0:.4f} | Fixed: 0")

        # ------------------------------------------------------------------
        # PH main loop
        # ------------------------------------------------------------------
        k                  = 0
        converged          = False
        max_dev_bin        = max_dev_bin0
        avg_dev_bin        = avg_dev_bin0
        _slam_triggered    = False
        _slam_triggered_at = None

        print(f"Starting PH iterations (max {K})")

        while k < K and not converged:
            iter_start = time()
            k += 1
            self._progress_counter[0] = 0
            phase_t = iter_start

            # 10a: Update linear coefficients (only changed entries)
            for sc_idx in range(n_scenarios):
                self._update_obj_linear(sc_idx, x_bar, self.multipliers, self.rho_bin)

            # 10b: Solve subproblems in parallel
            def _ph_solve(sc_idx, _k=k):
                model = self.scenarios[sc_idx]
                model.Params.MIPGap     = self.mip_gap
                model.Params.OutputFlag = 0
                model.Params.Threads    = threads_per_sub
                model.Params.MIPFocus   = 1   # find feasible solutions fast; PH doesn't need optimality proof
                model.Params.TimeLimit = GRB.INFINITY
                vidx      = self.scenario_vidx[sc_idx]
                vars_flat = self.var_cache_flat[sc_idx]
                for _, (vi, var) in enumerate(zip(vidx, vars_flat)):
                    if var is not None:
                        var.Start = self.sol_matrix[sc_idx, vi]

                _t0       = time()
                _last_log = [_t0]
                _last_inc = [None]

                def _cb(__, where):
                    now = time()
                    elapsed = now - _t0
                    # Report new incumbents immediately (any time)
                    if where == GRB.Callback.MIPSOL and self.show_subproblem_incumbents:
                        try:
                            obj = model.cbGet(GRB.Callback.MIPSOL_OBJ)
                            if _last_inc[0] is None or abs(obj - _last_inc[0]) > 1e-6:
                                _last_inc[0] = obj
                                print(
                                    f"  [Iter {_k} sc={sc_idx:02d} {elapsed:.1f}s] "
                                    f"new incumbent  obj={obj:.4e}",
                                    flush=True,
                                )
                        except Exception:
                            pass
                        return
                    # Report B&B progress every 2 s for slow solves (> 1 s)
                    if where != GRB.Callback.MIP or not self.show_subproblem_mip_progress:
                        return
                    if elapsed < 1.0 or now - _last_log[0] < 2.0:
                        return
                    _last_log[0] = now
                    try:
                        best = model.cbGet(GRB.Callback.MIP_OBJBST)
                        bnd  = model.cbGet(GRB.Callback.MIP_OBJBND)
                        gap  = abs(bnd - best) / (abs(best) + 1e-10)
                        node = int(model.cbGet(GRB.Callback.MIP_NODCNT))
                        print(
                            f"  [Iter {_k} sc={sc_idx:02d} {elapsed:.0f}s] "
                            f"gap={gap:.1%}  best={best:.4e}  bnd={bnd:.4e}  nodes={node}",
                            flush=True,
                        )
                    except Exception:
                        pass

                model.optimize(_cb)

                rescued = False
                if model.SolCount == 0:
                    aug_obj = model.getObjective()
                    model.setObjective(self.original_objectives[sc_idx], GRB.MAXIMIZE)
                    model.Params.MIPGap = max(self.mip_gap, 0.05)
                    model.optimize()
                    rescued = model.SolCount > 0
                    model.setObjective(aug_obj, GRB.MAXIMIZE)

                row     = self.sol_matrix[sc_idx].copy()
                obj_val = 0.0
                if model.SolCount > 0:
                    for _, (vi, var) in enumerate(zip(vidx, vars_flat)):
                        if var is not None:
                            row[vi] = self.fix_val[vi] if self.fixed[vi] else var.X
                    obj_val = self.original_objectives[sc_idx].getValue()
                elapsed_total = time() - _t0
                return sc_idx, row, obj_val, (model.SolCount > 0), rescued, elapsed_total

            futures_ph = {executor.submit(_ph_solve, sc_idx): sc_idx for sc_idx in range(n_scenarios)}
            fut_start = {f: time() for f in futures_ph}
            ph_results_dict = {}
            pending = set(futures_ph.keys())
            last_hb = time()
            hb_every_sec = 5.0
            last_progress = 0
            last_completion = time()
            last_stall_alert = time()
            stall_alert_every_sec = 15.0

            while pending:
                done, pending = wait(pending, timeout=1.0, return_when=FIRST_COMPLETED)

                for fut in done:
                    sc_idx, row, obj_val, has_sol, rescued, elapsed_total = fut.result()
                    ph_results_dict[sc_idx] = (sc_idx, row, obj_val, has_sol)
                    last_completion = time()
                    if not has_sol:
                        print(f"  [Iter {k} sc={sc_idx}] NO SOL (status={self.scenarios[sc_idx].Status})", flush=True)
                    elif rescued:
                        print(f"  [Iter {k} sc={sc_idx}] rescued", flush=True)
                    if elapsed_total >= 10.0:
                        print(f"  [Iter {k} sc={sc_idx}] solved in {elapsed_total:.1f}s", flush=True)

                done_count = len(ph_results_dict)
                if done_count != last_progress and (done_count % 5 == 0 or done_count == n_scenarios):
                    print(f"  [Iter {k}] progress: {done_count}/{n_scenarios} done, {len(pending)} running", flush=True)
                    last_progress = done_count

                now = time()
                if pending and (now - last_hb) >= hb_every_sec:
                    pending_ids = sorted(futures_ph[f] for f in pending)
                    pending_preview = ",".join(f"{pid:02d}" for pid in pending_ids[:12])
                    if len(pending_ids) > 12:
                        pending_preview += ",..."
                    oldest_wait = max(now - fut_start[f] for f in pending)
                    print(
                        f"  [Iter {k}] heartbeat: {done_count}/{n_scenarios} done, "
                        f"{len(pending_ids)} running, oldest={oldest_wait:.1f}s | pending: {pending_preview}",
                        flush=True,
                    )
                    last_hb = now

                if pending and (now - last_completion) >= stall_alert_every_sec and (now - last_stall_alert) >= stall_alert_every_sec:
                    pending_with_elapsed = sorted(
                        ((futures_ph[f], now - fut_start[f]) for f in pending),
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    top_slow = ", ".join(f"sc={sid:02d}:{elap:.1f}s" for sid, elap in pending_with_elapsed[:5])
                    print(
                        f"  [Iter {k}] no completions for {now - last_completion:.1f}s; "
                        f"{len(pending)} still running | slowest: {top_slow}",
                        flush=True,
                    )
                    last_stall_alert = now

            print(f"  [Iter {k}] subproblem solves complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()
            ph_results = [ph_results_dict[sc_idx] for sc_idx in range(n_scenarios)]

            new_sol_matrix = np.empty_like(self.sol_matrix)
            iter_obj       = 0.0
            n_no_sol_iter = 0
            for sc_idx, row, obj_val, has_solution in ph_results:
                new_sol_matrix[sc_idx] = row
                iter_obj              += probs[sc_idx] * obj_val
                if not has_solution:
                    n_no_sol_iter += 1
            self.sol_matrix = new_sol_matrix
            self.objective_history.append(iter_obj)

            print(f"  [Iter {k}] result aggregation complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()

            if n_no_sol_iter > 0:
                print(f"  WARNING: {n_no_sol_iter}/{n_scenarios} subproblems had no incumbent in iter {k}.")

            # 10c: Update x̄ — node-aware (single vectorised operation)
            x_bar      = self._compute_xbar()
            self.x_bar = x_bar

            print(f"  [Iter {k}] x_bar update complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()

            # 10d: Variable fixing — node-aware consensus check
            # For a node variable, we check agreement only within the scenarios
            # that share that node (not across all 27).
            newly_fixed = 0
            if n_binary_na > 0:
                bin_cols = np.where(is_binary & ~self.fixed)[0]
                fix_scan_start = time()
                fix_scan_last  = fix_scan_start
                n_bin_cols     = len(bin_cols)
                if n_bin_cols > 0:
                    print(f"  [Iter {k}] variable fixing scan start ({n_bin_cols} binary NA vars)", flush=True)
                for i, vi in enumerate(bin_cols, start=1):
                    node  = self._vi_to_node[vi]
                    scens = self._node_scens[node]
                    vals  = self.sol_matrix[scens, vi]
                    ref_v = np.round(vals[0])
                    if np.all(np.abs(vals - ref_v) < self.epsilon_bin):
                        self.unanimous_count[vi] += 1
                    else:
                        self.unanimous_count[vi] = 0
                    now_fix = time()
                    if now_fix - fix_scan_last >= 5.0:
                        print(
                            f"  [Iter {k}] variable fixing scan: {i}/{n_bin_cols} checked "
                            f"({now_fix - fix_scan_start:.1f}s)",
                            flush=True,
                        )
                        fix_scan_last = now_fix
                to_fix = bin_cols[self.unanimous_count[bin_cols] >= self.fix_threshold]
                n_to_fix = len(to_fix)
                if n_to_fix > 0:
                    print(f"  [Iter {k}] applying bounds for {n_to_fix} newly fixed vars", flush=True)
                apply_fix_last = time()
                for i, vi in enumerate(to_fix, start=1):
                    node  = self._vi_to_node[vi]
                    scens = self._node_scens[node]
                    fv    = float(np.round(self.sol_matrix[scens[0], vi]))
                    self.fixed[vi]   = True
                    self.fix_val[vi] = fv
                    newly_fixed     += 1
                    for sc_idx, local_i in self._vi_occurrences[vi]:
                        v_obj = self.var_cache_flat[sc_idx][local_i]
                        if v_obj is not None:
                            v_obj.LB = fv
                            v_obj.UB = fv
                    now_apply = time()
                    if now_apply - apply_fix_last >= 5.0:
                        print(
                            f"  [Iter {k}] applying fixed bounds: {i}/{n_to_fix} "
                            f"({now_apply - fix_scan_start:.1f}s)",
                            flush=True,
                        )
                        apply_fix_last = now_apply
                if n_to_fix > 0:
                    self.multipliers[:, to_fix] = 0.0

            print(f"  [Iter {k}] variable fixing complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()

            # 10e: Adaptive slam trigger
            if not _slam_triggered and k >= self.min_iters_before_slam:
                recent_disagree = self.disagree_history[-self.slam_stall_window:]
                recent_devs     = self.convergence_history[-self.slam_stall_window:]
                peak_disagree   = max(self.disagree_history) if self.disagree_history else 1
                if len(recent_disagree) >= self.slam_stall_window:
                    disagree_stagnated  = min(recent_disagree) >= (1 - self.slam_stall_tol) * recent_disagree[0]
                    dev_stagnated       = (recent_devs[0] <= 0 or
                                          (recent_devs[0] - min(recent_devs)) / recent_devs[0] < self.slam_stall_tol)
                    made_progress       = recent_disagree[-1] < self.slam_max_dev_threshold * peak_disagree
                    if disagree_stagnated and dev_stagnated and made_progress:
                        _slam_triggered    = True
                        _slam_triggered_at = k
                        tqdm.write(f"  [Iter {k}] Slam triggered — "
                                   f"Disagree stagnated ({recent_disagree[0]}→{min(recent_disagree)}, "
                                   f"peak={peak_disagree}) and MaxDev stagnated "
                                   f"({recent_devs[0]:.3e}→{min(recent_devs):.3e})")

            # 10e: Cycle-breaking slam
            n_slammed = 0
            if _slam_triggered:
                slam_hist_last = time()
                n_hist = len(self.xbar_binary_history)
                for i, (vi, hist) in enumerate(self.xbar_binary_history.items(), start=1):
                    if self.fixed[vi] or len(hist) < self.slam_stall_window:
                        continue
                    recent = hist[-self.slam_stall_window:]
                    if max(recent) - min(recent) > 0.15:
                        slam_v = float(np.round(np.mean(recent)))
                        self.fixed[vi]   = True
                        self.fix_val[vi] = slam_v
                        n_slammed       += 1
                    now_slam = time()
                    if now_slam - slam_hist_last >= 5.0:
                        print(f"  [Iter {k}] slam scan: {i}/{n_hist} checked", flush=True)
                        slam_hist_last = now_slam

            # Consistency repair on newly slammed variables: z=1 must have ≥1 h=1
            _to_apply: set[int] = set()
            if n_slammed > 0:
                _z_slam: dict[tuple, int] = {}
                _h_slam: dict[tuple, list] = {}
                slam_repair_last = time()
                n_all_na = len(self.all_na_names_ordered)
                for i, (vi2, qname2) in enumerate(zip(range(n_all_na), self.all_na_names_ordered), start=1):
                    if not (is_binary[vi2] and self.fixed[vi2]):
                        continue
                    vname2 = qname2.split("@")[0]  # strip @node
                    br2    = vname2.index("[")
                    pfx2   = vname2[:br2]
                    inner2 = vname2[br2+1:-1]
                    if pfx2 == "z":
                        u2, ts2 = inner2.rsplit(",", 1)
                        _z_slam[(u2, int(ts2))] = vi2
                    elif pfx2 == "h":
                        parts2 = inner2.rsplit(",", 2)
                        key2 = (parts2[0], int(parts2[1]))
                        _h_slam.setdefault(key2, []).append((vi2, int(parts2[2])))
                    now_repair = time()
                    if now_repair - slam_repair_last >= 5.0:
                        print(f"  [Iter {k}] slam repair scan: {i}/{n_all_na} vars", flush=True)
                        slam_repair_last = now_repair
                for (u2, ts2), z_vi2 in _z_slam.items():
                    if self.fix_val[z_vi2] < 0.5:
                        _to_apply.add(z_vi2)
                        continue
                    hlist2 = _h_slam.get((u2, ts2), [])
                    if any(self.fix_val[hv2] >= 0.5 for hv2, _ in hlist2):
                        _to_apply.add(z_vi2)
                        for hv2, _ in hlist2:
                            _to_apply.add(hv2)
                        continue
                    if hlist2:
                        best_hvi2, _ = max(hlist2, key=lambda hv2: self.x_bar[hv2[0]])
                        if self.x_bar[best_hvi2] > 0.0:
                            self.fix_val[best_hvi2] = 1.0
                            _to_apply.add(best_hvi2)
                        else:
                            self.fix_val[z_vi2] = 0.0
                    else:
                        self.fix_val[z_vi2] = 0.0
                    _to_apply.add(z_vi2)
                    for hv2, _ in hlist2:
                        _to_apply.add(hv2)

            # Apply bounds for newly slammed + repaired variables
            apply_slam_last = time()
            n_apply_slam = len(_to_apply)
            for i, vi2 in enumerate(_to_apply, start=1):
                fv2 = self.fix_val[vi2]
                for sc_idx, local_i in self._vi_occurrences[vi2]:
                    v_obj = self.var_cache_flat[sc_idx][local_i]
                    if v_obj is not None:
                        v_obj.LB = fv2
                        v_obj.UB = fv2
                now_apply_slam = time()
                if now_apply_slam - apply_slam_last >= 5.0:
                    print(
                        f"  [Iter {k}] applying slam bounds: {i}/{n_apply_slam}",
                        flush=True,
                    )
                    apply_slam_last = now_apply_slam
            if n_apply_slam > 0:
                apply_slam_idx = np.fromiter(_to_apply, dtype=np.int32, count=n_apply_slam)
                self.multipliers[:, apply_slam_idx] = 0.0

            # Update binary x_bar history (node-specific x̄)
            for vi in list(self.xbar_binary_history.keys()):
                if not self.fixed[vi]:
                    self.xbar_binary_history[vi].append(float(x_bar[vi]))

            self.n_fixed_history.append(int(self.fixed.sum()))

            # 10f: Convergence check — apply participating mask before computing deviations
            unfixed_mask = ~self.fixed
            diff_uf      = np.abs(self.sol_matrix - x_bar[np.newaxis, :])
            diff_uf[~self._participating] = 0.0  # zero out non-participating (sc_idx, vi)
            diff_uf[:, self.fixed]        = 0.0
            bin_mask_uf = is_binary & unfixed_mask

            max_dev_bin = diff_uf[:, bin_mask_uf].max()  if bin_mask_uf.any() else 0.0
            avg_dev_bin = diff_uf[:, bin_mask_uf].mean() if bin_mask_uf.any() else 0.0

            self.avg_dev_bin_history.append(avg_dev_bin)
            self.convergence_history.append(max_dev_bin)

            bin_disagree = int((diff_uf[:, bin_mask_uf] > self.epsilon_bin).sum()) if bin_mask_uf.any() else 0
            self.disagree_history.append(bin_disagree)
            converged = max_dev_bin <= self.epsilon_bin

            # Tail-stagnation escape: if disagreement is small but flat for many
            # iterations, force-fix the remaining binary NA vars to rounded x_bar.
            # This prevents very long tails where a handful of binaries never settle.
            tail_fixed = 0
            if (
                not converged
                and _slam_triggered
                and bin_disagree > 0
                and bin_disagree <= self.tail_fix_max_disagree
                and len(self.disagree_history) >= self.tail_fix_window
            ):
                recent_disagree = self.disagree_history[-self.tail_fix_window:]
                if min(recent_disagree) == max(recent_disagree):
                    rem_bin = np.where(is_binary & ~self.fixed)[0]
                    if rem_bin.size > 0:
                        print(
                            f"  [Iter {k}] tail stall detected: disagree={bin_disagree} "
                            f"for {self.tail_fix_window} iters -> force-fixing {rem_bin.size} vars",
                            flush=True,
                        )
                    for vi in rem_bin:
                        fv = float(np.round(x_bar[vi]))
                        self.fixed[vi] = True
                        self.fix_val[vi] = fv
                        tail_fixed += 1
                        for sc_idx, local_i in self._vi_occurrences[vi]:
                            v_obj = self.var_cache_flat[sc_idx][local_i]
                            if v_obj is not None:
                                v_obj.LB = fv
                                v_obj.UB = fv
                    if tail_fixed > 0:
                        self.multipliers[:, rem_bin] = 0.0
                        newly_fixed += tail_fixed
                        converged = True

            print(f"  [Iter {k}] convergence metrics complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()

            elapsed_iter = time() - iter_start
            slam_state = (
                f"@{_slam_triggered_at}" if _slam_triggered
                else (f"rdy<{self.slam_max_dev_threshold}"
                      if max_dev_bin < self.slam_max_dev_threshold else "wait")
            )
            print(
                f"Iter {k:4d}/{K} | obj={iter_obj:.3e} | BinDev={max_dev_bin:.2e} | "
                f"Disagree={bin_disagree} | Fixed={self.fixed.sum()}(+{newly_fixed}f+{n_slammed}s) | "
                f"rho_bin={self.rho_bin:.1e} | slam={slam_state} | t={elapsed_iter:.1f}s",
                flush=True,
            )

            # 10g: Multiplier update — zero out non-participating (sc_idx, vi) pairs
            self.multipliers += self.rho_bin * is_binary[np.newaxis, :] * (self.sol_matrix - x_bar[np.newaxis, :])
            self.multipliers[~self._participating] = 0.0  # only penalise participating pairs
            self.multipliers[:, self.fixed]        = 0.0

            print(f"  [Iter {k}] multiplier update complete in {time() - phase_t:.1f}s", flush=True)
            phase_t = time()

            # 10h: Adaptive rho increase if stalled
            if k >= self.stall_window and len(self.disagree_history) >= self.stall_window:
                recent_d = self.disagree_history[-self.stall_window:]
                if min(recent_d) > 0.8 * recent_d[0]:
                    self.rho_bin = min(self.rho_bin * self.rho_increase, self.rho_max_bin)
                elif recent_d[-1] < 0.8 * recent_d[0]:
                    self.rho_bin = max(self.rho_bin / self.rho_increase, self._rho_bin_calibrated)

            print(f"  [Iter {k}] rho adaptation complete in {time() - phase_t:.1f}s", flush=True)

        self.converged    = converged
        self.n_iters      = k
        self.total_time   = time() - total_ph_start
        self.max_dev_bin = max_dev_bin
        self.avg_dev_bin = avg_dev_bin

        self._post_SLP(executor)
        executor.shutdown(wait=False)

    # =========================================================================
    # _post_SLP — robust q SLP + per-scenario re-solve
    # =========================================================================

    def _post_SLP(self, executor):
        """Robust q LP across all scenarios, then per-scenario re-solve for stage-2 decisions."""
        n_scenarios = self.n_scenarios
        V           = self.V
        T           = self.T
        probs       = self.probs
        is_binary   = self.is_binary

        print("\n" + "=" * 70)
        print("POST-PROCESSING: robust q LP (all scenarios) + per-scenario re-solve")
        print("=" * 70)
        eval_start = time()

        # --- Binary consensus ---
        bin_consensus: dict[int, float] = {}
        for vi in range(V):
            if not is_binary[vi]:
                continue
            bin_consensus[vi] = self.fix_val[vi] if self.fixed[vi] else float(np.round(self.x_bar[vi]))

        n_hard = int(self.fixed.sum())
        n_soft = len(bin_consensus) - n_hard
        print(f"Consensus binary NA vars: {n_hard} hard-fixed by PH, {n_soft} from rounded x̄")

        self.bin_consensus = bin_consensus

        # Consistency repair: h[u,ss,th]=1 is impossible if z[u,ss]=0 in the same node.
        # PH fixes z and h independently so they can end up contradicting each other.
        # Note: h may be at a finer node granularity than z (e.g. h is stage-2, z is stage-1),
        # so we derive z's node key from h's node key by stripping the extra stage component.
        n_repaired = 0
        for vi, qname in enumerate(self.all_na_names_ordered):
            if not is_binary[vi] or bin_consensus.get(vi, 0.0) < 0.5:
                continue
            vname = qname.split("@")[0]
            if not vname.startswith("h["):
                continue
            h_nk  = qname.split("@")[1]
            inner = vname[2:-1]                        # "u,ss,th"
            u_str, ss_str, _ = inner.rsplit(",", 2)
            ss    = int(ss_str)
            # Derive the node key at the stage ss belongs to (z's granularity)
            p = h_nk.split("_")                        # e.g. ["s2","normal","good"]
            if ss < self.stage0_end:
                z_nk = "s0"
            elif ss < self.stage1_end:
                z_nk = f"s1_{p[1]}"
            else:
                z_nk = h_nk                            # same stage — key matches directly
            z_vi = self.name_to_idx.get(f"z[{u_str},{ss_str}]@{z_nk}")
            if z_vi is not None and bin_consensus.get(z_vi, 0.0) < 0.5:
                bin_consensus[vi] = 0.0
                n_repaired += 1
        if n_repaired:
            print(f"  [post-process] consistency repair: zeroed {n_repaired} h vars where z=0")

        # Apply binary bounds + restore penalty-free objectives
        for sc_idx in range(n_scenarios):
            model     = self.scenarios[sc_idx]
            vidx      = self.scenario_vidx[sc_idx]
            vars_flat = self.var_cache_flat[sc_idx]
            for _, (vi, var) in enumerate(zip(vidx, vars_flat)):
                if var is None or not is_binary[vi]:
                    continue
                val = bin_consensus[vi]
                var.LB = val
                var.UB = val
            model.setObjective(self.original_objectives[sc_idx], GRB.MAXIMIZE)
            model.update()

        # Parse consensus into stocking / harvest event sets (strip @node)
        _stock_na  = {}
        _harv_na   = {}
        _hexist_na = {}
        for vi, qname in enumerate(self.all_na_names_ordered):
            if not is_binary[vi] or bin_consensus.get(vi, 0.0) < 0.5:
                continue
            vname = qname.split("@")[0]  # strip @node
            br    = vname.index("[")
            pfx   = vname[:br]
            inner = vname[br+1:-1]
            if pfx == "z":
                u, ts = inner.rsplit(",", 1)
                _stock_na[(u, int(ts))] = True
            elif pfx == "h":
                p = inner.rsplit(",", 2)
                _harv_na[(p[0], int(p[1]), int(p[2]))] = True
            elif pfx == "h_exist":
                u, ts = inner.rsplit(",", 1)
                _hexist_na[(u, int(ts))] = True

        self._stock_na  = _stock_na
        self._harv_na   = _harv_na
        self._hexist_na = _hexist_na

        print(f"Active NA cohorts: {len(_stock_na)} stocked, "
              f"{len(_harv_na)} NA harvests, {len(_hexist_na)} exist. harvests")

        # --- Alive flags ---
        _alive: dict = {}
        _ref   = self.milp_objects[0]
        for (u, ss) in _stock_na:
            harvest_times = [t for (uh, ssh, t) in self._harv_na if uh == u and ssh == ss]
            t_h = harvest_times[0] if harvest_times else T
            alive_periods = [t for t in range(ss, t_h) if t < T]
            if alive_periods:
                _alive[(u, ss)] = alive_periods

        _alive_exist: dict = {}
        for u in _ref.U_exist:
            harvest_times = [t for (uh, t) in self._hexist_na if uh == u]
            t_h = harvest_times[0] if harvest_times else T
            alive_periods = [t for t in range(0, t_h) if t < T]
            if alive_periods:
                _alive_exist[u] = alive_periods

        # ================================================================
        # Three-stage stochastic LP for q (deterministic equivalent with NACs)
        # ================================================================
        recourse_q = gb.Model("recourse_q")
        recourse_q.Params.OutputFlag = 0
        T_end = T - 1
        Q_MAX_ROB = 1_000_000
        MAB_PENALTY = 1000

        _q_keys: set = set()
        for (u, ss) in _stock_na:
            _q_keys.add((u, ss))

        # Helpers: derive which tree node scenario s is in at time ss.
        _n_labels = len(self.labels)
        def _node_key(sc_idx: int, ss: int) -> str:
            l1 = self.labels[sc_idx // (_n_labels ** 2)]
            l2 = self.labels[(sc_idx // _n_labels) % _n_labels]
            if ss < self.stage0_end:
                return "s0"
            elif ss < self.stage1_end:
                return f"s1_{l1}"
            else:
                return f"s2_{l1}_{l2}"

        def _z_cons(sc_idx: int, u, ss: int) -> float:
            nk = _node_key(sc_idx, ss)
            vi = self.name_to_idx.get(f"z[{u},{ss}]@{nk}")
            return bin_consensus.get(vi, 0.0) if vi is not None else 0.0

        # Build the set of (node_key, u, ss) for all node-cohort combinations where z=1.
        # q_rob is indexed by node so that the NAC reads:
        #   all scenarios s passing through node n share q_s[u,ss] = q_rob[n, u, ss].
        _q_node_keys: set = set()
        for sc_idx in range(n_scenarios):
            for (u, ss) in _q_keys:
                if _z_cons(sc_idx, u, ss) >= 0.5:
                    _q_node_keys.add((_node_key(sc_idx, ss), u, ss))

        # UB per (node_key, u, ss): min over scenarios in that node.
        _q_node_ub: dict = {}
        for (nk, u, ss) in _q_node_keys:
            min_ub = Q_MAX_ROB
            for _ms in self.milp_objects:
                if (u, ss) in _ms.variables["q"]:
                    min_ub = min(min_ub, _ms.variables["q"][u, ss].UB)
            _q_node_ub[(nk, u, ss)] = min_ub

        _q_node_keys_list = list(_q_node_keys)
        q_rob = recourse_q.addVars(
            _q_node_keys_list,
            lb=0,
            ub=[_q_node_ub[k] for k in _q_node_keys_list],
            name=lambda k: f"q_rob_{k[0]}_{k[1]}_{k[2]}"
        )

        _ref0 = self.milp_objects[0]
        mab_slack_loc = recourse_q.addVars(
            _ref0.L_list, range(T), self.n_scenarios,
            lb=0, name="mab_slack_loc"
        )
        mab_slack_reg = recourse_q.addVars(
            range(T), self.n_scenarios,
            lb=0, name="mab_slack_reg"
        )
        dens_slack_new = recourse_q.addVars(
            _ref0.U, range(T), self.n_scenarios,
            lb=0, name="dens_slack_new"
        )
        dens_slack_exist = recourse_q.addVars(
            _ref0.U_exist, range(T), self.n_scenarios,
            lb=0, name="dens_slack_exist"
        )
        recourse_q.update()

        _obj = gb.LinExpr()

        for sc_idx in range(n_scenarios):
            _s_ref = self.milp_objects[sc_idx]
            p_s = self.probs[sc_idx]

            q_s = recourse_q.addVars(_s_ref.U, _s_ref.Tset, lb=0, name=f"q_{sc_idx}")

            for u, ss in _q_keys:
                nk = _node_key(sc_idx, ss)
                if _z_cons(sc_idx, u, ss) >= 0.5:
                    # NAC: scenarios sharing node nk must agree on q[u,ss]
                    recourse_q.addConstr(q_s[u, ss] == q_rob[nk, u, ss], name=f"q_link_{sc_idx}_{u}_{ss}")
                else:
                    # Cohort not stocked in this branch — fix q=0
                    recourse_q.addConstr(q_s[u, ss] == 0, name=f"q_zero_{sc_idx}_{u}_{ss}")

            _sbpf_alive_s: dict = {}
            for (u, ss), alive_t in _alive.items():
                for t in alive_t:
                    if (u, t) not in _sbpf_alive_s:
                        _sbpf_alive_s[(u, t)] = []
                    _sbpf_alive_s[(u, t)].append((ss, float(_s_ref.sbpf[ss][t])))

            def _biom_new(u, t):
                return gb.quicksum(c * q_s[u, ss] for ss, c in _sbpf_alive_s.get((u, t), []))

            def _biom_exist(u, t):
                if u in _alive_exist and t in _alive_exist[u]:
                    return float(_s_ref.biomass_exist[u][t])
                return 0.0

            for u in _s_ref.U:
                cap_kg = _s_ref.density_limit * _s_ref.vol[u]
                for t in range(T):
                    recourse_q.addConstr(
                        _biom_new(u, t) <= cap_kg + dens_slack_new[u, t, sc_idx],
                        name=f"dens_{sc_idx}_{u}_{t}"
                    )
            for u in _s_ref.U_exist:
                cap_kg = _s_ref.density_limit * _s_ref.vol[u]
                for t in range(T):
                    recourse_q.addConstr(
                        _biom_exist(u, t) <= cap_kg + dens_slack_exist[u, t, sc_idx],
                        name=f"dens_exist_{sc_idx}_{u}_{t}"
                    )

            for l in _s_ref.L_list:
                for t in range(T):
                    units_in_loc = [u for u in _s_ref.U if _s_ref.loc_of[u] == l]
                    exist_units_in_loc = [u for u in _s_ref.U_exist if _s_ref.loc_of[u] == l]
                    biomass_loc_t = (
                        gb.quicksum(_biom_new(u, t) for u in units_in_loc) +
                        sum(_biom_exist(u, t) for u in exist_units_in_loc)
                    )
                    recourse_q.addConstr(biomass_loc_t <= _s_ref.loc_mab[l] + mab_slack_loc[l, t, sc_idx],
                                         name=f"mab_loc_{sc_idx}_{l}_{t}")

            for t in range(T):
                biomass_reg_t = (
                    gb.quicksum(_biom_new(u, t) for u in _s_ref.U) +
                    sum(_biom_exist(u, t) for u in _s_ref.U_exist)
                )
                recourse_q.addConstr(biomass_reg_t <= self.regional_mab + mab_slack_reg[t, sc_idx],
                                     name=f"mab_reg_{sc_idx}_{t}")

            # end_bio_min constraint removed (see IP.py): terminal_val terms in
            # objective already incentivise keeping fish alive at T_end.

            def _price_new(ss, tt):
                return _s_ref._price_for_kg(_s_ref.wpf[ss][tt])

            revenue = gb.quicksum(
                float(_s_ref.sbpf[ss][t]) * q_s[u, ss] * _price_new(ss, t) * _s_ref.df[t]
                for (u, ss, t) in self._harv_na.keys()
                if (u, ss) in q_s
            )
            exist_harv_revenue = sum(
                float(_s_ref.biomass_exist[u][t_h])
                * _s_ref._price_for_kg(_s_ref.wpf_exist[u][t_h])
                * _s_ref.df[t_h]
                for (u, t_h) in _hexist_na.keys()
                if t_h < T
            )
            smolt_cost = gb.quicksum(
                _s_ref.smolt_cost_per_head * _s_ref.df[ss] * q_s[u, ss]
                for u, ss in q_s.keys()
            )
            feed_cost = gb.quicksum(
                _s_ref.feed_cost_per_kg_month * _s_ref.df[t] * float(_s_ref.sbpf[ss][t]) * q_s[u, ss]
                for (u, ss), alive_t in _alive.items()
                for t in alive_t
            ) + sum(
                _s_ref.feed_cost_per_kg_month * _s_ref.df[t] * float(_s_ref.biomass_exist[u][t])
                for u, alive_t in _alive_exist.items()
                for t in alive_t
            )
            terminal_value = gb.quicksum(
                _s_ref.terminal_value_per_kg * float(_s_ref.sbpf[ss][T_end]) * q_s[u, ss]
                for (u, ss), alive_t in _alive.items()
                if T_end in alive_t
            ) + sum(
                _s_ref.terminal_value_per_kg * float(_s_ref.biomass_exist[u][T_end])
                for u, alive_t in _alive_exist.items()
                if T_end in alive_t
            )
            _obj.add(p_s * (revenue + exist_harv_revenue - smolt_cost - feed_cost + terminal_value))

            _obj.add(-p_s * MAB_PENALTY * gb.quicksum(
                mab_slack_loc[l, t, sc_idx] for l in _s_ref.L_list for t in range(T)))
            _obj.add(-p_s * MAB_PENALTY * gb.quicksum(
                mab_slack_reg[t, sc_idx] for t in range(T)))
            _obj.add(-p_s * MAB_PENALTY * gb.quicksum(
                dens_slack_new[u, t, sc_idx] for u in _s_ref.U for t in range(T)))
            _obj.add(-p_s * MAB_PENALTY * gb.quicksum(
                dens_slack_exist[u, t, sc_idx] for u in _s_ref.U_exist for t in range(T)))
            # slack_end_bio_s removed with end_biomass_min constraint

        recourse_q.setObjective(_obj, GRB.MAXIMIZE)
        recourse_q.optimize()

        _rob_lp_time = time() - eval_start
        rob_feasible = recourse_q.SolCount > 0
        if not rob_feasible:
            print("WARNING: Recourse LP infeasible.")
            recourse_q.computeIIS()
            q_robust_vals = {}
        else:
            # q_robust_vals keyed by (node_key, u, ss)
            q_robust_vals = {k: v.X for k, v in q_rob.items()}
            self.q_robust_vals = q_robust_vals
            print(f"Three-stage stochastic LP in extensive form solved in {_rob_lp_time:.2f}s. Robust q found.")

        # --- Fix robust q in each scenario, re-solve for stage-2 decisions ---
        def _eval_solve(sc_idx):
            model = self.scenarios[sc_idx]
            milp  = self.milp_objects[sc_idx]
            start = time()

            q_vars = milp.variables["q"]
            # q_robust_vals is keyed by (node_key, u, ss); look up the node for this scenario.
            for (nk, u, ss), q_val in q_robust_vals.items():
                if _node_key(sc_idx, ss) != nk:
                    continue  # this entry belongs to a different node
                if (u, ss) not in q_vars:
                    continue
                if _z_cons(sc_idx, u, ss) < 0.5:
                    q_val = 0.0  # cohort not stocked in this branch
                elif q_val < 0.5:
                    q_val = 0.0
                elif q_val < 1.0:
                    q_val = 1.0
                q_vars[u, ss].lb = q_val
                q_vars[u, ss].ub = q_val

            model.optimize()
            elapsed  = time() - start
            feasible = model.SolCount > 0
            obj_val  = model.ObjVal if feasible else 0.0

            if not feasible:
                # Compute IIS to identify conflicting fixed bounds.
                model.computeIIS()
                with self._progress_lock:
                    if not getattr(self, '_iis_written', False):
                        self._iis_written = True
                        tqdm.write(f"  [post-proc sc={sc_idx}] INFEASIBLE")

                # Backtrack: relax only the bounds flagged by the IIS and re-solve.
                n_relaxed = 0
                for v in model.getVars():
                    if v.IISLB or v.IISUB:
                        v.LB = 0.0
                        v.UB = 1.0
                        n_relaxed += 1
                tqdm.write(f"  [post-proc sc={sc_idx}] relaxing {n_relaxed} IIS-bound vars, re-solving...")
                model.reset()
                model.optimize()
                elapsed  = time() - start
                feasible = model.SolCount > 0
                obj_val  = model.ObjVal if feasible else 0.0
                if feasible:
                    tqdm.write(f"  [post-proc sc={sc_idx}] feasible after relaxation  obj={obj_val:,.0f}")
                else:
                    tqdm.write(f"  [post-proc sc={sc_idx}] still infeasible after relaxation")

            return sc_idx, feasible, obj_val, elapsed

        eval_results = list(executor.map(_eval_solve, range(n_scenarios)))

        n_feasible, n_infeasible = 0, 0
        eval_obj             = 0.0
        infeasible_scenarios = []

        for sc_idx, feasible, obj_val, elapsed in eval_results:
            if feasible:
                n_feasible += 1
                eval_obj   += probs[sc_idx] * obj_val
                status_str  = f"obj={obj_val:>14,.0f}"
            else:
                n_infeasible += 1
                infeasible_scenarios.append(sc_idx)
                status_str   = "INFEASIBLE"
            print(f"  sc={sc_idx:2d}  {self.scenario_names[sc_idx][:40]:<40}  {status_str}  ({elapsed:.1f}s)")

        eval_time = time() - eval_start

        self.eval_obj             = eval_obj
        self.n_feasible           = n_feasible
        self.n_infeasible         = n_infeasible
        self.infeasible_scenarios = infeasible_scenarios

        print("\n" + "-" * 70)
        print(f"Feasibility : {n_feasible}/{n_scenarios} feasible"
              + (f"  [INFEASIBLE: {infeasible_scenarios}]" if infeasible_scenarios else "  [ALL OK]"))
        print(f"E[obj] robust q + re-solved: {eval_obj:,.2f}")
        print(f"E[obj] PH final iter:        {self.objective_history[-1]:,.2f}")
        print(f"Post-processing time: {eval_time:.1f}s")
        print("=" * 70)

    # =========================================================================
    # print_results()
    # =========================================================================

    def print_results(self):
        print("\n" + "=" * 70)
        print(f"{'CONVERGED' if self.converged else 'MAX ITERATIONS'} after {self.n_iters} iters ({self.total_time:.1f}s)")
        print(f"E[obj] post-processing:         {self.eval_obj:,.2f}  "
              f"({self.n_feasible}/{self.n_scenarios} feasible scenarios)")
        print(f"E[obj] PH final iter (aux):     {self.objective_history[-1]:,.2f}")
        print(f"Final max binary deviation:     {self.max_dev_bin:.2e}  (avg: {self.avg_dev_bin:.4f})")
        print(f"Variables fixed: {self.fixed.sum()} / {self.n_binary_na} binary NA vars")
        print(f"  (V={self.V} total: {self.n_binary_na} binary)")

        # --- Branch differentiation diagnostic ---
        # For each stage-1 and stage-2 variable that has a counterpart in all three branches,
        # report how many differ between bad/normal/good.
        print("\n--- Branch differentiation (stage-1 and stage-2 binary consensus) ---")
        # Build map: base_varname -> {node: consensus_val}
        branch_map: dict[str, dict[str, float]] = {}
        for vi, qn in enumerate(self.all_na_names_ordered):
            if not self.is_binary[vi]:
                continue
            vn, node = qn.split("@", 1)
            if node == "s0":
                continue  # stage-0 is shared by design
            branch_map.setdefault(vn, {})[node] = self.bin_consensus.get(vi, 0.0)

        # Count variables where bad != normal OR normal != good
        n_differ = 0
        examples = []
        for vn, node_vals in branch_map.items():
            vals = list(node_vals.values())
            if len(set(round(v) for v in vals)) > 1:
                n_differ += 1
                if len(examples) < 5:
                    examples.append(f"  {vn}: " + ", ".join(f"{nd}={v:.2f}" for nd, v in node_vals.items()))

        total_branch_vars = len(branch_map)
        print(f"  Branch-specific binary vars (stage-1/2): {total_branch_vars}")
        print(f"  Vars where branches differ in consensus:  {n_differ} / {total_branch_vars}")
        if n_differ == 0:
            print("  → All branches converged to the SAME binary decisions.")
            print("    This means the recourse model found one plan optimal across all temperature outcomes.")
            print("    Run more iterations or lower fix_threshold to allow more branch differentiation.")
        else:
            print(f"  → {n_differ} variables differ between branches (recourse IS adapting).")
            for ex in examples:
                print(ex)
        print("=" * 70)

    # =========================================================================
    # plot()
    # =========================================================================

    def plot(self, filename: str = "stats/sp_convergence.png"):
        """
        2×3 figure: Bad/Normal/Good path timelines (row 0) + objective/deviation/summary (row 1).
        """
        n_scenarios   = self.n_scenarios
        is_binary     = self.is_binary
        bin_consensus = self.bin_consensus
        n_binary_na   = self.n_binary_na

        fig, axes = plt.subplots(2, 3, figsize=(21, 10))

        _ref       = self.milp_objects[0]
        _units_all = _ref.U
        _loc_of    = _ref.loc_of
        _loc_names = sorted(set(_loc_of.values()))
        _base_palette = ["tab:blue", "tab:orange", "tab:green",
                         "tab:purple", "tab:brown", "tab:pink",
                         "tab:red", "tab:cyan", "tab:olive", "tab:gray"]
        _cmap    = plt.get_cmap("tab10")
        _palette = (_base_palette + [_cmap(i) for i in range(10)])[:len(_loc_names)]
        _loc_col   = {ln: _palette[i] for i, ln in enumerate(_loc_names)}
        _unit_to_y = {u: i for i, u in enumerate(_units_all)}
        _n_units   = len(_units_all)
        T          = self.T
        _BH        = 0.45
        _s2_start  = self.stage2_months[0] if self.stage2_months else T

        def _short_lbl(u):
            lp, up = u.split(" :: ")
            return f"L{lp.split()[-1]}:U{up.split()[-1]}"

        def _na_decisions(rep_nodes):
            stock = {}; harv = {}; hxist = {}
            for _vi, _qn in enumerate(self.all_na_names_ordered):
                if not is_binary[_vi] or bin_consensus.get(_vi, 0.0) < 0.5:
                    continue
                _nk = _qn.split("@")[1]
                if _nk not in rep_nodes:
                    continue
                _vn    = _qn.split("@")[0]
                _br    = _vn.index("[")
                _pfx   = _vn[:_br]
                _inner = _vn[_br + 1:-1]
                if _pfx == "z":
                    _u, _ts = _inner.rsplit(",", 1)
                    stock.setdefault(_u, []).append(int(_ts))
                elif _pfx == "h":
                    _p = _inner.rsplit(",", 2)
                    harv.setdefault(_p[0], []).append((int(_p[1]), int(_p[2])))
                elif _pfx == "h_exist":
                    _u, _ts = _inner.rsplit(",", 1)
                    hxist.setdefault(_u, []).append(int(_ts))
            return stock, harv, hxist

        def _draw_timeline(ax, rep_nodes, title, shade_color):
            stock, harv, hxist = _na_decisions(rep_nodes)

            ax.axvspan(_s2_start, T, alpha=0.07, color=shade_color, zorder=0)
            ax.text((_s2_start + T) / 2, _n_units - 0.15, "stage-2",
                    ha="center", va="top", fontsize=6, color=shade_color, style="italic")
            for _sb, _sl in [(self.stage0_end, "S1"), (self.stage1_end, "S2")]:
                ax.axvline(_sb, color="gray", ls="--", lw=0.8, alpha=0.6)
                ax.text(_sb + 0.4, _n_units - 0.1, _sl,
                        fontsize=6, color="gray", va="top")

            for _u in _units_all:
                _y   = _unit_to_y[_u]
                _col = _loc_col[_loc_of[_u]]
                if _u in _ref.U_exist:
                    _ehs = hxist.get(_u, [])
                    if _ehs:
                        for _th in _ehs:
                            ax.barh(_y, _th, left=0, height=_BH,
                                    color=_col, alpha=0.25, edgecolor="none", zorder=1)
                            ax.plot(_th, _y, "v", color="darkorange", ms=6, zorder=4)
                    else:
                        ax.barh(_y, T, left=0, height=_BH,
                                color=_col, alpha=0.10, edgecolor="none", zorder=1)
                for (_s0, _th) in harv.get(_u, []):
                    ax.barh(_y, _th - _s0, left=_s0, height=_BH,
                            color=_col, alpha=0.55, edgecolor=_col, lw=0.5, zorder=2)
                    ax.plot(_th, _y, "v", color="tab:red", ms=5, zorder=4)
                for _t in stock.get(_u, []):
                    ax.plot(_t, _y, "^", color="tab:green", ms=6, zorder=5)

                _harvested_starts = {_s0 for (_s0, _) in harv.get(_u, [])}
                for _t in stock.get(_u, []):
                    if _t not in _harvested_starts:
                        ax.barh(_y, T - _t, left=_t, height=_BH,
                                color=_col, alpha=0.20, edgecolor=_col, lw=0.5,
                                linestyle=":", zorder=1)

            ax.set_xlim(0, T); ax.set_ylim(-0.7, _n_units - 0.3)
            ax.set_yticks(range(_n_units))
            ax.set_yticklabels([_short_lbl(u) for u in _units_all], fontsize=7)
            ax.set_xlabel("Month"); ax.set_title(title)
            ax.grid(True, axis="x", alpha=0.25)
            _leg_h = [
                _L2D([0], [0], marker="^", color="w",
                     markerfacecolor="tab:green",  ms=7, label="Stocking"),
                _L2D([0], [0], marker="v", color="w",
                     markerfacecolor="tab:red",    ms=7, label="Harvest (new)"),
                _L2D([0], [0], marker="v", color="w",
                     markerfacecolor="darkorange", ms=7, label="Harvest (exist.)"),
                _MPatch(color="gray", alpha=0.20, linestyle=":",
                        label="Terminal (unharvested)"),
            ] + [_MPatch(color=_palette[i], alpha=0.55,
                         label=ln.replace("Location ", "Loc"))
                 for i, ln in enumerate(_loc_names)]
            ax.legend(handles=_leg_h, fontsize=6, loc="upper right", ncol=1)

        _draw_timeline(
            axes[0, 0],
            rep_nodes={"s0", "s1_bad", "s2_bad_bad"},
            title="Bad Path  (s0 → s1_bad → s2_bad_bad)",
            shade_color="tab:red",
        )
        _draw_timeline(
            axes[0, 1],
            rep_nodes={"s0", "s1_normal", "s2_normal_normal"},
            title="Normal Path  (s0 → s1_normal → s2_normal_normal)",
            shade_color="steelblue",
        )
        _draw_timeline(
            axes[0, 2],
            rep_nodes={"s0", "s1_good", "s2_good_good"},
            title="Good Path  (s0 → s1_good → s2_good_good)",
            shade_color="tab:green",
        )

        ax = axes[1, 0]
        ax.plot(list(range(len(self.objective_history))), self.objective_history,
                "s-", color="tab:blue", markersize=3)
        ax.set_xlabel("Iteration"); ax.set_ylabel("Expected Objective (NOK)")
        ax.set_title("Expected Objective"); ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(list(range(len(self.avg_dev_bin_history))), self.avg_dev_bin_history,
                "o-", color="tab:orange", markersize=3, label="Avg binary dev")
        ax.axhline(self.epsilon_bin, color="gray", ls="--", alpha=0.6, label="ε_bin")
        ax.set_xlabel("Iteration"); ax.set_ylabel("Avg Abs Deviation")
        ax.set_title("Avg Deviation: Binary NA Vars"); ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1, 2]
        feasibility_str = (
            f"{self.n_feasible}/{n_scenarios} feasible"
            + (" [ALL OK]" if self.n_infeasible == 0 else f" [{self.n_infeasible} INFEASIBLE]")
        )
        ax.text(0.5, 0.5,
                f"Total PH time: {self.total_time:.1f}s\n"
                f"Iterations: {self.n_iters}\n"
                f"Avg per iter: {self.total_time / max(self.n_iters, 1):.1f}s\n"
                f"Variables fixed: {self.fixed.sum()}/{n_binary_na}\n"
                f"Final avg bin dev: {self.avg_dev_bin:.4f}\n"
                f"Converged: {self.converged}\n\n"
                f"--- Tree-aware NA ---\n"
                f"s0: 27 scens | s1: 9/node | s2: 3/node\n"
                f"Total NA vars: {self.V} ({n_binary_na} binary)\n\n"
                f"--- Post-processing ---\n"
                f"Feasibility: {feasibility_str}\n"
                f"E[obj] (feasible): {self.eval_obj:,.0f}\n"
                f"E[obj] PH aux:     {self.objective_history[-1]:,.0f}",
                transform=ax.transAxes, fontsize=11, va="center", ha="center",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax.set_title("Summary"); ax.axis("off")

        fig.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Plot saved to {filename}")
        plt.close(fig)

        biomass_filename = filename.replace(".png", "_biomass.png")
        self._plot_biomass(biomass_filename)

    def _plot_biomass(self, filename: str = "stats/sp_biomass.png"):
        """
        Plot total regional biomass trajectories across all 27 scenarios.
        """
        n_scenarios = self.n_scenarios
        T = self.T
        probs = self.probs

        biomass_data = {sc_idx: np.zeros(T) for sc_idx in range(n_scenarios)}

        for sc_idx in range(n_scenarios):
            milp = self.milp_objects[sc_idx]
            model = self.scenarios[sc_idx]
            if model.SolCount == 0:
                continue

            for u in milp.U:
                for t in range(T):
                    for ss in milp.A_by_ut.get((u, t), []):
                        if (u, ss, t) in milp.A_set:
                            harvested_through_t = sum(
                                milp.variables["x"][u, ss, tau].X
                                for tau in milp.H_by_us[(u, ss)] if tau <= t
                            )
                            biomass_data[sc_idx][t] += float(milp.sbpf[ss][t]) * (
                                milp.variables["q"][u, ss].X - harvested_through_t
                            )
                    if u in milp.U_exist:
                        try:
                            e_val = milp.variables["e_alive"][u, t].X
                            h_val = milp.variables["h_exist"][u, t].X
                            biomass_data[sc_idx][t] += milp.biomass_exist[u][t] * (e_val - h_val)
                        except Exception:
                            pass

        months = list(range(T))
        fig, ax = plt.subplots(figsize=(14, 6))

        for sc_idx in range(n_scenarios):
            ax.plot(months, biomass_data[sc_idx] / 1e3, "-", color="gray", lw=0.8, alpha=0.3)

        exp_bio = np.zeros(T)
        for sc_idx in range(n_scenarios):
            exp_bio += probs[sc_idx] * biomass_data[sc_idx]
        ax.plot(months, exp_bio / 1e3, "o-", color="tab:blue", lw=2.5, ms=4,
                zorder=10, label="Expected biomass")

        reg_mab_t = self.regional_mab / 1e3
        ax.axhline(reg_mab_t, color="red", ls="--", lw=1.5, alpha=0.7,
                   label=f"Regional MAB ({reg_mab_t:,.0f} t)")

        for bnd in [self.stage0_end, self.stage1_end]:
            ax.axvline(bnd, color="purple", ls="--", lw=0.6, alpha=0.3)

        ax.set_xlabel("Month")
        ax.set_ylabel("Total Biomass (tonnes)")
        ax.set_title("Regional Biomass Trajectories (all 27 scenarios)")
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, T)
        ax.legend(fontsize=9, loc="upper right")

        fig.tight_layout()
        plt.savefig(filename, dpi=150)
        print(f"Biomass plot saved to {filename}")
        plt.close(fig)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    from instance import units_df, loc_mab, regional_mab, T, temps_normal_12, temps_bad_12, temps_good_12

    ald = AugmentedLagrangianDecomposition(
        units_df=units_df,
        loc_mab=loc_mab,
        regional_mab=regional_mab,
        T=T,
        temps_bad=temps_bad_12,
        temps_normal=temps_normal_12,
        temps_good=temps_good_12,
        K=1_000,
        epsilon_bin=0.01,
        mip_gap=0.02,
        tail_fix_window=5,
        tail_fix_max_disagree=30,
    )
    ald.build()
    ald.solve()
    ald.print_results()
    # ald.plot()
