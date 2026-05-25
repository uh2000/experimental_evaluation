"""
60-month rolling-SP variant for the long-vs-short horizon experiment.

Subclasses `AugmentedLagrangianDecomposition` from rolling_horizon_experiment
and adds support for a deterministic "tail" block (default: months 30-59)
that uses normal temperatures and S_normal for all scenarios.

Why this design
---------------
The first 30 months keep exactly the same structure as the 30-month rolling
SP (3-month deterministic prefix + 3 branching stages of 9 months each =
27 scenarios). The new tail block is appended after stage 3 with a fixed
"normal" temperature label and S_normal — so all 27 scenarios share
identical realisations in months 30-59.

Decisions in the tail are NOT part of the non-anticipativity index, so
they are recourse: each scenario optimises its own tail given its
stage-3 state. This is the desired behaviour — what we are testing is
whether the planner BENEFITS in months 0-29 from being aware of the
30-month deterministic outlook beyond month 30.

Tractability defence (for the thesis)
-------------------------------------
A fully stochastic 30-month tail would multiply scenarios. By treating
the tail as a deterministic "expected" outlook we keep the experiment
tractable across multiple reality paths while still giving the planner
the information it would have under any reasonable forecast at the
re-planning point (which is several rolls in the future anyway).
"""
from __future__ import annotations

import os
import sys
import itertools
import threading
from typing import List, Optional

import numpy as np

# Make the rolling_horizon_experiment package importable (instance.py / ip.py)
_RH_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "rolling_horizon_experiment")
)
if _RH_DIR not in sys.path:
    sys.path.insert(0, _RH_DIR)
# Local folder must come first so `import sp_rh` finds the local complete copy.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# `sp_rh.py` here is a complete copy of rolling_horizon_experiment/sp.py — used
# because the rolling_horizon_experiment/sp.py file is intermittently truncated
# in the OneDrive sync layer the bash mount sees.
from sp_rh import AugmentedLagrangianDecomposition  # noqa: E402
from ip import SalmonFarmingMILP                    # noqa: E402


class AugmentedLagrangianDecomposition60M(AugmentedLagrangianDecomposition):
    """
    Rolling SP with horizon `T` and a deterministic "tail" of normal
    temperatures appended after the last stochastic stage.

    `tail_months` defaults to every month in [0, T) that is not covered
    by `prefix_months` or any of `stage_slices`. So setting:

        T=60, prefix_months=[0,1,2], stage_slices=[3-11, 12-20, 21-29]

    gives a tail of months 30-59 automatically.
    """

    def __init__(self, *args, tail_months: Optional[List[int]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        if tail_months is None:
            covered = set(self.prefix_months)
            for sl in self.stage_slices:
                covered.update(sl)
            tail_months = [m for m in range(self.T) if m not in covered]
        self.tail_months = sorted(int(m) for m in tail_months)

    # ------------------------------------------------------------------
    # build() — copy of parent + tail block
    # ------------------------------------------------------------------
    def build(self):
        T = self.T
        labels = self.labels
        n_branching = self.n_branching_stages
        n_total = len(labels) ** n_branching
        temp_map = {
            "bad":    self.temps_bad,
            "normal": self.temps_normal,
            "good":   self.temps_good,
        }

        self._label_tuples = list(itertools.product(labels, repeat=n_branching))

        scenarios, milp_objects, probabilities, scenario_names, bundle_id = [], [], [], [], []

        for label_tuple in self._label_tuples:
            tag_parts = [f"s{i+1}_{l}" for i, l in enumerate(label_tuple)]
            if self.prefix_months:
                name = f"prefix_{self.prefix_label}__" + "__".join(tag_parts)
            else:
                name = "__".join(tag_parts)

            temps_t = np.zeros(T)
            S_t = np.full(T, self.S_normal)

            # Block sequence: optional prefix, stochastic stages, tail.
            blocks = []
            if self.prefix_months:
                blocks.append((self.prefix_label, list(self.prefix_months)))
            for i, lbl in enumerate(label_tuple):
                blocks.append((lbl, self.stage_slices[i]))
            if self.tail_months:
                blocks.append(("normal", list(self.tail_months)))

            for i, (sl, months) in enumerate(blocks):
                prev = blocks[i - 1][0] if i > 0 else None
                surv = self._stage_surv(sl, prev, self.S_normal, self.S_bad)
                for m in months:
                    S_t[m] = surv
                cal_temps = self._calendar_temps(sl, months, temp_map)
                for m, val in zip(months, cal_temps):
                    temps_t[m] = val

            # Sanity: every month in [0, T) should now have a non-zero temp.
            # (zero would indicate an uncovered month — a bug in stage_slices.)
            assert np.all(temps_t > 0.0), (
                f"Scenario '{name}' has uncovered months "
                f"{np.where(temps_t == 0.0)[0].tolist()}. "
                "Did you forget to include them in prefix/stages/tail?"
            )

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
            bundle_id.append(labels.index(label_tuple[0]))

        self.scenarios = scenarios
        self.milp_objects = milp_objects
        self.probabilities = probabilities
        self.scenario_names = scenario_names
        self.bundle_id = bundle_id
        self.n_scenarios = len(scenarios)
        self.probs = np.array(probabilities, dtype=np.float64)

        tail_msg = (f" + {len(self.tail_months)}-month deterministic tail"
                    if self.tail_months else "")
        prefix_msg = (f" + {len(self.prefix_months)}-month prefix"
                      if self.prefix_months else "")
        print(f"Built {self.n_scenarios} scenarios "
              f"({n_branching} stochastic stage(s){prefix_msg}{tail_msg}, T={T})")

        self._build_variable_index()

        self._n_workers_actual = min(self.n_scenarios, os.cpu_count() or 4)
        self._progress_lock = threading.Lock()
        self._progress_counter = [0]
        self.original_objectives = [scenarios[s].getObjective() for s in range(self.n_scenarios)]
