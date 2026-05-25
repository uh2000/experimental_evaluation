"""
Shared problem instance definition for the rolling-horizon experiment.

Defaults to a 30-month rolling SP with:
  - Months 0-2: deterministic prefix (3-month implementation period; temperature
    fixed by the realised first-3-months label).
  - Months 3-11: stage 1, blind commit (NA across all 27 scenarios).
  - Months 12-20: stage 2, NA per t1 (3 nodes of 9).
  - Months 21-29: stage 3, NA per (t1, t2) (9 nodes of 3).

Total scenarios = 3^3 = 27 (one per (t1, t2, t3) ∈ {bad, normal, good}^3).

The original 60-month, 81-scenario, 4-stage configuration is still reachable by
calling `build_scenarios(T=60, prefix_months=[], stage_slices=<old 4 slices>,
n_branching_stages=4)` — see the old defaults at the bottom of this file.
"""
import numpy as np
import pandas as pd

# Default horizon for the rolling-horizon experiment
T = 30

units_rows = [
    {"unit": "Unit 1",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 3",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 4",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 6",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 8",  "count": 187_000.0, "avg_weight_g": 550.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 9",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc1"},
    {"unit": "Unit 10", "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc1"},


    {"unit": "Unit 1",  "count": 152_000.0, "avg_weight_g": 4800.0, "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 2",  "count": 160_000.0, "avg_weight_g": 5100.0, "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},


    {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},


    {"unit": "Unit 1",  "count": 145_000.0, "avg_weight_g": 5700.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 2",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 3",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 4",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 5",  "count": 184_000.0, "avg_weight_g": 890.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 6",  "count": 173_000.0, "avg_weight_g": 1250.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 7",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 8",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},


    {"unit": "Unit 1",  "count": 148_000.0, "avg_weight_g": 5500.0, "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},


    {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g":   float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
]
units_df = pd.DataFrame(units_rows)

loc_mab = {
    "Location 1": 4_000_000,
    "Location 2": 5_600_000,
    "Location 3": 2_500_000,
    "Location 4": 4_300_000,
    "Location 5": 2_600_000,
    "Location 6": 5_500_000,
}
regional_mab = 35_000_000

# Scenario temperature and survival parameters
temps_bad_12    = np.array([3, 3, 3, 4,  7, 10, 12, 14.5, 13.5, 11,  8, 5.5])
temps_normal_12 = np.array([5, 5, 5, 6,  9, 12, 14, 16.5, 15.5, 13, 10, 7.5])
temps_good_12   = np.array([7, 7, 7, 8, 11, 14, 16, 18.5, 17.5, 15, 12, 9.5])

S_normal_v = (1.0 - 0.0002) ** 30
S_bad_v    = (1.0 - 0.001)  ** 30

labels       = ["bad", "normal", "good"]
temp_map     = {"bad": temps_bad_12, "normal": temps_normal_12, "good": temps_good_12}

# Default rolling-horizon stage structure: 4 conceptual blocks, 3 stochastic stages.
#   block 0 (months 0-2):   deterministic prefix (3 months)
#   block 1 (months 3-11):  blind commit       (9 months, label tA, NA across all)
#   block 2 (months 12-20): post-A             (9 months, label tB, NA per tA)
#   block 3 (months 21-29): post-B             (9 months, label tC, NA per tA,tB)
prefix_months  = list(range(0, 3))
stage_slices   = [
    list(range(3, 12)),
    list(range(12, 21)),
    list(range(21, 30)),
]


def _calendar_temps(label: str, months: list, start_calendar_month: int) -> np.ndarray:
    """Return monthly temperatures for the given block, indexed by absolute calendar
    month so that horizon-relative month m corresponds to calendar month
    (start_calendar_month + m) mod 12."""
    profile = temp_map[label]
    return np.array([profile[(start_calendar_month + m) % 12] for m in months])


def _stage_surv(label: str, prev_label: str, S_normal: float, S_bad: float) -> float:
    """High mortality only when this AND the preceding block are both 'good'."""
    return S_bad if (label == "good" and prev_label == "good") else S_normal


def build_scenarios(
    T: int = T,
    prefix_months: list = prefix_months,
    stage_slices: list = stage_slices,
    prefix_label: str = "normal",
    start_calendar_month: int = 0,
    S_normal: float = S_normal_v,
    S_bad: float = S_bad_v,
) -> list:
    """
    Build the scenario list for the rolling SP.

    Parameters
    ----------
    T : int
        Total horizon length in months.
    prefix_months : list[int]
        Months that share a single deterministic temperature (typically [0,1,2]).
        Empty list = no prefix, fully stochastic from month 0.
    stage_slices : list[list[int]]
        Months covered by each stochastic stage. Each stage has 3 branches
        (bad/normal/good); total scenarios = 3^len(stage_slices).
    prefix_label : str
        Temperature label applied to all months in `prefix_months`.
    start_calendar_month : int
        Calendar offset so that horizon-relative t=0 maps to this calendar month
        when tiling the 12-month seasonal profile (0 = Jan, 11 = Dec).
    S_normal, S_bad : float
        Survival rates for normal and high-mortality (good-after-good) months.

    Returns
    -------
    list[(name, temps_t, S_t, prob)]
    """
    n_stages = len(stage_slices)
    n_total  = len(labels) ** n_stages

    scenarios = []

    def _recurse(prefix_lbls: list, depth: int):
        if depth == n_stages:
            # Build full temperature/survival arrays from prefix + stage labels
            temps_t = np.zeros(T)
            S_t     = np.full(T, S_normal)

            # Prefix block (deterministic)
            blocks = []
            if prefix_months:
                blocks.append((prefix_label, prefix_months, None))
            # Stage blocks
            prev = prefix_label if prefix_months else None
            for i, lbl in enumerate(prefix_lbls):
                blocks.append((lbl, stage_slices[i], prev))
                prev = lbl

            for lbl, months, prev_lbl in blocks:
                surv = _stage_surv(lbl, prev_lbl, S_normal, S_bad)
                for m in months:
                    S_t[m] = surv
                profile_temps = _calendar_temps(lbl, months, start_calendar_month)
                for m, val in zip(months, profile_temps):
                    temps_t[m] = val

            stage_str = "__".join(f"s{i+1}_{l}" for i, l in enumerate(prefix_lbls))
            name = f"prefix_{prefix_label}__{stage_str}" if prefix_months else stage_str
            scenarios.append((name, temps_t, S_t, 1.0 / n_total))
            return

        for lbl in labels:
            _recurse(prefix_lbls + [lbl], depth + 1)

    _recurse([], 0)
    return scenarios
