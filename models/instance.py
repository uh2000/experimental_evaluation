"""
Shared problem instance definition used by IP.py, EEV.py, WS.py, SP.py, DE.py.
Import this module to get the canonical units_df, loc_mab, regional_mab, and T.
"""
import numpy as np
import pandas as pd

T = 60

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
    {"unit": "Unit 11",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"}, #artificial
    {"unit": "Unit 12",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc1"}, #artificial
    


    # {"unit": "Unit 1",  "count": 152_000.0, "avg_weight_g": 4800.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 2",  "count": 160_000.0, "avg_weight_g": 5100.0, "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},
    # {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc2"},


    # {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc3"},


    # {"unit": "Unit 1",  "count": 145_000.0, "avg_weight_g": 5700.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 2",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 3",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 4",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 5",  "count": 184_000.0, "avg_weight_g": 890.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 6",  "count": 173_000.0, "avg_weight_g": 1250.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 7",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 8",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc4"},


    # {"unit": "Unit 1",  "count": 148_000.0, "avg_weight_g": 5500.0, "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 30_000, "location": "Loc5"},


    # {"unit": "Unit 1",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 3",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 4",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 6",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 8",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 9",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 10", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 11", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},
    # {"unit": "Unit 12", "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc6"},


    # Loc7: mirror of Loc1 (+10 cages → 70 total), artificially added
    # {"unit": "Unit 1",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 2",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 3",  "count": 155_000.0, "avg_weight_g": 5000.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 4",  "count": 156_500.0, "avg_weight_g": 5300.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 5",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 6",  "count": 150_000.0, "avg_weight_g": 4500.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 7",  "count": float("nan"), "avg_weight_g": float("nan"), "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 8",  "count": 187_000.0, "avg_weight_g": 550.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 9",  "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc7"},
    # {"unit": "Unit 10", "count": 190_000.0, "avg_weight_g": 500.0, "volume_m3": 35_000, "location": "Loc7"},
]
units_df = pd.DataFrame(units_rows)

loc_mab = {
    "Location 1": 4_000_000,
    # "Location 2": 5_600_000,
    # "Location 3": 2_500_000,
    # "Location 4": 4_300_000,
    # "Location 5": 2_600_000,
    # "Location 6": 5_500_000,
    # "Location 7": 4_000_000, artificially added
}
regional_mab = 35_000_000*(1/6)

# Scenario temperature and survival parameters (matching SP.py / DE.py)
temps_bad_12    = np.array([3, 3, 3, 4,  7, 10, 12, 14.5, 13.5, 11,  8, 5.5])
temps_normal_12 = np.array([5, 5, 5, 6,  9, 12, 14, 16.5, 15.5, 13, 10, 7.5])
temps_good_12   = np.array([7, 7, 7, 8, 11, 14, 16, 18.5, 17.5, 15, 12, 9.5])
S_normal_v = (1.0 - 0.0002) ** 30
S_bad_v    = (1.0 - 0.001)  ** 30

labels       = ["bad", "normal", "good"]
temp_map     = {"bad": temps_bad_12, "normal": temps_normal_12, "good": temps_good_12}
stage_slices = [list(range(0, 15)), list(range(15, 30)),
                list(range(30, 45)), list(range(45, T))]


def _tile(arr, n):
    return np.tile(arr, (n // 12) + 1)[:n]


def build_scenarios():
    """Return list of (name, temps_sc, S_sc, prob) for all 81 scenarios."""
    scenarios = []
    for t1 in labels:
        for t2 in labels:
            for t3 in labels:
                for t4 in labels:
                    name = f"s1_{t1}__s2_{t2}__s3_{t3}__s4_{t4}"
                    temps_sc  = np.zeros(T)
                    S_sc      = np.full(T, S_normal_v)
                    stage_seq = [t1, t2, t3, t4]
                    for i, (sl, months) in enumerate(zip(stage_seq, stage_slices)):
                        prev = stage_seq[i - 1] if i > 0 else None
                        surv = S_bad_v if (sl == "good" and prev == "good") else S_normal_v
                        for mm in months:
                            S_sc[mm] = surv
                        temps_sc[months] = _tile(temp_map[sl], len(months))
                    scenarios.append((name, temps_sc, S_sc, 1.0 / 81))
    return scenarios
