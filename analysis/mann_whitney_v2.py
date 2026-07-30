"""
Mann-Whitney U Test v2
- 전체 Patient vs Control
- 성별/나이 그룹별
- 블록별 주기 그룹
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

OUT_DIR = Path("/home/hail/robot_ai2/analysis")
df = pd.read_csv(OUT_DIR / "subject_missing_rate_v2.csv")

BLOCKS = {
    "A_hr_rr":  ["missing_1h_hr", "missing_1h_rr"],
    "B1_temp":  ["missing_1h_core_temp", "missing_1h_skin_temp"],
    "B2_ppg":   ["missing_1h_bp_sys", "missing_1h_bp_dia", "missing_1h_glucose"],
    "A_spo2":   ["missing_1h_spo2"],
    "C_hrv":    ["missing_1h_hrv"],
    "D_event":  ["missing_1h_light_sensor", "missing_1h_proximity"],
    "E_daily":  ["missing_daily_step", "missing_daily_distance", "missing_daily_screen_time",
                 "missing_daily_wake_time", "missing_daily_sleep_time",
                 "missing_daily_deep_sleep_time", "missing_daily_rem_sleep_time",
                 "missing_daily_light_sleep_time", "missing_daily_total_sleep_time"],
    "F_ema":    ["missing_daily_EMA_Anxiety", "missing_daily_EMA_Depression",
                 "missing_daily_EMA_Sleep", "missing_daily_EMA_Stress"],
}

ALL_COLS = [c for cols in BLOCKS.values() for c in cols if c in df.columns]
ALL_COLS += [f"block_{b}" for b in BLOCKS if f"block_{b}" in df.columns]

def parse_age(df):
    df = df.copy()
    df["Age_group"] = pd.cut(df["Age"], bins=[0,30,40,50,60,100],
                              labels=["~30대","30대","40대","50대","60대+"])
    df["Sex_label"] = df["Sex"].map({1.0:"남", 2.0:"여", 0.0:"여"})
    return df

df = parse_age(df)
patient = df[df["phq9_label"]==1]
control = df[df["phq9_label"]==0]

def mw_test(p_vals, c_vals):
    p_vals = p_vals.dropna()
    c_vals = c_vals.dropna()
    if len(p_vals) < 5 or len(c_vals) < 5:
        return np.nan, np.nan, np.nan, np.nan
    stat, p = mannwhitneyu(p_vals, c_vals, alternative="two-sided")
    return p_vals.mean(), c_vals.mean(), p_vals.mean()-c_vals.mean(), p

# =====================================================
# 1. 전체
# =====================================================
rows = []
for col in ALL_COLS:
    pm, cm, diff, p = mw_test(patient[col], control[col])
    rows.append({
        "feature": col, "group_by": "전체", "group": "전체",
        "n_patient": len(patient[col].dropna()),
        "n_control": len(control[col].dropna()),
        "patient_mean": pm, "control_mean": cm,
        "diff": diff, "p_value": p,
        "significant": p < 0.05 if not np.isnan(p) else False
    })
df_overall = pd.DataFrame(rows)
df_overall.to_csv(OUT_DIR / "mann_whitney_v2_overall.csv", index=False)
print(f"전체: {len(df_overall)}개 feature")
print(df_overall[df_overall["significant"]][["feature","patient_mean","control_mean","diff","p_value"]])

# =====================================================
# 2. 성별/나이 그룹별
# =====================================================
rows = []
for group_col in ["Sex_label", "Age_group"]:
    for group_val in df[group_col].dropna().unique():
        sub = df[df[group_col] == group_val]
        p_sub = sub[sub["phq9_label"]==1]
        c_sub = sub[sub["phq9_label"]==0]
        for col in ALL_COLS:
            pm, cm, diff, p = mw_test(p_sub[col], c_sub[col])
            rows.append({
                "feature": col, "group_by": group_col, "group": group_val,
                "n_patient": len(p_sub[col].dropna()),
                "n_control": len(c_sub[col].dropna()),
                "patient_mean": pm, "control_mean": cm,
                "diff": diff, "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False
            })
df_grouped = pd.DataFrame(rows)
df_grouped.to_csv(OUT_DIR / "mann_whitney_v2_grouped.csv", index=False)
print(f"\n그룹별: {len(df_grouped[df_grouped['significant']])}개 significant")

# =====================================================
# 3. 블록별 주기 그룹 (block_ 컬럼만)
# =====================================================
block_cols = [f"block_{b}" for b in BLOCKS if f"block_{b}" in df.columns]
rows = []
for group_col in ["전체", "Sex_label", "Age_group"]:
    if group_col == "전체":
        groups = [("전체", df)]
    else:
        groups = [(v, df[df[group_col]==v]) for v in df[group_col].dropna().unique()]
    for group_val, sub in groups:
        p_sub = sub[sub["phq9_label"]==1]
        c_sub = sub[sub["phq9_label"]==0]
        for col in block_cols:
            pm, cm, diff, p = mw_test(p_sub[col], c_sub[col])
            rows.append({
                "block": col, "group_by": group_col, "group": group_val,
                "n_patient": len(p_sub[col].dropna()),
                "n_control": len(c_sub[col].dropna()),
                "patient_mean": pm, "control_mean": cm,
                "diff": diff, "p_value": p,
                "significant": p < 0.05 if not np.isnan(p) else False
            })
df_block = pd.DataFrame(rows)
df_block.to_csv(OUT_DIR / "mann_whitney_v2_block.csv", index=False)
print(f"\n블록별: {len(df_block[df_block['significant']])}개 significant")
print(df_block[df_block["significant"]][["block","group_by","group","patient_mean","control_mean","p_value"]])

print("\n완료!")
