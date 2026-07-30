"""
Missing Rate Analysis v2
- 블록 기준 수정 (저 사람 슬라이드 기준 맞춤)
- Diurnal missingness 추가 (hr, rr, core_temp, skin_temp, bp_sys, bp_dia, glucose)
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

CACHE_DIR = Path("/home/hail/robot_ai2/data/cache_all")
OUT_DIR   = Path("/home/hail/robot_ai2/analysis")

# =====================================================
# 블록 정의
# =====================================================
BLOCKS = {
    "A_hr_rr":  ["hr", "rr"],
    "B1_temp":  ["core_temp", "skin_temp"],
    "B2_ppg":   ["bp_sys", "bp_dia", "glucose"],
    "A_spo2":   ["spo2"],
    "C_hrv":    ["hrv"],
    "D_event":  ["light_sensor", "proximity"],
    "E_daily":  ["step", "distance", "screen_time", "wake_time", "sleep_time",
                 "deep_sleep_time", "rem_sleep_time", "light_sleep_time", "total_sleep_time"],
    "F_ema":    ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"],
}

# 블록별 수집 주기 (슬롯 수 / day)
BLOCK_SLOTS_PER_DAY = {
    "A_hr_rr":  24,    # 1분 → 1h slot
    "B1_temp":  24,    # 5분 → 1h slot (step1에서 1h aggregation)
    "B2_ppg":   24,    # 5분 → 1h slot
    "A_spo2":   7,     # nightly 00~07h
    "C_hrv":    6,     # 3x daily (10~12, 15~17, 22~24)
    "D_event":  24,    # 연속
    "E_daily":  1,     # daily
    "F_ema":    1,     # weekly (1/7 per day이지만 1로 통일)
}

# =====================================================
# 캐시 로드
# =====================================================
print("캐시 로드 중...")
with open(CACHE_DIR / "survey_all.pkl", "rb") as f:
    survey_all = pickle.load(f)
with open(CACHE_DIR / "minute_df_cache.pkl", "rb") as f:
    minute_df_cache = pickle.load(f)
with open(CACHE_DIR / "daily_df_cache.pkl", "rb") as f:
    daily_df_cache = pickle.load(f)

subject_info = survey_all.groupby("ID").agg(
    Sex=("Sex", "first"),
    Age=("Age", "first"),
    phq9_label=("phq9_label", "max"),
).reset_index()

def parse_age(x):
    if pd.isna(x): return np.nan
    dt = pd.to_datetime(x, errors="coerce")
    if pd.notna(dt):
        today = pd.Timestamp.today()
        return float(today.year - dt.year - ((today.month, today.day) < (dt.month, dt.day)))
    try: return float(x)
    except: return np.nan

subject_info["Age"] = subject_info["Age"].apply(parse_age)
subject_info.loc[(subject_info["Age"]<10)|(subject_info["Age"]>100), "Age"] = np.nan

# =====================================================
# Subject별 블록별 missing rate 계산
# =====================================================
MINUTE_FEATS = ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia",
                "glucose", "spo2", "hrv", "light_sensor", "proximity"]
SPO2_HOURS   = list(range(0, 7))
HRV_HOURS    = list(range(10,12)) + list(range(15,17)) + list(range(22,24))

results = []
for _, srow in tqdm(subject_info.iterrows(), total=len(subject_info)):
    sid = srow["ID"]
    row = {"subject_id": sid, "Sex": srow["Sex"],
           "Age": srow["Age"], "phq9_label": srow["phq9_label"]}

    # --- Minute features ---
    if sid in minute_df_cache:
        mdf = minute_df_cache[sid].copy()
        mdf['datetime'] = pd.to_datetime(mdf['datetime'])
        mdf['hour'] = mdf['datetime'].dt.hour

        for feat in MINUTE_FEATS:
            if feat not in mdf.columns:
                row[f"missing_1h_{feat}"] = 1.0
                continue

            if feat == "spo2":
                valid = mdf[mdf['hour'].isin(SPO2_HOURS)]
            elif feat == "hrv":
                valid = mdf[mdf['hour'].isin(HRV_HOURS)]
            else:
                valid = mdf

            if len(valid) == 0:
                row[f"missing_1h_{feat}"] = 1.0
            else:
                row[f"missing_1h_{feat}"] = valid[feat].isna().sum() / len(valid)
    else:
        for feat in MINUTE_FEATS:
            row[f"missing_1h_{feat}"] = 1.0

    # --- Daily features ---
    DAILY_FEATS = ["step", "distance", "screen_time", "wake_time", "sleep_time",
                   "deep_sleep_time", "rem_sleep_time", "light_sleep_time",
                   "total_sleep_time", "EMA_Anxiety", "EMA_Depression",
                   "EMA_Sleep", "EMA_Stress"]
    if sid in daily_df_cache:
        ddf = daily_df_cache[sid]
        for feat in DAILY_FEATS:
            if feat not in ddf.columns or len(ddf) == 0:
                row[f"missing_daily_{feat}"] = 1.0
            else:
                row[f"missing_daily_{feat}"] = ddf[feat].isna().sum() / len(ddf)
    else:
        for feat in DAILY_FEATS:
            row[f"missing_daily_{feat}"] = 1.0

    results.append(row)

df_result = pd.DataFrame(results)

# =====================================================
# 블록별 missing rate 추가
# =====================================================
for block, feats in BLOCKS.items():
    cols = []
    for f in feats:
        if f"missing_1h_{f}" in df_result.columns:
            cols.append(f"missing_1h_{f}")
        elif f"missing_daily_{f}" in df_result.columns:
            cols.append(f"missing_daily_{f}")
    if cols:
        df_result[f"block_{block}"] = df_result[cols].mean(axis=1)

df_result.to_csv(OUT_DIR / "subject_missing_rate_v2.csv", index=False, encoding="utf-8-sig")
print(f"\n저장: subject_missing_rate_v2.csv ({len(df_result)}명)")

# =====================================================
# Diurnal missingness (시간대별 결측률)
# hr, rr, core_temp, skin_temp, bp_sys, bp_dia, glucose
# =====================================================
print("\nDiurnal missingness 계산 중...")
DIURNAL_FEATS = ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia", "glucose"]

diurnal_results = []
for _, srow in tqdm(subject_info.iterrows(), total=len(subject_info)):
    sid = srow["ID"]
    if sid not in minute_df_cache:
        continue
    mdf = minute_df_cache[sid].copy()
    mdf['datetime'] = pd.to_datetime(mdf['datetime'])
    mdf['hour'] = mdf['datetime'].dt.hour

    for h in range(24):
        hour_df = mdf[mdf['hour'] == h]
        if len(hour_df) == 0:
            continue
        row = {"subject_id": sid, "hour": h, "phq9_label": srow["phq9_label"]}
        for feat in DIURNAL_FEATS:
            if feat not in hour_df.columns:
                row[f"missing_{feat}"] = 1.0
            else:
                row[f"missing_{feat}"] = hour_df[feat].isna().sum() / len(hour_df)
        diurnal_results.append(row)

df_diurnal = pd.DataFrame(diurnal_results)
df_diurnal.to_csv(OUT_DIR / "diurnal_missing_rate.csv", index=False, encoding="utf-8-sig")
print(f"저장: diurnal_missing_rate.csv ({len(df_diurnal)}행)")

print("\n완료!")
