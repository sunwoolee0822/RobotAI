"""
visualize_v5_window.py
======================
분석 단위: (subject, survey, direction) 윈도우 단위 (14일)
입력: labeled_all_chunk/{sample_id}/  ← step2_coformer.py 산출물 (source of truth)

이번 버전 수정 사항 (vs 이전 v5):
  [Fix 1] 01w: errorbar를 subject-level로 (사람별 평균의 mean±std, n_subjects 기준)
  [Fix 2] 02w: HRV(daily_3x)도 24h 분포 추출해서 hour pattern에 표시
  [Fix 3] 02w: Group C 패널에 sync time(10/15/21h) 강조 + spillover 안내
  [Fix 4] 03w 우측: EMA boxplot → 응답 횟수별(0/1/2) % 막대 그래프

확인된 사실 (bash 검증):
- 윈도우 겹침 0건
- HRV는 10/15/21시 ±2h 안에 97~98% 측정됨 (spillover 11/16/19시 등 소수 존재)
"""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from sklearn.cluster import KMeans
from tqdm import tqdm

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


# =====================================================
# 경로 / 상수
# =====================================================

SAMPLE_DIR = Path("/home/hail/robot_ai2/data/labeled_all_chunk")
STATS_DIR = Path("/home/hail/robot_ai2/analysis")
PER_SUBJECT_DIR = STATS_DIR / "per_subject"
OUT_DIR = STATS_DIR / "plots1"
STATS_DIR.mkdir(parents=True, exist_ok=True)
PER_SUBJECT_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 14
SOURCE_FOOTER = (
    "Source: per-window (14d) computation from labeled_all_chunk/ "
    "(same data path as model training)"
)

MISSING_HEAVY_THRESHOLD = 0.80


# =====================================================
# 피처 정의
# =====================================================

GROUP_DEF = [
    ("A",  "1-min Direct",          ["hr", "rr", "spo2"]),
    ("B1", "5-min Direct",          ["core_temp", "skin_temp"]),
    ("B2", "5-min Estimated (PPG)", ["bp_sys", "bp_dia", "glucose"]),
    ("C",  "HRV (Sync 3x/day)",     ["hrv"]),
    ("D",  "Event-based",           ["light_sensor", "proximity"]),
    ("E",  "Daily",                 ["step", "distance", "screen_time", "wake_time", "sleep_time",
                                     "deep_sleep_time", "rem_sleep_time", "light_sleep_time",
                                     "total_sleep_time"]),
    ("F",  "Weekly EMA",            ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"]),
]
FEATURE_TO_GROUP = {f: g for g, _, fs in GROUP_DEF for f in fs}
GROUP_COLORS = {"A": "#e74c3c", "B1": "#3498db", "B2": "#2980b9",
                "C": "#f39c12", "D": "#9b59b6", "E": "#2ecc71", "F": "#34495e"}

FEATURE_UNIT = {
    "hr": ("1min", 1), "rr": ("1min", 1), "spo2": ("1min", 1),
    "core_temp": ("5min", 5), "skin_temp": ("5min", 5),
    "bp_sys": ("5min", 5), "bp_dia": ("5min", 5), "glucose": ("5min", 5),
    "hrv": ("daily_3x", None),
    "light_sensor": ("event_1h", 60), "proximity": ("event_1h", 60),
}

EXPECTED_N_14D = {
    "1min":     WINDOW_DAYS * 24 * 60,
    "5min":     WINDOW_DAYS * 24 * 12,
    "daily_3x": WINDOW_DAYS * 3,
    "event_1h": WINDOW_DAYS * 24,
    "1day":     WINDOW_DAYS,
    "1week":    2,
}

FEATURE_TYPE = {}
for f in ["hr", "rr", "spo2", "core_temp", "skin_temp",
          "bp_sys", "bp_dia", "glucose", "hrv",
          "light_sensor", "proximity"]:
    FEATURE_TYPE[f] = "minute"
for f in ["step", "distance", "screen_time", "wake_time", "sleep_time",
          "deep_sleep_time", "rem_sleep_time", "light_sleep_time", "total_sleep_time"]:
    FEATURE_TYPE[f] = "daily"
for f in ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"]:
    FEATURE_TYPE[f] = "weekly"

PPG_ESTIMATED = {"bp_sys", "bp_dia", "glucose"}

MINUTE_FEATS = ["hr", "rr", "spo2", "core_temp", "skin_temp",
                "bp_sys", "bp_dia", "glucose", "hrv", "light_sensor", "proximity"]
DAILY_FEATS = ["step", "distance", "screen_time", "wake_time", "sleep_time",
               "deep_sleep_time", "rem_sleep_time", "light_sleep_time", "total_sleep_time"]
WEEKLY_FEATS = ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"]
ALL_FEATS = MINUTE_FEATS + DAILY_FEATS + WEEKLY_FEATS

FEATURE_STYLES = {
    "hr": {"color": "#c0392b", "marker": "o"},
    "rr": {"color": "#e74c3c", "marker": "s"},
    "spo2": {"color": "#f1948a", "marker": "^"},
    "core_temp": {"color": "#1f618d", "marker": "o"},
    "skin_temp": {"color": "#3498db", "marker": "s"},
    "bp_sys": {"color": "#1a5490", "marker": "^"},
    "bp_dia": {"color": "#2874a6", "marker": "v"},
    "glucose": {"color": "#5499c7", "marker": "D"},
    "hrv": {"color": "#f39c12", "marker": "o"},
    "light_sensor": {"color": "#9b59b6", "marker": "o"},
    "proximity": {"color": "#bb8fce", "marker": "s"},
    "step": {"color": "#117a65", "marker": "o"},
    "distance": {"color": "#1abc9c", "marker": "s"},
    "screen_time": {"color": "#48c9b0", "marker": "^"},
    "wake_time": {"color": "#7dcea0", "marker": "v"},
    "sleep_time": {"color": "#a9dfbf", "marker": "D"},
    "deep_sleep_time": {"color": "#2ecc71", "marker": "p"},
    "rem_sleep_time": {"color": "#27ae60", "marker": "*"},
    "light_sleep_time": {"color": "#52be80", "marker": "X"},
    "total_sleep_time": {"color": "#82e0aa", "marker": "h"},
    "EMA_Anxiety": {"color": "#34495e", "marker": "o"},
    "EMA_Depression": {"color": "#5d6d7e", "marker": "s"},
    "EMA_Sleep": {"color": "#85929e", "marker": "^"},
    "EMA_Stress": {"color": "#aeb6bf", "marker": "v"},
}


def feat_color(f): return FEATURE_STYLES.get(f, {}).get("color", "#888")
def feat_marker(f): return FEATURE_STYLES.get(f, {}).get("marker", "o")
def sorted_features(features):
    return sorted(features, key=lambda f: (FEATURE_TO_GROUP.get(f, "Z"), f))
def add_footer(fig):
    fig.text(0.5, 0.005, SOURCE_FOOTER, ha="center", fontsize=8,
             style="italic", color="#777")


def get_survey_key(sample_id):
    return sample_id.rsplit("_", 1)[0]


# =====================================================
# Sequence builders
# =====================================================

def build_minute_sequence(m_df, feat, unit_name, window_start, window_end):
    if unit_name == "1min":
        full_idx = pd.date_range(window_start, window_end - pd.Timedelta(minutes=1), freq="1min")
        if m_df is None or feat not in m_df.columns:
            return np.zeros(len(full_idx), dtype=np.int8), full_idx.hour.values
        tmp = m_df[["datetime", feat]].copy().dropna(subset=["datetime"])
        tmp = tmp.set_index("datetime")
        tmp = tmp[~tmp.index.duplicated(keep="first")]
        reindexed = tmp.reindex(full_idx)
        seq = reindexed[feat].notna().astype(np.int8).values
        hours = full_idx.hour.values

    elif unit_name == "5min":
        full_idx = pd.date_range(window_start, window_end - pd.Timedelta(minutes=5), freq="5min")
        if m_df is None or feat not in m_df.columns:
            return np.zeros(len(full_idx), dtype=np.int8), full_idx.hour.values
        tmp = m_df[["datetime", feat]].copy()
        tmp["bin"] = tmp["datetime"].dt.floor("5min")
        tmp["_p"] = tmp[feat].notna().astype(int)
        binned = tmp.groupby("bin")["_p"].max()
        binned = binned.reindex(full_idx, fill_value=0)
        seq = binned.values.astype(np.int8)
        hours = full_idx.hour.values

    elif unit_name == "event_1h":
        full_idx = pd.date_range(window_start, window_end - pd.Timedelta(hours=1), freq="1h")
        if m_df is None or feat not in m_df.columns:
            return np.zeros(len(full_idx), dtype=np.int8), full_idx.hour.values
        tmp = m_df[["datetime", feat]].copy()
        tmp["bin"] = tmp["datetime"].dt.floor("1h")
        tmp["_p"] = tmp[feat].notna().astype(int)
        binned = tmp.groupby("bin")["_p"].max()
        binned = binned.reindex(full_idx, fill_value=0)
        seq = binned.values.astype(np.int8)
        hours = full_idx.hour.values

    elif unit_name == "daily_3x":
        if m_df is None or feat not in m_df.columns:
            return (np.zeros(WINDOW_DAYS * 3, dtype=np.int8),
                    np.tile([10, 15, 21], WINDOW_DAYS))
        tmp = m_df[["datetime", feat]].copy()
        tmp["date"] = tmp["datetime"].dt.date
        tmp["hour"] = tmp["datetime"].dt.hour
        tmp["_p"] = tmp[feat].notna().astype(int)
        seq_records, hour_records = [], []
        for day_offset in range(WINDOW_DAYS):
            target_date = (window_start + pd.Timedelta(days=day_offset)).date()
            day_df = tmp[tmp["date"] == target_date]
            for sync_h in [10, 15, 21]:
                in_win = day_df[(day_df["hour"] >= sync_h - 2) & (day_df["hour"] <= sync_h + 2)]
                seq_records.append(int(in_win["_p"].sum() > 0))
                hour_records.append(sync_h)
        seq = np.array(seq_records, dtype=np.int8)
        hours = np.array(hour_records, dtype=int)
    else:
        raise ValueError(f"Unknown unit: {unit_name}")

    return seq, hours


def build_daily_sequence(d_df, feat, window_start):
    full_idx = pd.date_range(
        window_start, window_start + pd.Timedelta(days=WINDOW_DAYS - 1), freq="1d"
    )
    if d_df is None or feat not in d_df.columns:
        return np.zeros(WINDOW_DAYS, dtype=np.int8)
    tmp = d_df[["date", feat]].copy()
    tmp["date_only"] = pd.to_datetime(tmp["date"]).dt.normalize()
    tmp = tmp.drop_duplicates("date_only", keep="first")
    tmp = tmp.set_index("date_only")
    reindexed = tmp.reindex(full_idx)
    return reindexed[feat].notna().to_numpy(dtype=np.int8)


def build_weekly_sequence(d_df, feat, window_start):
    if d_df is None or feat not in d_df.columns:
        return np.zeros(2, dtype=np.int8)
    tmp = d_df[["date", feat]].copy()
    tmp["date"] = pd.to_datetime(tmp["date"])
    tmp["week"] = ((tmp["date"] - window_start).dt.days // 7).clip(0, 1)
    wg = tmp.groupby("week")[feat].apply(lambda x: int(x.notna().any()))
    seq = np.zeros(2, dtype=np.int8)
    for w_idx, val in wg.items():
        if 0 <= w_idx <= 1:
            seq[int(w_idx)] = int(val)
    return seq


# =====================================================
# Transition / gap / duration
# =====================================================

def fast_transition_stats(seq):
    if len(seq) < 2:
        return None, np.array([]), np.array([])
    seq = np.asarray(seq, dtype=np.int8)
    pairs = seq[:-1] * 2 + seq[1:]
    n_mm = int((pairs == 0).sum())
    n_mp = int((pairs == 1).sum())
    n_pm = int((pairs == 2).sum())
    n_pp = int((pairs == 3).sum())
    p_mm = n_mm / (n_mm + n_mp) if (n_mm + n_mp) > 0 else np.nan
    p_pp = n_pp / (n_pp + n_pm) if (n_pp + n_pm) > 0 else np.nan
    diff = np.diff(np.concatenate(([1 - seq[0]], seq, [1 - seq[-1]])))
    starts = np.where(diff != 0)[0]
    if len(starts) < 2:
        return ({"p_mm": p_mm, "p_pp": p_pp,
                 "max_gap": 0, "mean_gap": 0.0,
                 "max_duration": 0, "mean_duration": 0.0},
                np.array([]), np.array([]))
    lengths = np.diff(starts)
    values = seq[starts[:-1]]
    gaps = lengths[values == 0]
    durs = lengths[values == 1]
    stats = {
        "p_mm": p_mm, "p_pp": p_pp,
        "max_gap": int(gaps.max()) if len(gaps) > 0 else 0,
        "mean_gap": float(gaps.mean()) if len(gaps) > 0 else 0.0,
        "max_duration": int(durs.max()) if len(durs) > 0 else 0,
        "mean_duration": float(durs.mean()) if len(durs) > 0 else 0.0,
    }
    return stats, gaps, durs


# =====================================================
# Per-window 처리
# =====================================================

def process_one_sample(sd):
    meta_path = sd / "metadata.json"
    if not meta_path.exists():
        return None

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    sid = meta["subject_id"]
    sample_id = meta["sample_id"]
    direction = meta["direction"]
    origin = pd.Timestamp(meta["origin"])
    phq9 = meta.get("phq9_label")
    gad7 = meta.get("gad7_label")
    phq9_score = meta.get("phq9_score")
    gad7_score = meta.get("gad7_score")

    window_start = origin
    window_end = origin + pd.Timedelta(days=WINDOW_DAYS)

    m_df = None
    d_df = None
    m_path = sd / "minute_labeled.csv"
    d_path = sd / "daily_labeled.csv"
    if m_path.exists():
        try:
            m_df = pd.read_csv(m_path)
            if len(m_df) == 0:
                m_df = None
            else:
                m_df["datetime"] = pd.to_datetime(m_df["datetime"], errors="coerce")
                m_df = m_df.dropna(subset=["datetime"])
                if len(m_df) == 0:
                    m_df = None
        except Exception:
            m_df = None
    if d_path.exists():
        try:
            d_df = pd.read_csv(d_path)
            if len(d_df) == 0:
                d_df = None
            else:
                d_df["date"] = pd.to_datetime(d_df["date"], errors="coerce")
                d_df = d_df.dropna(subset=["date"])
                if len(d_df) == 0:
                    d_df = None
        except Exception:
            d_df = None

    base_meta = dict(
        subject_id=sid, sample_id=sample_id, direction=direction,
        phq9_label=phq9, gad7_label=gad7,
        phq9_score=phq9_score, gad7_score=gad7_score,
    )

    rows = []
    gaps_per_feat = {}
    durs_per_feat = {}
    gap_starts_per_feat = {}
    gap_ends_per_feat = {}
    hour_rows = []
    dow_rows = []

    # Minute features
    for feat in MINUTE_FEATS:
        unit_name = FEATURE_UNIT[feat][0]
        expected_n = EXPECTED_N_14D[unit_name]
        seq, hours = build_minute_sequence(m_df, feat, unit_name, window_start, window_end)
        if len(seq) == 0:
            continue
        observed_n = int(seq.sum())
        missing_rate = max(0.0, min(1.0, 1 - observed_n / expected_n))
        stats, gaps, durs = fast_transition_stats(seq)
        if stats is None:
            stats = {"p_mm": np.nan, "p_pp": np.nan,
                     "max_gap": 0, "mean_gap": 0.0,
                     "max_duration": 0, "mean_duration": 0.0}
        rows.append({**base_meta, "feature": feat, "unit": unit_name,
                     "expected_n": expected_n, "observed_n": observed_n,
                     "missing_rate": missing_rate, **stats})
        gaps_per_feat[feat] = gaps.tolist()
        durs_per_feat[feat] = durs.tolist()

        if unit_name in ["1min", "5min", "event_1h"]:
            hr_df = pd.DataFrame({"hour": hours, "p": seq})
            hr_agg = hr_df.groupby("hour")["p"].mean().reset_index()
            for _, hrow in hr_agg.iterrows():
                hour_rows.append({**base_meta, "feature": feat,
                                  "hour": int(hrow["hour"]),
                                  "missing_rate": float(1 - hrow["p"])})

            diff = np.diff(np.concatenate(([1 - seq[0]], seq, [1 - seq[-1]])))
            starts_idx = np.where(diff != 0)[0]
            if len(starts_idx) >= 2:
                values = seq[starts_idx[:-1]]
                g_s, g_e = [], []
                for i in range(len(starts_idx) - 1):
                    if values[i] == 0:
                        if starts_idx[i] < len(hours):
                            g_s.append(int(hours[starts_idx[i]]))
                        if starts_idx[i + 1] < len(hours):
                            g_e.append(int(hours[starts_idx[i + 1]]))
                gap_starts_per_feat[feat] = g_s
                gap_ends_per_feat[feat] = g_e

    # [Fix 2] HRV는 daily_3x라 위 hour_rows 루프에서 빠짐.
    # raw minute에서 직접 24h 분포 추출.
    # missing_rate(hour) = 1 - (그 시간대에 측정된 일수) / WINDOW_DAYS
    if m_df is not None and "hrv" in m_df.columns:
        hrv_data = m_df[m_df["hrv"].notna()].copy()
        if len(hrv_data) > 0:
            hrv_data["date"] = hrv_data["datetime"].dt.date
            hrv_data["hour"] = hrv_data["datetime"].dt.hour
            ws_date = window_start.date()
            we_date = window_end.date()
            hrv_data = hrv_data[(hrv_data["date"] >= ws_date) & (hrv_data["date"] < we_date)]
            for hour in range(24):
                day_set = set(hrv_data[hrv_data["hour"] == hour]["date"])
                n_days = len(day_set)
                missing_rate = 1 - n_days / WINDOW_DAYS
                hour_rows.append({**base_meta, "feature": "hrv",
                                  "hour": hour,
                                  "missing_rate": float(missing_rate)})
        else:
            # HRV 측정 없는 윈도우: 24시간 모두 missing 1.0
            for hour in range(24):
                hour_rows.append({**base_meta, "feature": "hrv",
                                  "hour": hour,
                                  "missing_rate": 1.0})

    # Daily features
    for feat in DAILY_FEATS:
        seq = build_daily_sequence(d_df, feat, window_start)
        observed_n = int(seq.sum())
        expected_n = WINDOW_DAYS
        missing_rate = 1 - observed_n / expected_n
        stats, gaps, durs = fast_transition_stats(seq)
        if stats is None:
            stats = {"p_mm": np.nan, "p_pp": np.nan,
                     "max_gap": 0, "mean_gap": 0.0,
                     "max_duration": 0, "mean_duration": 0.0}
        rows.append({**base_meta, "feature": feat, "unit": "1day",
                     "expected_n": expected_n, "observed_n": observed_n,
                     "missing_rate": missing_rate, **stats})
        gaps_per_feat[feat] = gaps.tolist()
        durs_per_feat[feat] = durs.tolist()
        full_idx = pd.date_range(
            window_start, window_start + pd.Timedelta(days=WINDOW_DAYS - 1), freq="1d"
        )
        for i, dt in enumerate(full_idx):
            dow_rows.append({**base_meta, "feature": feat,
                             "dow": int(dt.dayofweek),
                             "missing_rate": float(1 - seq[i])})

    # Weekly EMA
    for feat in WEEKLY_FEATS:
        seq = build_weekly_sequence(d_df, feat, window_start)
        observed_n = int(seq.sum())
        expected_n = 2
        missing_rate = 1 - observed_n / expected_n
        stats, gaps, durs = fast_transition_stats(seq)
        if stats is None:
            stats = {"p_mm": np.nan, "p_pp": np.nan,
                     "max_gap": 0, "mean_gap": 0.0,
                     "max_duration": 0, "mean_duration": 0.0}
        rows.append({**base_meta, "feature": feat, "unit": "1week",
                     "expected_n": expected_n, "observed_n": observed_n,
                     "missing_rate": missing_rate, **stats})
        gaps_per_feat[feat] = gaps.tolist()
        durs_per_feat[feat] = durs.tolist()

    return rows, gaps_per_feat, durs_per_feat, gap_starts_per_feat, gap_ends_per_feat, hour_rows, dow_rows


def collect_all_window_stats():
    print("\n[per-window stats 수집]")
    print(f"  source: {SAMPLE_DIR}")

    sample_dirs = sorted([p for p in SAMPLE_DIR.iterdir() if p.is_dir()])
    print(f"  발견된 샘플: {len(sample_dirs)}")

    all_rows = []
    gap_dist = {f: [] for f in ALL_FEATS}
    dur_dist = {f: [] for f in ALL_FEATS}
    gap_start = {f: [] for f in MINUTE_FEATS}
    gap_end = {f: [] for f in MINUTE_FEATS}
    hour_rows_all = []
    dow_rows_all = []

    skipped = 0
    for sd in tqdm(sample_dirs, desc="  samples"):
        try:
            result = process_one_sample(sd)
            if result is None:
                skipped += 1
                continue
            rows, gpf, dpf, gspf, gepf, hrs, dws = result
            all_rows.extend(rows)
            hour_rows_all.extend(hrs)
            dow_rows_all.extend(dws)
            for f, vals in gpf.items():
                gap_dist[f].extend(vals)
            for f, vals in dpf.items():
                dur_dist[f].extend(vals)
            for f, vals in gspf.items():
                gap_start[f].extend(vals)
            for f, vals in gepf.items():
                gap_end[f].extend(vals)
        except Exception as e:
            print(f"  [SKIP] {sd.name}: {e}")
            skipped += 1

    print(f"  처리됨: {len(sample_dirs) - skipped}, skipped: {skipped}")

    summary_df = pd.DataFrame(all_rows)
    hour_df = pd.DataFrame(hour_rows_all)
    dow_df = pd.DataFrame(dow_rows_all)

    summary_df.to_csv(STATS_DIR / "all_windows_summary.csv",
                      index=False, encoding="utf-8-sig")
    hour_df.to_csv(STATS_DIR / "per_window_hour.csv",
                   index=False, encoding="utf-8-sig")
    dow_df.to_csv(STATS_DIR / "per_window_dow.csv",
                  index=False, encoding="utf-8-sig")
    n_unique = summary_df['sample_id'].nunique() if len(summary_df) > 0 else 0
    print(f"  ✓ all_windows_summary.csv ({len(summary_df)} rows, unique windows={n_unique})")
    print(f"  ✓ per_window_hour.csv ({len(hour_df)} rows)")
    print(f"  ✓ per_window_dow.csv ({len(dow_df)} rows)")

    return summary_df, hour_df, dow_df, gap_dist, dur_dist, gap_start, gap_end


# =====================================================
# Per-subject 집계
# =====================================================

def build_per_subject_summary(summary_df, hour_df, dow_df):
    print("\n[per-subject 집계]")

    agg_funcs = {
        "expected_n": ["sum", "first"],
        "observed_n": "sum",
        "missing_rate": ["mean", "std", "min", "max"],
        "p_mm": ["mean", "std"],
        "p_pp": ["mean", "std"],
        "mean_gap": "mean",
        "max_gap": "max",
        "mean_duration": "mean",
        "max_duration": "max",
        "sample_id": "nunique",
    }

    grouped = summary_df.groupby(["subject_id", "feature", "unit"]).agg(agg_funcs)
    grouped.columns = ["_".join([str(c) for c in col if c]).strip("_") for col in grouped.columns]
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={
        "expected_n_sum": "total_expected",
        "expected_n_first": "expected_n_per_window",
        "observed_n_sum": "total_observed",
        "sample_id_nunique": "n_windows",
    })

    grouped["missing_rate_agg"] = 1 - grouped["total_observed"] / grouped["total_expected"].replace(0, np.nan)
    grouped["missing_rate_agg"] = grouped["missing_rate_agg"].clip(0, 1)

    dir_count = summary_df.groupby(["subject_id", "feature", "direction"])["sample_id"].nunique().unstack(fill_value=0)
    if "before" not in dir_count.columns:
        dir_count["before"] = 0
    if "after" not in dir_count.columns:
        dir_count["after"] = 0
    dir_count = dir_count[["before", "after"]].rename(
        columns={"before": "n_before", "after": "n_after"}).reset_index()
    grouped = grouped.merge(dir_count, on=["subject_id", "feature"], how="left")

    grouped["type"] = grouped["feature"].map(FEATURE_TYPE)
    grouped["group"] = grouped["feature"].map(FEATURE_TO_GROUP)
    grouped["total_days"] = grouped["n_windows"] * WINDOW_DAYS

    cols = ["subject_id", "feature", "type", "group", "unit",
            "n_windows", "n_before", "n_after",
            "expected_n_per_window", "total_observed", "total_expected",
            "missing_rate_agg",
            "missing_rate_mean", "missing_rate_std",
            "missing_rate_min", "missing_rate_max",
            "p_mm_mean", "p_mm_std", "p_pp_mean", "p_pp_std",
            "mean_gap_mean", "max_gap_max",
            "mean_duration_mean", "max_duration_max",
            "total_days"]
    cols = [c for c in cols if c in grouped.columns]
    summary_subject = grouped[cols].copy()

    summary_subject.to_csv(STATS_DIR / "per_subject_summary.csv",
                           index=False, encoding="utf-8-sig")
    print(f"  ✓ per_subject_summary.csv ({len(summary_subject)} rows, "
          f"{summary_subject['subject_id'].nunique()} subjects)")

    hour_subject = hour_df.groupby(
        ["subject_id", "feature", "hour"]
    )["missing_rate"].mean().reset_index()

    dow_subject = dow_df.groupby(
        ["subject_id", "feature", "dow"]
    )["missing_rate"].mean().reset_index()

    print("  사람별 파일 저장 중...")
    subject_ids = summary_subject["subject_id"].unique()

    for sid in tqdm(subject_ids, desc="  per_subject"):
        sub_summary = summary_subject[summary_subject["subject_id"] == sid].drop(
            columns=["subject_id"]
        )
        sub_summary.to_csv(PER_SUBJECT_DIR / f"{sid}.csv",
                           index=False, encoding="utf-8-sig")

        sub_hour = hour_subject[hour_subject["subject_id"] == sid].drop(
            columns=["subject_id"]
        )
        if len(sub_hour) > 0:
            sub_hour.to_csv(PER_SUBJECT_DIR / f"{sid}_hour.csv",
                            index=False, encoding="utf-8-sig")

        sub_dow = dow_subject[dow_subject["subject_id"] == sid].drop(
            columns=["subject_id"]
        )
        if len(sub_dow) > 0:
            sub_dow.to_csv(PER_SUBJECT_DIR / f"{sid}_dow.csv",
                           index=False, encoding="utf-8-sig")

    print(f"  ✓ per_subject/ 폴더에 {len(subject_ids)} subject × 3 file 저장")
    return summary_subject


# =====================================================
# 기존 figure 11장 (window 단위)
# =====================================================

def plot_missing_rate(df, filename):
    """[Fix 1] subject-level errorbar로 변경"""
    feats = sorted_features(df["feature"].unique())

    # 사람별 평균 먼저
    subject_mean = df.groupby(["subject_id", "feature"])["missing_rate"].mean().reset_index()
    n_subjects = subject_mean["subject_id"].nunique()

    rows = []
    for f in feats:
        sub = subject_mean[subject_mean["feature"] == f]["missing_rate"].dropna()
        rows.append({"feature": f, "mean": sub.mean(), "std": sub.std(),
                     "n": len(sub)})
    agg = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(agg))
    colors = [GROUP_COLORS[FEATURE_TO_GROUP[f]] for f in agg["feature"]]
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4, color=colors,
           edgecolor="black", alpha=0.85)
    for i, f in enumerate(agg["feature"]):
        if f in PPG_ESTIMATED:
            ax.text(i, agg["mean"].iloc[i] + agg["std"].iloc[i] + 0.04,
                    "★", ha="center", fontsize=14, color="#c0392b")
    ax.set_xticks(x)
    ax.set_xticklabels(agg["feature"], rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Missing rate per subject (mean ± std across subjects)")
    ax.set_xlabel("Feature")
    ax.set_title(
        f"[WINDOW (14d), Subject-level] Per-Feature Missing Rate "
        f"(n={n_subjects} subjects)\n"
        f"Each subject's missing rate = average across their windows; bars = mean±std across subjects\n"
        "★ = PPG-estimated", weight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.2)
    legend = [Rectangle((0, 0), 1, 1, fc=GROUP_COLORS[g],
                        label=f"Group {g}: {name}") for g, name, _ in GROUP_DEF]
    ax.legend(handles=legend, loc="upper right", fontsize=9, title="Feature Group")
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_hour_pattern(hour_df, filename):
    """[Fix 2 + 3] HRV 패널에 데이터 + sync time spillover 안내"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    axes = axes.flatten()

    grouped_minute = {"A": [], "B1": [], "B2": [], "C": [], "D": []}
    for f in MINUTE_FEATS:
        grouped_minute[FEATURE_TO_GROUP[f]].append(f)

    panels = [
        ("A", "Group A: 1-min Direct (hr/rr/spo2)"),
        ("B1+B2", "Group B: 5-min (Direct vs PPG-estimated)"),
        ("C", "Group C: HRV — Sync 3 times/day (10/15/21h)"),
        ("D", "Group D: Event-based (light_sensor/proximity)"),
    ]

    for ax, (grp, name) in zip(axes, panels):
        if grp == "B1+B2":
            feats_in = grouped_minute["B1"] + grouped_minute["B2"]
        else:
            feats_in = grouped_minute.get(grp, [])

        for f in feats_in:
            sub = hour_df[hour_df["feature"] == f]
            if len(sub) == 0:
                continue
            agg = sub.groupby("hour")["missing_rate"].agg(["mean", "std"]).reset_index()
            label = f"{f} ★" if f in PPG_ESTIMATED else f
            if f == "light_sensor":
                ax.errorbar(agg["hour"], agg["mean"], yerr=agg["std"],
                            marker="o", label=label, capsize=2, alpha=0.9,
                            linewidth=2.5, color="#6c3483", markersize=11, linestyle="-")
            elif f == "proximity":
                ax.errorbar(agg["hour"], agg["mean"], yerr=agg["std"],
                            marker="s", label=label, capsize=2, alpha=0.9,
                            linewidth=1.5, color="#d2b4de", markersize=5, linestyle="--")
            else:
                ax.errorbar(agg["hour"], agg["mean"], yerr=agg["std"],
                            marker=feat_marker(f), label=label, capsize=2, alpha=0.85,
                            linewidth=2, color=feat_color(f), markersize=7)

        ax.set_title(name, fontsize=12, weight="bold")
        ax.set_xlabel("Hour of day (0–23)")
        ax.set_ylabel("Missing rate per window (mean ± std)")
        ax.set_xticks(range(0, 24, 2))
        ax.set_ylim(-0.05, 1.15)

        if grp == "C":
            # [Fix 3] sync time 강조 + spillover 안내
            for sync_h in [10, 15, 21]:
                ax.axvline(sync_h, color="orange", linestyle=":", alpha=0.7, linewidth=2)
            ax.text(0.02, 0.10,
                    "Orange dotted lines = scheduled HRV sync times (10/15/21h)\n"
                    "Spillover at neighboring hours = users syncing slightly off-schedule\n"
                    "(small bumps at 11/14/16/19/22h ≈ 1-hr lag from scheduled time)",
                    transform=ax.transAxes, fontsize=9, style="italic", color="#cc6600",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        if grp == "D":
            ax.text(0.02, 0.10,
                    "light_sensor (solid, large) and proximity (dotted, small)\n"
                    "share the same phone hardware → identical patterns",
                    transform=ax.transAxes, fontsize=9, style="italic", color="#6c3483",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        ax.legend(fontsize=9, loc="best")
        ax.grid(alpha=0.3)

    fig.suptitle("[WINDOW (14d), 1h binning] Hourly Missing Rate Pattern\n"
                 "★ = PPG-estimated", fontsize=15, weight="bold")
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.96))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_dow_weekly(dow_df, summary_df, filename):
    """[Fix 4] 우측 EMA boxplot → 응답 횟수별 % 막대"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # 좌측: Daily by DOW (그대로)
    ax = axes[0]
    for f in DAILY_FEATS:
        sub = dow_df[dow_df["feature"] == f]
        if len(sub) == 0:
            continue
        agg = sub.groupby("dow")["missing_rate"].agg(["mean", "std"]).reset_index()
        ax.errorbar(agg["dow"], agg["mean"], yerr=agg["std"],
                    marker=feat_marker(f), label=f, capsize=3, alpha=0.85,
                    linewidth=2, color=feat_color(f), markersize=8)
    ax.set_xlabel("Day of week")
    ax.set_ylabel("Missing rate per window (mean ± std)")
    ax.set_title("[WINDOW (14d)] Group E: Daily Features by Day of Week", weight="bold")
    ax.set_xticks(range(7))
    ax.set_xticklabels(dow_names)
    ax.set_ylim(-0.05, 1.15)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)

    # [Fix 4] 우측: Weekly EMA — 응답 횟수별 stacked bar
    ax = axes[1]
    response_data = {}  # feature -> [%_0resp, %_1resp, %_2resp]
    for f in WEEKLY_FEATS:
        sub = summary_df[summary_df["feature"] == f]
        if len(sub) == 0:
            response_data[f] = [0, 0, 0]
            continue
        obs = sub["observed_n"].values.astype(int)
        # 0, 1, 2 카운트
        counts = np.array([
            (obs == 0).sum(),
            (obs == 1).sum(),
            (obs == 2).sum(),
        ])
        total = counts.sum()
        if total == 0:
            pct = [0, 0, 0]
        else:
            pct = (counts / total * 100).tolist()
        response_data[f] = pct

    labels = list(response_data.keys())
    data = np.array([response_data[f] for f in labels])  # (4, 3)
    x = np.arange(len(labels))
    width = 0.65
    cat_labels = ["0 responses (no EMA)",
                  "1 response (1 of 2 weeks)",
                  "2 responses (both weeks)"]
    cat_colors = ["#c0392b", "#f39c12", "#27ae60"]

    bottom = np.zeros(len(labels))
    for i, (cat_lab, color) in enumerate(zip(cat_labels, cat_colors)):
        bars = ax.bar(x, data[:, i], width, bottom=bottom, label=cat_lab,
                      color=color, edgecolor="white")
        # 막대 위에 % 표시
        for j, bar in enumerate(bars):
            if data[j, i] >= 5:  # 5% 이상만 표시
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bottom[j] + data[j, i] / 2,
                        f"{data[j, i]:.0f}%",
                        ha="center", va="center", fontsize=9,
                        color="white" if i != 1 else "black", weight="bold")
        bottom += data[:, i]

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("% of windows")
    ax.set_title("[WINDOW (14d)] Group F: Weekly EMA — # of Responses\n"
                 "(out of 2 expected per 14d window)", weight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3, axis="y")

    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_distribution(df, filename):
    bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
    bin_labels = ["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"]
    bin_colors = ["#27ae60", "#f1c40f", "#e67e22", "#e74c3c", "#7f1d1d"]
    feats = sorted_features(df["feature"].unique())

    data = np.zeros((len(feats), len(bin_labels)))
    n_total = []
    for i, f in enumerate(feats):
        sub = df[df["feature"] == f]["missing_rate"].dropna()
        n_total.append(len(sub))
        if len(sub) == 0:
            continue
        counts, _ = np.histogram(sub, bins=bins)
        data[i] = counts / len(sub) * 100

    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(feats))
    bottom = np.zeros(len(feats))
    for j, (lab, c) in enumerate(zip(bin_labels, bin_colors)):
        ax.bar(x, data[:, j], bottom=bottom, label=lab, color=c, edgecolor="white")
        bottom += data[:, j]
    ax.set_xticks(x)
    ax.set_xticklabels([f"{f}\n(n={n})" for f, n in zip(feats, n_total)],
                       rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% of windows")
    ax.set_title("[WINDOW (14d)] Distribution of Per-Window Missing Rate per Feature",
                 weight="bold")
    ax.legend(title="Missing rate", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3, axis="y")
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_window_overall(df, filename):
    from scipy import stats as scstats
    win_mean = df.groupby("sample_id")["missing_rate"].mean().dropna()
    vals = win_mean.values
    mu, sigma = vals.mean(), vals.std()
    threshold_2s = min(mu + 2 * sigma, 1.0)
    threshold_3s = min(mu + 3 * sigma, 1.0)
    mask_2s = vals <= threshold_2s
    mask_95 = vals < 0.95
    clean_2s = vals[mask_2s]
    clean_95 = vals[mask_95]

    fig, axes = plt.subplots(2, 2, figsize=(20, 14))

    # (A) 원본 분포
    ax = axes[0, 0]
    ax.hist(vals, bins=30, color="#3498db", edgecolor="black", alpha=0.7)
    ax.axvline(mu, color="red", linestyle="--", linewidth=2, label=f"mean={mu:.3f}")
    ax.axvline(np.median(vals), color="orange", linestyle="--", linewidth=2,
               label=f"median={np.median(vals):.3f}")
    ax.set_xlabel("Average missing rate per window")
    ax.set_ylabel("# of windows")
    ax.set_title(f"(A) Original Distribution (n={len(vals)})", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (B) Gaussian fit + threshold
    ax = axes[0, 1]
    ax.hist(vals, bins=30, density=True, color="#3498db", edgecolor="black", alpha=0.5)
    x_range = np.linspace(0, 1, 200)
    gauss = scstats.norm.pdf(x_range, mu, sigma)
    ax.plot(x_range, gauss, "r-", linewidth=2,
            label=f"Gaussian (\u03bc={mu:.3f}, \u03c3={sigma:.3f})")
    ax.axvline(threshold_2s, color="#e67e22", linestyle="--", linewidth=2,
               label=f"\u03bc+2\u03c3 = {threshold_2s:.3f}")
    ax.axvline(threshold_3s, color="#c0392b", linestyle="--", linewidth=2,
               label=f"\u03bc+3\u03c3 = {threshold_3s:.3f}")
    ax.axvspan(threshold_2s, 1.0, alpha=0.15, color="red")
    ax.set_xlabel("Average missing rate per window")
    ax.set_ylabel("Density")
    ax.set_title(f"(B) Gaussian Fit + Outlier Thresholds\n"
                 f"2\u03c3: remove {(~mask_2s).sum()} | "
                 f"3\u03c3: remove {(vals > threshold_3s).sum()}", weight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (C) 2-sigma 제거 후
    ax = axes[1, 0]
    ax.hist(clean_2s, bins=30, color="#27ae60", edgecolor="black", alpha=0.7)
    ax.axvline(clean_2s.mean(), color="red", linestyle="--", linewidth=2,
               label=f"mean={clean_2s.mean():.3f}")
    ax.axvline(np.median(clean_2s), color="orange", linestyle="--", linewidth=2,
               label=f"median={np.median(clean_2s):.3f}")
    ax.set_xlabel("Average missing rate per window")
    ax.set_ylabel("# of windows")
    ax.set_title(f"(C) After 2\u03c3 Removal (n={len(clean_2s)}, "
                 f"removed {(~mask_2s).sum()})", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    # (D) 95% 기준 제거 후
    ax = axes[1, 1]
    ax.hist(clean_95, bins=30, color="#9b59b6", edgecolor="black", alpha=0.7)
    ax.axvline(clean_95.mean(), color="red", linestyle="--", linewidth=2,
               label=f"mean={clean_95.mean():.3f}")
    ax.axvline(np.median(clean_95), color="orange", linestyle="--", linewidth=2,
               label=f"median={np.median(clean_95):.3f}")
    ax.set_xlabel("Average missing rate per window")
    ax.set_ylabel("# of windows")
    ax.set_title(f"(D) After 95% Threshold (n={len(clean_95)}, "
                 f"removed {(~mask_95).sum()})", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle("[WINDOW (14d)] Missing Rate Distribution + Gaussian Outlier Detection\n"
                 f"\u03bc={mu:.3f}, \u03c3={sigma:.3f} | "
                 f"2\u03c3 threshold={threshold_2s:.3f} | "
                 f"3\u03c3 threshold={threshold_3s:.3f}",
                 fontsize=14, weight="bold", y=1.01)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")
    print(f"    원본: n={len(vals)}, mean={mu:.3f}")
    print(f"    2σ 제거: n={len(clean_2s)}, removed={int((~mask_2s).sum())}, new_mean={clean_2s.mean():.3f}")
    print(f"    95% 제거: n={len(clean_95)}, removed={int((~mask_95).sum())}, new_mean={clean_95.mean():.3f}")


def plot_gap_duration(gap_dist, dur_dist, filename):
    feats = sorted_features([f for f in ALL_FEATS if len(gap_dist.get(f, [])) > 0])
    n = len(feats); ncols = 4; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.5 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        gaps = np.array(gap_dist[feat]); durs = np.array(dur_dist[feat])
        gaps = gaps[gaps > 0]; durs = durs[durs > 0]
        if len(gaps) == 0 and len(durs) == 0:
            ax.set_visible(False); continue
        ftype = FEATURE_TO_GROUP[feat]
        unit_name = FEATURE_UNIT.get(feat, ("?", None))[0] if feat in FEATURE_UNIT else \
                    ("1day" if feat in DAILY_FEATS else "1week")
        UNIT_TO_HOURS = {"1min": 1/60, "5min": 5/60, "event_1h": 1,
                         "daily_3x": 8, "1day": 24, "1week": 168}
        h_factor = UNIT_TO_HOURS.get(unit_name, 1)
        gaps_h = gaps * h_factor; durs_h = durs * h_factor
        ax.set_xlabel("Length (hours)")
        bins = np.arange(0, 25, 1)
        gaps_clip = np.clip(gaps_h, 0, 24)
        durs_clip = np.clip(durs_h, 0, 24)
        if len(gaps_clip) > 0:
            ax.hist(gaps_clip, bins=bins, alpha=0.6, label=f"missing gap (n={len(gaps)}, med={np.median(gaps_h):.1f}h)",
                    color="#e74c3c", edgecolor="black", linewidth=0.3)
        if len(durs_clip) > 0:
            ax.hist(durs_clip, bins=bins, alpha=0.6, label=f"present dur (n={len(durs)}, med={np.median(durs_h):.1f}h)",
                    color="#27ae60", edgecolor="black", linewidth=0.3)
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.set_title(f"{feat} (Group {ftype}, unit={unit_name}→hours)",
                     fontsize=10, weight="bold", color=GROUP_COLORS[ftype])
        ax.set_ylabel("# of segments")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("[WINDOW (14d)] Gap (red) and Duration (green) Length Distribution\n"
                 "→ AUGMENTATION: sample gap/dur lengths from these distributions",
                 fontsize=14, weight="bold", y=1.00)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_transition(df, filename):
    UNIT_TO_HOURS = {"1min": 1/60, "5min": 5/60, "event_1h": 1,
                     "daily_3x": 8, "1day": 24, "1week": 168}
    feats = sorted_features(df["feature"].unique())
    n = len(feats); ncols = 4; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4.5 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        sub = df[df["feature"] == feat]
        pmm = sub["p_mm"].dropna().values
        ppp = sub["p_pp"].dropna().values
        if len(pmm) == 0 and len(ppp) == 0:
            ax.set_visible(False); continue
        unit = sub["unit"].iloc[0] if len(sub) > 0 else "?"
        h_factor = UNIT_TO_HOURS.get(unit, 1)
        pmm_valid = pmm[(pmm > 0) & (pmm < 1)]
        ppp_valid = ppp[(ppp > 0) & (ppp < 1)]
        gap_hours = np.clip((1 / (1 - pmm_valid)) * h_factor, 0, 336)
        dur_hours = np.clip((1 / (1 - ppp_valid)) * h_factor, 0, 336)
        bins = np.arange(0, 25, 1)
        gap_clip = np.clip(gap_hours, 0, 24)
        dur_clip = np.clip(dur_hours, 0, 24)
        if len(gap_clip) > 0:
            ax.hist(gap_clip, bins=bins, alpha=0.6,
                    label=f"E[gap] med={np.median(gap_hours):.1f}h",
                    color="#e74c3c", edgecolor="black", linewidth=0.3)
        if len(dur_clip) > 0:
            ax.hist(dur_clip, bins=bins, alpha=0.6,
                    label=f"E[dur] med={np.median(dur_hours):.1f}h",
                    color="#27ae60", edgecolor="black", linewidth=0.3)
        grp = FEATURE_TO_GROUP[feat]
        ax.set_title(f"{feat} (G{grp})",
                     fontsize=10, weight="bold", color=GROUP_COLORS[grp])
        ax.set_xlabel("Expected consecutive hours")
        ax.set_ylabel("# of windows")
        ax.set_xticks(range(0, 25, 4))
        ax.set_xlim(0, 24)
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("[WINDOW (14d)] Markov Transition Probabilities Across Windows\n"
                 "→ AUGMENTATION: use P(M→M)/P(P→P) per feature for sequence simulation",
                 fontsize=14, weight="bold", y=1.00)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_gap_start_end(gap_start, gap_end, filename):
    feats = [f for f in MINUTE_FEATS if len(gap_start.get(f, [])) > 0]
    feats = sorted_features(feats)
    n = len(feats); ncols = 4; nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(22, 4 * nrows))
    axes = axes.flatten()
    for i, feat in enumerate(feats):
        ax = axes[i]
        starts = gap_start.get(feat, []); ends = gap_end.get(feat, [])
        if not starts and not ends:
            ax.set_visible(False); continue
        s_hist, _ = np.histogram(starts, bins=range(25))
        e_hist, _ = np.histogram(ends, bins=range(25))
        s_pct = s_hist / s_hist.sum() * 100 if s_hist.sum() > 0 else s_hist
        e_pct = e_hist / e_hist.sum() * 100 if e_hist.sum() > 0 else e_hist
        ax.plot(range(24), s_pct, marker="o", color="#e74c3c", linewidth=2,
                label="Missing START", markersize=6)
        ax.plot(range(24), e_pct, marker="s", color="#27ae60", linewidth=2,
                label="Missing END", markersize=6)
        ax.fill_between(range(24), 0, s_pct, alpha=0.2, color="#e74c3c")
        ax.fill_between(range(24), 0, e_pct, alpha=0.2, color="#27ae60")
        grp = FEATURE_TO_GROUP[feat]
        unit = FEATURE_UNIT.get(feat, ("?",))[0]
        ax.set_title(f"{feat} (Group {grp}, unit={unit})",
                     fontsize=10, weight="bold", color=GROUP_COLORS[grp])
        ax.set_xlabel("Hour of day (0-23h)"); ax.set_ylabel("% of all gap events")
        ax.set_xticks(range(0, 24, 3)); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("[WINDOW (14d)] When Do Missing Gaps Start and End?",
                 fontsize=14, weight="bold", y=1.00)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_feature_correlation(df, filename):
    pivot = df.pivot_table(index="sample_id", columns="feature",
                           values="missing_rate", aggfunc="mean")
    feats_in = sorted_features([f for f in ALL_FEATS if f in pivot.columns])
    pivot = pivot[feats_in]
    corr = pivot.corr(method="pearson")
    pairs = []
    for i, f1 in enumerate(feats_in):
        for j, f2 in enumerate(feats_in):
            if i < j and corr.iloc[i, j] >= 0.9:
                pairs.append((f1, f2, corr.iloc[i, j]))
    pairs.sort(key=lambda x: -x[2])

    fig, axes = plt.subplots(1, 2, figsize=(22, 11),
                             gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feats_in)))
    ax.set_xticklabels(feats_in, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(feats_in))); ax.set_yticklabels(feats_in, fontsize=9)
    ax.set_title("[WINDOW (14d)] Feature × Feature Missing-Rate Correlation",
                 weight="bold")
    plt.colorbar(im, ax=ax, label="Pearson r")
    ax = axes[1]; ax.axis("off")
    table_data = [[f1, f2, f"{c:.3f}"] for f1, f2, c in pairs[:25]]
    if table_data:
        table = ax.table(cellText=table_data,
                         colLabels=["Feature 1", "Feature 2", "Correlation"],
                         loc="center", cellLoc="left", colColours=["#dcdcdc"] * 3)
        table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.5)
    ax.set_title("Strongly Co-Missing Pairs (r ≥ 0.9)", weight="bold")
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_consistency(hour_df, filename):
    rows = []
    for feat in MINUTE_FEATS:
        sub = hour_df[hour_df["feature"] == feat]
        if len(sub) == 0: continue
        pivot = sub.pivot_table(index="sample_id", columns="hour",
                                values="missing_rate", aggfunc="mean").dropna()
        if len(pivot) < 10: continue
        mean_hour_std = float(pivot.std(axis=0).mean())
        np.random.seed(42)
        n_sample = min(100, len(pivot))
        sample = pivot.iloc[np.random.choice(len(pivot), n_sample, replace=False)].values
        corrs = []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                c = np.corrcoef(sample[i], sample[j])[0, 1]
                if not np.isnan(c): corrs.append(c)
        rows.append({"feature": feat, "group": FEATURE_TO_GROUP[feat],
                     "mean_hour_std": mean_hour_std,
                     "mean_pair_corr": float(np.mean(corrs)) if corrs else np.nan})
    cdf = pd.DataFrame(rows).sort_values("mean_pair_corr", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(20, 7))
    colors = [GROUP_COLORS[g] for g in cdf["group"]]
    ax = axes[0]
    ax.bar(cdf["feature"], cdf["mean_pair_corr"], color=colors, edgecolor="black")
    ax.axhline(0.5, color="red", linestyle="--", label="threshold = 0.5")
    ax.set_title("[WINDOW (14d)] Inter-Window Consistency", weight="bold")
    ax.set_ylabel("Mean Pairwise Pearson Correlation")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.set_ylim(0, 1.05)
    ax = axes[1]
    ax.bar(cdf["feature"], cdf["mean_hour_std"], color=colors, edgecolor="black")
    ax.axhline(0.25, color="red", linestyle="--", label="threshold = 0.25")
    ax.set_title("Variability Across Windows", weight="bold")
    ax.set_ylabel("Mean Std of missing rate per hour")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_clusters(df, filename, n_clusters=4):
    pivot = df.pivot_table(index="sample_id", columns="feature",
                           values="missing_rate", aggfunc="mean").fillna(1.0)
    feats_in = sorted_features(pivot.columns.tolist())
    pivot = pivot[feats_in]
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)  # type: ignore[arg-type]
    labels = km.fit_predict(pivot.values)
    centers = km.cluster_centers_
    cluster_avg = centers.mean(axis=1)
    order = np.argsort(cluster_avg)
    name_map = {old: new for new, old in enumerate(order)}
    labels_remap = np.array([name_map[l] for l in labels])
    cluster_names = ["Heavy Wearer\n(low missing)", "Regular Wearer",
                     "Occasional Wearer", "Rare Wearer\n(high missing)"]
    cluster_colors_list = ["#27ae60", "#f1c40f", "#e67e22", "#c0392b"]
    cluster_sizes = pd.Series(labels_remap).value_counts().sort_index()

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))
    ax = axes[0]
    for c in range(n_clusters):
        center = centers[order[c]]
        n_in = cluster_sizes.get(c, 0)
        ax.plot(range(len(feats_in)), center, marker="o",
                label=f"{cluster_names[c]} (n={n_in})", linewidth=2.5,
                color=cluster_colors_list[c], markersize=8)
    ax.set_xticks(range(len(feats_in)))
    ax.set_xticklabels(feats_in, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Missing rate (cluster center)")
    ax.set_title(f"[WINDOW (14d)] K={n_clusters} Window Clusters", weight="bold")
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3); ax.set_ylim(0, 1.1)
    ax = axes[1]
    pivot["cluster"] = labels_remap
    for c in range(n_clusters):
        sub = pivot[pivot["cluster"] == c].drop(columns="cluster")
        per_win = sub.mean(axis=1).values
        ax.hist(per_win, bins=20, alpha=0.6,
                label=f"{cluster_names[c]} (n={len(sub)})",
                color=cluster_colors_list[c])
    ax.set_xlabel("Average missing rate per window")
    ax.set_ylabel("# of windows")
    ax.set_title("Window Distribution by Cluster", weight="bold")
    ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.grid(alpha=0.3)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


# =====================================================
# Tier 1 신규 figure 3장
# =====================================================

def plot_direction_effect(df, filename):
    df = df.copy()
    df["survey_key"] = df["sample_id"].apply(get_survey_key)

    win_mean = df.groupby(["sample_id", "direction"])["missing_rate"].mean().reset_index()
    win_mean["survey_key"] = win_mean["sample_id"].apply(get_survey_key)

    pre = win_mean[win_mean["direction"] == "before"]["missing_rate"].values
    post = win_mean[win_mean["direction"] == "after"]["missing_rate"].values

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    ax = axes[0]
    bins = np.linspace(0, 1, 30)
    ax.hist(pre, bins=bins, alpha=0.55, label=f"Before (n={len(pre)})",
            color="#3498db", edgecolor="black")
    ax.hist(post, bins=bins, alpha=0.55, label=f"After (n={len(post)})",
            color="#e67e22", edgecolor="black")
    if len(pre) > 0:
        ax.axvline(np.mean(pre), color="#3498db", linestyle="--",
                   label=f"Before mean={np.mean(pre):.3f}")
    if len(post) > 0:
        ax.axvline(np.mean(post), color="#e67e22", linestyle="--",
                   label=f"After mean={np.mean(post):.3f}")
    ax.set_xlabel("Avg missing rate per window")
    ax.set_ylabel("# of windows")
    ax.set_title("Before vs After Survey: Window-Avg Missing Rate", weight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    pivot2 = win_mean.pivot_table(index="survey_key", columns="direction",
                                   values="missing_rate", aggfunc="mean")
    paired = pivot2.dropna()
    if len(paired) > 0:
        ax.scatter(paired["before"], paired["after"], alpha=0.3, s=15, color="#34495e")
        ax.plot([0, 1], [0, 1], "r--", label="y=x (no change)")
        diff = (paired["after"] - paired["before"]).mean()
        try:
            from scipy import stats as scstats
            t_stat, p_val = scstats.ttest_rel(paired["after"], paired["before"])
            stat_text = f"paired t={t_stat:.2f}, p={p_val:.2e}"
        except ImportError:
            stat_text = ""
        ax.set_xlabel("Before missing rate"); ax.set_ylabel("After missing rate")
        ax.set_title(f"Paired (same survey): n={len(paired)}\n"
                     f"After−Before mean={diff:+.3f}  {stat_text}", weight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_missing_heavy_vs_others(df, filename, threshold=MISSING_HEAVY_THRESHOLD):
    win_mean = df.groupby("sample_id").agg(
        missing_rate=("missing_rate", "mean"),
        phq9_score=("phq9_score", "first"),
        gad7_score=("gad7_score", "first"),
    ).dropna()

    heavy_mask = win_mean["missing_rate"] >= threshold
    heavy = win_mean[heavy_mask]
    rest = win_mean[~heavy_mask]

    fig, axes = plt.subplots(2, 2, figsize=(18, 11))

    for ax, score_col, name in [(axes[0, 0], "phq9_score", "PHQ9"),
                                  (axes[1, 0], "gad7_score", "GAD7")]:
        bins = np.arange(0, 28, 1)
        if len(rest) > 0:
            ax.hist(rest[score_col].dropna(), bins=bins, alpha=0.6,
                    label=f"Others (n={len(rest)})", color="#3498db", edgecolor="black")
            ax.axvline(rest[score_col].mean(), color="#3498db", linestyle="--",
                       label=f"Others μ={rest[score_col].mean():.2f}")
        if len(heavy) > 0:
            ax.hist(heavy[score_col].dropna(), bins=bins, alpha=0.6,
                    label=f"Missing≥{int(threshold*100)}% (n={len(heavy)})",
                    color="#e74c3c", edgecolor="black")
            ax.axvline(heavy[score_col].mean(), color="#e74c3c", linestyle="--",
                       label=f"Heavy μ={heavy[score_col].mean():.2f}")
        ax.set_xlabel(f"{name} score"); ax.set_ylabel("# of windows")
        ax.set_title(f"{name} Score Distribution", weight="bold")
        ax.legend(fontsize=9); ax.grid(alpha=0.3)

    for ax, score_col, name in [(axes[0, 1], "phq9_score", "PHQ9"),
                                  (axes[1, 1], "gad7_score", "GAD7")]:
        data = [rest[score_col].dropna().values, heavy[score_col].dropna().values]
        labels = [f"Others\n(n={len(rest)})",
                  f"Missing≥{int(threshold*100)}%\n(n={len(heavy)})"]
        bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=True)
        for patch, c in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
            patch.set_facecolor(c); patch.set_alpha(0.7)
        for i, d in enumerate(data, start=1):
            mean = np.mean(d) if len(d) > 0 else 0
            se = np.std(d) / np.sqrt(len(d)) if len(d) > 0 else 0
            ax.errorbar(i, mean, yerr=se, fmt="kD", markersize=8, capsize=8)
        try:
            from scipy import stats as scstats
            if len(data[0]) > 1 and len(data[1]) > 1:
                t_stat, p_val = scstats.ttest_ind(data[0], data[1], equal_var=False)
                stat_text = f"\nWelch t={t_stat:.2f}, p={p_val:.2e}"
            else:
                stat_text = ""
        except (ImportError, ValueError):
            stat_text = ""
        ax.set_ylabel(f"{name} score")
        ax.set_title(f"{name}: Heavy vs Others{stat_text}", weight="bold")
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle(
        f"Missing-Heavy Windows vs Others — PHQ9/GAD7 Score Distribution\n"
        f"(Heavy = window with avg missing rate ≥ {int(threshold*100)}%)",
        fontsize=14, weight="bold", y=1.00,
    )
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


def plot_augmentation_schematic(df, gap_dist, filename):
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    ax = axes[0, 0]
    np.random.seed(42)
    example_hr = np.random.binomial(1, 0.5, 14 * 24)
    ax.imshow(example_hr.reshape(14, 24), cmap="RdYlGn", aspect="auto",
              vmin=0, vmax=1)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day (1~14)")
    ax.set_title("STEP 1: Observed window (14d × 24h)\n"
                 "Red = missing, Green = present", weight="bold")

    ax = axes[0, 1]
    feat_example = "hr"
    sub = df[df["feature"] == feat_example]
    pmm = sub["p_mm"].dropna().values
    ppp = sub["p_pp"].dropna().values
    bins = np.linspace(0, 1, 21)
    if len(pmm) > 0:
        ax.hist(pmm, bins=bins, alpha=0.55, label=f"P(M→M) μ={pmm.mean():.2f}",
                color="#e74c3c")
    if len(ppp) > 0:
        ax.hist(ppp, bins=bins, alpha=0.55, label=f"P(P→P) μ={ppp.mean():.2f}",
                color="#27ae60")
    ax.set_xlabel("Transition probability"); ax.set_ylabel("# of windows")
    ax.set_title(f"STEP 2: Empirical transition distribution\n"
                 f"(across all windows, feature={feat_example})", weight="bold")
    ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1, 0]
    gaps = np.array(gap_dist.get(feat_example, []))
    gaps = gaps[gaps > 0]
    if len(gaps) > 0:
        max_v = gaps.max()
        bins = np.logspace(0, np.log10(max(max_v, 10)), 30)
        ax.hist(gaps, bins=bins, color="#e74c3c", alpha=0.7, edgecolor="black")
        ax.set_xscale("log")
    ax.set_xlabel("Gap length (minutes)"); ax.set_ylabel("# of segments")
    ax.set_title(f"STEP 3: Empirical gap length distribution\n"
                 f"(feature={feat_example})", weight="bold")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    ax.text(0.5, 0.95, "AUGMENTATION PIPELINE", ha="center", fontsize=16,
            weight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.78,
            "INPUT:  14d window with observed missingness pattern\n"
            "         (per feature, per direction)\n",
            fontsize=11, transform=ax.transAxes, family="monospace")
    ax.text(0.05, 0.62,
            "PARAMS (estimated from this analysis, per feature):\n"
            "   - P(M->M), P(P->P)  Markov transitions\n"
            "   - gap length distribution\n"
            "   - duration length distribution\n"
            "   - hour-of-day pattern (when gaps start/end)\n"
            "   - co-missing groups (sleep_*, BP, EMA, ...)\n",
            fontsize=10, transform=ax.transAxes, family="monospace")
    ax.text(0.05, 0.34,
            "AUGMENTATION (per training step):\n"
            "   1. For each feature, simulate missingness mask\n"
            "      using estimated Markov + gap distribution\n"
            "   2. Apply mask consistently within co-missing groups\n"
            "   3. (Optional) condition on direction (before/after)\n",
            fontsize=10, transform=ax.transAxes, family="monospace")
    ax.text(0.05, 0.10,
            "OUTPUT: realistic synthetic missingness on training data\n"
            "        -> robustness to test-time missingness patterns",
            fontsize=11, transform=ax.transAxes, family="monospace",
            color="#117a65")

    fig.suptitle("Augmentation Design — How These Statistics Are Used",
                 fontsize=15, weight="bold", y=1.00)
    add_footer(fig)
    plt.tight_layout(rect=(0, 0.02, 1, 0.97))
    plt.savefig(OUT_DIR / filename, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {filename}")


# =====================================================
# 메인
# =====================================================

def main():
    print("=" * 60)
    print("  visualize_v5_window.py — window 단위 결측 분석")
    print("  [Fix 1] 01w subject-level errorbar")
    print("  [Fix 2] 02w HRV 24h 분포 추가")
    print("  [Fix 3] 02w sync time + spillover 안내")
    print("  [Fix 4] 03w EMA 응답 횟수별 막대")
    print("=" * 60)

    summary_df, hour_df, dow_df, gap_dist, dur_dist, gap_start, gap_end = \
        collect_all_window_stats()

    if len(summary_df) == 0:
        print("\n  [ERROR] no windows processed. abort.")
        return

    build_per_subject_summary(summary_df, hour_df, dow_df)

    print("\n[기존 figure 11장]")
    plot_missing_rate(summary_df, "01w_missing_rate.png")
    plot_hour_pattern(hour_df, "02w_hour_pattern.png")
    plot_dow_weekly(dow_df, summary_df, "03w_dow_weekly_pattern.png")
    plot_distribution(summary_df, "04w_missing_distribution.png")
    plot_window_overall(summary_df, "05w_window_overall.png")
    plot_gap_duration(gap_dist, dur_dist, "06w_gap_duration.png")
    plot_transition(summary_df, "07w_transition.png")
    plot_gap_start_end(gap_start, gap_end, "08w_gap_start_end.png")
    plot_feature_correlation(summary_df, "09w_feature_correlation.png")
    plot_consistency(hour_df, "10w_consistency.png")
    plot_clusters(summary_df, "11w_window_clusters.png")

    print("\n[Tier 1 신규 figure 3장]")
    plot_direction_effect(summary_df, "T1a_direction_effect.png")
    plot_missing_heavy_vs_others(summary_df, "T1b_missing_heavy_vs_others.png")
    plot_augmentation_schematic(summary_df, gap_dist, "T1c_augmentation_schematic.png")

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)
    print("\n  요약:")
    print(f"  - 처리된 윈도우: {summary_df['sample_id'].nunique()}")
    print(f"  - 처리된 subject: {summary_df['subject_id'].nunique()}")
    n_before = summary_df[summary_df['direction']=='before']['sample_id'].nunique()
    n_after = summary_df[summary_df['direction']=='after']['sample_id'].nunique()
    print(f"  - direction별: before={n_before}, after={n_after}")
    print(f"  - 평균 missing rate: {summary_df['missing_rate'].mean():.3f}")


if __name__ == "__main__":
    main()