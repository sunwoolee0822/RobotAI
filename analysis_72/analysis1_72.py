"""
analysis1_72.py — 결측률 분석 (72h 전처리 데이터 기준)
=====================================================
analysis1_336.py 의 결측률 그림들을 336(14일, 분단위 원본) 대신
72(3일 슬라이딩 윈도우, 1시간 집계) **전처리 산출물**에서 다시 계산한다.

입력 : data/numpy_all_chunk_72_24feat/{time.npy, feature_columns.json, subject_ids.json}
출력 : analysis_72/

[중요] time.npy 저장 규약
  step2_coformer.py:218-220 은 관측값을 **슬롯 위치가 아니라 배열 앞쪽부터 순서대로**
  채운다 (left-packing). 즉 array[f, j] 는 "j번째 슬롯"이 아니라 "j번째 관측"이고,
  실제 시각은 time[f, j] (윈도우 원점으로부터의 분) 에만 들어 있다.
  → 일자/시간대를 슬롯 인덱스로 나누면 틀린다. 반드시 time 값을 써야 한다.
  윈도우 시작은 항상 자정 (w_start_min 이 1440 의 배수) 이므로
    hour-of-day = (time // 60) % 24,  day = time // 1440
  이 성립하고, 한 윈도우(72slot=3일)에서 각 시간대는 정확히 3회씩 기대된다.

[측정 단위] 피처마다 실제 수집 주기가 달라 결측률의 단위를 맞춘다.
  - 시간(1h) 단위 : hr, rr, core_temp, skin_temp, bp_sys, bp_dia, glucose
                    (관측일당 15~22시간 → 연속 측정)
  - 일(day) 단위  : 그 외 전부. spo2/hrv/light_sensor/proximity 는 관측일당
                    1.8~6.3시간, 하루 1회꼴이라 시간 단위로 재면 결측률이
                    과대평가된다. daily/EMA 는 forward-fill 로 이미 일 단위다.

산출:
  analysis_72/missing_rate_by_feature_72.csv
  analysis_72/missing_rate_by_subject_feature_72.csv
  analysis_72/hourly_missing_rate_72.csv
  analysis_72/never_measured_72.csv
  analysis_72/plots/fig1_missing_rate_feature.png
  analysis_72/plots/fig2_hourly_missing_rate.png
  analysis_72/plots/fig3_daily_cadence.png

사용법: python analysis1_72.py
"""
import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 19,
    "axes.labelsize": 17,
    "xtick.labelsize": 13,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
})

# =====================================================
# 피처 그룹 (수집 주기 기준으로 재정의)
#   - spo2 를 A(1-min) 에서 C 로 옮김: 실측 결과 하루 1회꼴이다.
#   - C 를 "Daily-sync" 로 개명 (기존 HRV 전용 → spo2 + hrv)
# =====================================================

# 블록명은 2026-07-22 미팅 체크리스트의 측정주기 블록표 표기를 그대로 따른다.
# spo2 는 명목상 1-min(A) 이지만 실측 주기가 daily 라 A_spo2 로 독립시킨다.
GROUP_DEF = [
    ("A",  "1-min, no fill",     ["hr", "rr"]),
    ("B1", "5-min, no fill",     ["core_temp", "skin_temp"]),
    ("B2", "5-min PPG, no fill", ["bp_sys", "bp_dia", "glucose"]),
    ("C",  "1-min, no fill",     ["light_sensor"]),
    ("D",  "fill 24",            ["spo2", "hrv", "proximity"]),
    ("E",  "fill 24",            ["step", "distance", "screen_time", "wake_time", "sleep_time",
                                  "deep_sleep_time", "rem_sleep_time", "light_sleep_time",
                                  "total_sleep_time"]),
    ("F",  "fill 168",           ["EMA_Anxiety", "EMA_Depression", "EMA_Sleep", "EMA_Stress"]),
]
FEATURE_TO_GROUP = {f: g for g, _, fs in GROUP_DEF for f in fs}
GROUP_NAME = {g: name for g, name, _ in GROUP_DEF}
GROUP_ORDER = [g for g, _, _ in GROUP_DEF]

# 색 규칙: 그룹 = 색, 그룹 내부 = 마커/선종류. 전 그림 공통.
#  - hr/rr 은 어느 그림에서든 빨강으로 고정
#  - B1(직접측정)과 B2(PPG 추정)는 ★ 없이 색만으로 구분해야 하므로 명도차를 크게 벌린다
#    (★ 는 통계적 유의성 표기로 예약)
GROUP_COLORS = {"A": "#e74c3c", "B1": "#5dade2", "B2": "#154360",
                "C": "#9b59b6", "D": "#d35400", "E": "#27ae60", "F": "#34495e"}

# 피처별 라인 스타일 (시간대별 그림에서 같은 그룹 안 구분용)
FEATURE_STYLE = {
    "hr":        ("-",  "o"), "rr":        ("--", "s"),
    "core_temp": ("-",  "o"), "skin_temp": ("--", "s"),
    "bp_sys":    ("-",  "o"), "bp_dia":    ("--", "s"), "glucose": (":", "D"),
    "spo2":      ("-",  "o"), "hrv":       ("--", "s"),
    "light_sensor": ("-", "o"), "proximity": ("--", "s"),
}

# 시간(1h) 해상도가 실재하는 피처 = 시간대별 그림 대상
HOURLY_FEATS = ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia",
                "glucose", "light_sensor"]
HOUR_PANELS = [("A", ["hr", "rr"]),
               ("B1", ["core_temp", "skin_temp"]),
               ("B2", ["bp_sys", "bp_dia", "glucose"]),
               ("C", ["light_sensor"])]
# 하루 1회꼴이라 시간대 분해가 무의미한 센서 (fig3 에서 관측 시각 분포로만 본다)
DAILY_SENSORS = ["spo2", "hrv", "proximity"]

WINDOW_DAYS = 3
SLOTS_PER_HOUR_PER_WINDOW = WINDOW_DAYS   # 한 윈도우에서 각 시간대는 3회 기대


def canonical(col):
    """72 npy 컬럼명 -> 피처명 (분당센서는 _mean 접미사)"""
    return col[:-5] if col.endswith("_mean") else col


def sorted_features(feats):
    return sorted(feats, key=lambda f: (GROUP_ORDER.index(FEATURE_TO_GROUP[f]), f))


# =====================================================
# 집계
# =====================================================

def accumulate(root, chunk=4000):
    """subject × feature × hour 관측수, subject × feature 결측률(시간/일 단위) 누적"""
    cols = [canonical(c) for c in json.loads((root / "feature_columns.json").read_text())]
    subjects = [str(s) for s in json.loads((root / "subject_ids.json").read_text())]
    uniq = sorted(set(subjects))
    s_index = {s: i for i, s in enumerate(uniq)}
    srow = np.array([s_index[s] for s in subjects])

    time = np.load(root / "time.npy", mmap_mode="r")
    n, n_feat, n_slot = time.shape
    assert n_feat == len(cols) and n == len(subjects)
    n_subj = len(uniq)

    obs_hour = np.zeros((n_subj, n_feat, 24), dtype=np.int64)   # 관측 수
    obs_slot = np.zeros((n_subj, n_feat), dtype=np.int64)       # 관측 슬롯 수
    obs_day  = np.zeros((n_subj, n_feat), dtype=np.int64)       # 관측 일수
    n_win    = np.zeros(n_subj, dtype=np.int64)                 # 윈도우 수

    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        T = np.asarray(time[s:e])
        O = T >= 0
        rows = srow[s:e]

        np.add.at(n_win, rows, 1)
        np.add.at(obs_slot, (rows, slice(None)), O.sum(axis=2))

        # 관측 일수: time 은 관측 순으로 정렬되어 있으므로 day 가 바뀌는 지점을 센다
        day = np.where(O, T // 1440, -1)
        changed = (day[:, :, 1:] != day[:, :, :-1]) & O[:, :, 1:]
        np.add.at(obs_day, (rows, slice(None)), O[:, :, 0] + changed.sum(axis=2))

        # 시간대별 관측 수
        hour = np.where(O, (T // 60) % 24, -1).astype(np.int16)
        for f in range(n_feat):
            hf = hour[:, f, :]
            v = hf >= 0
            if not v.any():
                continue
            r = np.repeat(rows[:, None], n_slot, axis=1)[v]
            np.add.at(obs_hour, (r, f, hf[v].astype(np.int64)), 1)

    return dict(cols=cols, subjects=uniq, obs_hour=obs_hour, obs_slot=obs_slot,
                obs_day=obs_day, n_win=n_win, n_windows=n, n_slot=n_slot)


def build_tables(acc):
    cols, uniq = acc["cols"], acc["subjects"]
    n_win = acc["n_win"]

    # subject × feature 결측률 (시간 단위 / 일 단위)
    miss_h = 1.0 - acc["obs_slot"] / (n_win[:, None] * acc["n_slot"])
    miss_d = 1.0 - acc["obs_day"] / (n_win[:, None] * WINDOW_DAYS)

    rows = []
    for j, f in enumerate(cols):
        unit = "1 hour" if f in HOURLY_FEATS else "1 day"
        mr = miss_h[:, j] if f in HOURLY_FEATS else miss_d[:, j]
        never = acc["obs_slot"][:, j] == 0
        rows.append(pd.DataFrame({"subject_id": uniq, "feature": f, "unit": unit,
                                  "missing_rate": mr,
                                  "missing_rate_hourly": miss_h[:, j],
                                  "missing_rate_daily": miss_d[:, j],
                                  "never_measured": never}))
    subj_tab = pd.concat(rows, ignore_index=True)

    feats = sorted_features(cols)
    agg = []
    for f in feats:
        sub = subj_tab[subj_tab["feature"] == f]
        agg.append({
            "feature": f, "group": FEATURE_TO_GROUP[f], "unit": sub["unit"].iloc[0],
            "mean": sub["missing_rate"].mean(), "std": sub["missing_rate"].std(),
            "mean_hourly": sub["missing_rate_hourly"].mean(),
            "mean_daily": sub["missing_rate_daily"].mean(),
            "n_subject": len(sub),
            "n_never": int(sub["never_measured"].sum()),
            "pct_never": sub["never_measured"].mean() * 100,
            # 측정 이력이 있는 subject 만 (아예 안 잰 사람 제외)
            "mean_measured": sub.loc[~sub["never_measured"], "missing_rate"].mean(),
            "std_measured": sub.loc[~sub["never_measured"], "missing_rate"].std(),
        })
    return subj_tab, pd.DataFrame(agg)


def hourly_frame(acc):
    """subject × feature × hour 결측률 long table"""
    cols, uniq, n_win = acc["cols"], acc["subjects"], acc["n_win"]
    exp = (n_win[:, None, None] * SLOTS_PER_HOUR_PER_WINDOW).astype(float)
    mr = 1.0 - acc["obs_hour"] / exp                      # (subj, feat, 24)
    never = acc["obs_slot"] == 0                          # (subj, feat)
    recs = []
    for j, f in enumerate(cols):
        for h in range(24):
            recs.append(pd.DataFrame({"subject_id": uniq, "feature": f, "hour": h,
                                      "missing_rate": mr[:, j, h],
                                      "never_measured": never[:, j]}))
    return pd.concat(recs, ignore_index=True)


# =====================================================
# 그림
# =====================================================

def ampm(h):
    """0->12 AM, 6->6 AM, 12->12 PM, 18->6 PM, 24->12 AM"""
    hh = h % 24
    suffix = "AM" if hh < 12 else "PM"
    disp = hh % 12 or 12
    return f"{disp} {suffix}"


def hour_ticks(ax, step=3):
    """step: 라벨 간격(시간). 패널이 좁으면 6으로 줄여 라벨 충돌을 피한다."""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, step))
    ax.set_xticklabels([ampm(h) for h in range(0, 25, step)])
    if step > 3:
        ax.set_xticks(range(0, 25, 3), minor=True)


def wrap24(y):
    """00:00~24:00 로 그리기 위해 24시 지점에 0시 값을 붙인다"""
    return np.append(y, y[0])


def declutter(items, gap, lo, hi):
    """직접 라벨용 y좌표 충돌 해소.

    items 는 [name, y] 리스트. y 순으로 정렬한 뒤 아래에서부터 최소 gap 만큼
    벌리고, 위쪽으로 밀려 hi 를 넘으면 반대 방향으로 되민다. 레전드 박스를
    쓰지 않고 곡선 옆에 이름을 붙이기 위한 최소 구현.
    """
    out = sorted(([n, float(y)] for n, y in items), key=lambda t: t[1])
    for i in range(1, len(out)):
        out[i][1] = max(out[i][1], out[i - 1][1] + gap)
    over = out[-1][1] - hi
    if over > 0:
        for it in out:
            it[1] -= over
    for i in range(len(out)):                       # 아래쪽 경계도 지킨다
        out[i][1] = min(max(out[i][1], lo + i * gap), hi)
    return out


def fig1_missing_rate(agg, n_subjects, n_windows, out_png):
    fig, ax = plt.subplots(figsize=(20, 8))
    x = np.arange(len(agg))
    colors = [GROUP_COLORS[g] for g in agg["group"]]
    ax.bar(x, agg["mean"], yerr=agg["std"], capsize=4,
           color=colors, edgecolor="black", alpha=0.9)

    ax.set_xticks(x)
    labels = [f"{f}\n({u})" for f, u in zip(agg["feature"], agg["unit"])]
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Missing rate")
    ax.set_xlabel("Feature (measurement unit)")
    ax.set_title("Missing Rate by Feature — preprocessed 72h windows, native measurement unit\n"
                 f"(n={n_subjects:,} subjects, {n_windows:,} windows; mean ± SD across subjects)",
                 weight="bold")
    ax.grid(alpha=0.3, axis="y")
    ax.set_ylim(0, 1.0)

    # ★ 대신 레전드에 그룹명을 실어 PPG 추정군(B2)을 명시한다
    legend = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS[g], edgecolor="black",
                            label=f"{g} ({GROUP_NAME[g]})")
              for g in GROUP_ORDER]
    ax.legend(handles=legend, loc="upper right", ncol=2, framealpha=0.95)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def fig2_hourly(hour_df, subj_tab, agg, n_subjects, out_png):
    """시간대별 결측률 — 시간 해상도가 실재하는 피처만.
    윗줄: 시간대별 곡선 / 아랫줄: subject 단위 결측률 분포.

    아랫줄을 붙인 이유: "아예 측정하지 않는 사람이 많아서 평균 결측률이 높아 보인다"
    는 가설을 검증하기 위함. 실제로는 미측정자가 1~4% 뿐이고 분포도 이봉이 아니라
    연속적이라, 평균이 특정 소집단에 끌려간 게 아님을 같이 보여야 한다.
    """
    fig, axes = plt.subplots(2, 4, figsize=(27, 11.5))
    a_map = agg.set_index("feature")

    # --- 윗줄: 시간대별 결측률 ---
    # 레전드 박스는 쓰지 않는다 — 곡선/에러바를 가리기 때문에, 각 곡선 끝에
    # 피처명을 직접 붙이고(흰 테두리로 가독성 확보) 세로 충돌만 밀어낸다.
    for col, (grp, feats) in enumerate(HOUR_PANELS):
        ax = axes[0, col]
        ends = []
        for f in feats:
            g = (hour_df[hour_df["feature"] == f]
                 .groupby("hour")["missing_rate"].agg(["mean", "std"]).reindex(range(24)))
            ls, mk = FEATURE_STYLE[f]
            y = wrap24(g["mean"].values)
            ax.errorbar(range(25), y, yerr=wrap24(g["std"].values),
                        color=GROUP_COLORS[grp], linestyle=ls, marker=mk,
                        markersize=6, linewidth=2, capsize=2, alpha=0.9)
            ends.append([f, float(y[-2])])          # x=23 지점 값 기준으로 라벨 배치
        hour_ticks(ax, step=6)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        ax.set_title(f"{grp}  ({GROUP_NAME[grp]})", weight="bold", color=GROUP_COLORS[grp])
        ax.set_xlabel("Time of day")
        for f, ylab in declutter(ends, gap=0.075, lo=0.03, hi=0.97):
            ax.text(24.4, ylab, f, color=GROUP_COLORS[grp], fontsize=13, weight="bold",
                    va="center", ha="left", clip_on=False,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        if col == 0:
            ax.set_ylabel("Missing rate")

    # --- 아랫줄: subject 단위 결측률 분포 ---
    bins = np.linspace(0, 1, 21)
    for col, (grp, feats) in enumerate(HOUR_PANELS):
        ax = axes[1, col]
        ends = []
        for f in feats:
            v = subj_tab.loc[subj_tab["feature"] == f, "missing_rate"].values
            ls, mk = FEATURE_STYLE[f]
            cnt, _, _ = ax.hist(v, bins=bins, histtype="step", linewidth=2.2,
                                linestyle=ls, color=GROUP_COLORS[grp])
            ends.append([f, float(cnt[-1])])        # 마지막 bin 높이 기준으로 라벨 배치
        ax.set_xlim(0, 1)
        ax.set_xlabel("Per-subject missing rate")
        ax.grid(alpha=0.3, axis="y")
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
        # 통계 박스는 뺐다 — y축 눈금을 가렸고, 같은 수치가
        # never_measured_72.csv / missing_rate_by_feature_72.csv 에 이미 들어있다.
        top = ax.get_ylim()[1]
        for f, ylab in declutter(ends, gap=0.09 * top, lo=0.04 * top, hi=0.98 * top):
            ax.text(1.02, ylab, f, color=GROUP_COLORS[grp], fontsize=13, weight="bold",
                    va="center", ha="left", clip_on=False,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        if col == 0:
            ax.set_ylabel("Number of subjects")

    axes[0, 0].annotate("Hourly pattern", xy=(-0.30, 0.5), xycoords="axes fraction",
                        rotation=90, va="center", ha="center", weight="bold", fontsize=17)
    axes[1, 0].annotate("Subject-level distribution", xy=(-0.30, 0.5),
                        xycoords="axes fraction", rotation=90, va="center",
                        ha="center", weight="bold", fontsize=17)

    fig.suptitle(
        "Hourly Missing Rate — measured on the preprocessed data "
        "(1-hour aggregation, 3-day windows; mean ± SD across subjects)\n"
        "Sub-daily features only — SpO2/HRV/light_sensor/proximity are daily-cadence "
        "and are shown in Fig.3",
        weight="bold", fontsize=19)
    # w_pad: 축 바깥에 직접 라벨을 두므로 패널 사이를 넉넉히 벌린다
    plt.tight_layout(rect=(0.01, 0, 0.99, 0.93), w_pad=4.5)
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def fig3_daily(agg, acc, out_png):
    """하루 1회꼴 센서 + daily/EMA — 시간대 분해가 무의미한 피처들"""
    fig, axes = plt.subplots(1, 2, figsize=(21, 7.5),
                             gridspec_kw={"width_ratios": [1.25, 1]})

    # (a) 일 단위 결측률
    ax = axes[0]
    d = agg[~agg["feature"].isin(HOURLY_FEATS)].reset_index(drop=True)
    x = np.arange(len(d))
    ax.bar(x, d["mean"], yerr=d["std"], capsize=4,
           color=[GROUP_COLORS[g] for g in d["group"]], edgecolor="black", alpha=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(d["feature"], rotation=45, ha="right")
    ax.set_ylabel("Missing rate (per day)")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("(a) Daily-cadence features — missing rate on a daily basis", weight="bold")
    legend = [plt.Rectangle((0, 0), 1, 1, fc=GROUP_COLORS[g], edgecolor="black",
                            label=f"{g} ({GROUP_NAME[g]})")
              for g in ["D", "E", "F"]]
    ax.legend(handles=legend, loc="upper left", framealpha=0.95, fontsize=12)

    # (b) 관측 시각 분포 — "왜 시간대로 나누지 않는가" 의 근거
    ax = axes[1]
    cols = acc["cols"]
    tot = acc["obs_hour"].sum(axis=0)          # (feat, 24)
    for f in DAILY_SENSORS:
        j = cols.index(f)
        y = tot[j] / max(tot[j].sum(), 1)
        ls, mk = FEATURE_STYLE[f]
        ax.plot(range(25), wrap24(y), label=f, linestyle=ls, marker=mk,
                color=GROUP_COLORS[FEATURE_TO_GROUP[f]], linewidth=2, markersize=6)
    hour_ticks(ax)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Share of all observations")
    ax.grid(alpha=0.3)
    ax.legend(framealpha=0.95)
    ax.set_title("(b) When these sensors are actually sampled\n"
                 "(concentrated in a few hours → hourly split is uninformative)",
                 weight="bold")

    fig.suptitle("Daily-cadence Features — preprocessed 72h windows", weight="bold", fontsize=19)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="data/numpy_all_chunk_72_24feat")
    ap.add_argument("--out_dir", default="analysis_72")
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    figs = out / "plots"; figs.mkdir(parents=True, exist_ok=True)

    print("집계 중...")
    acc = accumulate(root)
    subj_tab, agg = build_tables(acc)
    hour_df = hourly_frame(acc)

    n_subjects = len(acc["subjects"])
    n_windows = acc["n_windows"]

    subj_tab.to_csv(out / "missing_rate_by_subject_feature_72.csv",
                    index=False, encoding="utf-8-sig")
    agg.to_csv(out / "missing_rate_by_feature_72.csv", index=False, encoding="utf-8-sig")
    hour_df.groupby(["feature", "hour"])["missing_rate"].agg(["mean", "std"]).reset_index() \
        .to_csv(out / "hourly_missing_rate_72.csv", index=False, encoding="utf-8-sig")
    agg[["feature", "group", "n_subject", "n_never", "pct_never",
         "mean", "mean_measured"]].to_csv(out / "never_measured_72.csv",
                                          index=False, encoding="utf-8-sig")

    fig1_missing_rate(agg, n_subjects, n_windows, figs / "fig1_missing_rate_feature.png")
    fig2_hourly(hour_df, subj_tab, agg, n_subjects, figs / "fig2_hourly_missing_rate.png")
    fig3_daily(agg, acc, figs / "fig3_daily_cadence.png")

    print(f"\nsubjects={n_subjects:,}  windows={n_windows:,}\n")
    show = agg[["feature", "group", "unit", "mean", "std",
                "mean_hourly", "mean_daily", "n_never", "pct_never", "mean_measured"]]
    print(show.to_string(index=False, formatters={
        c: "{:.3f}".format for c in
        ["mean", "std", "mean_hourly", "mean_daily", "pct_never", "mean_measured"]}))
    print(f"\n✓ {out}/  (csv 4개, plots/ 그림 3장)")


if __name__ == "__main__":
    main()
