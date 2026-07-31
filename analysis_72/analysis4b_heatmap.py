"""
analysis4b_heatmap.py — 2-5 subgroup 결과 히트맵 (Cliff's δ)
성별/나이대 subgroup × feature 의 δ를 히트맵으로. (구 plot_step3_heatmap의 δ 버전)

입력 : analysis_336/stats_missing_by_group_feature.csv (analysis4_336.py 산출)
출력 : analysis_336/plots1/fig_subgroup_delta_age.png, fig_subgroup_delta_sex.png

사용법: python analysis4b_heatmap.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("/home/hail/robot_ai2/analysis_336")
FIG = OUT / "plots1"; FIG.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(OUT / "stats_missing_by_group_feature.csv")

FEATS = ['hr','rr','core_temp','skin_temp','bp_sys','bp_dia','glucose','spo2','hrv',
         'light_sensor','proximity','step','distance','screen_time','wake_time',
         'sleep_time','deep_sleep_time','rem_sleep_time','light_sleep_time',
         'total_sleep_time','EMA_Anxiety','EMA_Depression','EMA_Sleep','EMA_Stress']

PANELS = [
    ("Age", ["<30","30s","40s","50s","60+"], "fig_subgroup_delta_age.png"),
    ("Sex", ["Male","Female"], "fig_subgroup_delta_sex.png"),
]

for gb, groups, fname in PANELS:
    sub = df[df["group_by"] == gb]
    M = np.full((len(FEATS), len(groups)), np.nan)
    sig = np.full((len(FEATS), len(groups)), "", dtype=object)
    for j, g in enumerate(groups):
        gg = sub[sub["group"] == g].set_index("feature")
        for i, f in enumerate(FEATS):
            if f in gg.index:
                M[i, j] = gg.loc[f, "delta"]
                s = gg.loc[f, "sig"]
                sig[i, j] = "" if s in ("ns", "NA") else "*"

    fig, ax = plt.subplots(figsize=(2.2*len(groups)+3, 12))
    vmax = 0.35
    im = ax.imshow(M, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i in range(len(FEATS)):
        for j in range(len(groups)):
            if not np.isnan(M[i, j]):
                txt = f"{M[i,j]:+.2f}{sig[i,j]}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i,j]) > 0.2 else "black")
    ax.set_xticks(range(len(groups))); ax.set_xticklabels(groups, fontsize=13)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label("Cliff's delta  (neg = control missing more / pos = depressed more)", fontsize=11)
    ax.set_title(f"Subgroup x Feature: Depressed vs Control missing, Cliff's delta ({gb})\n"
                 "* = p<.05 (no multiple-comparison correction). |delta|: 0.11 small / 0.28 medium",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(FIG / fname, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved: {FIG/fname}")