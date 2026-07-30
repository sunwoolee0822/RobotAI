import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path("/home/hail/robot_ai2/analysis")
FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
})

df = pd.read_csv(OUT_DIR / "diurnal_missing_rate.csv")
patient = df[df["phq9_label"]==1]
control = df[df["phq9_label"]==0]

FEATS = ["hr", "rr", "core_temp", "skin_temp", "bp_sys", "bp_dia", "glucose"]


def get_hourly_stats(data, feat):
    col = f"missing_{feat}"
    grp = data.groupby("hour")[col].agg(["mean","sem"]).reset_index()
    return grp

fig, axes = plt.subplots(2, 4, figsize=(26, 12), sharey=False)
axes = axes.flatten()

for i, feat in enumerate(FEATS):
    ax = axes[i]
    p_stat = get_hourly_stats(patient, feat)
    c_stat = get_hourly_stats(control, feat)

    p_mean = p_stat["mean"] * 100
    c_mean = c_stat["mean"] * 100
    p_sem  = p_stat["sem"]  * 100
    c_sem  = c_stat["sem"]  * 100

    ax.plot(c_stat["hour"], c_mean, color='steelblue',
            marker='o', markersize=3, label='Control', linewidth=1.5)
    ax.fill_between(c_stat["hour"], c_mean - c_sem, c_mean + c_sem,
                    alpha=0.2, color='steelblue')

    ax.plot(p_stat["hour"], p_mean, color='salmon',
            marker='o', markersize=3, label='Patient', linewidth=1.5)
    ax.fill_between(p_stat["hour"], p_mean - p_sem, p_mean + p_sem,
                    alpha=0.2, color='salmon')

    # night shading
    ax.axvspan(22, 23.9, alpha=0.08, color='gray')
    ax.axvspan(0, 6,     alpha=0.08, color='gray')

    # auto y-range (min-0.5% ~ max+0.5%)
    all_vals = pd.concat([p_mean, c_mean])
    ymin = max(0, all_vals.min() - 0.5)
    ymax = min(100, all_vals.max() + 0.5)
    ax.set_ylim(ymin, ymax)

    ax.set_title(feat, fontsize=22, fontweight='bold')
    ax.set_xlabel('Hour of day (0-23)', fontsize=16)
    ax.set_ylabel('Missing Rate (%)', fontsize=16)
    ax.set_xticks(range(0, 24, 4))
    ax.legend(fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[-1].set_visible(False)

plt.suptitle('Diurnal Missingness - High Frequency Sensors\n(shaded = night, band = SEM)',
             fontsize=26, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_diurnal_missing.png", dpi=150, bbox_inches='tight')
plt.close()
print("Done")
