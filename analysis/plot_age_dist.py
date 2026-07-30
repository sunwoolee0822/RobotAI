import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt

OUT_DIR = Path("/home/hail/robot_ai2/analysis")
FIG_DIR = OUT_DIR / "figures"

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
})

df = pd.read_csv(OUT_DIR / "subject_missing_rate_v2.csv")

def parse_age(df):
    df = df.copy()
    df["Age_group"] = pd.cut(df["Age"], bins=[0,30,40,50,60,100],
                              labels=["<30s","30s","40s","50s","60s+"])
    return df

df = parse_age(df)

# significant features per age group
SIG_FEATS = {
    "<30s": ["missing_1h_rr", "missing_daily_wake_time", "missing_daily_total_sleep_time",
             "missing_1h_bp_sys", "missing_1h_bp_dia"],
    "30s":  ["missing_daily_total_sleep_time", "missing_daily_screen_time",
             "missing_daily_distance", "missing_daily_step", "missing_1h_hrv"],
    "40s":  ["missing_daily_total_sleep_time", "missing_1h_skin_temp"],
    "50s":  ["missing_daily_total_sleep_time", "missing_daily_screen_time", "missing_1h_glucose"],
    "60s+": ["missing_daily_EMA_Anxiety", "missing_daily_EMA_Depression",
             "missing_daily_EMA_Sleep", "missing_daily_EMA_Stress"],
}

FNAME_MAP = {"<30s": "under30s", "30s": "30s", "40s": "40s", "50s": "50s", "60s+": "60splus"}

AGE_GROUPS = ["<30s", "30s", "40s", "50s", "60s+"]
bins = np.linspace(0, 1, 21)

for age_grp in AGE_GROUPS:
    feats = SIG_FEATS[age_grp]
    sub = df[df["Age_group"] == age_grp]
    patient = sub[sub["phq9_label"]==1]
    control = sub[sub["phq9_label"]==0]

    n_feats = len(feats)
    fig, axes = plt.subplots(1, n_feats, figsize=(7*n_feats, 6))
    if n_feats == 1:
        axes = [axes]

    for ax, col in zip(axes, feats):
        feat_name = col.replace("missing_1h_","").replace("missing_daily_","")
        p_vals = patient[col].dropna().values
        c_vals = control[col].dropna().values

        p_w = np.ones(len(p_vals)) / len(p_vals) * 100
        c_w = np.ones(len(c_vals)) / len(c_vals) * 100

        ax.hist(c_vals, bins=bins, weights=c_w, alpha=0.6, color='steelblue',
                label=f'Control (n={len(c_vals)})')
        ax.hist(p_vals, bins=bins, weights=p_w, alpha=0.6, color='salmon',
                label=f'Patient (n={len(p_vals)})')
        ax.axvline(p_vals.mean(), color='darkred',  linestyle='--', linewidth=2)
        ax.axvline(c_vals.mean(), color='darkblue', linestyle='--', linewidth=2)

        ax.set_xlabel('Missing Rate')
        ax.set_ylabel('% of Subjects')
        ax.set_xlim(0, 1)
        ax.legend(fontsize=14)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        from scipy.stats import mannwhitneyu
        _, p = mannwhitneyu(p_vals, c_vals, alternative='two-sided')
        diff = p_vals.mean() - c_vals.mean()
        direction = 'P < C' if diff < 0 else 'P > C'
        ax.set_title(f'{feat_name}\np={p:.4f} ({direction})', fontsize=18, fontweight='bold')

    plt.suptitle(f'{age_grp} (Patient n={len(patient)}, Control n={len(control)})',
                 fontsize=24, fontweight='bold')
    plt.tight_layout()
    fname = f"fig_dist_{FNAME_MAP[age_grp]}.png"
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"saved: {fname}")

print("Done")
