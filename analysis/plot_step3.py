import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

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
df['Age_group'] = pd.cut(df['Age'], bins=[0,30,40,50,60,100],
                          labels=['<30s','30s','40s','50s','60s+'])
df['Sex_label'] = df['Sex'].map({1.0:'Male', 2.0:'Female', 0.0:'Female'})

FEAT_LIST = ['hr','rr','core_temp','skin_temp','bp_sys','bp_dia','glucose','spo2','hrv',
             'light_sensor','proximity','step','distance','screen_time','wake_time',
             'sleep_time','deep_sleep_time','rem_sleep_time','light_sleep_time',
             'total_sleep_time','EMA_Anxiety','EMA_Depression','EMA_Sleep','EMA_Stress']

FEAT_COLS = []
for f in FEAT_LIST:
    if f'missing_1h_{f}' in df.columns:
        FEAT_COLS.append(f'missing_1h_{f}')
    elif f'missing_daily_{f}' in df.columns:
        FEAT_COLS.append(f'missing_daily_{f}')

def get_stats(group_df, col):
    p = group_df[group_df['phq9_label']==1][col].dropna()
    c = group_df[group_df['phq9_label']==0][col].dropna()
    if len(p) > 5 and len(c) > 5:
        _, pval = mannwhitneyu(p, c, alternative='two-sided')
    else:
        pval = 1.0
    return p.mean()*100, p.sem()*100, c.mean()*100, c.sem()*100, pval

def draw_barplot(ax, group_df, title):
    x = np.arange(len(FEAT_LIST))
    w = 0.35
    p_means, p_sems, c_means, c_sems, pvals = [], [], [], [], []
    for col in FEAT_COLS:
        pm, ps, cm, cs, pv = get_stats(group_df, col)
        p_means.append(pm); p_sems.append(ps)
        c_means.append(cm); c_sems.append(cs)
        pvals.append(pv)

    n_p = (group_df['phq9_label']==1).sum()
    n_c = (group_df['phq9_label']==0).sum()

    ax.bar(x-w/2, c_means, width=w, yerr=c_sems, color='steelblue', alpha=0.7,
           capsize=3, label=f'Control (n={n_c})', error_kw={'linewidth':1.2})
    ax.bar(x+w/2, p_means, width=w, yerr=p_sems, color='salmon', alpha=0.8,
           capsize=3, label=f'Patient (n={n_p})', error_kw={'linewidth':1.2})

    for i, pv in enumerate(pvals):
        if pv < 0.05:
            ymax = max(p_means[i]+p_sems[i], c_means[i]+c_sems[i])
            ax.text(i, ymax+1.0, '*', ha='center', fontsize=28, color='red', fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(FEAT_LIST, rotation=45, ha='right', fontsize=22)
    ax.set_ylabel('Mean Missing Rate (%)', fontsize=24)
    ax.set_ylim(0, 110)
    ax.set_title(title, fontweight='bold', fontsize=28)
    ax.legend(fontsize=22)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# =====================
# Figure 1: Sex (2 rows)
# =====================
sex_groups = ['Male', 'Female']
fig, axes = plt.subplots(2, 1, figsize=(24, 18))

for ax, sex in zip(axes, sex_groups):
    sub = df[df['Sex_label']==sex]
    draw_barplot(ax, sub, sex)

plt.suptitle('Step 3: Sex Group - Patient vs Control Missing Rate\n(* = p<0.05, error bar = SEM)',
             fontsize=30, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_step3_sex.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_step3_sex.png saved")

# =====================
# Figure 2: Age (5 rows)
# =====================
age_groups = ['<30s','30s','40s','50s','60s+']
fig, axes = plt.subplots(5, 1, figsize=(24, 45))

for ax, age in zip(axes, age_groups):
    sub = df[df['Age_group']==age]
    draw_barplot(ax, sub, age)

plt.suptitle('Step 3: Age Group - Patient vs Control Missing Rate\n(* = p<0.05, error bar = SEM)',
             fontsize=30, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_step3_age.png", dpi=150, bbox_inches='tight')
plt.close()
print("fig_step3_age.png saved")
