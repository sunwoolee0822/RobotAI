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

df_mw = pd.read_csv(OUT_DIR / "mann_whitney_v2_overall.csv")

# shorten feature names (missing_1h_hr -> hr)
df_mw['feat_short'] = df_mw['feature'].str.replace('missing_1h_','').str.replace('missing_daily_','')

# group definition and order
GROUPS = {
    'Continuous\n(Band)': ['hr','rr','skin_temp','core_temp','bp_sys','bp_dia','glucose','light_sensor','proximity'],
    'Nightly\n(00~07h)':  ['spo2'],
    '3x Daily\n(HRV)':    ['hrv'],
    'Daily\n(App)':       ['step','distance','screen_time','wake_time','sleep_time',
                           'deep_sleep_time','rem_sleep_time','light_sleep_time','total_sleep_time'],
    'Weekly\n(Survey)':   ['EMA_Anxiety','EMA_Depression','EMA_Sleep','EMA_Stress'],
}

GROUP_COLORS = {
    'Continuous\n(Band)':  '#4C72B0',
    'Nightly\n(00~07h)':   '#55A868',
    '3x Daily\n(HRV)':     '#C44E52',
    'Daily\n(App)':        '#DD8452',
    'Weekly\n(Survey)':    '#8172B2',
}

ordered = []
boundaries = []
labels = []
for grp, feats in GROUPS.items():
    labels.append((len(ordered) + len(feats)/2 - 0.5, grp))
    ordered += feats
    boundaries.append(len(ordered) - 0.5)

df_mw = df_mw.set_index('feat_short').reindex(ordered).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(26, 14))

# left panel
ax = axes[0]
y = np.arange(len(df_mw))
h = 0.35
ax.barh([i+h/2 for i in y], df_mw['control_mean'], height=h, color='steelblue', alpha=0.6, label='Control')
ax.barh([i-h/2 for i in y], df_mw['patient_mean'], height=h, color='salmon',   alpha=0.8, label='Patient')
ax.set_yticks(y)
ax.set_yticklabels(df_mw['feat_short'], fontsize=16)
ax.set_xlabel('Mean Missing Rate', fontsize=20)
ax.set_title('Patient vs Control\nMean Missing Rate by Feature', fontsize=22, fontweight='bold')
ax.legend(fontsize=18)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_xlim(0, 1.25)

for i, row in df_mw.iterrows():
    if row['significant']:
        xmax = max(row['patient_mean'], row['control_mean'])
        ax.text(xmax+0.01, i, '* p='+str(round(row['p_value'],4)),
                va='center', fontsize=16, color='black', fontweight='bold')

for b in boundaries[:-1]:
    ax.axhline(b, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
for y_pos, grp in labels:
    ax.text(-0.32, y_pos, grp, ha='center', va='center', fontsize=14,
            color=GROUP_COLORS[grp], fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='gray', alpha=0.8))

# right panel
ax2 = axes[1]
colors_p = ['crimson' if s else 'lightgray' for s in df_mw['significant']]
ax2.barh(y, -np.log10(df_mw['p_value'].clip(1e-10)), color=colors_p, alpha=0.8)
ax2.axvline(-np.log10(0.05), color='black', linestyle='--', linewidth=1.5, label='p=0.05')
ax2.set_yticks(y)
ax2.set_yticklabels(df_mw['feat_short'], fontsize=16)
ax2.set_xlabel('-log10(p-value)', fontsize=20)
ax2.set_title('Statistical Significance\n(red = p<0.05)', fontsize=22, fontweight='bold')
ax2.legend(fontsize=18)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, row in df_mw.iterrows():
    ax2.text(-np.log10(row['p_value'])+0.05, i,
             'p='+str(round(row['p_value'],4)), va='center', fontsize=14)

for b in boundaries[:-1]:
    ax2.axhline(b, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

plt.suptitle('Step 2: Mann-Whitney U Test\nPatient (n=733) vs Control (n=2098) Missing Rate Comparison',
             fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_step2.png", dpi=150, bbox_inches='tight')
plt.close()
print("Done")
