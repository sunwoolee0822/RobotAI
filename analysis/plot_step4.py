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

df_blk = pd.read_csv(OUT_DIR / "mann_whitney_v2_block.csv")
df_blk['block_short'] = df_blk['block'].str.replace('block_','')

# raw group values stored in the CSV (kept as-is for filtering only, never plotted)
OVERALL_KEY = '전체'
AGE_KEYS = ['~30대','30대','40대','50대','60대+']
SEX_KEYS = ['남','여']

AGE_LABELS = ['<30s','30s','40s','50s','60s+']
SEX_LABELS = ['Male','Female']

BLOCKS = ['A_hr_rr','B1_temp','B2_ppg','A_spo2','C_hrv','D_event','E_daily','F_ema']

def get_vals(sub, block, group_val):
    row = sub[(sub['block_short']==block) & (sub['group']==group_val)]
    if len(row) == 0:
        return np.nan, np.nan, False
    r = row.iloc[0]
    return r['patient_mean']*100, r['control_mean']*100, r['significant']

fig, axes = plt.subplots(3, 1, figsize=(20, 22))

# =====================
# 1. Overall
# =====================
ax = axes[0]
sub = df_blk[df_blk['group_by']==OVERALL_KEY]
x = np.arange(len(BLOCKS))
w = 0.35
p_vals = [get_vals(sub, b, OVERALL_KEY)[0] for b in BLOCKS]
c_vals = [get_vals(sub, b, OVERALL_KEY)[1] for b in BLOCKS]
sigs   = [get_vals(sub, b, OVERALL_KEY)[2] for b in BLOCKS]

ax.bar(x-w/2, c_vals, width=w, color='steelblue', alpha=0.7, label='Control')
ax.bar(x+w/2, p_vals, width=w, color='salmon',    alpha=0.8, label='Patient')
for i, s in enumerate(sigs):
    if s:
        ymax = max(p_vals[i], c_vals[i])
        ax.text(i, ymax+0.5, '*', ha='center', fontsize=28, color='red', fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(BLOCKS, fontsize=18)
ax.set_ylabel('Mean Missing Rate (%)')
ax.set_title('Overall (Patient n=733 vs Control n=2098)', fontweight='bold')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =====================
# 2. Age Group
# =====================
ax = axes[1]
sub = df_blk[df_blk['group_by']=='Age_group']
colors_age = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
n = len(AGE_KEYS)
total_w = 0.8
w = total_w / n
offsets = np.linspace(-total_w/2+w/2, total_w/2-w/2, n)

for gi, (grp_data, grp_label, col) in enumerate(zip(AGE_KEYS, AGE_LABELS, colors_age)):
    p_vals = [get_vals(sub, b, grp_data)[0] for b in BLOCKS]
    sigs   = [get_vals(sub, b, grp_data)[2] for b in BLOCKS]
    ax.bar(x+offsets[gi], p_vals, width=w, color=col, alpha=0.8, label=grp_label)
    for i, s in enumerate(sigs):
        if s and not np.isnan(p_vals[i]):
            ax.text(i+offsets[gi], p_vals[i]+0.5, '*', ha='center', fontsize=18, color='black', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(BLOCKS, fontsize=18)
ax.set_ylabel('Patient Mean Missing Rate (%)')
ax.set_title('Age Group - Patient Missing Rate (* = p<0.05)', fontweight='bold')
ax.legend(fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# =====================
# 3. Sex Group
# =====================
ax = axes[2]
sub = df_blk[df_blk['group_by']=='Sex_label']
colors_sex = ['steelblue','salmon']
n = len(SEX_KEYS)
total_w = 0.7
w = total_w / (n*2)
base_offsets = [-0.35, 0.0]

for gi, (grp_data, grp_label, col) in enumerate(zip(SEX_KEYS, SEX_LABELS, colors_sex)):
    off_c = base_offsets[gi]
    off_p = base_offsets[gi] + w + 0.02
    c_vals = [get_vals(sub, b, grp_data)[1] for b in BLOCKS]
    p_vals = [get_vals(sub, b, grp_data)[0] for b in BLOCKS]
    sigs   = [get_vals(sub, b, grp_data)[2] for b in BLOCKS]
    ax.bar(x+off_c, c_vals, width=w, color=col, alpha=0.3, label=f'Control ({grp_label})')
    ax.bar(x+off_p, p_vals, width=w, color=col, alpha=0.9, label=f'Patient ({grp_label})')
    for i, s in enumerate(sigs):
        if s and not np.isnan(p_vals[i]):
            ymax = max(p_vals[i], c_vals[i])
            ax.text(i+off_p, ymax+0.5, '*', ha='center', fontsize=20, color='red', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(BLOCKS, fontsize=18)
ax.set_ylabel('Mean Missing Rate (%)')
ax.set_title('Sex Group - Patient vs Control (* = p<0.05)', fontweight='bold')
ax.legend(fontsize=14, ncol=2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Step 4: Block-level Missing Rate Comparison',
             fontsize=26, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / "fig_step4.png", dpi=150, bbox_inches='tight')
plt.close()
print("Done")
