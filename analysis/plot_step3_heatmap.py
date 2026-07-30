"""Step 3 heatmaps: -log10(p) of Mann-Whitney per feature x group (age / sex)."""
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

df = pd.read_csv(OUT_DIR / "mann_whitney_v2_grouped.csv")
df['feat_short'] = (df['feature'].str.replace('missing_1h_', '')
                                 .str.replace('missing_daily_', ''))

FEAT_LIST = ['hr','rr','core_temp','skin_temp','bp_sys','bp_dia','glucose','spo2','hrv',
             'light_sensor','proximity','step','distance','screen_time','wake_time',
             'sleep_time','deep_sleep_time','rem_sleep_time','light_sleep_time',
             'total_sleep_time','EMA_Anxiety','EMA_Depression','EMA_Sleep','EMA_Stress']

# raw group values in the CSV (filter keys only, never plotted)
PANELS = [
    ('Age', 'Age_group', ['~30대','30대','40대','50대','60대+'], ['<30s','30s','40s','50s','60s+'],
     'fig_step3_age_feat.png', (16, 24)),
    ('Sex', 'Sex_label', ['남','여'], ['Male','Female'],
     'fig_step3_sex_feat.png', (11, 24)),
]

for panel_name, group_by, keys, disp_labels, fname, figsize in PANELS:
    sub = df[df['group_by'] == group_by]

    logp = np.zeros((len(FEAT_LIST), len(keys)))
    sig_dir = np.full((len(FEAT_LIST), len(keys)), '', dtype=object)

    for j, key in enumerate(keys):
        g = sub[sub['group'] == key].set_index('feat_short')
        for i, feat in enumerate(FEAT_LIST):
            if feat not in g.index:
                continue
            r = g.loc[feat]
            logp[i, j] = -np.log10(max(r['p_value'], 1e-10))
            if r['significant']:
                sig_dir[i, j] = 'v' if r['diff'] < 0 else '^'

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(logp, aspect='auto', cmap='Reds', vmin=0, vmax=4)

    for i in range(len(FEAT_LIST)):
        for j in range(len(keys)):
            if sig_dir[i, j]:
                ax.text(j, i, sig_dir[i, j], ha='center', va='center',
                        color='white', fontsize=24, fontweight='bold')

    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(disp_labels, fontsize=24)
    ax.set_yticks(range(len(FEAT_LIST)))
    ax.set_yticklabels(FEAT_LIST, fontsize=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cbar.set_label('-log10(p-value)', fontsize=22)
    cbar.ax.tick_params(labelsize=20)

    ax.set_title(f'Step 3: Mann-Whitney by {panel_name}\n'
                 '(v=Patient missing less, ^=Patient missing more)\n'
                 f'{panel_name} Group - Feature',
                 fontsize=26, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIG_DIR / fname, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'saved: {fname}')
