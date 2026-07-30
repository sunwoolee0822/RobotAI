"""Re-plot the hierarchical clustering figure from the arrays saved by
hierarchical_clustering.py (no model / GPU needed)."""
import matplotlib
import numpy as np

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

OUT_DIR = Path('/home/hail/robot_ai2/analysis')
FIG_DIR = OUT_DIR / 'figures'

plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 24,
    'axes.labelsize': 20,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 24,
})

embeddings     = np.load(OUT_DIR / 'embeddings.npy')
cluster_labels = np.load(OUT_DIR / 'cluster_labels.npy')
labels         = np.load(OUT_DIR / 'labels.npy')

emb_pca = PCA(n_components=50, random_state=0).fit_transform(embeddings)
n_clusters = int(cluster_labels.max()) + 1

fig, axes = plt.subplots(1, 3, figsize=(24, 8))

# Patient rate per cluster
patient_rates, cluster_sizes = [], []
for c in range(n_clusters):
    mask_c = cluster_labels == c
    patient_rates.append((labels[mask_c] == 1).mean() * 100)
    cluster_sizes.append(mask_c.sum())

ax = axes[0]
bars = ax.bar(range(n_clusters), patient_rates, color='salmon', alpha=0.8)
ax.axhline(labels.mean()*100, color='steelblue', linestyle='--',
           label=f'Overall ({labels.mean()*100:.1f}%)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Patient Rate (%)')
ax.set_title('Patient Rate per Cluster')
ax.legend(fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for bar, n in zip(bars, cluster_sizes):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'n={n}', ha='center', fontsize=16)

# PCA scatter (PC1 vs PC2) colored by cluster
ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for c in range(n_clusters):
    mask_c = cluster_labels == c
    ax.scatter(emb_pca[mask_c, 0], emb_pca[mask_c, 1],
               c=[colors[c]], alpha=0.3, s=5, label=f'C{c}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Scatter (colored by cluster)')
ax.legend(markerscale=3, fontsize=14)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Patient vs Control scatter
ax = axes[2]
ax.scatter(emb_pca[labels==0, 0], emb_pca[labels==0, 1],
           c='steelblue', alpha=0.2, s=5, label='Control')
ax.scatter(emb_pca[labels==1, 0], emb_pca[labels==1, 1],
           c='salmon', alpha=0.4, s=8, label='Patient')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Scatter (Patient vs Control)')
ax.legend(markerscale=3, fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle(f'Hierarchical Clustering (Ward, k={n_clusters})\nCoFormer Embedding -> PCA(50)',
             fontsize=26, fontweight='bold')
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_hierarchical_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print('saved: fig_hierarchical_clustering.png')
