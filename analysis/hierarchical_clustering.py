import os
import sys

sys.path.insert(0, '/home/hail/robot_ai2')
sys.path.insert(0, '/home/hail/robot_ai2/CoFormer')

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA

from CoFormer.models.model_medical_attn_aggre import make_model
from data.dataset import RobotAIDataModule
from module import CoFormerModule

# 24feat baseline(run mqe2zp1x) 체크포인트. top-3 중 택1 — 다른 후보:
#   epoch=10-step=53610-val_loss=0.8466.ckpt / epoch=9-step=51499-val_loss=0.5532.ckpt
CKPT       = '/home/hail/robot_ai2/logs/coformer_24feat/mqe2zp1x/checkpoints/epoch=10-step=55485-val_loss=0.6666.ckpt'
DATA_ROOT  = '/home/hail/robot_ai2/data/numpy_all_chunk_72_24feat/'
SPLIT_PATH = '/home/hail/robot_ai2/data/numpy_all_chunk_72_24feat/split.npy'
DEVICE     = 'cuda:0'   # high 학습이 gpu1을 쓰는 중이면 0, 반대면 1로
OUT_DIR    = Path('/home/hail/robot_ai2/analysis/figures')
OUT_DIR.mkdir(exist_ok=True)

# =====================
# 1. 모델 로드
# =====================
print("모델 로드 중...")
dm = RobotAIDataModule(DATA_ROOT, SPLIT_PATH, batch_size=16)
dm.setup()
num_agents = dm.train_dset.data.shape[1]

backbone = make_model(
    src_vocab=1, tgt_vocab=2, N=8, d_model=256, d_ff=256,
    h=8, dropout=0.1, num_agents=num_agents,
    num_neighbors=30, agent_encoding_dim=32, static_dim=4
)

with open('/home/hail/robot_ai2/cfg/robotai.yaml') as f:
    cfg = yaml.safe_load(f)

model = CoFormerModule.load_from_checkpoint(
    CKPT, model=backbone, cfg=cfg, strict=False
)
model.eval()
device = torch.device(DEVICE)
model = model.to(device)

# =====================
# 2. Hook으로 all_feature 추출
# =====================
print("Embedding 추출 중...")

embedding_buffer = []

def hook_fn(module, input, output):
    embedding_buffer.append(input[0].detach().cpu())

# generator 직전 all_feature를 뽑으려면 generator에 hook
hook = model.model.generator.register_forward_hook(hook_fn)

labels  = []
ages    = []
sexes   = []

with torch.no_grad():
    for batch in dm.test_dataloader():
        data   = batch['data'].to(device)
        time   = batch['time'].to(device)
        mask   = batch['mask'].to(device)
        static = batch['static'].to(device)
        gt     = batch['gt'].squeeze(-1).long()

        _ = model.model(data, time, mask, static)

        labels.append(gt.numpy())
        ages.append(static[:, 1].cpu().numpy())
        sexes.append(static[:, 0].cpu().numpy())

hook.remove()

embeddings = torch.cat(embedding_buffer, dim=0).numpy()
labels     = np.concatenate(labels)
ages       = np.concatenate(ages)
sexes      = np.concatenate(sexes)
print(f"Embedding shape: {embeddings.shape}")

# =====================
# 3. PCA 256 → 50
# =====================
print("PCA 중...")
pca = PCA(n_components=50)
emb_pca = pca.fit_transform(embeddings)
print(f"설명된 분산: {pca.explained_variance_ratio_.sum():.3f}")

# =====================
# 4. Hierarchical Clustering (Ward)
# =====================
print("Clustering 중...")
n_clusters = 6
clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
cluster_labels = clustering.fit_predict(emb_pca)

# =====================
# 5. 클러스터별 분석
# =====================
print("\n=== 클러스터별 분석 ===")
for c in range(n_clusters):
    mask_c = cluster_labels == c
    n_total   = mask_c.sum()
    n_patient = (labels[mask_c] == 1).sum()
    patient_rate = n_patient / n_total * 100
    mean_age  = ages[mask_c].mean()
    female_rate = (sexes[mask_c] == 1).mean() * 100
    print(f"Cluster {c}: n={n_total:4d} | Patient={patient_rate:.1f}% | "
          f"Age={mean_age:.1f} | Female={female_rate:.1f}%")

# =====================
# 6. Figure - 클러스터별 Patient rate
# =====================
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Patient rate per cluster
patient_rates = []
cluster_sizes = []
for c in range(n_clusters):
    mask_c = cluster_labels == c
    patient_rates.append((labels[mask_c] == 1).mean() * 100)
    cluster_sizes.append(mask_c.sum())

ax = axes[0]
bars = ax.bar(range(n_clusters), patient_rates, color='salmon', alpha=0.8)
ax.axhline(labels.mean()*100, color='steelblue', linestyle='--', label=f'Overall ({labels.mean()*100:.1f}%)')
ax.set_xlabel('Cluster')
ax.set_ylabel('Patient Rate (%)')
ax.set_title('Patient Rate per Cluster')
ax.legend()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
for i, (bar, n) in enumerate(zip(bars, cluster_sizes)):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            f'n={n}', ha='center', fontsize=8)

# PCA scatter (PC1 vs PC2)
ax = axes[1]
colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
for c in range(n_clusters):
    mask_c = cluster_labels == c
    ax.scatter(emb_pca[mask_c, 0], emb_pca[mask_c, 1],
               c=[colors[c]], alpha=0.3, s=5, label=f'C{c}')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA Scatter (colored by cluster)')
ax.legend(markerscale=3, fontsize=7)
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
ax.legend(markerscale=3, fontsize=8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('Hierarchical Clustering (Ward, k=6)\nCoFormer Embedding → PCA(50)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig_hierarchical_clustering.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n저장: fig_hierarchical_clustering.png")
print("Done")

# 저장
np.save('/home/hail/robot_ai2/analysis/embeddings.npy', embeddings)
np.save('/home/hail/robot_ai2/analysis/cluster_labels.npy', cluster_labels)
np.save('/home/hail/robot_ai2/analysis/labels.npy', labels)
np.save('/home/hail/robot_ai2/analysis/ages.npy', ages)
np.save('/home/hail/robot_ai2/analysis/sexes.npy', sexes)
print("저장 완료")
