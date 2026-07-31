import os
import sys

sys.path.insert(0, '/home/hail/robot_ai2')
sys.path.insert(0, '/home/hail/robot_ai2/CoFormer')

import json

import matplotlib
import numpy as np
import torch

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from CoFormer.models.model_medical_attn_aggre import make_model
from data.dataset import RobotAIDataModule
from module import CoFormerModule

CKPT_PATH  = '/home/hail/robot_ai2/logs/origin_mtm/7sp6nqdy/checkpoints/epoch=2-step=12502-val_loss=0.4946.ckpt'
DATA_ROOT  = '/home/hail/robot_ai2/data/numpy_all_chunk/'
SPLIT_PATH = '/home/hail/robot_ai2/data/numpy_all_chunk/split.npy'
OUTPUT_DIR = '/home/hail/robot_ai2/attention_maps/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(DATA_ROOT + 'feature_columns.json') as f:
    features = json.load(f)
n_feat = len(features)

# _mean 또는 max/min 없는 feature만
mean_features = [f for f in features if f.endswith('_mean') or
                 not any(f.endswith(s) for s in ['_max', '_min'])]
mean_indices  = [i for i, f in enumerate(features) if f in mean_features]
mean_labels   = [f.replace('_mean', '') for f in mean_features]
n_mean        = len(mean_features)
print(f"선택된 features: {n_mean}개")

print("모델 로드 중...")
backbone = make_model(
    src_vocab=1, tgt_vocab=2,
    N=8, d_model=256, d_ff=256, h=8,
    dropout=0.1, num_agents=n_feat,
    num_neighbors=11, agent_encoding_dim=32, static_dim=4,
)
model = CoFormerModule.load_from_checkpoint(
    CKPT_PATH,
    model=backbone,
    cfg={'lr': 1e-4, 'weight_decay': 0.0, 'lr_factor': 0.2, 'patience': 25},
)
model.eval()
model = model.model
device = torch.device('cuda:0')
model = model.to(device)

print("데이터 로드 중...")
dm = RobotAIDataModule(DATA_ROOT, SPLIT_PATH, batch_size=32)
dm.setup()

patient_batch = None
control_batch = None
for batch in dm.test_dataloader():
    gt = batch['gt'].squeeze()
    p_idx = (gt == 1).nonzero(as_tuple=True)[0]
    c_idx = (gt == 0).nonzero(as_tuple=True)[0]
    if len(p_idx) >= 8 and patient_batch is None:
        patient_batch = {k: v[p_idx[:8]] for k, v in batch.items()}
    if len(c_idx) >= 8 and control_batch is None:
        control_batch = {k: v[c_idx[:8]] for k, v in batch.items()}
    if patient_batch is not None and control_batch is not None:
        break

def extract_all_attentions(batch):
    data   = batch['data'].to(device)
    time   = batch['time'].to(device)
    mask   = batch['mask'].to(device)
    static = batch['static'].to(device)

    with torch.no_grad():
        _ = model(data, time, mask, static)

    # 1. Temporal attention (layer1s 마지막)
    # self.attn: (B, agent, head, T, T)
    t_attn = model.encoder.layer1s[-1].self_attn.attn
    t_attn = t_attn.mean(dim=2).mean(dim=-2).mean(dim=0)  # (agent, T)
    temporal = t_attn.cpu().numpy()

    # 2. Channel attention (layer2s GAT - GATv2Conv attention)
    # hook으로 뽑기
    channel_attn = [None]
    def hook_fn(module, input, output):
        # GATv2Conv output: (N*B, head, out_dim) → attention weights 별도
        pass

    # GAT attention은 dgl GATv2Conv에서 직접 뽑기 어려우므로
    # agentwise_memory를 proxy로 사용
    # layer2s의 self_attn(GATlayer)은 attn 저장 안 함
    # 대신 agent_aggre attention 뽑기

    # 3. Agent aggregation attention
    # agent_aggre: MultiHeadedAttention
    # self.attn: (B, agent, head, 1, agent) → agent가 어떤 agent를 봤는지
    a_attn = model.agent_aggre.attn  # (B, agent, head, 1, agent)
    a_attn = a_attn.mean(dim=2).squeeze(dim=-2).mean(dim=0)  # (agent, agent)
    aggre = a_attn.cpu().numpy()

    return temporal, aggre

print("Patient attention 추출...")
p_temporal, p_aggre = extract_all_attentions(patient_batch)
print("Control attention 추출...")
c_temporal, c_aggre = extract_all_attentions(control_batch)

# mean feature 인덱스로 추출
p_t = p_temporal[mean_indices]
c_t = c_temporal[mean_indices]
p_a = p_aggre[np.ix_(mean_indices, mean_indices)]
c_a = c_aggre[np.ix_(mean_indices, mean_indices)]

day_ticks  = list(range(0, 73))
day_labels = [str(h) for h in range(0, 73)]

def plot_triple(p, c, title, fname, xlabel, ylabel_p='Feature', cmap='YlOrRd', figsize=(60,8)):
    diff  = p - c
    vmax  = max(abs(p).max(), abs(c).max())
    vlim  = abs(diff).max()
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, data, ttl, cm, vmin_val, vmax_val in [
        (axes[0], p,    f'Patient - {title}',      cmap,     0,     vmax),
        (axes[1], c,    f'Control - {title}',      cmap,     0,     vmax),
        (axes[2], diff, f'Diff (Patient-Control)', 'RdBu_r', -vlim, vlim),
    ]:
        im = ax.imshow(data, aspect='auto', cmap=cm, vmin=vmin_val, vmax=vmax_val)
        ax.set_title(ttl, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel_p, fontsize=9)
        ax.set_yticks(range(n_mean))
        ax.set_yticklabels(mean_labels, fontsize=7)
        if data.shape[1] == 72:
            ax.set_xticks(day_ticks[::6])
            ax.set_xticklabels(day_labels[::6], fontsize=7)
            for x in [24, 48]:
                ax.axvline(x=x, color='gray', linewidth=1, linestyle='--', alpha=0.5)
        else:
            ax.set_xticks(range(n_mean))
            ax.set_xticklabels(mean_labels, fontsize=6, rotation=90)
        plt.colorbar(im, ax=ax)
    plt.suptitle(f'CoFormer {title}', fontsize=13, weight='bold')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/{fname}', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  저장: {fname}")

# 1. Temporal attention
print("Temporal attention 저장 중...")
plot_triple(p_t, c_t,
            title='Temporal Attention (Layer4)',
            fname='1_temporal_attn.png',
            xlabel='Time (hour)')

# 2. Agent aggregation attention
print("Agent aggregation attention 저장 중...")
plot_triple(p_a, c_a,
            title='Agent Aggregation Attention',
            fname='2_agent_aggre_attn.png',
            xlabel='Key Feature',
            figsize=(30, 10))

print(f"\n완료! → {OUTPUT_DIR}")
print("※ Channel attention (GAT)은 DGL GATv2Conv 구조상 별도 추출 필요")
