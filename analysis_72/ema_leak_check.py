"""
EMA 블록의 Δ(low-high F1)가 EMA_Depression 누수 때문인지 검증
"""
import json

import numpy as np
from sklearn.metrics import f1_score

D='/home/hail/robot_ai2/data/numpy_all_chunk_72/'
P=np.load('analysis/test_predictions_ckpt_e4s22819.npz')
pred,gt=P['pred'],P['gt'].reshape(-1)
static=np.load(D+'static.npy'); time=np.load(D+'time.npy',mmap_mode='r')
cols=json.load(open(D+'feature_columns.json'))
split=np.load(D+'split.npy',allow_pickle=True); test_idx=split[2]
age=static[test_idx][:,1]
obs=(time[test_idx]>=0)

def idx(names): return [i for i,c in enumerate(cols) if c in names]
variants={
 'EMA all (4종)'          : idx(['EMA_Anxiety','EMA_Depression','EMA_Sleep','EMA_Stress']),
 'EMA_Depression 만'      : idx(['EMA_Depression']),
 'EMA - Depression 제외'  : idx(['EMA_Anxiety','EMA_Sleep','EMA_Stress']),
 'EMA_Sleep+Stress 만'    : idx(['EMA_Sleep','EMA_Stress']),
}

v=age>=18
pred_v,gt_v=pred[v],gt[v]
def f1s(y,p): return f1_score(y,p,zero_division=0) if len(y)>5 else np.nan

print(f"{'variant':<24}{'N_low':>7}{'N_high':>7}{'low_F1':>8}{'high_F1':>9}{'gap':>8}")
print('-'*63)
for name,ci in variants.items():
    if not ci:
        print(f"{name:<24}  (feature 없음)"); continue
    mr = 1.0-obs[:,ci,:].mean(axis=(1,2))
    mr = mr[v]
    med=np.median(mr); lo,hi=mr<=med, mr>med
    # median이 0 또는 1이면 split 실패
    f1_lo=f1s(gt_v[lo],pred_v[lo]); f1_hi=f1s(gt_v[hi],pred_v[hi])
    print(f"{name:<24}{lo.sum():>7}{hi.sum():>7}{f1_lo:>8.3f}{f1_hi:>9.3f}{f1_lo-f1_hi:>+8.3f}")

# 추가: EMA 관측 여부(있다/없다)로 이분 - median split보다 직관적
print("\n[EMA 관측 유무로 직접 비교]")
for name,ci in variants.items():
    if not ci: continue
    has = obs[:,ci,:].any(axis=(1,2))[v]     # 하나라도 관측되면 True
    if has.sum()<6 or (~has).sum()<6:
        print(f"  {name:<24} split 불가 (has={has.sum()}, none={(~has).sum()})"); continue
    f1_has=f1s(gt_v[has],pred_v[has]); f1_no=f1s(gt_v[~has],pred_v[~has])
    print(f"  {name:<24} 관측있음 F1={f1_has:.3f} (n={has.sum()})  |  관측없음 F1={f1_no:.3f} (n={(~has).sum()})  gap={f1_has-f1_no:+.3f}")
