import json

import numpy as np
from sklearn.metrics import f1_score

D='/home/hail/robot_ai2/data/numpy_all_chunk_72/'
P=np.load('analysis/test_predictions_ckpt_e4s22819.npz')
pred,gt=P['pred'],P['gt'].reshape(-1)
static=np.load(D+'static.npy'); time=np.load(D+'time.npy',mmap_mode='r')
split=np.load(D+'split.npy',allow_pickle=True); test_idx=split[2]
st=static[test_idx]; sex,age=st[:,0],st[:,1]
obs=(time[test_idx]>=0)                      # (N,46,72)
miss_all=1.0-obs.mean(axis=(1,2))            # 샘플 전체 결측률

valid=age>=18
pred,gt,age,mr=pred[valid],gt[valid],age[valid],miss_all[valid]
def band(a): return '<30' if a<30 else '30s' if a<40 else '40s' if a<50 else '50s' if a<60 else '60+'
bands=np.array([band(a) for a in age])

print("="*72)
print("[1] 나이대 안에서 결측 low/high로 쪼갠 F1 (나이 통제, 결측 자체 효과)")
print("="*72)
print(f"{'age':<6}{'N':>6}  {'low결측 F1':>11}{'high결측 F1':>12}{'Δ(low-high)':>13}")
for b in ['<30','30s','40s','50s','60+']:
    m=bands==b
    y,p,r=gt[m],pred[m],mr[m]
    med=np.median(r)
    lo=r<=med; hi=r>med
    f1_lo=f1_score(y[lo],p[lo],zero_division=0) if lo.sum()>5 else np.nan
    f1_hi=f1_score(y[hi],p[hi],zero_division=0) if hi.sum()>5 else np.nan
    print(f"{b:<6}{m.sum():>6}  {f1_lo:>11.3f}{f1_hi:>12.3f}{f1_lo-f1_hi:>+13.3f}")

print("\n"+"="*72)
print("[2] 나이대별 환자/정상 결측 격차 vs 성능 (언밸런스 효과)")
print("="*72)
print(f"{'age':<6}{'N':>6}{'F1':>7}  {'환자결측%':>9}{'정상결측%':>9}{'격차Δ':>8}")
for b in ['<30','30s','40s','50s','60+']:
    m=bands==b
    y,p,r=gt[m],pred[m],mr[m]
    f1=f1_score(y,p,zero_division=0)
    pat=r[y==1].mean()*100; ctl=r[y==0].mean()*100
    print(f"{b:<6}{m.sum():>6}{f1:>7.3f}  {pat:>9.1f}{ctl:>9.1f}{pat-ctl:>+8.1f}")

print("\n(格차Δ = 환자결측 - 정상결측 %p. 이게 크면 언밸런스 큰 나이대)")
