"""
그룹별 분석 figure 생성 (예측 파일에서 매번 재계산)
사용법:
  python make_figures.py
  python make_figures.py --pred analysis/test_predictions_full.npz --tag _full
"""
import matplotlib

matplotlib.use('Agg')
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

ap=argparse.ArgumentParser()
ap.add_argument('--pred', default='analysis/test_predictions_ckpt_e4s22819.npz')
ap.add_argument('--data', default='/home/hail/robot_ai2/data/numpy_all_chunk_72/')
ap.add_argument('--out',  default='analysis')
ap.add_argument('--tag',  default='')
args=ap.parse_args()

plt.rcParams['font.size']=11; plt.rcParams['axes.unicode_minus']=False

# ---- load ----
P=np.load(args.pred); pred,gt=P['pred'],P['gt'].reshape(-1)
D=args.data
static=np.load(D+'static.npy'); time=np.load(D+'time.npy',mmap_mode='r')
cols=json.load(open(D+'feature_columns.json'))
split=np.load(D+'split.npy',allow_pickle=True); test_idx=split[2]
st=static[test_idx]; sex,age=st[:,0],st[:,1]
obs=(time[test_idx]>=0)                 # (N,46,72)
miss=1.0-obs.mean(axis=(1,2))           # 전체 결측률
assert len(pred)==len(st), f"length mismatch {len(pred)} vs {len(st)}"

# 블록 정의 (수집주기 그룹, prefix 매칭)
def cidx(prefixes): return [i for i,c in enumerate(cols) if any(c.startswith(p) for p in prefixes)]
BLOCKS={
 'A hr/rr':cidx(['hr_','rr_']),
 'A spo2':cidx(['spo2']),
 'B temp/bp/glu':cidx(['core_temp','skin_temp','bp_','glucose']),
 'C hrv':cidx(['hrv']),
 'D phone':cidx(['light_sensor','proximity']),
 'E daily':cidx(['step','distance','screen_time','wake_time','sleep_time','deep_sleep','rem_sleep','light_sleep','total_sleep']),
 'F ema':cidx(['EMA_']),
}
BLOCKS={k:v for k,v in BLOCKS.items() if len(v)>0}
# 블록별 결측률 (샘플별)
block_miss={k: 1.0-obs[:,ci,:].mean(axis=(1,2)) for k,ci in BLOCKS.items()}

# age 오염 제거
v=age>=18
pred,gt,sex,age,miss=pred[v],gt[v],sex[v],age[v],miss[v]
block_miss={k:m[v] for k,m in block_miss.items()}
def band(a): return '<30' if a<30 else '30s' if a<40 else '40s' if a<50 else '50s' if a<60 else '60+'
bands=np.array([band(a) for a in age])
ages=['<30','30s','40s','50s','60+']; x=np.arange(5); w=0.38
t=args.tag

def f1s(y,p): return f1_score(y,p,zero_division=0) if len(y)>5 else np.nan

# ===== Fig1: 성별x나이 F1 =====
F_f1=[f1s(gt[(sex==1.0)&(bands==b)],pred[(sex==1.0)&(bands==b)]) for b in ages]
M_f1=[f1s(gt[(sex==0.0)&(bands==b)],pred[(sex==0.0)&(bands==b)]) for b in ages]
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(x-w/2,F_f1,w,label='Female',color='#e87ba4'); ax.bar(x+w/2,M_f1,w,label='Male',color='#2a78d6')
ax.set_xticks(x);ax.set_xticklabels(ages);ax.set_xlabel('Age group');ax.set_ylabel('F1');ax.set_ylim(0,0.7)
ax.set_title('(1) F1 by Sex x Age  (performance drops with age)')
ax.legend();ax.grid(axis='y',alpha=0.3)
for i,val in enumerate(F_f1): ax.text(i-w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=8)
for i,val in enumerate(M_f1): ax.text(i+w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=8)
plt.tight_layout();plt.savefig(f'{args.out}/fig1_f1_by_age_sex{t}.png',dpi=150);plt.close()

# ===== Fig2: 언밸런스 격차 vs F1 =====
f1_age=[f1s(gt[bands==b],pred[bands==b]) for b in ages]
gap=[]
for b in ages:
    m=bands==b; y=gt[m]
    gap.append((miss[m][y==1].mean()-miss[m][y==0].mean())*100)
fig,ax1=plt.subplots(figsize=(7,4))
ax1.bar(x,f1_age,0.5,color='#2a78d6'); ax1.set_ylabel('F1',color='#2a78d6');ax1.set_ylim(0,0.6)
ax1.set_xticks(x);ax1.set_xticklabels(ages);ax1.set_xlabel('Age group')
ax2=ax1.twinx(); ax2.plot(x,gap,'o-',color='#eda100',lw=2)
ax2.set_ylabel('Patient - Control missing gap (%p)',color='#eda100'); ax2.axhline(0,color='#ccc',lw=0.8)
ax1.set_title('(2) Imbalance (P-C gap) vs F1  (no matching)'); ax1.grid(axis='y',alpha=0.3)
plt.tight_layout();plt.savefig(f'{args.out}/fig2_imbalance_vs_f1{t}.png',dpi=150);plt.close()

# ===== Fig3: 나이통제 결측 low/high =====
lo,hi=[],[]
for b in ages:
    m=bands==b; y,p,r=gt[m],pred[m],miss[m]; med=np.median(r)
    lo.append(f1s(y[r<=med],p[r<=med])); hi.append(f1s(y[r>med],p[r>med]))
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(x-w/2,lo,w,label='Low missing',color='#1baf7a'); ax.bar(x+w/2,hi,w,label='High missing',color='#e34948')
ax.set_xticks(x);ax.set_xticklabels(ages);ax.set_xlabel('Age group');ax.set_ylabel('F1');ax.set_ylim(0,0.75)
ax.set_title('(3) Missing effect (age-controlled): low<high, 60+ exception')
ax.legend();ax.grid(axis='y',alpha=0.3)
for i,val in enumerate(lo): ax.text(i-w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=8)
for i,val in enumerate(hi): ax.text(i+w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=8)
plt.tight_layout();plt.savefig(f'{args.out}/fig3_missing_effect{t}.png',dpi=150);plt.close()

# ===== Fig4: 블록별 결측 low/high F1 =====
bl_names=list(block_miss.keys())
bl_lo,bl_hi=[],[]
for k in bl_names:
    r=block_miss[k]; med=np.median(r)
    l,h=r<=med,r>med
    bl_lo.append(f1s(gt[l],pred[l])); bl_hi.append(f1s(gt[h],pred[h]))
xb=np.arange(len(bl_names))
fig,ax=plt.subplots(figsize=(9,4))
ax.bar(xb-w/2,bl_lo,w,label='Low missing (this block)',color='#1baf7a')
ax.bar(xb+w/2,bl_hi,w,label='High missing (this block)',color='#e34948')
ax.set_xticks(xb);ax.set_xticklabels(bl_names,rotation=20,ha='right');ax.set_ylabel('F1');ax.set_ylim(0,0.6)
ax.set_title('(4) Missing effect by device/block  (larger gap = more critical)')
ax.legend();ax.grid(axis='y',alpha=0.3)
for i,val in enumerate(bl_lo): ax.text(i-w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=7)
for i,val in enumerate(bl_hi): ax.text(i+w/2,val+0.01,f'{val:.2f}',ha='center',fontsize=7)
plt.tight_layout();plt.savefig(f'{args.out}/fig4_missing_by_block{t}.png',dpi=150);plt.close()

# ===== Fig5: 결측률 quartile별 F1 =====
q=np.quantile(miss,[0,0.25,0.5,0.75,1.0])
qlabel=['Q1\n(least missing)','Q2','Q3','Q4\n(most missing)']
qf1=[]
for i in range(4):
    lo_,hi_=q[i],q[i+1]
    m=(miss>=lo_)&(miss<=hi_) if i==3 else (miss>=lo_)&(miss<hi_)
    qf1.append(f1s(gt[m],pred[m]))
fig,ax=plt.subplots(figsize=(7,4))
ax.bar(np.arange(4),qf1,0.55,color=['#1baf7a','#9acd6e','#eda100','#e34948'])
ax.set_xticks(np.arange(4));ax.set_xticklabels(qlabel);ax.set_ylabel('F1');ax.set_ylim(0,0.6)
ax.set_title('(5) F1 by overall missing-rate quartile  (more missing -> lower F1)')
ax.grid(axis='y',alpha=0.3)
for i,val in enumerate(qf1): ax.text(i,val+0.01,f'{val:.2f}',ha='center',fontsize=9)
plt.tight_layout();plt.savefig(f'{args.out}/fig5_f1_by_quartile{t}.png',dpi=150);plt.close()

print("saved:")
for n in ['fig1_f1_by_age_sex','fig2_imbalance_vs_f1','fig3_missing_effect','fig4_missing_by_block','fig5_f1_by_quartile']:
    print(f"  {args.out}/{n}{t}.png")

# 콘솔에 fig4/fig5 수치도 출력
print("\n[Fig4] block   low_f1  high_f1   gap")
for k,l,h in zip(bl_names,bl_lo,bl_hi): print(f"  {k:<14}{l:>6.3f}{h:>8.3f}{l-h:>+8.3f}")
print("\n[Fig5] quartile F1:", [round(v,3) for v in qf1])
