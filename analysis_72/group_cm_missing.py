import json

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

D = '/home/hail/robot_ai2/data/numpy_all_chunk_72/'
P = np.load('analysis/test_predictions_ckpt_e4s22819.npz')
pred, gt = P['pred'], P['gt'].reshape(-1)

static = np.load(D+'static.npy')
time   = np.load(D+'time.npy', mmap_mode='r')   # (N,46,72), 관측=>=0
cols   = json.load(open(D+'feature_columns.json'))
split  = np.load(D+'split.npy', allow_pickle=True)
test_idx = split[2]

st = static[test_idx]; sex, age = st[:,0], st[:,1]
assert len(pred)==len(st), f"{len(pred)} vs {len(st)}"

# ---- 블록 정의 (수집주기 그룹) ----
def cidx(names): return [cols.index(n) for n in names if n in cols]
# feature명은 mean/max/min 접미사가 있으니 prefix로 매칭
def cidx_pref(prefixes):
    return [i for i,c in enumerate(cols) if any(c.startswith(p) for p in prefixes)]
blocks = {
  'A_hr_rr': cidx_pref(['hr_','rr_']),
  'A_spo2':  cidx_pref(['spo2']),
  'B_temp_bp_glu': cidx_pref(['core_temp','skin_temp','bp_','glucose']),
  'C_hrv':   cidx_pref(['hrv']),
  'D_event': cidx_pref(['light_sensor','proximity']),
  'E_daily': cidx_pref(['step','distance','screen_time','wake_time','sleep_time',
                        'deep_sleep','rem_sleep','light_sleep','total_sleep']),
  'F_ema':   cidx_pref(['EMA_']),
}

# ---- 샘플별 블록 결측률: NaN slot 비율 (관측=time>=0) ----
# time을 test 순서로 맞춤
time_test = time[test_idx]        # (N,46,72)
obs = (time_test >= 0)            # 관측 True
# 블록별 결측률(샘플별): 1 - (관측 slot / 전체 slot), 해당 블록 feature 평균
def block_missing_rate(block_cols):
    o = obs[:, block_cols, :]     # (N, k, 72)
    return 1.0 - o.mean(axis=(1,2))   # (N,)

miss = {b: block_missing_rate(ci) for b,ci in blocks.items() if len(ci)>0}

# ---- age 오염 제거 ----
valid = age >= 18
pred_v, gt_v, sex_v, age_v = pred[valid], gt[valid], sex[valid], age[valid]
miss_v = {b: m[valid] for b,m in miss.items()}

def band(a): return '<30' if a<30 else '30s' if a<40 else '40s' if a<50 else '50s' if a<60 else '60+'
sexname={0.0:'M',1.0:'F'}

# 어느 블록을 표에 보여줄지 (핵심: sleep=E, EMA=F, hr_rr=A)
show_blocks = ['E_daily','F_ema','A_hr_rr']
hdr = f"{'group':<9}{'N':>6}{'F1':>7}{'Recall':>8}"
for b in show_blocks: hdr += f"{'ΔP-C_'+b.split('_')[0]:>11}"
print(hdr)
print('-'*len(hdr))

for s in [1.0,0.0]:
    for b in ['<30','30s','40s','50s','60+']:
        m = (sex_v==s) & np.array([band(a)==b for a in age_v])
        if m.sum()==0: continue
        y,p = gt_v[m], pred_v[m]
        f1 = f1_score(y,p,zero_division=0)
        cm = confusion_matrix(y,p,labels=[0,1]); tn,fp,fn,tp=cm.ravel()
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        row = f"{sexname[s]+' '+b:<9}{m.sum():>6}{f1:>7.3f}{rec:>8.3f}"
        # 각 블록: 환자 결측률 - 정상 결측률 (%p)
        for blk in show_blocks:
            mr = miss_v[blk][m]
            pat = mr[y==1].mean() if (y==1).any() else np.nan
            ctl = mr[y==0].mean() if (y==0).any() else np.nan
            d = (pat-ctl)*100
            row += f"{d:>+11.1f}"
        print(row)

print("\n(ΔP-C = 환자 결측률 - 정상 결측률, %p. 양수=환자가 더 결측많음, 음수=환자가 덜 결측)")
print("E=수면/일단위, F=EMA설문, A=hr/rr")
