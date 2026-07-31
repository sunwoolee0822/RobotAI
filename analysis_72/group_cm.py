import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

# 예측 로드
P = np.load('analysis/test_predictions_ckpt_e4s22819.npz')
pred, prob, gt = P['pred'], P['prob'], P['gt'].reshape(-1)

# static (test 순서와 매칭: split[2] 순서 = shuffle=False라 동일)
D = '/home/hail/robot_ai2/data/numpy_all_chunk_72/'
static = np.load(D+'static.npy')
split = np.load(D+'split.npy', allow_pickle=True)
test_idx = split[2]
st = static[test_idx]                 # test 샘플의 static
sex, age = st[:,0], st[:,1]

assert len(pred)==len(st), f"길이불일치 {len(pred)} vs {len(st)}"

# age 오염(0살) 제거
valid = age >= 18
pred, gt, sex, age = pred[valid], gt[valid], sex[valid], age[valid]

def band(a):
    return '<30' if a<30 else '30s' if a<40 else '40s' if a<50 else '50s' if a<60 else '60+'
sexname = {0.0:'M', 1.0:'F'}

print(f"{'group':<12}{'N':>7}{'환자':>6}{'정상':>6}  {'TN FP FN TP':>16}  {'F1':>6}{'Recall':>8}")
for s in [1.0, 0.0]:          # F 먼저(언밸런스 그룹), M
    for b in ['<30','30s','40s','50s','60+']:
        m = (sex==s) & np.array([band(a)==b for a in age])
        if m.sum()==0: continue
        y, p = gt[m], pred[m]
        n_pat = int((y==1).sum()); n_ctrl = int((y==0).sum())
        cm = confusion_matrix(y, p, labels=[0,1])
        tn,fp,fn,tp = cm.ravel()
        f1 = f1_score(y, p, zero_division=0)
        rec = tp/(tp+fn) if (tp+fn)>0 else 0
        print(f"{sexname[s]+' '+b:<12}{m.sum():>7}{n_pat:>6}{n_ctrl:>6}  "
              f"{tn:>3} {fp:>3} {fn:>3} {tp:>3}  {f1:>6.3f}{rec:>8.3f}")
