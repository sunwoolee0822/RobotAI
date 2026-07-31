"""
analysis3b_age.py
=================
LMM에서 age_z coef가 양수(+0.103)로 나온 게 형 예전 발견("60+ F1 급락")과
반대라, 나이를 구간으로 쪼개 비선형/지표별로 확인한다.

입력 : analysis_336/lmm_table.csv  (analysis3_336.py 산출: subject/correct/missing_rate/age/sex/phq9_label)
       + test_predictions_baseline.npz (F1 계산용 pred/gt/prob)

(a) 나이대 구간별 정답률(correct 평균)
(b) 나이대 구간별 accuracy vs F1  → "acc는 높은데 F1은 낮은" 착시 확인
(c) LMM에 age를 구간 범주로 넣어 어느 구간이 음수인지

사용법:
  python analysis3b_age.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="analysis_336/lmm_table.csv")
    ap.add_argument("--pred", default="test_predictions_baseline.npz")
    ap.add_argument("--out_dir", default="analysis_336")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.table)

    # pred/gt를 표에 붙임 (lmm_table 행 순서 = test 순서 = pred 순서)
    d = np.load(args.pred)
    assert len(d["pred"]) == len(df), f"pred({len(d['pred'])}) != table({len(df)})"
    df["pred"] = d["pred"]; df["gt"] = d["gt"]
    if "prob" in d: df["prob"] = d["prob"]

    # 나이대 구간
    bins = [0, 30, 45, 60, 200]
    labels = ["<30", "30-45", "45-60", "60+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)

    # ---- (a)+(b) 구간별 accuracy vs F1 ----
    print("=" * 70)
    print("  (a)(b) 나이대 구간별 accuracy vs F1")
    print("=" * 70)
    rows = []
    for band in labels:
        sub = df[df["age_band"] == band]
        if len(sub) == 0:
            continue
        acc = accuracy_score(sub["gt"], sub["pred"])
        # F1: 양성(우울=1) 기준. 한 클래스만 있으면 0 처리
        try:
            f1 = f1_score(sub["gt"], sub["pred"], zero_division=0)
        except Exception:
            f1 = np.nan
        pos_rate_true = (sub["gt"] == 1).mean()       # 실제 양성 비율
        pos_rate_pred = (sub["pred"] == 1).mean()     # 모델이 양성 예측한 비율
        rows.append({"age_band": band, "n": len(sub),
                     "accuracy": acc, "f1": f1,
                     "true_pos_rate": pos_rate_true,
                     "pred_pos_rate": pos_rate_pred})
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tab.to_csv(out_dir / "age_band_acc_f1.csv", index=False, encoding="utf-8-sig")
    print(f"\n  → acc는 높은데 f1은 낮은 구간이 있으면 = 다수클래스 쏠림(착시).")
    print(f"  → pred_pos_rate가 true_pos_rate보다 크게 낮으면 = 양성 예측 회피.")

    # ---- (c) LMM: age를 구간 범주로 ----
    print("\n" + "=" * 70)
    print("  (c) 로지스틱: age_band(범주) + missing + sex + phq9")
    print("=" * 70)
    try:
        import statsmodels.formula.api as smf
        df["missing_rate_z"] = (df["missing_rate"] - df["missing_rate"].mean()) / df["missing_rate"].std()
        # 기준(reference) = "<30"
        df["age_band"] = pd.Categorical(df["age_band"], categories=labels, ordered=False)
        formula = "correct ~ missing_rate_z + C(age_band) + C(sex)"
        if df["phq9_label"].notna().all() and df["phq9_label"].nunique() > 1:
            formula += " + C(phq9_label)"
        md = smf.logit(formula, data=df)
        mdf = md.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["subject"]})
        print(mdf.summary())
        print("\n[해석] C(age_band)[T.60+] 의 coef 부호가 핵심.")
        print("  음수면 = <30 대비 60+에서 정답 오즈↓ (형 발견 '60+ 급락' 지지).")
        print("  근데 (a)(b)에서 f1으로 봤을 때랑 acc로 봤을 때가 다르면,")
        print("  '60+는 acc 높지만 f1 낮음(다수클래스 쏠림)' 으로 정리.")
    except Exception as e:
        print(f"[logit 실패] {e}")


if __name__ == "__main__":
    main()