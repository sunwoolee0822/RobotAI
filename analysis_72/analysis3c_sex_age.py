"""
analysis3c_sexage.py
====================
성별 × 나이대 교차 subgroup 분석 (형 예전 sex×age 셀 분석의 통계 버전).
LMM에서 sex는 전체적으로 ns였지만, 성별 효과가 나이대에 따라 다른지(상호작용) 확인.

입력 : analysis_336/lmm_table.csv + test_predictions_baseline.npz

(a) 8셀(2 sex × 4 age_band)별 n / accuracy / F1 / missing_rate / pred_pos_rate
(b) 상호작용 로지스틱: correct ~ missing_z + C(sex)*C(age_band)
(c) 셀별 결측→성능 방향 (참고)

사용법: python analysis3c_sexage.py
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
    d = np.load(args.pred)
    assert len(d["pred"]) == len(df)
    df["pred"] = d["pred"]; df["gt"] = d["gt"]

    bins = [0, 30, 45, 60, 200]; labels = ["<30", "30-45", "45-60", "60+"]
    df["age_band"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    sex_name = {0: "M", 1: "F"}   # static sex 코딩: 확인필요(0/1) — 실제 매핑은 데이터 기준
    df["sex_name"] = df["sex"].map(sex_name).fillna(df["sex"].astype(str))

    # ---- (a) 8셀 표 ----
    print("=" * 78)
    print("  (a) 성별 × 나이대 셀별 성능/결측")
    print("=" * 78)
    rows = []
    for s in sorted(df["sex"].unique()):
        for band in labels:
            sub = df[(df["sex"] == s) & (df["age_band"] == band)]
            if len(sub) == 0:
                continue
            try:
                f1 = f1_score(sub["gt"], sub["pred"], zero_division=0)
            except Exception:
                f1 = np.nan
            rows.append({
                "sex": sex_name.get(s, s), "age_band": band, "n": len(sub),
                "accuracy": accuracy_score(sub["gt"], sub["pred"]),
                "f1": f1,
                "missing_rate": sub["missing_rate"].mean(),
                "true_pos": (sub["gt"] == 1).mean(),
                "pred_pos": (sub["pred"] == 1).mean(),
            })
    tab = pd.DataFrame(rows)
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    tab.to_csv(out_dir / "sexage_cells.csv", index=False, encoding="utf-8-sig")

    # ---- (b) 상호작용 로지스틱 ----
    print("\n" + "=" * 78)
    print("  (b) 상호작용: correct ~ missing_z + C(sex)*C(age_band)")
    print("=" * 78)
    try:
        import statsmodels.formula.api as smf
        df["missing_rate_z"] = (df["missing_rate"] - df["missing_rate"].mean()) / df["missing_rate"].std()
        df["age_band"] = pd.Categorical(df["age_band"], categories=labels)
        md = smf.logit("correct ~ missing_rate_z + C(sex)*C(age_band)", data=df)
        mdf = md.fit(disp=False, cov_type="cluster", cov_kwds={"groups": df["subject"]})
        print(mdf.summary())

        # 상호작용항만 뽑아서 유의성 확인
        inter = [n for n in mdf.params.index if ":" in n]
        print("\n[상호작용항 (성별×나이대) 유의성]")
        any_sig = False
        for n in inter:
            p = mdf.pvalues[n]
            sig = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
            if sig != "ns":
                any_sig = True
            print(f"  {n}: coef={mdf.params[n]:+.4f}, p={p:.4f} {sig}")
        print("\n[해석] 상호작용항이 모두 ns면 → 성별 효과가 나이대에 따라 다르지 않음")
        print("  (= 성별은 나이대 무관하게 성능에 영향 없음, LMM 결과와 일관).")
        print("  하나라도 유의하면 → 특정 성별×나이대 조합에서만 차이 존재.")
        if not any_sig:
            print("\n  결론: 상호작용 없음. 성별은 subgroup 관계없이 non-factor.")
    except Exception as e:
        print(f"[logit 실패] {e}")

    print("\n" + "=" * 78)
    print(f"  저장: {out_dir}/sexage_cells.csv")
    print("=" * 78)
    print("\n⚠ sex 코딩(0=M/1=F) 가정임. static.npy sex 실제 매핑 확인 필요.")


if __name__ == "__main__":
    main()