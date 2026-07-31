"""
analysis0_demo.py — 기초 인구통계 (문서 맨 위 표용)
labeled_all_chunk_336 metadata에서 subject 단위 성별/나이/PHQ9 집계.
사용법: python analysis0_demo.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_DIR = Path("/home/hail/robot_ai2/data/labeled_all_chunk_336")
OUT = Path("/home/hail/robot_ai2/analysis_336")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for sd in sorted(p for p in SAMPLE_DIR.iterdir() if p.is_dir()):
    mp = sd / "metadata.json"
    if not mp.exists():
        continue
    m = json.loads(mp.read_text(encoding="utf-8"))
    rows.append({
        "subject_id": m.get("subject_id"),
        "sample_id": m.get("sample_id"),
        "phq9_label": m.get("phq9_label"),
        "phq9_score": m.get("phq9_score"),
        "gad7_label": m.get("gad7_label"),
        "gad7_score": m.get("gad7_score"),
    })
df = pd.DataFrame(rows)
print(f"windows(samples): {len(df)}, subjects: {df['subject_id'].nunique()}")

# static.npy에서 subject별 성별/나이 (numpy 데이터와 subject_ids 매핑)
root = Path("/home/hail/robot_ai2/data/numpy_all_chunk_72_24feat")
static = np.load(root / "static.npy")           # (N,4): sex,age,height,weight
sids = [str(x) for x in json.load(open(root / "subject_ids.json"))]
sdf = pd.DataFrame({"subject_id": sids, "sex": static[:,0], "age": static[:,1]})
sdf = sdf.drop_duplicates("subject_id")         # subject 단위

print("\n=== 성별 (subject 단위) ===")
print(sdf["sex"].value_counts().to_string())
print(f"  total subjects with static: {len(sdf)}")

print("\n=== 나이 (subject 단위) ===")
print(f"  mean={sdf['age'].mean():.1f}, median={sdf['age'].median():.1f}, "
      f"min={sdf['age'].min():.0f}, max={sdf['age'].max():.0f}, sd={sdf['age'].std():.1f}")

print("\n=== 나이대 구간 (subject 단위) ===")
bins=[0,30,45,60,200]; labs=["<30","30-45","45-60","60+"]
print(pd.cut(sdf["age"], bins=bins, labels=labs, right=False).value_counts().sort_index().to_string())

print("\n=== PHQ9 (survey/window 단위) ===")
n_survey = len(df)
n_dep = (df["phq9_label"]==1).sum()
print(f"  surveys={n_survey}, depressed(label=1)={n_dep} ({100*n_dep/n_survey:.1f}%)")
print(f"  phq9_score mean={df['phq9_score'].mean():.1f}, median={df['phq9_score'].median():.0f}")

print("\n=== GAD7 (참고, 나중 분석용) ===")
n_anx = (df["gad7_label"]==1).sum()
print(f"  anxious(label=1)={n_anx} ({100*n_anx/n_survey:.1f}%), "
      f"gad7_score mean={df['gad7_score'].mean():.1f}")

# 저장
summary = {
    "n_windows": len(df),
    "n_subjects": df["subject_id"].nunique(),
    "n_subjects_static": len(sdf),
    "sex_counts": sdf["sex"].value_counts().to_dict(),
    "age_mean": round(float(sdf["age"].mean()),1),
    "age_median": round(float(sdf["age"].median()),1),
    "age_sd": round(float(sdf["age"].std()),1),
    "phq9_depressed": int(n_dep), "phq9_pct": round(100*n_dep/n_survey,1),
    "gad7_anxious": int(n_anx), "gad7_pct": round(100*n_anx/n_survey,1),
}
json.dump(summary, open(OUT/"demographics_summary.json","w"), ensure_ascii=False, indent=2)
print(f"\n저장: {OUT}/demographics_summary.json")