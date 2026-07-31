"""
make_group_splits.py
====================
subject 단위 missing high/low 그룹으로 train/val/test 를 각각 필터해서
그룹별 학습용 split 을 만든다.

- split.npy: [train_idx, val_idx, test_idx] (샘플 인덱스)
- subject_ids.json: 샘플별 subject_id (길이 = 전체 샘플 수)
- missing_csv: subject_id, group(high/low) 컬럼

출력 (out_dir, 기본 data_root):
  split_group_high.npy  -> high-missing subject 만으로 train/val/test
  split_group_low.npy   -> low-missing  subject 만으로 train/val/test

사용법:
  python make_group_splits.py \
    --data_root data/numpy_all_chunk_72_24feat \
    --missing_csv analysis_new/subject_missing_rate_new.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_groups(missing_csv):
    grp = {}
    with open(missing_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            grp[str(r["subject_id"])] = str(r["group"]).lower()
    return grp


def filter_idx(idx, subject_ids, keep_sids):
    return np.array([i for i in idx if subject_ids[i] in keep_sids], dtype=int)


def summarize(name, tr, va, te, subject_ids, y):
    def stat(idx):
        yy = y[idx].astype(int)
        return len(idx), len(set(subject_ids[i] for i in idx)), int((yy == 0).sum()), int((yy == 1).sum())
    print(f"\n[{name}]")
    for split_name, idx in [("train", tr), ("val", va), ("test", te)]:
        n, ns, n0, n1 = stat(idx)
        warn = "  <-- one-class!" if (n0 == 0 or n1 == 0) else ""
        print(f"  {split_name:5s}: {n:6d} samples, {ns:4d} subjects, PHQ9 0={n0}, 1={n1}{warn}")


def save_split(path, tr, va, te):
    np.save(path, np.array([np.asarray(tr, int), np.asarray(va, int),
                            np.asarray(te, int)], dtype=object), allow_pickle=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--missing_csv", required=True)
    ap.add_argument("--split_name", default="split.npy")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir) if args.out_dir else data_root

    split = np.load(data_root / args.split_name, allow_pickle=True)
    train_idx, val_idx, test_idx = [np.asarray(x, int) for x in split]
    subject_ids = [str(x) for x in json.load(open(data_root / "subject_ids.json"))]
    y = np.load(data_root / "gt.npy").squeeze()

    grp = load_groups(args.missing_csv)
    high_sids = {s for s, g in grp.items() if "high" in g}
    low_sids = {s for s, g in grp.items() if "low" in g}
    print(f"groups: high={len(high_sids)} subjects, low={len(low_sids)} subjects")

    for gname, sids in [("high", high_sids), ("low", low_sids)]:
        tr = filter_idx(train_idx, subject_ids, sids)
        va = filter_idx(val_idx, subject_ids, sids)
        te = filter_idx(test_idx, subject_ids, sids)
        out = out_dir / f"split_group_{gname}.npy"
        save_split(out, tr, va, te)
        summarize(f"group_{gname}", tr, va, te, subject_ids, y)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    main()
