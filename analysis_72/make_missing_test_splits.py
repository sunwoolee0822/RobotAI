import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def find_col(df, candidates, contains=None):
    cols = list(df.columns)

    for c in candidates:
        if c in cols:
            return c

    if contains:
        for c in cols:
            lc = c.lower()
            if all(x.lower() in lc for x in contains):
                return c

    raise ValueError(f"Column not found. candidates={candidates}, columns={cols}")


def summarize(name, idx, subject_ids, y):
    idx = np.asarray(idx, dtype=int)
    sids = [subject_ids[i] for i in idx]
    yy = y[idx].astype(int)

    n0 = int((yy == 0).sum())
    n1 = int((yy == 1).sum())

    print(f"\n[{name}]")
    print(f"  samples : {len(idx)}")
    print(f"  subjects: {len(set(sids))}")
    print(f"  PHQ9 0  : {n0}")
    print(f"  PHQ9 1  : {n1}")

    if n0 == 0 or n1 == 0:
        print("  WARNING: one-class test set. AUROC may be invalid.")


def save_split(path, train_idx, val_idx, test_idx):
    np.save(
        path,
        np.array(
            [
                np.asarray(train_idx, dtype=int),
                np.asarray(val_idx, dtype=int),
                np.asarray(test_idx, dtype=int),
            ],
            dtype=object,
        ),
        allow_pickle=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--missing_csv", required=True)
    ap.add_argument("--split_name", default="split.npy")
    ap.add_argument("--out_dir", default=None)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    missing_csv = Path(args.missing_csv)
    out_dir = Path(args.out_dir) if args.out_dir else data_root
    out_dir.mkdir(parents=True, exist_ok=True)

    split_path = data_root / args.split_name
    subject_path = data_root / "subject_ids.json"
    gt_path = data_root / "gt.npy"

    split = np.load(split_path, allow_pickle=True)
    train_idx, val_idx, test_idx = [np.asarray(x, dtype=int) for x in split]

    with open(subject_path, "r", encoding="utf-8") as f:
        subject_ids = [str(x) for x in json.load(f)]

    y = np.load(gt_path).squeeze()

    miss = pd.read_csv(missing_csv)

    sid_col = find_col(
        miss,
        candidates=["subject_id", "sid", "ID", "id", "patient_id"],
    )

    group_col = None
    for c in miss.columns:
        if c.lower() in ["group", "missing_group", "missing_level", "missing_bin"]:
            group_col = c
            break

    miss[sid_col] = miss[sid_col].astype(str)

    if group_col is not None:
        g = miss[group_col].astype(str).str.lower()
        low_sids = set(miss.loc[g.str.contains("low"), sid_col])
        high_sids = set(miss.loc[g.str.contains("high"), sid_col])
        print(f"Using group column: {group_col}")
    else:
        rate_col = find_col(
            miss,
            candidates=["missing_rate", "subject_missing_rate", "mean_missing_rate"],
            contains=["missing", "rate"],
        )
        med = miss[rate_col].median()
        low_sids = set(miss.loc[miss[rate_col] <= med, sid_col])
        high_sids = set(miss.loc[miss[rate_col] > med, sid_col])
        print(f"Using median split from column: {rate_col}")
        print(f"Median missing rate: {med:.6f}")

    low_test_idx = np.array(
        [i for i in test_idx if subject_ids[i] in low_sids],
        dtype=int,
    )
    high_test_idx = np.array(
        [i for i in test_idx if subject_ids[i] in high_sids],
        dtype=int,
    )

    covered = set(low_test_idx.tolist()) | set(high_test_idx.tolist())
    missing_from_groups = set(test_idx.tolist()) - covered

    low_path = out_dir / "split_test_low.npy"
    high_path = out_dir / "split_test_high.npy"

    save_split(low_path, train_idx, val_idx, low_test_idx)
    save_split(high_path, train_idx, val_idx, high_test_idx)

    print("\nSaved:")
    print(f"  {low_path}")
    print(f"  {high_path}")

    summarize("original_test", test_idx, subject_ids, y)
    summarize("low_test", low_test_idx, subject_ids, y)
    summarize("high_test", high_test_idx, subject_ids, y)

    print("\n[coverage]")
    print(f"  original test samples: {len(test_idx)}")
    print(f"  low + high samples   : {len(low_test_idx) + len(high_test_idx)}")
    print(f"  not assigned samples : {len(missing_from_groups)}")

    train_sids = set(subject_ids[i] for i in train_idx)
    val_sids = set(subject_ids[i] for i in val_idx)
    low_test_sids = set(subject_ids[i] for i in low_test_idx)
    high_test_sids = set(subject_ids[i] for i in high_test_idx)

    print("\n[leakage check]")
    print(f"  train ∩ low_test subjects : {len(train_sids & low_test_sids)}")
    print(f"  train ∩ high_test subjects: {len(train_sids & high_test_sids)}")
    print(f"  val ∩ low_test subjects   : {len(val_sids & low_test_sids)}")
    print(f"  val ∩ high_test subjects  : {len(val_sids & high_test_sids)}")


if __name__ == "__main__":
    main()
