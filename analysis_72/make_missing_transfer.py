"""
make_missing_transfer.py
========================
rule-based missing 주입:
  high-missing 그룹의 (feature, slot)별 결측 패턴을 low-missing 그룹 샘플에 입혀
  high 원본 + low(degraded) 를 합쳐 새 data_root 로 저장한다.

규칙:
  p_miss[f,t] = high 그룹 샘플에서 (time<0) 인 비율   # (F, 72)
  low 샘플의 관측 slot(time>=0) 을 확률적으로 드롭 -> array=0, time=-1, mask 갱신
  high 샘플은 그대로 둔다 -> "합쳐서" 하나의 data_root

  --mode over  : 관측 slot 을 Bernoulli(p_miss) 로 드롭 (low 결측이 high 위에 얹혀 overshoot)
  --mode match : 최종 low 결측률이 high 와 같아지게 조정
                 p_drop = clip((p_miss - m_low)/(1 - m_low), 0, 1)
                 (m_low = low 그룹의 (f,t) 결측 확률; 이미 high 보다 결측 많으면 0)

관측 규약: time>=0 관측 / time<0(=-1) 결측, 결측 slot 의 array=0

사용법:
  python make_missing_transfer.py \
    --src data/numpy_all_chunk_72_24feat \
    --dst data/numpy_all_chunk_72_24feat_transfer \
    --missing_csv analysis_new/subject_missing_rate_new.csv
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np


def load_groups(missing_csv):
    grp = {}
    with open(missing_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            grp[str(r["subject_id"])] = str(r["group"]).lower()
    return grp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="baseline data_root")
    ap.add_argument("--dst", required=True, help="output data_root")
    ap.add_argument("--missing_csv", required=True)
    ap.add_argument("--match", action="store_true",
                    help="설정 시 최종 low 결측률을 high 와 일치(match). 기본은 overshoot")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    array = np.load(src / "array.npy")            # (N, F, 72)
    time = np.load(src / "time.npy")              # (N, F, 72)
    mask = np.load(src / "mask.npy")              # (N, F)
    subject_ids = [str(x) for x in json.load(open(src / "subject_ids.json"))]
    N, F, T = array.shape
    print(f"src: array={array.shape}")

    grp = load_groups(args.missing_csv)
    is_high = np.array([grp.get(s, "") == "high" for s in subject_ids])
    is_low = np.array([grp.get(s, "") == "low" for s in subject_ids])
    print(f"samples: high={is_high.sum()}, low={is_low.sum()}, "
          f"ungrouped={N - is_high.sum() - is_low.sum()}")

    # high 그룹의 (feature, slot)별 결측 확률
    obs = time >= 0                                # True=관측
    p_miss = 1.0 - obs[is_high].mean(axis=0)       # (F, T)
    print(f"high p_miss: mean={p_miss.mean():.3f}, "
          f"min={p_miss.min():.3f}, max={p_miss.max():.3f}")

    # 관측 slot 을 드롭할 확률 p_drop (F, T)
    if args.match:
        # 최종 low 결측률이 high 와 같아지게: 이미 빠진 만큼 감안
        m_low = 1.0 - obs[is_low].mean(axis=0)     # (F, T) low 현재 결측
        with np.errstate(divide="ignore", invalid="ignore"):
            p_drop = (p_miss - m_low) / (1.0 - m_low)
        p_drop = np.clip(np.nan_to_num(p_drop, nan=0.0), 0.0, 1.0)
        print(f"[match] m_low mean={m_low.mean():.3f}, p_drop mean={p_drop.mean():.3f}")
    else:
        # high 패턴을 그대로 얹음 -> low 결측이 high 위로 overshoot
        p_drop = p_miss

    # low 샘플의 관측 slot 을 Bernoulli(p_drop) 로 드롭
    low_idx = np.where(is_low)[0]
    dropped_total = 0
    obs_total = 0
    for i in low_idx:
        obs_i = obs[i]                             # (F, T)
        obs_total += int(obs_i.sum())
        # 관측 slot 마다 p_drop 확률로 드롭
        draw = rng.random((F, T))
        drop = obs_i & (draw < p_drop)             # 관측 & 드롭당첨
        if drop.any():
            array[i][drop] = 0.0
            time[i][drop] = -1.0
            dropped_total += int(drop.sum())
        # mask(feature별 관측수) 재계산
        mask[i] = (time[i] >= 0).sum(axis=1)

    print(f"low observed slots: {obs_total}, dropped: {dropped_total} "
          f"({100 * dropped_total / max(obs_total, 1):.1f}%)")

    # 결측률 확인
    def missing_rate(sel):
        return 1.0 - (time[sel] >= 0).mean()
    print(f"after: high miss={missing_rate(is_high):.3f}, "
          f"low miss={missing_rate(is_low):.3f}")

    np.save(dst / "array.npy", array)
    np.save(dst / "time.npy", time)
    np.save(dst / "mask.npy", mask)
    for name in ["static.npy", "gt.npy", "split.npy",
                 "subject_ids.json", "sample_ids.json",
                 "feature_columns.json", "meta.json"]:
        if (src / name).exists():
            shutil.copy(src / name, dst / name)
    print(f"\ndone -> {dst}")


if __name__ == "__main__":
    main()
