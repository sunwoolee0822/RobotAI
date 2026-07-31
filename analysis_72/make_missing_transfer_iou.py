"""
make_missing_transfer_iou.py
============================
rule-based missing 주입 (IOU 포함관계 이식, first-union 버전).

  ① feature별 IOU 포함관계 선택: coverage=|L∩U|/|L|=1 완전포함 U 우선, high 빈도 가중.
     결과 결측 = L ∪ U (union). 완전포함 없으면 best_cov fallback.
  ② 블록 단위 룰 제거: broadcast feature 의 부분 결측 블록이 BLOCK_THRESH 이상이면 통째 결측.
  ③ 시간대 고려: TIME_MATCH_SENSORS 는 동점에서 hour-of-day 유사도 반영.

관측 규약: time>=0 관측 / time<0(-1) 결측, 결측 slot 의 array=0.

사용법:
  python make_missing_transfer_iou.py \
    --src data/numpy_all_chunk_72_24feat \
    --dst data/numpy_all_chunk_72_24feat_transfer_iou \
    --missing_csv analysis_new/subject_missing_rate_new.csv
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

TIME_MATCH_SENSORS = ["bp_dia_mean", "bp_sys_mean", "core_temp_mean"]

COV_EPS = 1e-6
TOD_W = 4.0
BLOCK_THRESH = 0.5


def load_groups(missing_csv):
    grp = {}
    with open(missing_csv, encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            grp[str(r["subject_id"])] = str(r["group"]).lower()
    return grp


def get_blocks(fill_slots, feat, T):
    v = fill_slots.get(feat)
    if v is None:
        return None
    v = min(int(v), T)
    if v >= T:
        return [(0, T)]
    return [(s, min(s + v, T)) for s in range(0, T, v)]


def hour_profile(mask_2d, hod):
    return np.stack([mask_2d[:, hod == h].mean(axis=1) for h in range(24)], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--missing_csv", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    array = np.load(src / "array.npy")
    time = np.load(src / "time.npy")
    mask = np.load(src / "mask.npy")
    cols = json.load(open(src / "feature_columns.json"))
    subject_ids = [str(x) for x in json.load(open(src / "subject_ids.json"))]
    meta = json.load(open(src / "meta.json")) if (src / "meta.json").exists() else {}
    fill_slots = meta.get("fill_slots", {})
    N, F, T = array.shape
    hod = np.arange(T) % 24
    print(f"src: array={array.shape}, TIME_MATCH_SENSORS={TIME_MATCH_SENSORS}")

    grp = load_groups(args.missing_csv)
    is_high = np.array([grp.get(s, "") == "high" for s in subject_ids])
    is_low = np.array([grp.get(s, "") == "low" for s in subject_ids])
    low_idx = np.where(is_low)[0]
    print(f"samples: high={is_high.sum()}, low={is_low.sum()}, "
          f"ungrouped={N - is_high.sum() - is_low.sum()}")

    miss = time < 0

    dropped_total = 0
    obs_total = 0
    block_cleaned = 0

    for fi, feat in enumerate(cols):
        is_tod = feat in TIME_MATCH_SENSORS

        Hf = miss[is_high, fi, :]

        Lf = miss[low_idx, fi, :]
        nL = Lf.sum(axis=1)
        n_low = len(nL)

        # --- 목표 결측량 매칭 ---
        # U 를 통째로 씌우면 |L∪U| >= |U| 라 목표를 구조적으로 초과한다(관측을 되살릴 수 없으므로).
        # 그래서 U 는 "어느 슬롯이 잘 빠지는지"의 패턴 가이드로만 쓰고,
        # 추가할 슬롯 수를 (목표 t - 현재 |L|) 로 정확히 통제한다 -> 결측률이 high 와 일치.
        t = rng.choice(Hf.sum(axis=1), size=n_low)      # high 결측량 분포에서 목표치

        # 보정: 최종은 max(|L|, t) 라 항상 목표 이상으로 뜬다(관측 복원 불가).
        # 그래서 목표를 δ 만큼 낮춰 최종 평균이 high 평균과 일치하도록 이분탐색.
        tgt_mean = Hf.sum(axis=1).mean()
        lo_d, hi_d = -float(T), float(T)
        for _ in range(40):
            mid = (lo_d + hi_d) / 2
            if np.maximum(nL, t - mid).mean() > tgt_mean:
                lo_d = mid                              # 더 많이 낮춰야 함
            else:
                hi_d = mid
        t = t - (lo_d + hi_d) / 2
        need = np.maximum(0, np.rint(t).astype(int) - nL)   # 더 빼야 할 슬롯 수

        # 슬롯 선호도 = high 의 슬롯별 결측 성향(시간대 구조 내포) + Gumbel 노이즈
        p_slot = Hf.mean(axis=0)                        # (T,) high 결측 성향
        pref = np.log(p_slot + 1e-6)[None, :] + rng.gumbel(size=(n_low, T))

        blocks = get_blocks(fill_slots, feat, T)
        if blocks is None:
            # 분단위 센서: 슬롯 단위로 정확히 need 개 추가
            pref = np.where(Lf, -np.inf, pref)          # 이미 결측인 곳 제외
            order = np.argsort(-pref, axis=1)
            rank = np.empty_like(order)
            np.put_along_axis(rank, order, np.broadcast_to(np.arange(T), (n_low, T)), axis=1)
            new_miss = Lf | (rank < need[:, None])
        else:
            # broadcast feature(daily/EMA): 블록 통째로만 빼서 경계 잔존값이 안 생기게(②)
            blen = blocks[0][1] - blocks[0][0]
            n_blk = len(blocks)
            blk_pref = np.stack([pref[:, a:b].mean(axis=1) for (a, b) in blocks], axis=1)
            blk_full = np.stack([Lf[:, a:b].all(axis=1) for (a, b) in blocks], axis=1)
            blk_pref = np.where(blk_full, np.inf, blk_pref)   # 이미 전부 결측인 블록은 유지
            # 블록은 blen 단위라 양자화가 큼(EMA는 72슬롯 1블록).
            # 확률적 반올림으로 기대값을 보존하고, 스케일 s 를 이분탐색해 평균을 맞춘다.
            base_blk = t / blen
            lo_s, hi_s = 0.0, float(n_blk) + 1.0
            u_rnd = rng.random(n_low)
            for _ in range(40):
                mid_s = (lo_s + hi_s) / 2
                b = np.clip(base_blk * mid_s, 0, n_blk)
                nb = np.floor(b).astype(int) + (u_rnd < (b - np.floor(b)))
                nb = np.clip(nb, 0, n_blk)
                est = np.maximum(nL, nb * blen).mean()
                if est > tgt_mean:
                    hi_s = mid_s
                else:
                    lo_s = mid_s
            b = np.clip(base_blk * (lo_s + hi_s) / 2, 0, n_blk)
            need_blk = np.clip(np.floor(b).astype(int) + (u_rnd < (b - np.floor(b))), 0, n_blk)
            order = np.argsort(-blk_pref, axis=1)
            rank = np.empty_like(order)
            np.put_along_axis(rank, order, np.broadcast_to(np.arange(n_blk), (n_low, n_blk)), axis=1)
            take = rank < need_blk[:, None]                   # 결측 처리할 블록
            new_miss = Lf.copy()
            for bi, (a, b) in enumerate(blocks):
                sel = take[:, bi]
                if sel.any():
                    before = int(new_miss[sel, a:b].sum())
                    new_miss[np.where(sel)[0], a:b] = True
                    block_cleaned += int(new_miss[sel, a:b].sum()) - before

        print(f"  [{feat:18s}] 목표 {t.mean():5.1f} -> 실제 {new_miss.sum(axis=1).mean():5.1f}")

        newly = new_miss & ~Lf
        obs_total += int((~Lf).sum())
        dropped_total += int(newly.sum())
        for k, i in enumerate(low_idx):
            drp = newly[k]
            if drp.any():
                array[i, fi, drp] = 0.0
                time[i, fi, drp] = -1.0

        print(f"  [{fi:2d}] {feat:18s} "
              f"{'[TOD]' if is_tod else '     '} added={int(newly.sum())}")

    mask = (time >= 0).sum(axis=2).astype(mask.dtype)

    print(f"\nlow observed slots: {obs_total}, newly dropped: {dropped_total} "
          f"({100 * dropped_total / max(obs_total, 1):.1f}%)  "
          f"block_cleaned+={block_cleaned}")

    def missing_rate(sel):
        return 1.0 - (time[sel] >= 0).mean()
    hr_after = missing_rate(is_high)
    lr_after = missing_rate(is_low)
    print(f"after: high miss={hr_after:.3f}, low miss={lr_after:.3f} "
          f"(overshoot={max(0.0, lr_after - hr_after):.3f})")

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