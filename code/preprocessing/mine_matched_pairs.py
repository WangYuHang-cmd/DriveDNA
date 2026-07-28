#!/usr/bin/env python3
"""
T4 — Matched-context pair mining (the naturalistic answer to PDB's controlled design).

Build balanced same-driver vs different-driver window pairs that are MATCHED on:
  scenario bucket × speed regime × THW regime (car-following) × consolidated vehicle model.
Constraints: positives are CROSS-ROUTE (same driver, different route — no route leakage);
negatives are different drivers within the same match key (vehicle-controlled comparison).
Seed-locked; released as a split manifest.

    ../openpilot/.venv/bin/python mine_matched_pairs.py
Output: DriveDNA/data/splits/matched_context_pairs.parquet
"""
import os
import numpy as np
import pandas as pd

WQ = "./data/segments/windows.parquet"
OUT = "./data/splits/matched_context_pairs.parquet"
SEED = 20260709
MAX_POS_PER_KEY = 30
MAX_NEG_PER_KEY = 30
TARGET = 25000  # per class


def main():
    W = pd.read_parquet(WQ).reset_index(drop=True)
    W["win_id"] = np.arange(len(W))
    # match key: scenario × speed bin × THW bin (car-following only) × consolidated model
    W["v_bin"] = pd.cut(W.v_mean, [0, 10, 20, 30, np.inf], labels=["v0", "v1", "v2", "v3"])
    W["thw_bin"] = "na"
    cf = W.scenario == "car_following"
    W.loc[cf, "thw_bin"] = pd.cut(W.loc[cf, "thw_median"], [0, 1.2, 2.0, np.inf],
                                  labels=["t0", "t1", "t2"]).astype(str)
    W["key"] = (W.scenario.astype(str) + "|" + W.v_bin.astype(str) + "|" +
                W.thw_bin.astype(str) + "|" + W.model_canon.astype(str))

    rng = np.random.default_rng(SEED)
    pos_rows, neg_rows = [], []
    for key, g in W.groupby("key"):
        if len(g) < 4 or g.driver.nunique() < 2:
            continue
        ids = g.win_id.to_numpy(); drv = g.driver.to_numpy(); rts = g.route.to_numpy()
        # positives: same driver, different route
        by_drv = {}
        for i in range(len(g)):
            by_drv.setdefault(drv[i], []).append(i)
        cand_pos = []
        for d, idxs in by_drv.items():
            if len(idxs) < 2:
                continue
            idxs = np.array(idxs)
            for _ in range(min(len(idxs), 8)):
                a, b = rng.choice(idxs, 2, replace=False)
                if rts[a] != rts[b]:
                    cand_pos.append((ids[a], ids[b]))
        if cand_pos:
            take = min(len(cand_pos), MAX_POS_PER_KEY)
            sel = rng.choice(len(cand_pos), take, replace=False)
            pos_rows += [(key, *cand_pos[i], 1) for i in sel]
        # negatives: different drivers, same key
        cand_neg = []
        for _ in range(min(len(g) * 2, 200)):
            a, b = rng.choice(len(g), 2, replace=False)
            if drv[a] != drv[b]:
                cand_neg.append((ids[a], ids[b]))
        if cand_neg:
            take = min(len(cand_neg), MAX_NEG_PER_KEY)
            sel = rng.choice(len(cand_neg), take, replace=False)
            neg_rows += [(key, *cand_neg[i], 0) for i in sel]

    # balance classes to TARGET each
    rng.shuffle(pos_rows); rng.shuffle(neg_rows)
    n = min(len(pos_rows), len(neg_rows), TARGET)
    pairs = pd.DataFrame(pos_rows[:n] + neg_rows[:n],
                         columns=["key", "win_a", "win_b", "same_driver"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    pairs.to_parquet(OUT, index=False)
    print(f"wrote {OUT}")
    print(f"pairs: {len(pairs)}  (pos={n}, neg={n})  keys={pairs.key.nunique()}")
    ex = pairs.key.value_counts()
    print(f"pairs/key: median={ex.median():.0f} max={ex.max()}")
    # sanity: scenario mix of pairs
    sc = pairs.key.str.split("|").str[0].value_counts()
    print("scenario mix:\n" + sc.to_string())


if __name__ == "__main__":
    main()
