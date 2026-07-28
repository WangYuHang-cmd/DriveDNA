#!/usr/bin/env python3
"""
V3 — TS foundation-model row: MOMENT-1 (CMU, 2024) frozen zero-shot embeddings on T2.

No training at all: embed each window with the pretrained MOMENT-1-large encoder
(channel-independent, 512-step context → take the last 512 of our 600 steps),
mean over channels → embedding → harness.enrollment_protocol on UNSEEN drivers.
One "TS foundation model (zero-shot)" row for the unified T2 table — answers
"where do 2024/25 pretrained TS models stand on this benchmark without adaptation?"

    PYTHONPATH= .../python v3_moment.py [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch

import sys
BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")

SEED = 20260709


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(SEED)

    X = np.load(f"{args.bundle}/windows_x.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{args.bundle}/windows_meta.parquet")
    folds = json.load(open(f"{args.bundle}/driver_folds.json"))
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    if args.smoke:
        ev_idx = ev_idx[:1000]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3

    from momentfm import MOMENTPipeline
    model = MOMENTPipeline.from_pretrained(
        "AutonLab/MOMENT-1-large", model_kwargs={"task_name": "embedding"},
        cache_dir=f"{BASE}/data/.hfhub")
    model.init()
    model = model.to(dev).bfloat16().eval()
    print("MOMENT-1-large loaded (frozen, zero-shot)", flush=True)

    Z = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 16):
            sl = ev_idx[i:i + 16]
            w = (X[np.sort(sl)].astype(np.float32) - mu) / sd
            w = w[np.argsort(np.argsort(sl))][:, -512:]        # last 512 steps
            x = torch.from_numpy(w).permute(0, 2, 1).to(dev).bfloat16()   # [B, C, 512]
            out = model(x_enc=x)
            Z.append(out.embeddings.float().cpu().numpy())     # [B, d] (mean over channels inside)
            if (i // 16) % 160 == 0:
                print(f"  {i}/{len(ev_idx)}", flush=True)
    Z = np.concatenate(Z)
    print("embeddings:", Z.shape, flush=True)

    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
    print("\n== V3 MOMENT-1-large zero-shot on T2 (UNSEEN drivers) ==")
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    if not args.smoke:
        json.dump({str(k): v for k, v in res.items()},
                  open(f"{BASE}/results/v3_moment.json", "w"), indent=1)
        print("saved results/v3_moment.json")


if __name__ == "__main__":
    main()
