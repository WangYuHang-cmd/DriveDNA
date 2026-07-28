#!/usr/bin/env python3
"""
M4 — CAN↔video contrastive alignment (CLIP-style, full data).

Generic multimodal SSL with NO driver labels: align each window's CAN embedding with
its co-occurring video embedding via symmetric InfoNCE. Shows DriveDNA supports
multimodal representation learning research. Reports:
  1. cross-modal retrieval R@1/R@5/R@10 (eval windows, both directions)
  2. T2 enrollment probe of the aligned CAN encoder (frozen; SupCon pooling probe NOT
     retrained — the embedding is used as-is, cosine) — does scene-alignment inject
     identity? (expected: modest; alignment is about context)
  3. scenario linear probe from the aligned CAN embedding vs from a random-init CAN
     encoder — does alignment inject scene semantics into the CAN side? (expected: yes)

    python m4_clip_align.py [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")
sys.path.insert(0, f"{BASE}/code/model")
from s1_supcon import Encoder  # CAN side
from m7_video_probe import VidPool  # video side

SEED = 20260709
torch.manual_seed(SEED)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(SEED)

    X = np.load(f"{args.bundle}/windows_x.npy", mmap_mode="r")
    V = np.load(f"{args.bundle}/windows_vid.npy", mmap_mode="r")
    VN = np.load(f"{args.bundle}/windows_vid_n.npy")
    meta = pd.read_parquet(f"{args.bundle}/windows_meta.parquet")
    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    folds = json.load(open(f"{args.bundle}/driver_folds.json"))
    has = VN > 0
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy() & has)
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & has)
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:1500]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    print(f"[M4] train {len(tr_idx)}  eval {len(ev_idx)} (no driver labels used in training)", flush=True)

    can = Encoder(c_in=X.shape[2]).to(dev)
    vid = VidPool().to(dev)
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=dev))
    opt = torch.optim.AdamW(list(can.parameters()) + list(vid.parameters()) + [logit_scale],
                            lr=3e-4, weight_decay=1e-4)

    def batch(idx):
        srt = np.sort(idx); inv = np.argsort(np.argsort(idx))
        x = torch.from_numpy((X[srt].astype(np.float32) - mu) / sd).to(dev)[inv]
        v = torch.from_numpy(V[srt].astype(np.float32)[inv]).to(dev)
        n = torch.from_numpy(VN[srt][inv].astype(np.int64)).to(dev)
        return x, v, n

    epochs = 2 if args.smoke else args.epochs
    steps = 60 if args.smoke else 300
    for ep in range(epochs):
        can.train(); vid.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 192, replace=False)
            x, v, n = batch(bidx)
            zc, zv = can(x), vid(v, n)                       # both L2-normalized
            logits = logit_scale.exp().clamp(max=100) * zc @ zv.T
            y = torch.arange(len(bidx), device=dev)
            loss = 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  infonce={tot/steps:.4f}", flush=True)

    can.eval(); vid.eval()
    Zc, Zv = [], []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 512):
            x, v, n = batch(ev_idx[i:i + 512])
            Zc.append(can(x).cpu().numpy()); Zv.append(vid(v, n).cpu().numpy())
    Zc, Zv = np.concatenate(Zc), np.concatenate(Zv)

    # 1. cross-modal retrieval
    sim = Zc @ Zv.T
    out = {}
    for name, S in [("can->vid", sim), ("vid->can", sim.T)]:
        rank = (-S).argsort(1)
        pos = np.array([np.where(rank[i] == i)[0][0] for i in range(len(S))])
        out[name] = {f"R@{k}": float((pos < k).mean()) for k in (1, 5, 10)}
        print(f"  retrieval {name}: " + "  ".join(f"R@{k}={out[name][f'R@{k}']:.3f}" for k in (1, 5, 10))
              + f"  (chance R@1={1/len(S):.4f})", flush=True)

    # 2. T2 enrollment with aligned CAN embedding (as-is)
    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Zc, me.driver.to_numpy(), me.route.to_numpy())
    print("  T2 probe of aligned CAN embedding (unseen drivers):")
    for k, r in res.items():
        print(f"   k={k:>3}  top1={r['top1']:.3f}  AUROC={r['auroc']:.3f}  EER={r['eer']:.3f}")
    out["t2_probe"] = {str(k): v for k, v in res.items()}

    # 3. scenario linear probe: aligned vs random-init CAN encoder
    from sklearn.linear_model import LogisticRegression
    scen = W["scenario"].to_numpy()[ev_idx]
    rnd = Encoder(c_in=X.shape[2]).to(dev).eval()
    Zr = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 512):
            x, _, _ = batch(ev_idx[i:i + 512])
            Zr.append(rnd(x).cpu().numpy())
    Zr = np.concatenate(Zr)
    half = rng.random(len(ev_idx)) < 0.5
    for nm, Z in [("aligned", Zc), ("random-init", Zr)]:
        clf = LogisticRegression(max_iter=400).fit(Z[half], scen[half])
        acc = float((clf.predict(Z[~half]) == scen[~half]).mean())
        out[f"scenario_probe_{nm}"] = acc
        print(f"  scenario probe ({nm}): acc={acc:.3f}  (chance {1/len(np.unique(scen)):.3f})", flush=True)

    if not args.smoke:
        json.dump(out, open(f"{BASE}/results/m4_clip_align.json", "w"), indent=1)
        torch.save({"can": can.state_dict(), "vid": vid.state_dict(), "mu": mu, "sd": sd},
                   f"{BASE}/experiments/checkpoints/m4_clip.pt")
        print("saved results/m4_clip_align.json")


if __name__ == "__main__":
    main()
