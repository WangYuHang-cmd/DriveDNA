#!/usr/bin/env python3
"""
M7 — video-only driver-ID probe [DIAGNOSTIC].

Question: can forward-road video alone identify the driver? Expected answer: weakly,
and mostly via route/scene leakage — which empirically supports DriveDNA's
video-as-CONTEXT (not identity) design.

Three results:
  1. zero-shot: mean-pooled DINOv2 window embedding → T2 enrollment protocol (unseen drivers)
  2. trained:   attentive-pooling SupCon probe over the 120 frame tokens (video ONLY)
                → same enrollment protocol   [compare with S1 CAN: AUROC .931]
  3. leakage contrast: linear probes on eval windows — predict DRIVER vs predict ROUTE
                from the same video embedding (window-split). Expected: route >> driver.

    python m7_video_probe.py [--smoke]
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
from s1_supcon import supcon, PKSampler  # noqa: E402

SEED = 20260709
torch.manual_seed(SEED)


class VidPool(nn.Module):
    def __init__(self, d_in=768, d=192, d_emb=128):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d_emb))

    def forward(self, v, n):                                 # v [B,120,768], n frames valid
        h = self.proj(v)
        mask = (torch.arange(h.shape[1], device=h.device)[None] < n[:, None])
        a = self.att(h).masked_fill(~mask.unsqueeze(-1), -1e9)
        w = torch.softmax(a, dim=1)
        mu = (w * h).sum(1)
        sd = torch.sqrt(((h - mu.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6))
        return F.normalize(self.head(torch.cat([mu, sd], -1)), dim=-1)


def linear_probe_acc(Z, labels, rng, min_per=4):
    """50/50 within-class window split -> logistic regression accuracy + chance."""
    from sklearn.linear_model import LogisticRegression
    ok = pd.Series(labels).groupby(labels).transform("size") >= min_per
    Z, labels = Z[ok.to_numpy()], np.asarray(labels)[ok.to_numpy()]
    tr = np.zeros(len(labels), bool)
    for lb in pd.unique(labels):
        ix = np.flatnonzero(labels == lb)
        tr[rng.choice(ix, len(ix) // 2, replace=False)] = True
    clf = LogisticRegression(max_iter=400, C=1.0).fit(Z[tr], labels[tr])
    acc = float((clf.predict(Z[~tr]) == labels[~tr]).mean())
    return acc, 1.0 / len(pd.unique(labels)), len(pd.unique(labels))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    V = np.load(os.path.join(args.bundle, "windows_vid.npy"), mmap_mode="r")
    VN = np.load(os.path.join(args.bundle, "windows_vid_n.npy"))
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    has = VN > 0
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy() & has)
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & has)
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:2000]
    print(f"video-aligned: train {len(tr_idx)}  eval {len(ev_idx)} "
          f"({meta.iloc[ev_idx].driver.nunique()} unseen drivers)", flush=True)

    from harness import enrollment_protocol
    me = meta.iloc[ev_idx]

    # ---- 1. zero-shot mean-pool ----
    def meanpool(idx):
        out = np.zeros((len(idx), 768), np.float32)
        for j in range(0, len(idx), 1024):
            sl = idx[j:j + 1024]
            v = V[np.sort(sl)].astype(np.float32)[np.argsort(np.argsort(sl))]
            n = VN[np.sort(sl)][np.argsort(np.argsort(sl))].astype(int)
            for i in range(len(sl)):
                out[j + i] = v[i, :max(n[i], 1)].mean(0)
        return out
    Z0 = meanpool(ev_idx)
    Z0n = Z0 / (np.linalg.norm(Z0, axis=1, keepdims=True) + 1e-9)
    res0 = enrollment_protocol(Z0n, me.driver.to_numpy(), me.route.to_numpy())

    # ---- 2. SupCon-trained video-only probe ----
    model = VidPool().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    drivers_tr = meta.driver.to_numpy()[tr_idx]
    dmap = {d: i for i, d in enumerate(pd.unique(drivers_tr))}
    epochs = 2 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train()
        sampler = PKSampler(drivers_tr, P=16, K=4, seed=SEED + ep)
        tot = nb = 0
        for bi in sampler:
            gidx = tr_idx[bi]
            srt = np.sort(gidx); inv = np.argsort(np.argsort(gidx))
            v = torch.from_numpy(V[srt].astype(np.float32)[inv]).to(dev)
            n = torch.from_numpy(VN[srt][inv].astype(np.int64)).to(dev)
            y = torch.tensor([dmap[drivers_tr[i]] for i in bi], device=dev)
            loss = supcon(model(v, n), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"epoch {ep+1}/{epochs}  supcon={tot/max(nb,1):.4f}", flush=True)
    model.eval()
    Z1 = []
    with torch.no_grad():
        for j in range(0, len(ev_idx), 512):
            sl = ev_idx[j:j + 512]
            srt = np.sort(sl); inv = np.argsort(np.argsort(sl))
            v = torch.from_numpy(V[srt].astype(np.float32)[inv]).to(dev)
            n = torch.from_numpy(VN[srt][inv].astype(np.int64)).to(dev)
            Z1.append(model(v, n).cpu().numpy())
    Z1 = np.concatenate(Z1)
    res1 = enrollment_protocol(Z1, me.driver.to_numpy(), me.route.to_numpy())

    # ---- 3. driver-vs-route linear-probe contrast (same embedding) ----
    acc_d, ch_d, n_d = linear_probe_acc(Z1, me.driver.to_numpy(), rng)
    acc_r, ch_r, n_r = linear_probe_acc(Z1, me.route.to_numpy(), rng)

    print("\n== M7 video-only driver-ID probe (UNSEEN drivers) ==")
    for tag, res in [("zero-shot mean-pool", res0), ("SupCon-trained probe", res1)]:
        print(f"-- {tag} --")
        print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
        for k, r in res.items():
            print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
                  f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    print(f"-- leakage contrast (linear probe, window-split) --")
    print(f"  driver: acc {acc_d:.3f} (chance {ch_d:.4f}, {n_d} classes)  = {acc_d/ch_d:.1f}x")
    print(f"  route : acc {acc_r:.3f} (chance {ch_r:.4f}, {n_r} classes)  = {acc_r/ch_r:.1f}x")
    if not args.smoke:
        json.dump({"zero_shot": {str(k): v for k, v in res0.items()},
                   "trained": {str(k): v for k, v in res1.items()},
                   "probe": {"driver_acc": acc_d, "driver_chance": ch_d,
                             "route_acc": acc_r, "route_chance": ch_r}},
                  open(f"{BASE}/results/m7_video_probe.json", "w"), indent=1)
        torch.save({"model": model.state_dict()},
                   f"{BASE}/experiments/checkpoints/m7_vidpool.pt")
        print("saved results/m7_video_probe.json + checkpoints/m7_vidpool.pt")


if __name__ == "__main__":
    main()
