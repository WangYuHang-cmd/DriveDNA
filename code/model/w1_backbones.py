#!/usr/bin/env python3
"""
W1 — additional modern TS backbones/objectives on the SAME T2 protocol as S1 (full data).

Purpose: instantiate the T2 protocol with multiple modern representation baselines →
one unified T2 table (descriptors / S1-SupCon / PatchTST-CI / iTransformer / ArcFace).
These are benchmark instantiations, not "our stronger model".

Variants (--arch):
  ci       true channel-independent PatchTST (per-channel patching, shared weights,
           attention within channel only, attentive channel aggregation) + SupCon
  itr      iTransformer (ICLR'24): one token per variate (channel), attention across
           variates + SupCon
  arcface  S1's joint patch encoder trained with ArcFace (margin classification over
           train drivers) instead of SupCon — isolates the objective axis

Everything else matches s1_supcon.py exactly: folds, PKSampler P16×K4, channel-dropout,
standardization, 20 epochs, harness.enrollment_protocol on UNSEEN drivers.
    python w1_backbones.py --arch ci [--smoke]
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
from s1_supcon import Encoder as JointEncoder, supcon, PKSampler  # noqa: E402

SEED = 20260709
torch.manual_seed(SEED)


class CIPatchTST(nn.Module):
    """Channel-independent PatchTST: each channel is a univariate series; shared
    patch-embed + Transformer attend WITHIN a channel; attentive pooling over
    (channel × time) token summaries."""
    def __init__(self, c_in=17, d=128, patch=20, depth=3, heads=4, d_emb=128, t=600):
        super().__init__()
        self.patch, self.c_in = patch, c_in
        self.proj = nn.Linear(patch, d)
        self.pos = nn.Parameter(torch.randn(1, t // patch, d) * 0.02)
        self.chan = nn.Parameter(torch.randn(c_in, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d_emb))

    def forward(self, x):                                   # x [B,600,C]
        B, T, C = x.shape
        u = x.permute(0, 2, 1).reshape(B * C, T // self.patch, self.patch)
        h = self.enc(self.proj(u) + self.pos)               # [B*C, n_tok, d] within-channel attn
        h = h.mean(1).reshape(B, C, -1) + self.chan          # per-channel summary + channel id
        w = torch.softmax(self.att(h), dim=1)                # attentive stats over channels
        mu = (w * h).sum(1)
        sd = torch.sqrt(((h - mu.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6))
        return F.normalize(self.head(torch.cat([mu, sd], -1)), dim=-1)


class ITransformer(nn.Module):
    """iTransformer: each variate's full series -> one token; attention ACROSS variates."""
    def __init__(self, c_in=17, d=192, depth=4, heads=4, d_emb=128, t=600):
        super().__init__()
        self.proj = nn.Linear(t, d)
        self.chan = nn.Parameter(torch.randn(1, c_in, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d_emb))

    def forward(self, x):                                   # x [B,600,C]
        h = self.enc(self.proj(x.permute(0, 2, 1)) + self.chan)  # [B,C,d]
        w = torch.softmax(self.att(h), dim=1)
        mu = (w * h).sum(1)
        sd = torch.sqrt(((h - mu.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6))
        return F.normalize(self.head(torch.cat([mu, sd], -1)), dim=-1)


class ArcFaceHead(nn.Module):
    def __init__(self, d_emb, n_cls, s=30.0, m=0.3):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, d_emb) * 0.02)
        self.s, self.m = s, m

    def forward(self, z, y):                                 # z L2-normalized
        cos = z @ F.normalize(self.W, dim=-1).T
        th = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        cos_m = torch.cos(th + self.m)
        logits = self.s * torch.where(
            F.one_hot(y, cos.shape[1]).bool(), cos_m, cos)
        return F.cross_entropy(logits, y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, choices=["ci", "itr", "arcface"])
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    tr_drv = set(folds["train"]); ev_drv = set(folds["val"]) | set(folds["few_shot_heldout"])
    tr_idx = np.flatnonzero(meta.driver.isin(tr_drv).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(ev_drv).to_numpy())
    if args.smoke:
        tr_idx = tr_idx[:4000]; ev_idx = ev_idx[:2000]
    samp = X[np.sort(np.random.default_rng(args.seed).choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3

    drivers_tr = meta.driver.to_numpy()[tr_idx]
    dmap = {d: i for i, d in enumerate(pd.unique(drivers_tr))}
    print(f"[{args.arch}] train windows={len(tr_idx)} ({len(dmap)} drivers)  "
          f"eval windows={len(ev_idx)} ({meta.iloc[ev_idx].driver.nunique()} unseen)  dev={dev}", flush=True)

    if args.arch == "ci":
        model = CIPatchTST(c_in=X.shape[2]).to(dev)
    elif args.arch == "itr":
        model = ITransformer(c_in=X.shape[2]).to(dev)
    else:
        model = JointEncoder(c_in=X.shape[2]).to(dev)
    arc = ArcFaceHead(128, len(dmap)).to(dev) if args.arch == "arcface" else None
    params = list(model.parameters()) + (list(arc.parameters()) if arc else [])
    opt = torch.optim.AdamW(params, lr=3e-4, weight_decay=1e-4)
    epochs = 2 if args.smoke else args.epochs

    def fetch(idx_global):
        return torch.from_numpy((X[np.sort(idx_global)].astype(np.float32) - mu) / sd)

    for ep in range(epochs):
        model.train()
        sampler = PKSampler(drivers_tr, P=16, K=4, seed=args.seed + ep)
        tot = nb = 0
        for bi in sampler:
            gidx = tr_idx[bi]
            order = np.argsort(gidx)
            x = fetch(gidx).to(dev)[np.argsort(order)]
            if np.random.rand() < 0.5:                       # channel dropout (as S1)
                drop = torch.rand(x.shape[2], device=dev) < 0.15
                x[:, :, drop] = 0
            y = torch.tensor([dmap[drivers_tr[i]] for i in bi], device=dev)
            z = model(x)
            loss = arc(z, y) if arc else supcon(z, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"epoch {ep+1}/{epochs}  loss={tot/max(nb,1):.4f}", flush=True)

    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 512):
            embs.append(model(fetch(ev_idx[i:i + 512]).to(dev)).cpu().numpy())
    Z = np.concatenate(embs)
    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
    name = {"ci": "PatchTST-CI + SupCon", "itr": "iTransformer + SupCon",
            "arcface": "joint-patch + ArcFace"}[args.arch]
    print(f"\n== W1 {name} on UNSEEN drivers ==")
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    sfx = "" if args.seed == SEED else f"_s{args.seed % 100}"
    ck = f"{BASE}/experiments/checkpoints/w1_{args.arch}{sfx}.pt"
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd, "arch": args.arch}, ck)
    if not args.smoke:
        os.makedirs(f"{BASE}/results", exist_ok=True)
        json.dump({str(k): v for k, v in res.items()},
                  open(f"{BASE}/results/w1_{args.arch}{sfx}.json", "w"), indent=1)
    print("saved", ck)


if __name__ == "__main__":
    main()
