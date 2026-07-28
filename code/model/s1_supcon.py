#!/usr/bin/env python3
"""
S1 — PatchTST-style SupCon driver embedding (the first learned baseline; T2).

Encoder: patch the [600,17] window (patch=10 → 60 tokens) → linear embed → Transformer
→ attentive statistics pooling (x-vector style) → 128-d embedding.
Loss: supervised contrastive (drivers as classes) with P×K batch sampling.
Eval: harness.enrollment_protocol on UNSEEN drivers (val/test folds, route-disjoint).
Robustness: channel-dropout augmentation (handles per-vehicle missing channels).

Runs from the colab_bundle (works identically on 5080 and Colab A100):
    python s1_supcon.py --bundle .../data/colab_bundle --epochs 20 [--smoke]
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
sys.path.insert(0, "./code/eval")

SEED = 20260709
torch.manual_seed(SEED)


class Encoder(nn.Module):
    def __init__(self, c_in=17, d=192, patch=10, depth=4, heads=4, d_emb=128, t=600):
        super().__init__()
        self.patch = patch
        self.proj = nn.Linear(c_in * patch, d)
        self.pos = nn.Parameter(torch.randn(1, t // patch, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d_emb))

    def forward(self, x, mask=None):                     # x [B,600,C]
        B, T, C = x.shape
        p = x.reshape(B, T // self.patch, self.patch * C)
        h = self.enc(self.proj(p) + self.pos)            # [B,60,d]
        w = torch.softmax(self.att(h), dim=1)            # attentive stats pooling
        mu = (w * h).sum(1)
        sd = torch.sqrt(((h - mu.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6))
        z = self.head(torch.cat([mu, sd], -1))
        return F.normalize(z, dim=-1)


def supcon(z, y, tau=0.1):
    sim = z @ z.T / tau
    n = len(y)
    eye = torch.eye(n, dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    sim = sim.masked_fill(eye, -1e9)
    logp = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    denom = pos.sum(1).clamp_min(1)
    return -(logp * pos).sum(1).div(denom)[pos.any(1)].mean()


class PKSampler:
    """Yields batches of P drivers × K windows."""
    def __init__(self, drivers, P=16, K=4, seed=SEED):
        self.rng = np.random.default_rng(seed)
        self.by = {}
        for i, d in enumerate(drivers):
            self.by.setdefault(d, []).append(i)
        self.by = {d: np.array(v) for d, v in self.by.items() if len(v) >= K}
        self.P, self.K = P, K
        self.keys = list(self.by)

    def __iter__(self):
        n_batches = max(1, sum(len(v) for v in self.by.values()) // (self.P * self.K))
        for _ in range(n_batches):
            ds = self.rng.choice(len(self.keys), min(self.P, len(self.keys)), replace=False)
            idx = np.concatenate([self.rng.choice(self.by[self.keys[d]], self.K,
                                                  replace=len(self.by[self.keys[d]]) < self.K)
                                  for d in ds])
            yield idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./experiments/checkpoints/s1_supcon.pt")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    if args.seed != SEED and args.out.endswith("s1_supcon.pt"):
        args.out = args.out.replace(".pt", f"_s{args.seed % 100}.pt")
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    tr_drv = set(folds["train"]); ev_drv = set(folds["val"]) | set(folds["few_shot_heldout"])

    # per-channel standardization from a train subsample
    tr_idx = np.flatnonzero(meta.driver.isin(tr_drv).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(ev_drv).to_numpy())
    if args.smoke:
        tr_idx = tr_idx[:4000]; ev_idx = ev_idx[:2000]
    samp = X[np.sort(np.random.default_rng(args.seed).choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    print(f"train windows={len(tr_idx)} ({len(tr_drv)} drivers)  eval windows={len(ev_idx)} "
          f"({meta.iloc[ev_idx].driver.nunique()} unseen drivers)  device={dev}", flush=True)

    model = Encoder(c_in=X.shape[2]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    drivers_tr = meta.driver.to_numpy()[tr_idx]
    dmap = {d: i for i, d in enumerate(pd.unique(drivers_tr))}
    epochs = 2 if args.smoke else args.epochs

    def fetch(idx_global):
        x = torch.from_numpy(((X[np.sort(idx_global)].astype(np.float32) - mu) / sd))
        return x

    for ep in range(epochs):
        model.train()
        sampler = PKSampler(drivers_tr, P=16, K=4, seed=args.seed + ep)
        tot = nb = 0
        for bi in sampler:
            gidx = tr_idx[bi]
            order = np.argsort(gidx)
            x = fetch(gidx).to(dev)[np.argsort(order)]     # restore batch order
            if model.training and np.random.rand() < 0.5:  # channel dropout (missing-channel robustness)
                drop = torch.rand(x.shape[2], device=dev) < 0.15
                x[:, :, drop] = 0
            y = torch.tensor([dmap[drivers_tr[i]] for i in bi], device=dev)
            loss = supcon(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss); nb += 1
        print(f"epoch {ep+1}/{epochs}  supcon={tot/max(nb,1):.4f}", flush=True)

    # ---- embed eval windows (unseen drivers) + harness enrollment protocol ----
    model.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 512):
            x = fetch(ev_idx[i:i + 512]).to(dev)
            embs.append(model(x).cpu().numpy())
    Z = np.concatenate(embs)
    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
    print(f"\n== S1 SupCon on UNSEEN drivers ==")
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
