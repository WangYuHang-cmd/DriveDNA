#!/usr/bin/env python3
"""
W2/W3 — self-supervised CAN pretraining baselines on the T2 protocol (full data).

  --mode masked   W2: masked-TS reconstruction pretrain (SimMTM/TimeMAE-style):
                  mask 40% of patch tokens, reconstruct raw patches (MSE on masked).
  --mode jepa     W3: JEPA-style latent-predictive SSL baseline (APPENDIX ONLY —
                  strictly this name; it predicts EMA-target latents at masked
                  positions instead of raw values. NOT a world model.)

Both: pretrain the shared patch-Transformer trunk on train-fold windows WITHOUT any
driver labels, then FREEZE the trunk and train only the attentive-stats pooling head
with SupCon (probe) → harness.enrollment_protocol on UNSEEN drivers. Answers: does
DriveDNA support label-free pretraining research, and how far does SSL close the gap
to supervised S1?

    python w2_ssl.py --mode masked [--smoke]
"""
import os
import json
import copy
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
PATCH, D, T = 10, 192, 600
MASK_FRAC = 0.4


class Trunk(nn.Module):
    """Patch tokenizer + Transformer (token-level outputs, no pooling)."""
    def __init__(self, c_in=17, d=D, patch=PATCH, depth=4, heads=4, t=T):
        super().__init__()
        self.patch, self.c_in = patch, c_in
        self.proj = nn.Linear(c_in * patch, d)
        self.pos = nn.Parameter(torch.randn(1, t // patch, d) * 0.02)
        self.mask_tok = nn.Parameter(torch.randn(d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)

    def tokens(self, x):                                     # x [B,600,C] -> [B,60,d] pre-enc
        B, T_, C = x.shape
        return self.proj(x.reshape(B, T_ // self.patch, self.patch * C)) + self.pos

    def forward(self, x, mask=None):                         # mask [B,60] bool: True=masked
        h = self.tokens(x)
        if mask is not None:
            h = torch.where(mask.unsqueeze(-1), self.mask_tok.expand_as(h), h)
        return self.enc(h)                                   # [B,60,d]


class PoolHead(nn.Module):
    """Attentive-stats pooling + projection (the probe; matches S1's pooling)."""
    def __init__(self, d=D, d_emb=128):
        super().__init__()
        self.att = nn.Sequential(nn.Linear(d, d // 2), nn.Tanh(), nn.Linear(d // 2, 1))
        self.head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d_emb))

    def forward(self, h):                                    # h [B,60,d]
        w = torch.softmax(self.att(h), dim=1)
        mu = (w * h).sum(1)
        sd = torch.sqrt(((h - mu.unsqueeze(1)) ** 2 * w).sum(1).clamp_min(1e-6))
        return F.normalize(self.head(torch.cat([mu, sd], -1)), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["masked", "jepa"])
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--pre-epochs", type=int, default=15)
    ap.add_argument("--probe-epochs", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:2000]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3

    def fetch(idx_global):
        return torch.from_numpy((X[np.sort(idx_global)].astype(np.float32) - mu) / sd)

    trunk = Trunk(c_in=X.shape[2]).to(dev)
    n_tok = T // PATCH
    name = ("masked-TS reconstruction (SimMTM/TimeMAE-style)" if args.mode == "masked"
            else "JEPA-style latent-predictive SSL baseline")
    print(f"[W2/W3 {args.mode}] {name}: pretrain {len(tr_idx)} windows (no driver labels)", flush=True)

    if args.mode == "masked":
        dec = nn.Linear(D, PATCH * X.shape[2]).to(dev)
        opt = torch.optim.AdamW(list(trunk.parameters()) + list(dec.parameters()), lr=3e-4, weight_decay=1e-4)
    else:
        tgt = copy.deepcopy(trunk).to(dev)
        for p in tgt.parameters():
            p.requires_grad_(False)
        pred = nn.Sequential(nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)).to(dev)
        opt = torch.optim.AdamW(list(trunk.parameters()) + list(pred.parameters()), lr=3e-4, weight_decay=1e-4)

    pre_epochs = 2 if args.smoke else args.pre_epochs
    steps = 60 if args.smoke else 350
    for ep in range(pre_epochs):
        trunk.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 96, replace=False)
            x = fetch(bidx).to(dev)
            m = torch.from_numpy(rng.random((len(bidx), n_tok)) < MASK_FRAC).to(dev)
            h = trunk(x, mask=m)
            if args.mode == "masked":
                target = x.reshape(len(bidx), n_tok, PATCH * X.shape[2])
                loss = ((dec(h) - target) ** 2)[m].mean()
            else:
                with torch.no_grad():
                    ht = tgt(x)                              # full-view target latents
                    ht = F.layer_norm(ht, (D,))
                loss = F.smooth_l1_loss(pred(h)[m], ht[m])
            opt.zero_grad(); loss.backward(); opt.step()
            if args.mode == "jepa":                          # EMA target update
                with torch.no_grad():
                    for pt, ps in zip(tgt.parameters(), trunk.parameters()):
                        pt.mul_(0.996).add_(ps, alpha=0.004)
            tot += float(loss.detach())
        print(f"pretrain epoch {ep+1}/{pre_epochs}  loss={tot/steps:.4f}", flush=True)

    # ---- frozen-trunk SupCon probe ----
    for p in trunk.parameters():
        p.requires_grad_(False)
    trunk.eval()
    head = PoolHead().to(dev)
    hopt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=1e-4)
    drivers_tr = meta.driver.to_numpy()[tr_idx]
    dmap = {d: i for i, d in enumerate(pd.unique(drivers_tr))}
    probe_epochs = 2 if args.smoke else args.probe_epochs
    for ep in range(probe_epochs):
        head.train()
        sampler = PKSampler(drivers_tr, P=16, K=4, seed=SEED + ep)
        tot = nb = 0
        for bi in sampler:
            gidx = tr_idx[bi]
            order = np.argsort(gidx)
            x = fetch(gidx).to(dev)[np.argsort(order)]
            y = torch.tensor([dmap[drivers_tr[i]] for i in bi], device=dev)
            with torch.no_grad():
                h = trunk(x)
            loss = supcon(head(h), y)
            hopt.zero_grad(); loss.backward(); hopt.step()
            tot += float(loss.detach()); nb += 1
        print(f"probe epoch {ep+1}/{probe_epochs}  supcon={tot/max(nb,1):.4f}", flush=True)

    head.eval()
    embs = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), 512):
            embs.append(head(trunk(fetch(ev_idx[i:i + 512]).to(dev))).cpu().numpy())
    Z = np.concatenate(embs)
    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
    print(f"\n== {name} → frozen-trunk SupCon probe, UNSEEN drivers ==")
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    ck = f"{BASE}/experiments/checkpoints/w2_{args.mode}.pt"
    torch.save({"trunk": trunk.state_dict(), "head": head.state_dict(), "mu": mu, "sd": sd}, ck)
    if not args.smoke:
        json.dump({str(k): v for k, v in res.items()},
                  open(f"{BASE}/results/w2_{args.mode}.json", "w"), indent=1)
    print("saved", ck)


if __name__ == "__main__":
    main()
