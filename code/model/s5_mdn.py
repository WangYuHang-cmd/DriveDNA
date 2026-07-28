#!/usr/bin/env python3
"""
S5 — MDN distributional future-behavior head (must-land #6).

Same protocol as S3 (5 s history → 5 s future, condition-dropout for generic-vs-personalized),
but the decoder outputs a K-component Gaussian mixture over the future trajectory:
  p(Y | X, z_d) = Σ_k π_k N(μ_k, σ_k²)   (diagonal, per step × target)
Metrics: NLL (distributional) + RMSE of the mixture mean; PG on BOTH → "distributional PG".

    python s5_mdn.py --epochs 15 [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from s3_personalized import SupportEncoder  # reuse

SEED = 20260709
torch.manual_seed(SEED)
L_HIST, L_FUT, K = 50, 50, 5
TGT = ["aEgo", "actual_curvature"]


class MDNPredictor(nn.Module):
    def __init__(self, c_in, n_tgt, d=128, dz=64, k=K):
        super().__init__()
        self.enc = nn.GRU(c_in, d, num_layers=2, batch_first=True)
        self.film = nn.Linear(dz, 2 * d)
        self.k, self.n = k, L_FUT * n_tgt
        self.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, k * (1 + 2 * self.n)))

    def forward(self, x, z):
        h = self.enc(x)[0][:, -1]
        g, b = self.film(z).chunk(2, -1)
        h = h * (1 + g) + b
        out = self.head(h)                                   # [B, K(1+2n)]
        logit = out[:, :self.k]
        mu, logv = out[:, self.k:].view(-1, self.k, 2, self.n).unbind(2)
        return logit, mu, logv.clamp(-6, 3)

    def nll(self, x, z, y):
        logit, mu, logv = self(x, z)
        yf = y.reshape(len(y), 1, self.n)
        lp = -0.5 * (((yf - mu) ** 2) / logv.exp() + logv + np.log(2 * np.pi)).sum(-1)
        return -(torch.logsumexp(torch.log_softmax(logit, -1) + lp, dim=1)).mean(), (logit, mu)

    @staticmethod
    def mean_pred(logit, mu):
        w = torch.softmax(logit, -1).unsqueeze(-1)
        return (w * mu).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./experiments/checkpoints/s5_mdn.pt")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    ch = json.load(open(os.path.join(args.bundle, "channels.json")))["channels"]
    tgt_ix = [ch.index(c) for c in TGT]
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:1500]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu_ = samp.reshape(-1, X.shape[2]).mean(0); sd_ = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    norm = lambda a: (a.astype(np.float32) - mu_) / sd_

    by_drv = meta.iloc[tr_idx].groupby("driver").indices
    by_drv = {d: tr_idx[v] for d, v in by_drv.items() if len(v) >= 3}

    sup = SupportEncoder(X.shape[2]).to(dev)
    net = MDNPredictor(X.shape[2], len(TGT)).to(dev)
    opt = torch.optim.AdamW(list(sup.parameters()) + list(net.parameters()), lr=3e-4, weight_decay=1e-4)

    def make(bidx, train=True):
        w = norm(X[np.sort(bidx)])[np.argsort(np.argsort(bidx))]
        anchors = rng.integers(L_HIST, 600 - L_FUT, len(bidx))
        xs = torch.from_numpy(np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])).to(dev)
        ys = torch.from_numpy(np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])).to(dev)
        z = torch.zeros(len(bidx), 64, device=dev)
        for i, gi in enumerate(bidx):
            pool = by_drv.get(meta.driver.iat[gi], np.array([gi]))
            pool = pool[pool != gi]
            if train and (len(pool) == 0 or rng.random() < 0.3):
                continue
            pick = np.sort(rng.choice(pool, min(3, len(pool)), replace=False)) if len(pool) else [gi]
            z[i] = sup(torch.from_numpy(norm(X[pick])).to(dev)).mean(0)
        return xs, ys, z

    epochs = 2 if args.smoke else args.epochs
    steps = 80 if args.smoke else 400
    for ep in range(epochs):
        sup.train(); net.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 64, replace=False)
            xs, ys, z = make(bidx, True)
            loss, _ = net.nll(xs, z, ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  nll={tot/steps:.4f}", flush=True)

    sup.eval(); net.eval()
    rng_eval = np.random.default_rng(20260709)  # eval protocol FIXED across seeds
    me = meta.iloc[ev_idx].reset_index(drop=True)
    agg = {"g": {"nll": [], "rmse": []}, "p": {"nll": [], "rmse": []}}
    with torch.no_grad():
        for d, g in me.groupby("driver"):
            rts = g.route.unique()
            if len(rts) < 2 or len(g) < 8:
                continue
            rng_eval.shuffle(rts)
            sup_r = set(rts[: max(1, len(rts) // 2)])
            sup_pool = ev_idx[g.index[g.route.isin(sup_r)].to_numpy()]
            qry = ev_idx[g.index[~g.route.isin(sup_r)].to_numpy()]
            if len(sup_pool) < 1 or len(qry) < 4:
                continue
            w = norm(X[np.sort(qry)])[np.argsort(np.argsort(qry))]
            anchors = rng_eval.integers(L_HIST, 600 - L_FUT, len(qry))
            xs = torch.from_numpy(np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])).to(dev)
            ys = torch.from_numpy(np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])).to(dev)
            pick = np.sort(rng_eval.choice(sup_pool, min(5, len(sup_pool)), replace=False))
            zd = sup(torch.from_numpy(norm(X[pick])).to(dev)).mean(0, keepdim=True).expand(len(qry), -1)
            for tag, z in [("g", torch.zeros_like(zd)), ("p", zd)]:
                nll, (logit, mu) = net.nll(xs, z, ys)
                pm = net.mean_pred(logit, mu).view(-1, L_FUT, len(TGT))
                agg[tag]["nll"].append(float(nll))
                agg[tag]["rmse"] += list(np.sqrt(((pm - ys) ** 2).mean((1, 2)).cpu().numpy()))
    gn, pn = np.mean(agg["g"]["nll"]), np.mean(agg["p"]["nll"])
    gr, pr = np.mean(agg["g"]["rmse"]), np.mean(agg["p"]["rmse"])
    print("\n== S5 MDN distributional prediction (unseen drivers, k=5) ==")
    print(f"  NLL : generic {gn:.3f}  personal {pn:.3f}  distributional-PG {gn-pn:+.3f}")
    print(f"  RMSE: generic {gr:.4f}  personal {pr:.4f}  PG {(gr-pr)/gr*100:+.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"sup": sup.state_dict(), "net": net.state_dict(), "mu": mu_, "sd": sd_}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
