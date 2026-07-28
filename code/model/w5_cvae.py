#!/usr/bin/env python3
"""
W5/CVAE — CVAE distributional future-behavior head (main-table row; confirms the MDN
"style is distributional" finding is not an MDN artifact).

EXACTLY the s5_mdn.py protocol (same targets, support encoder, condition-dropout,
driver-disjoint eval, generic-vs-personalized) with the mixture head replaced by a CVAE:
  prior     p(z_l | h, z_d)          posterior q(z_l | h, z_d, Y)
  decoder   p(Y | h, z_d, z_l) = N(mu, sigma^2)  (diagonal, per step x target)
Train: ELBO (recon NLL + KL(q||p)).  Eval NLL: importance-sampled with S prior draws
(comparable to MDN NLL); RMSE from the prior-mean decode.

    python w5_cvae.py --epochs 15 [--smoke]
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
L_HIST, L_FUT = 50, 50
DZL = 16          # latent dim
NS = 32           # prior samples for eval NLL
TGT = ["aEgo", "actual_curvature"]


class CVAEPredictor(nn.Module):
    def __init__(self, c_in, n_tgt, d=128, dz=64, dzl=DZL):
        super().__init__()
        self.n = L_FUT * n_tgt
        self.enc = nn.GRU(c_in, d, num_layers=2, batch_first=True)
        self.film = nn.Linear(dz, 2 * d)
        self.prior = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * dzl))
        self.post = nn.Sequential(nn.Linear(d + self.n, d), nn.GELU(), nn.Linear(d, 2 * dzl))
        self.dec = nn.Sequential(nn.Linear(d + dzl, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, 2 * self.n))

    def ctx(self, x, z):
        h = self.enc(x)[0][:, -1]
        g, b = self.film(z).chunk(2, -1)
        return h * (1 + g) + b

    @staticmethod
    def gauss_nll(y, mu, logv):                            # sum over dims
        return 0.5 * (((y - mu) ** 2) / logv.exp() + logv + np.log(2 * np.pi)).sum(-1)

    def elbo(self, x, z, y):
        h = self.ctx(x, z)
        yf = y.reshape(len(y), self.n)
        mp, lp = self.prior(h).chunk(2, -1)
        mq, lq = self.post(torch.cat([h, yf], -1)).chunk(2, -1)
        lp, lq = lp.clamp(-6, 3), lq.clamp(-6, 3)
        zl = mq + torch.randn_like(mq) * (0.5 * lq).exp()
        mu, logv = self.dec(torch.cat([h, zl], -1)).chunk(2, -1)
        rec = self.gauss_nll(yf, mu, logv.clamp(-3, 3)).mean()
        kl = 0.5 * ((lq - lp).exp() + (mq - mp) ** 2 / lp.exp() - 1 + lp - lq).sum(-1).mean()
        return rec, kl

    @torch.no_grad()
    def eval_nll_rmse(self, x, z, y, ns=NS):
        h = self.ctx(x, z)
        yf = y.reshape(len(y), self.n)
        mp, lp = self.prior(h).chunk(2, -1)
        lp = lp.clamp(-6, 3)
        lls = []
        for _ in range(ns):
            zl = mp + torch.randn_like(mp) * (0.5 * lp).exp()
            mu, logv = self.dec(torch.cat([h, zl], -1)).chunk(2, -1)
            lls.append(-self.gauss_nll(yf, mu, logv.clamp(-3, 3)))
        nll = -(torch.logsumexp(torch.stack(lls, 1), dim=1) - np.log(ns)).mean()
        mu0, _ = self.dec(torch.cat([h, mp], -1)).chunk(2, -1)   # prior-mean decode
        rmse = torch.sqrt(((mu0 - yf) ** 2).mean(-1))
        return float(nll), rmse.cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./experiments/checkpoints/w5_cvae.pt")
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
    net = CVAEPredictor(X.shape[2], len(TGT)).to(dev)
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
        beta = min(1.0, (ep + 1) / 3)                        # KL annealing (3-epoch warmup)
        tot = tr_ = tk = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 64, replace=False)
            xs, ys, z = make(bidx, True)
            rec, kl = net.elbo(xs, z, ys)
            loss = rec + beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); tr_ += float(rec.detach()); tk += float(kl.detach())
        print(f"epoch {ep+1}/{epochs}  elbo={tot/steps:.3f} (rec {tr_/steps:.3f} kl {tk/steps:.3f} beta {beta:.2f})", flush=True)

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
                nll, rmse = net.eval_nll_rmse(xs, z, ys)
                agg[tag]["nll"].append(nll)
                agg[tag]["rmse"] += list(rmse)
    gn, pn = np.mean(agg["g"]["nll"]), np.mean(agg["p"]["nll"])
    gr, pr = np.mean(agg["g"]["rmse"]), np.mean(agg["p"]["rmse"])
    print("\n== W5 CVAE distributional prediction (unseen drivers, k=5) ==")
    print(f"  NLL : generic {gn:.3f}  personal {pn:.3f}  distributional-PG {gn-pn:+.3f}")
    print(f"  RMSE: generic {gr:.4f}  personal {pr:.4f}  PG {(gr-pr)/gr*100:+.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"sup": sup.state_dict(), "net": net.state_dict(), "mu": mu_, "sd": sd_}, args.out)
    if not args.smoke:
        json.dump({"nll_generic": float(gn), "nll_personal": float(pn), "dpg": float(gn - pn),
                   "rmse_generic": float(gr), "rmse_personal": float(pr)},
                  open("./results/w5_cvae.json", "w"), indent=1)
    print("saved", args.out)


if __name__ == "__main__":
    main()
