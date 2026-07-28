#!/usr/bin/env python3
"""
S3 — Few-shot personalized future realized-behavior prediction (T3; must-land #4).

Task (per plan): Ŷ_{t+1:t+H} = f(X_{t−5s:t}, S_d) — 5 s history @10 Hz (50 steps) →
future realized motion over 1/3/5 s (10/30/50 steps; 3 s primary).
Targets (Tier A): aEgo, actual_curvature (+vEgo for context error).

ONE model, two paths via condition-dropout (p=0.3 → z_d=0 during training):
  generic       Ŷ = f(X, 0)
  personalized  Ŷ = f(X, z_d),  z_d = mean support-window embeddings (few-shot, route-disjoint)
→ Personalization Gain PG = RMSE_generic − RMSE_personalized with identical capacity.
Eval on UNSEEN drivers; PG reported per-horizon AND stratified by T1 primitives.

    python s3_personalized.py --bundle .../colab_bundle --epochs 15 [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

SEED = 20260709
torch.manual_seed(SEED)
L_HIST, L_FUT = 50, 50            # 5 s history, 5 s future (evaluate 10/30/50)
TGT = ["aEgo", "actual_curvature"]


class SupportEncoder(nn.Module):
    """Window [600,C] → 64-d driver embedding (GRU + mean-pool over time)."""
    def __init__(self, c_in, d=64):
        super().__init__()
        self.gru = nn.GRU(c_in, d, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, d)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.out(h.mean(1))


class PredictorTF(nn.Module):
    """V4 variant: Transformer encoder over history tokens + FiLM, same interface."""
    def __init__(self, c_in, n_tgt, d=128, dz=64, depth=3, heads=4):
        super().__init__()
        self.proj = nn.Linear(c_in, d)
        self.pos = nn.Parameter(torch.randn(1, 50, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, depth)
        self.film = nn.Linear(dz, 2 * d)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, L_FUT * n_tgt))
        self.n_tgt = n_tgt

    def forward(self, x, z):
        h = self.enc(self.proj(x) + self.pos).mean(1)   # [B,d]
        g, b = self.film(z).chunk(2, -1)
        h = h * (1 + g) + b
        return self.dec(h).view(-1, L_FUT, self.n_tgt)


class Predictor(nn.Module):
    """History [50,C] + z_d → future [50, n_tgt] with FiLM conditioning."""
    def __init__(self, c_in, n_tgt, d=128, dz=64):
        super().__init__()
        self.enc = nn.GRU(c_in, d, num_layers=2, batch_first=True)
        self.film = nn.Linear(dz, 2 * d)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                 nn.Linear(2 * d, L_FUT * n_tgt))
        self.n_tgt = n_tgt

    def forward(self, x, z):
        h = self.enc(x)[0][:, -1]                       # [B,d]
        g, b = self.film(z).chunk(2, -1)
        h = h * (1 + g) + b
        return self.dec(h).view(-1, L_FUT, self.n_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--use-s1", action="store_true", help="frozen S1-SupCon embedding as z_d (dz=128)")
    ap.add_argument("--out", default="./experiments/checkpoints/s3_pred.pt")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--arch", default="gru", choices=["gru", "transformer"])
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    ch = json.load(open(os.path.join(args.bundle, "channels.json")))["channels"]
    tgt_ix = [ch.index(c) for c in TGT]
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    tr_drv = set(folds["train"])
    ev_drv = set(folds["val"]) | set(folds["few_shot_heldout"])

    tr_idx = np.flatnonzero(meta.driver.isin(tr_drv).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(ev_drv).to_numpy())
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:1500]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0)
    sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    tgt_mu, tgt_sd = mu[tgt_ix], sd[tgt_ix]

    # support-window lookup (train + eval; support drawn per driver, route-disjoint at eval)
    by_drv_tr = meta.iloc[tr_idx].groupby("driver").indices
    by_drv_tr = {d: tr_idx[v] for d, v in by_drv_tr.items() if len(v) >= 3}

    if args.use_s1:
        from s1_supcon import Encoder as S1Enc
        ck = torch.load("./experiments/checkpoints/s1_supcon.pt",
                        map_location=dev, weights_only=False)
        s1 = S1Enc(c_in=X.shape[2]).to(dev); s1.load_state_dict(ck["model"]); s1.eval()
        for p in s1.parameters():
            p.requires_grad_(False)
        s1_mu, s1_sd = ck["mu"], ck["sd"]
        DZ = 128
        sup_enc = lambda w: s1((w * torch.from_numpy(sd / s1_sd).to(dev)) +
                               torch.from_numpy((mu - s1_mu) / s1_sd).to(dev).float())
        params = []
    else:
        DZ = 64
        sup_mod = SupportEncoder(X.shape[2]).to(dev)
        sup_enc = sup_mod
        params = list(sup_mod.parameters())
    pred = (PredictorTF if args.arch == "transformer" else Predictor)(X.shape[2], len(TGT), dz=DZ).to(dev)
    opt = torch.optim.AdamW(params + list(pred.parameters()), lr=3e-4, weight_decay=1e-4)

    def norm(a):  # [B,600,C] float32 standardized
        return (a.astype(np.float32) - mu) / sd

    def batch(bidx, train=True):
        """Return history x [B,50,C], target y [B,50,K] (standardized), z_d [B,64]."""
        w = norm(X[np.sort(bidx)])
        order = np.argsort(np.argsort(bidx))
        w = w[order]
        anchors = rng.integers(L_HIST, 600 - L_FUT, len(bidx))
        xs = np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])
        ys = np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])
        xs = torch.from_numpy(xs).to(dev)
        ys = torch.from_numpy(ys).to(dev)
        # few-shot support: other windows of the same driver
        zs = []
        for i, gi in enumerate(bidx):
            d = meta.driver.iat[gi]
            pool = by_drv_tr.get(d, np.array([gi]))
            pool = pool[pool != gi]
            if train and (len(pool) == 0 or rng.random() < 0.3):   # condition dropout → generic path
                zs.append(None)
            else:
                pick = rng.choice(pool, min(3, len(pool)), replace=False) if len(pool) else [gi]
                zs.append(np.sort(pick))
        z = torch.zeros(len(bidx), DZ, device=dev)
        for i, pick in enumerate(zs):
            if pick is not None:
                sw = torch.from_numpy(norm(X[pick])).to(dev)
                z[i] = sup_enc(sw).mean(0)
        return xs, ys, z

    epochs = 2 if args.smoke else args.epochs
    steps = 80 if args.smoke else 400
    for ep in range(epochs):
        (sup_mod.train() if not args.use_s1 else None); pred.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 64, replace=False)
            xs, ys, z = batch(bidx, train=True)
            loss = nn.functional.mse_loss(pred(xs, z), ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  mse={tot/steps:.4f}", flush=True)

    # ---------- evaluation on UNSEEN drivers: generic vs few-shot, route-disjoint ----------
    (sup_mod.eval() if not args.use_s1 else None); pred.eval()
    me = meta.iloc[ev_idx].reset_index(drop=True)
    res = {k: {"g": [], "p": []} for k in (1, 3, 5, 10)}    # support size (windows ≈ minutes)
    prim_cols = [c for c in meta.columns if c.startswith("p_")]
    strat = {c: {"g": [], "p": []} for c in prim_cols}
    H = {"1s": 10, "3s": 30, "5s": 50}
    per_h = {h: {"g": [], "p": []} for h in H}

    with torch.no_grad():
        for d, g in me.groupby("driver"):
            rts = g.route.unique()
            if len(rts) < 2 or len(g) < 8:
                continue
            rng.shuffle(rts)
            sup_r = set(rts[: max(1, len(rts) // 2)])
            sup_pool = ev_idx[g.index[g.route.isin(sup_r)].to_numpy()]
            qry = g.index[~g.route.isin(sup_r)].to_numpy()
            if len(sup_pool) < 1 or len(qry) < 4:
                continue
            qidx = ev_idx[qry]
            w = norm(X[np.sort(qidx)])[np.argsort(np.argsort(qidx))]
            anchors = rng.integers(L_HIST, 600 - L_FUT, len(qidx))
            xs = torch.from_numpy(np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])).to(dev)
            ys = np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])
            e_g = pred(xs, torch.zeros(len(qidx), DZ, device=dev)).cpu().numpy()
            for k in res:
                pick = np.sort(rng.choice(sup_pool, min(k, len(sup_pool)), replace=False))
                z = sup_enc(torch.from_numpy(norm(X[pick])).to(dev)).mean(0, keepdim=True)
                e_p = pred(xs, z.expand(len(qidx), -1)).cpu().numpy()
                rg = np.sqrt(((e_g - ys) ** 2).mean((1, 2)))
                rp = np.sqrt(((e_p - ys) ** 2).mean((1, 2)))
                res[k]["g"] += list(rg); res[k]["p"] += list(rp)
                if k == 5:   # stratification + horizons at the 5-window operating point
                    for h, n in H.items():
                        per_h[h]["g"] += list(np.sqrt(((e_g[:, :n] - ys[:, :n]) ** 2).mean((1, 2))))
                        per_h[h]["p"] += list(np.sqrt(((e_p[:, :n] - ys[:, :n]) ** 2).mean((1, 2))))
                    for c in prim_cols:
                        mrows = (g.iloc[np.argsort(np.argsort(qry))][c] == 1).to_numpy()
                        if mrows.sum() >= 2:
                            strat[c]["g"] += list(rg[mrows]); strat[c]["p"] += list(rp[mrows])

    print("\n== S3 few-shot personalization on UNSEEN drivers (RMSE, standardized units) ==")
    print(f"{'k(sup)':>7} {'generic':>9} {'personal':>9} {'PG':>8} {'PG%':>6}")
    for k, r in res.items():
        if not r["g"]:
            continue
        g_, p_ = np.mean(r["g"]), np.mean(r["p"])
        print(f"{k:>7} {g_:>9.4f} {p_:>9.4f} {g_-p_:>8.4f} {(g_-p_)/g_*100:>5.1f}%")
    print("\n== per-horizon (k=5) ==")
    for h, r in per_h.items():
        g_, p_ = np.mean(r["g"]), np.mean(r["p"])
        print(f"  {h:>3}: generic {g_:.4f}  personal {p_:.4f}  PG {(g_-p_)/g_*100:+.1f}%")
    print("\n== PG stratified by T1 primitive (k=5, windows tagged +1) ==")
    for c, r in strat.items():
        if len(r["g"]) >= 30:
            g_, p_ = np.mean(r["g"]), np.mean(r["p"])
            print(f"  {c:20} n={len(r['g']):>5}  PG {(g_-p_)/g_*100:+.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"sup_enc": (sup_mod.state_dict() if not args.use_s1 else None), "pred": pred.state_dict(),
                "mu": mu, "sd": sd}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
    rng_eval = np.random.default_rng(20260709)  # eval protocol FIXED across seeds
