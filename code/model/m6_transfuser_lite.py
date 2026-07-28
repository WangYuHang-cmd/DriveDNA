#!/usr/bin/env python3
"""
M6 — TransFuser-lite style conditioning (full data; no LiDAR/BEV).

Architecturally distinct from MCPP (which FiLMs the fused state): here the few-shot
style vector is injected INTO the control QUERY of a transformer decoder that attends
over fused CAN + video tokens — à la StyleDrive's TransFuser-Style. Tests whether
DriveDNA discriminates conditioning ARCHITECTURES, not just conditioning signals.

  memory  = [CAN tokens (50×d) ; DINOv2 video tokens (10×d)]
  query   = learned control query + proj(z_driver)      (style-in-query)
  decoder = 2-layer cross-attention → future [50, 2]

Eval (unseen drivers, route-disjoint k=5 support): generic (z=0) vs style-query, each
with and without video → 4 rows; PG = generic − personalized.

    python m6_transfuser_lite.py --epochs 15 [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from s4_mcpp import SupportEncoder

SEED = 20260709
torch.manual_seed(SEED)
L_HIST, L_FUT, NV, DV = 50, 50, 10, 768
TGT = ["aEgo", "actual_curvature"]


class TransFuserLite(nn.Module):
    def __init__(self, c_in, n_tgt, d=128, dz=64, nq=4):
        super().__init__()
        self.cproj = nn.Linear(c_in, d)
        self.cpos = nn.Parameter(torch.randn(1, L_HIST, d) * 0.02)
        self.vproj = nn.Linear(DV, d)
        self.vpos = nn.Parameter(torch.randn(1, NV, d) * 0.02)
        enc_l = nn.TransformerEncoderLayer(d, 4, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.fuse = nn.TransformerEncoder(enc_l, 2)          # joint CAN+video fusion
        self.query = nn.Parameter(torch.randn(1, nq, d) * 0.02)
        self.zproj = nn.Linear(dz, d)                        # style INTO the query
        dec_l = nn.TransformerDecoderLayer(d, 4, d * 4, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.dec = nn.TransformerDecoder(dec_l, 2)
        self.head = nn.Linear(d * nq, L_FUT * n_tgt)
        self.n_tgt, self.nq = n_tgt, nq

    def forward(self, x, vid, z):
        B = len(x)
        mem = self.cproj(x) + self.cpos
        if vid is not None:
            mem = torch.cat([mem, self.vproj(vid) + self.vpos], 1)
        mem = self.fuse(mem)
        q = self.query.expand(B, -1, -1) + self.zproj(z).unsqueeze(1)
        h = self.dec(q, mem)                                 # [B,nq,d]
        return self.head(h.flatten(1)).view(B, L_FUT, self.n_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(SEED)
    BASE = "."

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    V = np.load(os.path.join(args.bundle, "windows_vid.npy"), mmap_mode="r")
    VN = np.load(os.path.join(args.bundle, "windows_vid_n.npy"))
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    ch = json.load(open(os.path.join(args.bundle, "channels.json")))["channels"]
    tgt_ix = [ch.index(c) for c in TGT]
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    has = VN > 0
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy() & has)
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & has)
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:1500]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    norm = lambda a: (a.astype(np.float32) - mu) / sd

    by_drv = meta.iloc[tr_idx].groupby("driver").indices
    by_drv = {d: tr_idx[v] for d, v in by_drv.items() if len(v) >= 3}

    sup = SupportEncoder(X.shape[2]).to(dev)
    net = TransFuserLite(X.shape[2], len(TGT)).to(dev)
    opt = torch.optim.AdamW(list(sup.parameters()) + list(net.parameters()), lr=3e-4, weight_decay=1e-4)
    print(f"[M6] train {len(tr_idx)}  eval {len(ev_idx)}", flush=True)

    def gather(bidx, anchors):
        w = norm(X[np.sort(bidx)])[np.argsort(np.argsort(bidx))]
        xs = np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])
        ys = np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])
        vraw = V[np.sort(bidx)][np.argsort(np.argsort(bidx))]
        vid = np.stack([vraw[i, max(0, (a - L_HIST) // 5):max(0, (a - L_HIST) // 5) + NV]
                        for i, a in enumerate(anchors)]).astype(np.float32)
        return (torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                torch.from_numpy(vid).to(dev))

    def zsup(bidx, train):
        z = torch.zeros(len(bidx), 64, device=dev)
        for i, gi in enumerate(bidx):
            pool = by_drv.get(meta.driver.iat[gi], np.array([gi]))
            pool = pool[pool != gi]
            if train and (len(pool) == 0 or rng.random() < 0.3):
                continue
            pick = np.sort(rng.choice(pool, min(3, len(pool)), replace=False)) if len(pool) else [gi]
            z[i] = sup(torch.from_numpy(norm(X[pick])).to(dev)).mean(0)
        return z

    epochs = 2 if args.smoke else args.epochs
    steps = 80 if args.smoke else 400
    for ep in range(epochs):
        sup.train(); net.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 64, replace=False)
            anchors = rng.integers(L_HIST, 600 - L_FUT, len(bidx))
            xs, ys, vid = gather(bidx, anchors)
            if rng.random() < 0.3:                           # video dropout
                vid = None
            z = zsup(bidx, train=True)
            loss = nn.functional.mse_loss(net(xs, vid, z), ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  mse={tot/steps:.4f}", flush=True)

    sup.eval(); net.eval()
    me = meta.iloc[ev_idx].reset_index(drop=True)
    combos = {"CAN generic": (0, 0), "CAN style-q": (0, 1),
              "CAN+vid generic": (1, 0), "CAN+vid style-q": (1, 1)}
    errs = {c: [] for c in combos}
    with torch.no_grad():
        for d, g in me.groupby("driver"):
            rts = g.route.unique()
            if len(rts) < 2 or len(g) < 8:
                continue
            rng.shuffle(rts)
            sup_r = set(rts[: max(1, len(rts) // 2)])
            sup_pool = ev_idx[g.index[g.route.isin(sup_r)].to_numpy()]
            qry = ev_idx[g.index[~g.route.isin(sup_r)].to_numpy()]
            if len(sup_pool) < 1 or len(qry) < 4:
                continue
            anchors = rng.integers(L_HIST, 600 - L_FUT, len(qry))
            xs, ys, vid = gather(qry, anchors)
            pick = np.sort(rng.choice(sup_pool, min(5, len(sup_pool)), replace=False))
            zd = sup(torch.from_numpy(norm(X[pick])).to(dev)).mean(0, keepdim=True).expand(len(qry), -1)
            z0 = torch.zeros_like(zd)
            for cname, (u_v, u_z) in combos.items():
                e = net(xs, vid if u_v else None, zd if u_z else z0).cpu().numpy()
                errs[cname] += list(np.sqrt(((e - ys.cpu().numpy()) ** 2).mean((1, 2))))
    print("\n== M6 TransFuser-lite style-in-query (UNSEEN drivers, k=5) ==")
    out = {}
    for c, v in errs.items():
        out[c] = float(np.mean(v))
        print(f"  {c:18} RMSE={out[c]:.4f}")
    pg_can = (out["CAN generic"] - out["CAN style-q"]) / out["CAN generic"] * 100
    pg_mm = (out["CAN+vid generic"] - out["CAN+vid style-q"]) / out["CAN+vid generic"] * 100
    print(f"  PG (CAN): {pg_can:+.2f}%   PG (CAN+vid): {pg_mm:+.2f}%")
    if not args.smoke:
        out["pg_can_pct"], out["pg_mm_pct"] = pg_can, pg_mm
        json.dump(out, open(f"{BASE}/results/m6_transfuser_lite.json", "w"), indent=1)
        torch.save({"sup": sup.state_dict(), "net": net.state_dict(), "mu": mu, "sd": sd},
                   f"{BASE}/experiments/checkpoints/m6_transfuser_lite.pt")
        print("saved results/m6_transfuser_lite.json")


if __name__ == "__main__":
    main()
