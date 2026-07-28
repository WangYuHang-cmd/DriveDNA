#!/usr/bin/env python3
"""
M2 — MCPP full 6-way ablation (flagship multimodal baseline, full data).

Conditioning ladder (ONE model, modality dropout → clean eval-time ablation):
  1. CAN                          2. CAN+veh
  3. CAN+veh+vid                  4. CAN+veh+drv
  5. CAN+veh+vid+drv              6. CAN+veh+vid+drv+res
where res = residual-style driver conditioning: the support encoder additionally sees
the driver's population-residual sequences (true − generic-CAN prediction on support
windows), i.e. "style = what remains after the population model" (S2) as a signal.

--video dinov2  (default): 10× 768-d per-frame tokens (5 s @2 fps)
--video vjepa2: 2–3× 1024-d temporal clip tokens (4 s clips) — run after M1 alignment
    python m2_mcpp6.py --epochs 15 [--smoke] [--video vjepa2]
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
L_HIST, L_FUT = 50, 50
TGT = ["aEgo", "actual_curvature"]


class MCPP6(nn.Module):
    def __init__(self, c_in, n_tgt, n_veh, dv=768, d=128, dz=64, dveh=16, dres=32):
        super().__init__()
        self.enc = nn.GRU(c_in, d, num_layers=2, batch_first=True)
        self.vproj = nn.Linear(dv, d)
        self.xattn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.vgate = nn.Parameter(torch.tensor(0.1))
        self.vemb = nn.Embedding(n_veh, dveh)
        self.resenc = nn.GRU(n_tgt, dres, batch_first=True)
        self.film = nn.Linear(dz + dveh + dres, 2 * d)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, L_FUT * n_tgt))
        self.n_tgt = n_tgt

    def encode_res(self, r):                                  # r [B,50,n_tgt] residual seq
        return self.resenc(r)[0][:, -1]

    def forward(self, x, vid, z, vehvec, zres):
        h = self.enc(x)[0][:, -1]
        if vid is not None:
            vtok = self.vproj(vid)
            att, _ = self.xattn(h.unsqueeze(1), vtok, vtok)
            h = h + self.vgate * att.squeeze(1)
        cond = torch.cat([z, vehvec, zres], -1)
        g, b = self.film(cond).chunk(2, -1)
        h = h * (1 + g) + b
        return self.dec(h).view(-1, L_FUT, self.n_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--video", default="dinov2", choices=["dinov2", "vjepa2", "siglip2", "dinov3"])
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--veh-dropout", type=float, default=0.2,
                    help="0 = vehicle embedding always on (S4 recipe)")
    ap.add_argument("--res-start", type=int, default=0,
                    help="epoch at which the residual pathway activates")
    ap.add_argument("--tag", default="", help="suffix for output files")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    BASE = "."

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    if args.video == "dinov2":
        V = np.load(os.path.join(args.bundle, "windows_vid.npy"), mmap_mode="r")
        VN = np.load(os.path.join(args.bundle, "windows_vid_n.npy"))
        NV, DV, VFPS = 10, 768, 2.0            # tokens covering the 5 s history
    elif args.video == "siglip2":
        V = np.load(os.path.join(args.bundle, "windows_vids2.npy"), mmap_mode="r")
        VN = np.load(os.path.join(args.bundle, "windows_vids2_n.npy"))
        NV, DV, VFPS = 10, 768, 2.0            # same geometry as dinov2
    elif args.video == "dinov3":
        V = np.load(os.path.join(args.bundle, "windows_vidd3.npy"), mmap_mode="r")
        VN = np.load(os.path.join(args.bundle, "windows_vidd3_n.npy"))
        NV, DV, VFPS = 10, 768, 2.0            # same geometry as dinov2
    else:
        V = np.load(os.path.join(args.bundle, "windows_vidj.npy"), mmap_mode="r")
        VN = np.load(os.path.join(args.bundle, "windows_vidj_n.npy"))
        NV, DV, VFPS = 3, 1024, 0.25           # 4 s clips → 2–3 overlap the history
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    ch = json.load(open(os.path.join(args.bundle, "channels.json")))["channels"]
    tgt_ix = [ch.index(c) for c in TGT]
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    veh_ids = {m: i for i, m in enumerate(sorted(meta.model_canon.unique()))}
    veh = meta.model_canon.map(veh_ids).to_numpy()

    has_vid = VN > 0
    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy() & has_vid)
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & has_vid)
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:1500]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    norm = lambda a: (a.astype(np.float32) - mu) / sd

    by_drv_tr = meta.iloc[tr_idx].groupby("driver").indices
    by_drv_tr = {d: tr_idx[v] for d, v in by_drv_tr.items() if len(v) >= 3}

    sup = SupportEncoder(X.shape[2]).to(dev)
    net = MCPP6(X.shape[2], len(TGT), n_veh=len(veh_ids), dv=DV).to(dev)
    opt = torch.optim.AdamW(list(sup.parameters()) + list(net.parameters()), lr=3e-4, weight_decay=1e-4)
    print(f"[M2 {args.video}] train {len(tr_idx)}  eval {len(ev_idx)}  "
          f"video tokens {NV}x{DV}", flush=True)

    def vid_slice(vraw, anchors):
        if args.video in ("dinov2", "siglip2", "dinov3"):
            return np.stack([vraw[i, max(0, (a - L_HIST) // 5):max(0, (a - L_HIST) // 5) + NV]
                             for i, a in enumerate(anchors)]).astype(np.float32)
        out = np.zeros((len(anchors), NV, DV), np.float32)
        for i, a in enumerate(anchors):                       # clips overlapping [a-50, a] (0.1 s steps; clip = 40)
            c0 = max(0, (a - L_HIST) // 40)
            sl = vraw[i, c0:c0 + NV].astype(np.float32)
            out[i, :len(sl)] = sl
        return out

    def gather(bidx, anchors):
        w = norm(X[np.sort(bidx)])[np.argsort(np.argsort(bidx))]
        xs = np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])
        ys = np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])
        vraw = V[np.sort(bidx)][np.argsort(np.argsort(bidx))]
        vid = vid_slice(vraw, anchors)
        return (torch.from_numpy(xs).to(dev), torch.from_numpy(ys).to(dev),
                torch.from_numpy(vid).to(dev))

    def gen_pred(xs):
        """Generic CAN-only prediction (all conditioning off) for residual computation."""
        B = len(xs)
        return net(xs, None, torch.zeros(B, 64, device=dev),
                   torch.zeros(B, 16, device=dev), torch.zeros(B, 32, device=dev))

    def support_z(picks, train_drop=False):
        """z_d and z_res from support windows (fixed mid-window anchor)."""
        w = norm(X[picks])
        zd = sup(torch.from_numpy(w).to(dev)).mean(0)
        xs = torch.from_numpy(w[:, 300 - L_HIST:300]).to(dev)
        ys = torch.from_numpy(w[:, 300:300 + L_FUT][:, :, tgt_ix]).to(dev)
        with torch.no_grad():
            r = ys - gen_pred(xs)
        zr = net.encode_res(r).mean(0)
        return zd, zr

    epochs = 2 if args.smoke else args.epochs
    steps = 80 if args.smoke else 400
    for ep in range(epochs):
        sup.train(); net.train()
        tot = 0.0
        for _ in range(steps):
            bidx = rng.choice(tr_idx, 64, replace=False)
            anchors = rng.integers(L_HIST, 600 - L_FUT, len(bidx))
            xs, ys, vid = gather(bidx, anchors)
            drop_v = rng.random() < 0.3
            drop_veh = args.veh_dropout > 0 and rng.random() < args.veh_dropout
            use_res = ep >= args.res_start
            zs = torch.zeros(len(bidx), 64, device=dev)
            zrs = torch.zeros(len(bidx), 32, device=dev)
            for i, gi in enumerate(bidx):
                pool = by_drv_tr.get(meta.driver.iat[gi], np.array([gi]))
                pool = pool[pool != gi]
                if len(pool) == 0 or rng.random() < 0.3:      # driver-condition dropout
                    continue
                pick = np.sort(rng.choice(pool, min(3, len(pool)), replace=False))
                zd, zr = support_z(pick)
                zs[i] = zd
                if use_res and rng.random() >= 0.3:            # residual-condition dropout
                    zrs[i] = zr
            vehvec = net.vemb(torch.from_numpy(veh[bidx]).to(dev))
            if drop_veh:
                vehvec = torch.zeros_like(vehvec)
            loss = nn.functional.mse_loss(
                net(xs, None if drop_v else vid, zs, vehvec, zrs), ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  mse={tot/steps:.4f}", flush=True)

    # ---- 6-way eval (unseen drivers, route-disjoint support, k=5) ----
    sup.eval(); net.eval()
    me = meta.iloc[ev_idx].reset_index(drop=True)
    combos = {"CAN": (0, 0, 0, 0), "CAN+veh": (1, 0, 0, 0), "CAN+veh+vid": (1, 1, 0, 0),
              "CAN+veh+drv": (1, 0, 1, 0), "CAN+veh+vid+drv": (1, 1, 1, 0),
              "CAN+veh+vid+drv+res": (1, 1, 1, 1)}
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
            zd1, zr1 = support_z(pick)
            zd = zd1.unsqueeze(0).expand(len(qry), -1)
            zr = zr1.unsqueeze(0).expand(len(qry), -1)
            vehvec = net.vemb(torch.from_numpy(veh[qry]).to(dev))
            z0, r0, v0 = torch.zeros_like(zd), torch.zeros_like(zr), torch.zeros_like(vehvec)
            for cname, (u_veh, u_vid, u_drv, u_res) in combos.items():
                e = net(xs, vid if u_vid else None, zd if u_drv else z0,
                        vehvec if u_veh else v0, zr if u_res else r0).cpu().numpy()
                errs[cname] += list(np.sqrt(((e - ys.cpu().numpy()) ** 2).mean((1, 2))))
    print(f"\n== M2 MCPP 6-way ablation ({args.video}; UNSEEN drivers, k=5) ==")
    base = np.mean(errs["CAN"])
    out = {}
    for c, v in errs.items():
        m = float(np.mean(v))
        out[c] = {"rmse": m, "delta_pct": float((base - m) / base * 100)}
        print(f"  {c:22} RMSE={m:.4f}  Δ vs CAN {(base-m)/base*100:+.2f}%")
    ckpt = f"./experiments/checkpoints/m2_mcpp6_{args.video}{args.tag}.pt"
    torch.save({"sup": sup.state_dict(), "net": net.state_dict(), "mu": mu, "sd": sd}, ckpt)
    if not args.smoke:
        json.dump(out, open(f"./results/m2_mcpp6_{args.video}{args.tag}.json", "w"), indent=1)
    print("saved", ckpt)


if __name__ == "__main__":
    main()
