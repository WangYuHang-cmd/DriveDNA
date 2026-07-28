#!/usr/bin/env python3
"""
S4 — MCPP: Multimodal Context-Conditioned Personalized Predictor (flagship; must-land #5).

z_scene(video, cross-attention) ⊕ z_beh(CAN GRU) ⊕ z_vehicle(model embedding) ⊕ z_driver(few-shot)
→ future realized behavior Ŷ_{t+1:t+H} (aEgo, actual_curvature; 50 steps).

ONE model trained with MODALITY DROPOUT (video→0, z_d→0 independently) → clean ablation at eval:
  CAN-only · +video · +driver · +video+driver     (vehicle embedding always on; its ablation = v_emb→0)
Video history = 10 frozen DINOv2 tokens (5 s @2 fps) attended by the CAN state (video = CONTEXT).

    python s4_mcpp.py --epochs 15 [--smoke]
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
L_HIST, L_FUT, NV, DV = 50, 50, 10, 768
TGT = ["aEgo", "actual_curvature"]


class SupportEncoder(nn.Module):
    def __init__(self, c_in, d=64):
        super().__init__()
        self.gru = nn.GRU(c_in, d, batch_first=True, bidirectional=True)
        self.out = nn.Linear(2 * d, d)

    def forward(self, x):
        return self.out(self.gru(x)[0].mean(1))


class MCPP(nn.Module):
    def __init__(self, c_in, n_tgt, n_veh, d=128, dz=64, dveh=16):
        super().__init__()
        self.enc = nn.GRU(c_in, d, num_layers=2, batch_first=True)
        self.vproj = nn.Linear(DV, d)
        self.xattn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.vgate = nn.Parameter(torch.tensor(0.1))
        self.vemb = nn.Embedding(n_veh, dveh)
        self.film = nn.Linear(dz + dveh, 2 * d)
        self.dec = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, L_FUT * n_tgt))
        self.n_tgt = n_tgt

    def forward(self, x, vid, z, veh):
        h = self.enc(x)[0][:, -1]                                   # z_beh
        vtok = self.vproj(vid)                                      # [B,10,d]
        att, _ = self.xattn(h.unsqueeze(1), vtok, vtok)             # z_scene via cross-attn
        h = h + self.vgate * att.squeeze(1)
        cond = torch.cat([z, self.vemb(veh)], -1)
        g, b = self.film(cond).chunk(2, -1)
        h = h * (1 + g) + b
        return self.dec(h).view(-1, L_FUT, self.n_tgt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./experiments/checkpoints/s4_mcpp.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    V = np.load(os.path.join(args.bundle, "windows_vid.npy"), mmap_mode="r")
    VN = np.load(os.path.join(args.bundle, "windows_vid_n.npy"))
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

    by_drv_tr = meta.iloc[tr_idx].groupby("driver").indices
    by_drv_tr = {d: tr_idx[v] for d, v in by_drv_tr.items() if len(v) >= 3}

    sup = SupportEncoder(X.shape[2]).to(dev)
    net = MCPP(X.shape[2], len(TGT), n_veh=len(veh_ids)).to(dev)
    opt = torch.optim.AdamW(list(sup.parameters()) + list(net.parameters()), lr=3e-4, weight_decay=1e-4)
    norm = lambda a: (a.astype(np.float32) - mu) / sd

    def gather(bidx, anchors):
        """x [B,50,C], y [B,50,K], vid [B,10,768]."""
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
            pool = by_drv_tr.get(meta.driver.iat[gi], np.array([gi]))
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
            if rng.random() < 0.3:                       # modality dropout: video
                vid = torch.zeros_like(vid)
            z = zsup(bidx, train=True)                   # condition dropout inside
            vb = torch.from_numpy(veh[bidx]).to(dev)
            loss = nn.functional.mse_loss(net(xs, vid, z, vb), ys)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach())
        print(f"epoch {ep+1}/{epochs}  mse={tot/steps:.4f}", flush=True)

    # ---- eval on unseen drivers: 4-way modality ablation (route-disjoint support) ----
    sup.eval(); net.eval()
    me = meta.iloc[ev_idx].reset_index(drop=True)
    combos = {"CAN": (False, False), "CAN+vid": (True, False),
              "CAN+drv": (False, True), "CAN+vid+drv": (True, True)}
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
            vb = torch.from_numpy(veh[qry]).to(dev)
            pick = np.sort(rng.choice(sup_pool, min(5, len(sup_pool)), replace=False))
            zd = sup(torch.from_numpy(norm(X[pick])).to(dev)).mean(0, keepdim=True).expand(len(qry), -1)
            z0 = torch.zeros_like(zd)
            v0 = torch.zeros_like(vid)
            for cname, (use_v, use_d) in combos.items():
                e = net(xs, vid if use_v else v0, zd if use_d else z0, vb).cpu().numpy()
                errs[cname] += list(np.sqrt(((e - ys.cpu().numpy()) ** 2).mean((1, 2))))
    print("\n== MCPP 4-way modality ablation on UNSEEN drivers (RMSE @5 s; k=5 support) ==")
    base = np.mean(errs["CAN"])
    for c, v in errs.items():
        m = np.mean(v)
        print(f"  {c:12} RMSE={m:.4f}  Δ vs CAN-only {(base-m)/base*100:+.1f}%")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"sup": sup.state_dict(), "net": net.state_dict(), "mu": mu, "sd": sd}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
