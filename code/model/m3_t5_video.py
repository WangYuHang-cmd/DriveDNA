#!/usr/bin/env python3
"""
M3 — temporal video cross-attention for T5 event forecasting (full data).

T5: given 5 s CAN history, does an event (hard-brake / sharp-steer onset) start within
the next 5 s? CAN-only baseline = t5_events.py (AUROC .846). Here video enters as
causal context: DINOv2 frame tokens STRICTLY BEFORE the anchor (never the future — the
event must not be visible), attended by the CAN state. Three eval settings from ONE
model (context-length augmentation at train time):
  CAN-only · CAN + video(5 s) · CAN + video(full past, up to 60 s)
Directly tests "video helps T5 most" (lead decel / cut-in / red light are visual) and
whether longer visual context helps.

    python m3_t5_video.py [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import sys
BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")
from t5_events import find_onsets  # noqa: E402

SEED = 20260709
torch.manual_seed(SEED)
L_H, L_F = 50, 50
RATE = 10.0
VFPS = 2          # DINOv2 tokens per second
NVMAX = 120


def build(meta_idx, X, acc_ix, cur_ix, rng, per_event=6):
    """Like t5_events.build but also returns (window_gi, anchor) per sample."""
    xs, ys, leads, kinds, gis, ancs = [], [], [], [], [], []
    for gi in meta_idx:
        w = X[gi].astype(np.float32)
        acc = w[:, acc_ix]
        cr = np.abs(np.diff(w[:, cur_ix], prepend=w[0, cur_ix])) * RATE
        events = [(i, "brake") for i in find_onsets(acc, -2.0, -1.0)] + \
                 [(i, "steer") for i in find_onsets(-cr, -0.08, -0.04)]
        used = set()
        for oi, kind in events:
            if oi < L_H + 5:
                continue
            for lead in (5, 10, 20, 30, 40, 50):
                a = oi - lead
                if a < L_H:
                    continue
                xs.append(w[a - L_H:a]); ys.append(1); leads.append(lead / RATE)
                kinds.append(kind); gis.append(gi); ancs.append(a)
                used.add(a // 25)
        ev_set = set(oi for oi, _ in events)
        for _ in range(per_event):
            a = int(rng.integers(L_H, 600 - L_F))
            if any(a < oi <= a + L_F for oi in ev_set) or (a // 25) in used:
                continue
            xs.append(w[a - L_H:a]); ys.append(0); leads.append(np.nan)
            kinds.append("neg"); gis.append(gi); ancs.append(a)
    return (np.stack(xs) if xs else np.zeros((0, L_H, X.shape[2]), np.float32),
            np.array(ys), np.array(leads), np.array(kinds),
            np.array(gis), np.array(ancs))


class VidT5(nn.Module):
    def __init__(self, c_in, dv=768, d=96):
        super().__init__()
        self.gru = nn.GRU(c_in, d, batch_first=True)
        self.vproj = nn.Linear(dv, d)
        self.xattn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.vgate = nn.Parameter(torch.tensor(0.1))
        self.fc = nn.Linear(d, 1)

    def forward(self, x, vid=None, vmask=None):
        """vid [B,120,768]; vmask True = PAD/disallowed (causal + validity)."""
        h = self.gru(x)[0][:, -1]
        if vid is not None:
            ok = ~vmask.all(1)                               # rows with ≥1 visible frame
            if ok.any():
                att, _ = self.xattn(h[ok].unsqueeze(1), self.vproj(vid[ok]),
                                    self.vproj(vid[ok]), key_padding_mask=vmask[ok])
                h = h.clone()
                h[ok] = h[ok] + self.vgate * att.squeeze(1)
        return self.fc(h).squeeze(-1)


def vid_mask(ancs, vn, mode):
    """True = masked. Causal: token j visible iff it ends at/before the anchor
    (dinov2: frame at step j*5 < a; vjepa2: clip [j*40, j*40+40) fully before a);
    mode '5s' limits to tokens overlapping the 5 s history."""
    B = len(ancs)
    m = np.ones((B, NVMAX), bool)
    for i, (a, n) in enumerate(zip(ancs, vn)):
        if STEP == 5:
            hi = min(int(n), (a + 4) // 5)                   # frame starts before a
            lo = 0 if mode == "full" else max(0, (a - L_H) // 5)
        else:
            hi = min(int(n), a // STEP)                      # clips fully before a
            lo = 0 if mode == "full" else max(0, hi - 2)     # last ~2 clips ≈ 8 s
        if hi > lo:
            m[i, lo:hi] = False
    return m


def main():
    from sklearn.metrics import roc_auc_score, average_precision_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default=f"{BASE}/data/colab_bundle")
    ap.add_argument("--video", default="dinov2", choices=["dinov2", "vjepa2", "siglip2", "dinov3"])
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(SEED)

    X = np.load(f"{args.bundle}/windows_x.npy", mmap_mode="r")
    global NVMAX, STEP, DVID
    if args.video == "dinov2":
        V = np.load(f"{args.bundle}/windows_vid.npy", mmap_mode="r")
        VN = np.load(f"{args.bundle}/windows_vid_n.npy")
        NVMAX, STEP, DVID = 120, 5, 768        # frame j at step j*5 (2 fps)
    elif args.video == "siglip2":
        V = np.load(f"{args.bundle}/windows_vids2.npy", mmap_mode="r")
        VN = np.load(f"{args.bundle}/windows_vids2_n.npy")
        NVMAX, STEP, DVID = 120, 5, 768        # same geometry as dinov2
    elif args.video == "dinov3":
        V = np.load(f"{args.bundle}/windows_vidd3.npy", mmap_mode="r")
        VN = np.load(f"{args.bundle}/windows_vidd3_n.npy")
        NVMAX, STEP, DVID = 120, 5, 768        # same geometry as dinov2
    else:
        V = np.load(f"{args.bundle}/windows_vidj.npy", mmap_mode="r")
        VN = np.load(f"{args.bundle}/windows_vidj_n.npy")
        NVMAX, STEP, DVID = 15, 40, 1024       # clip j spans steps [j*40, j*40+40)
    meta = pd.read_parquet(f"{args.bundle}/windows_meta.parquet")
    ch = json.load(open(f"{args.bundle}/channels.json"))["channels"]
    acc_ix, cur_ix = ch.index("aEgo"), ch.index("actual_curvature")
    folds = json.load(open(f"{args.bundle}/driver_folds.json"))
    has = VN > 0
    tr_i = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy() & has)
    ev_i = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & has)
    if args.smoke:
        tr_i = rng.choice(tr_i, min(4000, len(tr_i)), replace=False)
        ev_i = rng.choice(ev_i, min(2000, len(ev_i)), replace=False)
    print(f"windows: train {len(tr_i)}  eval {len(ev_i)} (FULL={not args.smoke})", flush=True)
    print("building event datasets ...", flush=True)
    Xtr, ytr, _, _, gtr, atr = build(tr_i, X, acc_ix, cur_ix, rng)
    Xev, yev, lev, kev, gev, aev = build(ev_i, X, acc_ix, cur_ix, rng)
    print(f"train {len(ytr)} ({ytr.mean()*100:.0f}% pos)  eval {len(yev)} ({yev.mean()*100:.0f}% pos)", flush=True)

    mu = Xtr.reshape(-1, X.shape[2]).mean(0); sd = Xtr.reshape(-1, X.shape[2]).std(0) + 1e-3
    net = VidT5(X.shape[2], dv=DVID).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.from_numpy((Xtr - mu) / sd)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    vn_tr = VN[gtr]

    epochs = 2 if args.smoke else args.epochs
    for ep in range(epochs):
        net.train(); perm = torch.randperm(len(ytr_t)); tot = 0.0; nb = 0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256].numpy()
            x = Xtr_t[b].to(dev)
            r = rng.random()
            if r < 0.25:                                     # context-length augmentation
                logit = net(x)
            else:
                mode = "5s" if r < 0.6 else "full"
                vid = torch.from_numpy(V[np.sort(gtr[b])].astype(np.float32)[np.argsort(np.argsort(gtr[b]))]).to(dev)
                vm = torch.from_numpy(vid_mask(atr[b], vn_tr[b], mode)).to(dev)
                logit = net(x, vid, vm)
            loss = nn.functional.binary_cross_entropy_with_logits(logit, ytr_t[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"epoch {ep+1}/{epochs}  bce={tot/nb:.4f}", flush=True)

    net.eval()
    Xev_t = torch.from_numpy((Xev - mu) / sd)
    vn_ev = VN[gev]
    res = {}
    for setting in ("CAN", "CAN+vid5s", "CAN+vidfull"):
        s = []
        with torch.no_grad():
            for i in range(0, len(yev), 512):
                b = slice(i, i + 512)
                x = Xev_t[b].to(dev)
                if setting == "CAN":
                    logit = net(x)
                else:
                    gb = gev[b]
                    vid = torch.from_numpy(V[np.sort(gb)].astype(np.float32)[np.argsort(np.argsort(gb))]).to(dev)
                    mode = "5s" if setting == "CAN+vid5s" else "full"
                    vm = torch.from_numpy(vid_mask(aev[b], vn_ev[b], mode)).to(dev)
                    logit = net(x, vid, vm)
                s.append(torch.sigmoid(logit).cpu().numpy())
        s = np.concatenate(s)
        row = {"auroc": float(roc_auc_score(yev, s)), "ap": float(average_precision_score(yev, s))}
        for kind in ("brake", "steer"):
            m = (kev == kind) | (kev == "neg")
            if (kev[m] == kind).sum() > 50:
                row[kind] = float(roc_auc_score(yev[m] * (kev[m] == kind), s[m]))
        row["by_lead"] = {}
        for lo, hi in [(0, 1.01), (1.01, 3.01), (3.01, 5.01)]:
            mpos = (yev == 1) & (lev > lo) & (lev <= hi)
            if mpos.sum() > 50:
                mm = mpos | (yev == 0)
                row["by_lead"][f"{lo:.0f}-{hi:.0f}s"] = float(roc_auc_score(yev[mm], s[mm]))
        res[setting] = row
        print(f"  {setting:12} AUROC={row['auroc']:.3f}  AP={row['ap']:.3f}  "
              f"brake={row.get('brake', float('nan')):.3f} steer={row.get('steer', float('nan')):.3f}  "
              f"lead: {row['by_lead']}", flush=True)
    if not args.smoke:
        json.dump(res, open(f"{BASE}/results/m3_t5_video_{args.video}.json", "w"), indent=1)
        torch.save({"net": net.state_dict(), "mu": mu, "sd": sd},
                   f"{BASE}/experiments/checkpoints/m3_t5_video_{args.video}.pt")
        print(f"saved results/m3_t5_video_{args.video}.json")


if __name__ == "__main__":
    main()
