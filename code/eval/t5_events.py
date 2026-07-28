#!/usr/bin/env python3
"""
T5 — Behavior-event forecasting (seq-to-label): given 5 s history, does the event start
within the next 5 s? Events: hard-brake onset (aEgo < −2 after ≥1 s calmer) and sharp-steer
onset (|curvature-rate| spike). Reports AP/AUROC on UNSEEN drivers + AUROC by lead-time bin.
Candidate taxonomy (labels rule-derived; not frozen).
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = "."
BUNDLE = f"{BASE}/data/colab_bundle"
SEED = 20260709
torch.manual_seed(SEED)
L_H, L_F = 50, 50
RATE = 10.0


def find_onsets(sig, thresh, calm, calm_win=10):
    """Indices where sig first crosses below thresh after calm_win steps above calm."""
    on = []
    below = sig < thresh
    for i in np.flatnonzero(below):
        if i >= calm_win and np.all(sig[i - calm_win:i] > calm):
            on.append(i)
    return on


def build(meta_idx, X, acc_ix, cur_ix, rng, per_event=6):
    xs, ys, leads, kinds = [], [], [], []
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
                xs.append(w[a - L_H:a]); ys.append(1); leads.append(lead / RATE); kinds.append(kind)
                used.add(a // 25)
        # negatives: anchors with no event in next 5 s
        ev_set = set(oi for oi, _ in events)
        for _ in range(per_event):
            a = int(rng.integers(L_H, 600 - L_F))
            if any(a < oi <= a + L_F for oi in ev_set) or (a // 25) in used:
                continue
            xs.append(w[a - L_H:a]); ys.append(0); leads.append(np.nan); kinds.append("neg")
    return (np.stack(xs) if xs else np.zeros((0, L_H, X.shape[2]), np.float32),
            np.array(ys), np.array(leads), np.array(kinds))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-tr", type=int, default=0, help="0 = ALL train windows")
    ap.add_argument("--cap-ev", type=int, default=0, help="0 = ALL eval windows")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    from sklearn.metrics import roc_auc_score, average_precision_score
    rng = np.random.default_rng(args.seed)
    X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{BUNDLE}/windows_meta.parquet")
    ch = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
    acc_ix, cur_ix = ch.index("aEgo"), ch.index("actual_curvature")
    folds = json.load(open(f"{BUNDLE}/driver_folds.json"))
    tr_i = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    ev_i = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    if args.cap_tr:
        tr_i = rng.choice(tr_i, min(args.cap_tr, len(tr_i)), replace=False)
    if args.cap_ev:
        ev_i = rng.choice(ev_i, min(args.cap_ev, len(ev_i)), replace=False)
    print(f"windows: train {len(tr_i)}  eval {len(ev_i)} (FULL={not args.cap_tr})", flush=True)
    print("building event datasets ...", flush=True)
    Xtr, ytr, _, _ = build(tr_i, X, acc_ix, cur_ix, rng)
    Xev, yev, lev, kev = build(ev_i, X, acc_ix, cur_ix, rng)
    print(f"train: {len(ytr)} samples ({ytr.mean()*100:.0f}% pos)  "
          f"eval: {len(yev)} ({yev.mean()*100:.0f}% pos)", flush=True)

    mu = Xtr.reshape(-1, X.shape[2]).mean(0); sd = Xtr.reshape(-1, X.shape[2]).std(0) + 1e-3
    dev = "cuda"
    net = nn.Sequential().to(dev)
    class Clf(nn.Module):
        def __init__(self, c):
            super().__init__()
            self.gru = nn.GRU(c, 64, batch_first=True)
            self.fc = nn.Linear(64, 1)
        def forward(self, x):
            return self.fc(self.gru(x)[0][:, -1]).squeeze(-1)
    net = Clf(X.shape[2]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xtr_t = torch.from_numpy((Xtr - mu) / sd)
    ytr_t = torch.from_numpy(ytr.astype(np.float32))
    for ep in range(5):
        net.train(); perm = torch.randperm(len(ytr_t)); tot = 0
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = nn.functional.binary_cross_entropy_with_logits(
                net(Xtr_t[b].to(dev)), ytr_t[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach())
        print(f"epoch {ep+1}/5  bce={tot/max(1,len(perm)//256):.4f}", flush=True)
    net.eval()
    with torch.no_grad():
        s = []
        Xev_t = torch.from_numpy((Xev - mu) / sd)
        for i in range(0, len(yev), 512):
            s.append(torch.sigmoid(net(Xev_t[i:i+512].to(dev))).cpu().numpy())
        s = np.concatenate(s)
    res = {"overall": {"auroc": float(roc_auc_score(yev, s)),
                       "ap": float(average_precision_score(yev, s)),
                       "pos_rate": float(yev.mean())}}
    print(f"\n== T5 event forecasting (unseen drivers) ==")
    print(f"  overall: AUROC={res['overall']['auroc']:.3f}  AP={res['overall']['ap']:.3f} "
          f"(pos {yev.mean()*100:.0f}%)")
    for kind in ("brake", "steer"):
        m = (kev == kind) | (kev == "neg")
        if (kev[m] == kind).sum() > 50:
            res[kind] = {"auroc": float(roc_auc_score(yev[m] * (kev[m] == kind), s[m]))}
            print(f"  {kind:6} AUROC={res[kind]['auroc']:.3f}")
    print("  -- AUROC by lead-time (positives at that lead vs all negatives) --")
    res["by_lead"] = {}
    for lo, hi in [(0, 1.01), (1.01, 3.01), (3.01, 5.01)]:
        mpos = (yev == 1) & (lev > lo) & (lev <= hi)
        if mpos.sum() > 50:
            mm = mpos | (yev == 0)
            a = roc_auc_score(yev[mm], s[mm])
            res["by_lead"][f"{lo:.0f}-{hi:.0f}s"] = float(a)
            print(f"  lead {lo:.0f}–{hi:.0f}s: AUROC={a:.3f}  (n_pos={int(mpos.sum())})")
    os.makedirs(f"{BASE}/results", exist_ok=True)
    json.dump(res, open(f"{BASE}/results/t5_results.json", "w"), indent=1)
    print("saved results/t5_results.json  [candidate taxonomy — labels rule-derived, not frozen]")


if __name__ == "__main__":
    main()
