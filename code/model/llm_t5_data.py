#!/usr/bin/env python3
"""
Build the FIXED T5 anchor subsample for LLM/VLM zero-shot rows, and re-score the
standard CAN GRU on that same subsample (fair comparison column).

Anchors reuse the exact rule-derived taxonomy of code/eval/t5_events.py
(hard-brake onset aEgo<-2 after calm; sharp-steer |curv-rate| spike), on the
same unseen-driver eval population (val + few-shot folds).  Subsample:
1,000 positives + 2,000 negatives, fixed seed (AUROC is prevalence-invariant;
AP is reported on this enriched prevalence and labeled as such).

Outputs:
  data/segments/t5_llm_subsample.parquet   (gi, a, y, lead, kind, driver, route, t_abs)
  results/llm_t5_gru_subsample.json        GRU AUROC/AP on the subsample
  results/llm_t5_gru_scores.npz            per-anchor GRU scores
"""
import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE = "."
BUNDLE = f"{BASE}/data/colab_bundle"
SEED = 20260709
L_H, L_F = 50, 50
RATE = 10.0
N_POS, N_NEG = 1000, 2000

sys.path.insert(0, f"{BASE}/code/eval")
from t5_events import find_onsets                     # exact same onset rules


def build_with_pos(meta_idx, X, acc_ix, cur_ix, rng, per_event=6):
    """t5_events.build, extended to record (gi, anchor)."""
    recs = []
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
                recs.append((int(gi), int(a), 1, lead / RATE, kind))
                used.add(a // 25)
        ev_set = set(oi for oi, _ in events)
        for _ in range(per_event):
            a = int(rng.integers(L_H, 600 - L_F))
            if any(a < oi <= a + L_F for oi in ev_set) or (a // 25) in used:
                continue
            recs.append((int(gi), int(a), 0, np.nan, "neg"))
    return pd.DataFrame(recs, columns=["gi", "a", "y", "lead", "kind"])


class GRUHead(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.g = nn.GRU(c, 64, batch_first=True)
        self.f = nn.Linear(64, 1)

    def forward(self, x):
        _, h = self.g(x)
        return self.f(h[-1]).squeeze(-1)


def main():
    rng = np.random.default_rng(SEED)
    torch.manual_seed(SEED)
    X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
    meta = pd.read_parquet(f"{BUNDLE}/windows_meta.parquet")
    ch = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
    acc_ix, cur_ix = ch.index("aEgo"), ch.index("actual_curvature")
    folds = json.load(open(f"{BUNDLE}/driver_folds.json"))
    tr_i = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    ev_i = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())

    print("building eval anchors ...", flush=True)
    ev = build_with_pos(ev_i, X, acc_ix, cur_ix, rng)
    print(f"eval anchors: {len(ev)} ({ev.y.mean()*100:.1f}% pos)", flush=True)
    pos = ev[ev.y == 1].sample(N_POS, random_state=SEED)
    neg = ev[ev.y == 0].sample(N_NEG, random_state=SEED)
    sub = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    sub["driver"] = meta.iloc[sub.gi].driver.values
    sub["route"] = meta.iloc[sub.gi].route.values
    sub["model"] = meta.iloc[sub.gi].model.values
    sub["t_abs"] = meta.iloc[sub.gi].t0.values + sub.a.values / RATE
    sub.to_parquet(f"{BASE}/data/segments/t5_llm_subsample.parquet")
    print(f"subsample saved: {len(sub)} anchors", flush=True)

    # ---- train the standard GRU (same recipe as t5_events) and score subsample ----
    print("building train anchors ...", flush=True)
    tr = build_with_pos(tr_i, X, acc_ix, cur_ix, rng)
    print(f"train anchors: {len(tr)} ({tr.y.mean()*100:.1f}% pos)", flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fetch(df, i, bs):
        rows = df.iloc[i:i + bs]
        xs = np.stack([X[int(r.gi)][int(r.a) - L_H:int(r.a)] for r in rows.itertuples()]).astype(np.float32)
        return np.nan_to_num(xs), rows.y.to_numpy().astype(np.float32)

    mu_sd_sample, _ = fetch(tr.sample(3000, random_state=SEED), 0, 3000)
    mu = mu_sd_sample.reshape(-1, X.shape[2]).mean(0)
    sd = mu_sd_sample.reshape(-1, X.shape[2]).std(0) + 1e-3

    net = GRUHead(X.shape[2]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    bs = 256
    for ep in range(5):
        tr_shuf = tr.sample(frac=1, random_state=SEED + ep).reset_index(drop=True)
        tot = 0.0
        for i in range(0, len(tr_shuf), bs):
            xb, yb = fetch(tr_shuf, i, bs)
            xb = torch.from_numpy((xb - mu) / sd).to(dev)
            yb = torch.from_numpy(yb).to(dev)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward(); opt.step()
            tot += float(loss) * len(yb)
        print(f"epoch {ep}: loss {tot/len(tr_shuf):.4f}", flush=True)

    net.eval()
    scores = []
    with torch.no_grad():
        for i in range(0, len(sub), bs):
            xb, _ = fetch(sub, i, bs)
            xb = torch.from_numpy((xb - mu) / sd).to(dev)
            scores.append(torch.sigmoid(net(xb)).cpu().numpy())
    scores = np.concatenate(scores)
    from sklearn.metrics import roc_auc_score, average_precision_score
    res = {"auroc": float(roc_auc_score(sub.y, scores)),
           "ap": float(average_precision_score(sub.y, scores)),
           "n": int(len(sub)), "pos_rate": float(sub.y.mean())}
    print("GRU on subsample:", res, flush=True)
    json.dump(res, open(f"{BASE}/results/llm_t5_gru_subsample.json", "w"), indent=1)
    np.savez(f"{BASE}/results/llm_t5_gru_scores.npz", scores=scores, y=sub.y.to_numpy())


if __name__ == "__main__":
    main()
