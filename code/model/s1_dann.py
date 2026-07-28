#!/usr/bin/env python3
"""
S2/H — DANN adversarial vehicle-invariance on the S1 embedding (must-land #7 completion).

Train the S1 SupCon encoder WITH a gradient-reversal vehicle-model classifier:
  loss = SupCon(driver) + λ · CE(vehicle | GRL(z))
→ the DANN point on the utility–leakage Pareto. Evaluates BOTH the plain-S1 checkpoint and
the DANN model on: T2 enrollment (utility) + vehicle-leakage probe (invariance).

    python s1_dann.py --epochs 20 --lam 0.3 [--smoke]
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from s1_supcon import Encoder, supcon, PKSampler
import sys
sys.path.insert(0, "./code/eval")
from harness import enrollment_protocol, leakage_probe  # noqa: E402

SEED = 20260709
torch.manual_seed(SEED)


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="./data/colab_bundle")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="./experiments/checkpoints/s1_dann.pt")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(SEED)

    X = np.load(os.path.join(args.bundle, "windows_x.npy"), mmap_mode="r")
    meta = pd.read_parquet(os.path.join(args.bundle, "windows_meta.parquet"))
    folds = json.load(open(os.path.join(args.bundle, "driver_folds.json")))
    veh_ids = {m: i for i, m in enumerate(sorted(meta.model_canon.unique()))}
    veh_all = meta.model_canon.map(veh_ids).to_numpy()

    tr_idx = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
    ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    if args.smoke:
        tr_idx, ev_idx = tr_idx[:4000], ev_idx[:2000]
    samp = X[np.sort(rng.choice(tr_idx, min(3000, len(tr_idx)), replace=False))].astype(np.float32)
    mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
    norm = lambda a: (a.astype(np.float32) - mu) / sd

    model = Encoder(c_in=X.shape[2]).to(dev)
    vhead = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, len(veh_ids))).to(dev)
    opt = torch.optim.AdamW(list(model.parameters()) + list(vhead.parameters()),
                            lr=3e-4, weight_decay=1e-4)
    drivers_tr = meta.driver.to_numpy()[tr_idx]
    dmap = {d: i for i, d in enumerate(pd.unique(drivers_tr))}

    epochs = 2 if args.smoke else args.epochs
    for ep in range(epochs):
        model.train()
        sampler = PKSampler(drivers_tr, P=16, K=4, seed=SEED + ep)
        tot_s = tot_v = nb = 0
        lam = args.lam * min(1.0, (ep + 1) / max(1, epochs // 3))     # warmup schedule
        for bi in sampler:
            gidx = tr_idx[bi]
            x = torch.from_numpy(norm(X[np.sort(gidx)])[np.argsort(np.argsort(gidx))]).to(dev)
            if rng.random() < 0.5:
                drop = torch.rand(x.shape[2], device=dev) < 0.15
                x[:, :, drop] = 0
            y = torch.tensor([dmap[drivers_tr[i]] for i in bi], device=dev)
            v = torch.from_numpy(veh_all[gidx]).to(dev)
            z = model(x)
            ls = supcon(z, y)
            lv = nn.functional.cross_entropy(vhead(GRL.apply(z, lam)), v)
            loss = ls + args.lam * lv
            opt.zero_grad(); loss.backward(); opt.step()
            tot_s += float(ls.detach()); tot_v += float(lv.detach()); nb += 1
        print(f"epoch {ep+1}/{epochs}  supcon={tot_s/nb:.4f}  veh_ce={tot_v/nb:.4f}  lam={lam:.2f}", flush=True)

    # ---------- evaluate BOTH models on the Pareto axes ----------
    def embed_all(m, mu_, sd_):
        m.eval(); out = []
        with torch.no_grad():
            for i in range(0, len(ev_idx), 512):
                a = X[np.sort(ev_idx[i:i+512])].astype(np.float32)
                a = (a - mu_) / sd_
                a = a[np.argsort(np.argsort(ev_idx[i:i+512]))]
                out.append(m(torch.from_numpy(a).to(dev)).cpu().numpy())
        return np.concatenate(out)

    me = meta.iloc[ev_idx]
    results = {}
    # DANN model
    Zd = embed_all(model, mu, sd)
    # plain S1 checkpoint
    ck = torch.load("./experiments/checkpoints/s1_supcon.pt",
                    map_location=dev, weights_only=False)
    plain = Encoder(c_in=X.shape[2]).to(dev); plain.load_state_dict(ck["model"])
    Zp = embed_all(plain, ck["mu"], ck["sd"])
    for name, Z in [("S1-plain", Zp), ("S1-DANN", Zd)]:
        res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
        k = max(res)
        lk = leakage_probe(Z, me.model_canon.to_numpy(), me.driver.to_numpy())
        results[name] = (res[k], lk)
        print(f"\n== {name} ==  (@{k}min, {res[k]['n_drivers']} unseen drivers)")
        print(f"  utility: top1={res[k]['top1']:.3f} AUROC={res[k]['auroc']:.3f} EER={res[k]['eer']:.3f}")
        print(f"  vehicle-leakage: bal_acc={lk['bal_acc']:.3f} (chance {lk['chance']:.3f})")
    print("\n== PARETO (utility top1 vs vehicle-leakage bal_acc) ==")
    for n, (r, l) in results.items():
        print(f"  {n:9} utility={r['top1']:.3f}  leakage={l['bal_acc']:.3f}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model": model.state_dict(), "mu": mu, "sd": sd}, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
