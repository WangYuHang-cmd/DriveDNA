#!/usr/bin/env python3
"""
T4 — Matched-context driver comparison: full evaluation on the released pair manifest.

Q: in comparable contexts (scenario × speed × THW × vehicle model), are drivers still
distinguishable? Evaluate representations on 14,868 balanced same/different-driver pairs:
  * S1 SupCon embedding (cosine)                     [learned]
  * descriptor vector (negative L2 on standardized)  [interpretable anchor]
Metrics: verification AUROC / EER via harness. Also per-scenario breakdown.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")
from s1_supcon import Encoder  # noqa: E402
from harness import verification_metrics, STATS  # noqa: E402
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

BUNDLE = f"{BASE}/data/colab_bundle"


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=f"{BASE}/experiments/checkpoints/s1_supcon.pt")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    pairs = pd.read_parquet(f"{BASE}/data/splits/matched_context_pairs.parquet")
    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
    ck = torch.load(args.ckpt, map_location="cuda", weights_only=False)
    enc = Encoder(c_in=X.shape[2]).cuda().eval(); enc.load_state_dict(ck["model"])
    mu, sd = ck["mu"], ck["sd"]

    need = np.unique(np.concatenate([pairs.win_a.to_numpy(), pairs.win_b.to_numpy()]))
    print(f"pairs={len(pairs)}  unique windows={len(need)}")
    # embed needed windows
    Z = np.zeros((len(W), 128), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(need), 512):
            idx = need[i:i + 512]
            a = (X[idx].astype(np.float32) - mu) / sd
            Z[idx] = enc(torch.from_numpy(a).cuda()).cpu().numpy()
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

    # descriptor representation
    D = W[STATS].to_numpy(float)
    D = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(D))

    y = pairs.same_driver.to_numpy()
    s_emb = (Zn[pairs.win_a.to_numpy()] * Zn[pairs.win_b.to_numpy()]).sum(1)
    s_desc = -np.linalg.norm(D[pairs.win_a.to_numpy()] - D[pairs.win_b.to_numpy()], axis=1)

    out = {}
    print("\n== T4 matched-context same-driver verification (14.9k balanced pairs) ==")
    for name, s in [("S1 embedding", s_emb), ("descriptors", s_desc)]:
        m = verification_metrics(s, y)
        out[name] = m
        print(f"  {name:14} AUROC={m['auroc']:.3f}  EER={m['eer']:.3f}")
    print("\n-- per-scenario (S1 embedding) --")
    sc = pairs.key.str.split("|").str[0]
    per = {}
    for s_name, g in pairs.groupby(sc):
        if g.same_driver.nunique() < 2 or len(g) < 200:
            continue
        m = verification_metrics(s_emb[g.index.to_numpy()], y[g.index.to_numpy()])
        per[s_name] = m
        print(f"  {s_name:14} n={len(g):>5}  AUROC={m['auroc']:.3f}  EER={m['eer']:.3f}")
    json.dump({"overall": out, "per_scenario": per},
              open(f"{BASE}/results/t4_results{args.tag}.json", "w"), indent=1)
    print("\nsaved results/t4_results.json")


if __name__ == "__main__":
    main()
