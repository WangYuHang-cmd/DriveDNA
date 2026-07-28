#!/usr/bin/env python3
"""
E1 (supplementary, appendix): within-driver CROSS-VEHICLE verification.

For every driver observed on >=2 consolidated vehicle models (>=20 windows each),
enroll a prototype from vehicle A and verify windows from vehicle B against
other drivers' windows on the SAME model as B (vehicle-controlled negatives).
Control condition: same-vehicle cross-window verification under the identical
protocol. Cluster bootstrap over drivers for the CI.

Writes results/e1_cross_vehicle.json.
"""
import sys, json
import numpy as np
import pandas as pd
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")
from harness import verification_metrics
from s1_supcon import Encoder

SEED = 20260709
MIN_WIN = 20

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location=DEV, weights_only=False)
enc = Encoder(c_in=X.shape[2]).to(DEV).eval(); enc.load_state_dict(ck["model"])
mu, sd = ck["mu"], ck["sd"]

# multi-vehicle drivers with >=2 canon models of >=MIN_WIN windows
cnt = W.groupby(["driver", "model_canon"]).size()
ok = cnt[cnt >= MIN_WIN].reset_index()
multi = ok.groupby("driver").filter(lambda g: g.model_canon.nunique() >= 2)
drivers = sorted(multi.driver.unique())
print(f"multi-vehicle drivers (>= {MIN_WIN} win on >=2 models): {len(drivers)}", flush=True)

# embed every window we might touch: multi drivers' windows + all windows on their models
models_needed = sorted(multi.model_canon.unique())
need_mask = W.driver.isin(drivers) | W.model_canon.isin(models_needed)
need = np.flatnonzero(need_mask.to_numpy())
Z = np.zeros((len(W), 128), np.float32)
with torch.no_grad():
    for i in range(0, len(need), 512):
        idx = need[i:i + 512]
        a = np.nan_to_num((X[idx].astype(np.float32) - mu) / sd)
        Z[idx] = enc(torch.from_numpy(a).to(DEV)).cpu().numpy()
Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
rng = np.random.default_rng(SEED)


def proto(ix):
    p = Z[ix].mean(0)
    return p / (np.linalg.norm(p) + 1e-9)


def eval_pair(d, ma, mb):
    """enroll on model ma, verify on model mb (vehicle-controlled negatives)."""
    ia = np.flatnonzero(((W.driver == d) & (W.model_canon == ma)).to_numpy())
    ib = np.flatnonzero(((W.driver == d) & (W.model_canon == mb)).to_numpy())
    ineg = np.flatnonzero(((W.driver != d) & (W.model_canon == mb)).to_numpy())
    if len(ineg) < MIN_WIN:
        return None
    ia = rng.permutation(ia)[:50]
    ib = rng.permutation(ib)[:200]
    ineg = rng.permutation(ineg)[:2000]
    p = proto(ia)
    s = np.concatenate([Z[ib] @ p, Z[ineg] @ p])
    y = np.concatenate([np.ones(len(ib)), np.zeros(len(ineg))])
    m = verification_metrics(s, y)
    m.update(driver=d, enroll=ma, query=mb, n_pos=len(ib), n_neg=len(ineg))
    return m


def eval_same(d, ma):
    """control: enroll and verify on the SAME vehicle (disjoint windows)."""
    ia = rng.permutation(np.flatnonzero(((W.driver == d) & (W.model_canon == ma)).to_numpy()))
    ineg = np.flatnonzero(((W.driver != d) & (W.model_canon == ma)).to_numpy())
    if len(ia) < 2 * MIN_WIN or len(ineg) < MIN_WIN:
        return None
    half = len(ia) // 2
    p = proto(ia[:min(half, 50)])
    ib = ia[half:half + 200]
    ineg = rng.permutation(ineg)[:2000]
    s = np.concatenate([Z[ib] @ p, Z[ineg] @ p])
    y = np.concatenate([np.ones(len(ib)), np.zeros(len(ineg))])
    m = verification_metrics(s, y)
    m.update(driver=d, model=ma, n_pos=len(ib), n_neg=len(ineg))
    return m


cross, same = [], []
for d in drivers:
    ms = ok[ok.driver == d].model_canon.tolist()
    for ma in ms:
        for mb in ms:
            if ma != mb:
                r = eval_pair(d, ma, mb)
                if r:
                    cross.append(r)
        r = eval_same(d, ma)
        if r:
            same.append(r)

def summarize(rows):
    per_driver = pd.DataFrame(rows).groupby("driver").auroc.mean()
    boots = [per_driver.sample(len(per_driver), replace=True, random_state=int(b)).mean()
             for b in range(2000)]
    return {"n_pairs": len(rows), "n_drivers": int(per_driver.size),
            "mean_auroc_over_drivers": float(per_driver.mean()),
            "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "per_driver": {k: float(v) for k, v in per_driver.items()}}

out = {"cross_vehicle": summarize(cross), "same_vehicle_control": summarize(same),
       "pairs": cross, "same_pairs": same, "min_windows": MIN_WIN}
print(json.dumps({k: out[k] for k in ("cross_vehicle", "same_vehicle_control")}, indent=1), flush=True)
json.dump(out, open(f"{BASE}/results/e1_cross_vehicle.json", "w"), indent=1)
print("saved results/e1_cross_vehicle.json", flush=True)
