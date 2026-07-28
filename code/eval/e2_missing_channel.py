#!/usr/bin/env python3
"""
E2 (appendix): missing-channel robustness of the frozen S1 SupCon embedding.

At evaluation time one signal group is removed (set to the training mean, i.e.
zero after normalization) and the full T2 enrollment protocol is re-run on the
val ∪ few_shot_heldout population. Reports k=5 AUROC / top-1 per ablation.
Writes results/e2_missing_channel.json.
"""
import sys, json
import numpy as np
import pandas as pd
import torch
DEV = "cuda" if torch.cuda.is_available() else "cpu"

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")
from harness import enrollment_protocol
from s1_supcon import Encoder

ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
GROUPS = {
    "none (full input)": [],
    "motion (vEgo, aEgo)": ["vEgo", "aEgo"],
    "curvature/yaw": ["actual_curvature", "yaw_rate", "curv_measured"],
    "lead vehicle (radar)": ["leadOne_status", "leadOne_dRel", "leadOne_vRel"],
    "lane lines": ["laneLeft_y", "laneRight_y"],
    "steering input": ["steeringAngleDeg", "steeringRateDeg", "steeringPressed"],
    "pedals": ["gas", "gasPressed", "brake", "brakePressed"],
}

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
folds = json.load(open(f"{BASE}/data/splits/driver_folds.json"))
pop = set(folds["val"]) | set(folds["few_shot_heldout"])
sel = np.flatnonzero(W.driver.isin(pop).to_numpy())
drv, rts = W.driver.iloc[sel].to_numpy(), W.route.iloc[sel].to_numpy()
print(f"eval windows {len(sel)}", flush=True)

ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location=DEV, weights_only=False)
enc = Encoder(c_in=X.shape[2]).to(DEV).eval(); enc.load_state_dict(ck["model"])
mu, sd = ck["mu"], ck["sd"]

out = {}
for name, drop in GROUPS.items():
    idx = [ch.index(c) for c in drop]
    Z = np.zeros((len(sel), 128), np.float32)
    with torch.no_grad():
        for i in range(0, len(sel), 512):
            j = sel[i:i + 512]
            a = np.nan_to_num((X[j].astype(np.float32) - mu) / sd)
            if idx:
                a[:, :, idx] = 0.0
            Z[i:i + len(j)] = enc(torch.from_numpy(a).to(DEV)).cpu().numpy()
    res = enrollment_protocol(Z, drv, rts, k_minutes=(5,))
    v = res[5]
    out[name] = {"dropped": drop, "auroc": v["auroc"], "eer": v["eer"],
                 "top1": v["top1"], "n_drivers": v["n_drivers"]}
    print(f"{name:24s} AUROC {v['auroc']:.3f}  top1 {v['top1']:.3f}", flush=True)

json.dump(out, open(f"{BASE}/results/e2_missing_channel.json", "w"), indent=1)
print("saved results/e2_missing_channel.json", flush=True)
