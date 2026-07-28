#!/usr/bin/env python3
"""
Export the compact Colab training bundle: per-window 10 Hz signal tensors + metadata + splits.

For every tagged window (windows.parquet), slice the route's cached parquet into a fixed
[600, C] float16 tensor (60 s @10 Hz; shorter windows zero-padded with a validity mask).
Channels = Tier-A realized-motion + Tier-B auxiliary + lead/lane context. NaN→0 with mask.

Output (data/colab_bundle/):
  windows_x.npy        [N, 600, C] float16 (memmap-written)
  windows_mask.npy     [N, 600] uint8   (1 = valid frame)
  windows_meta.parquet (driver/route/model_canon/scenario/primitives/span ids)
  channels.json        channel names + tier tags
  + copies of splits (driver_folds.json, matched_context_pairs.parquet)
Then: tar.gz for upload to Colab/Drive.

    nice -n 10 .../python export_colab_bundle.py
"""
import os
import json
import shutil
import numpy as np
import pandas as pd

CACHE = "./data/cache"
WQ = "./data/segments/windows.parquet"
SPLITS = "./data/splits"
OUTDIR = "./data/colab_bundle"
T = 600  # 60 s @10 Hz

CHANNELS = [  # (name, tier)
    ("vEgo", "A"), ("aEgo", "A"), ("actual_curvature", "A"), ("yaw_rate", "A"),
    ("curv_measured", "A"), ("leadOne_status", "ctx"), ("leadOne_dRel", "ctx"),
    ("leadOne_vRel", "ctx"), ("laneLeft_y", "A"), ("laneRight_y", "A"),
    ("steeringAngleDeg", "B"), ("steeringRateDeg", "B"), ("steeringPressed", "B"),
    ("gas", "B"), ("gasPressed", "B"), ("brake", "B"), ("brakePressed", "B"),
]
COLS = [c for c, _ in CHANNELS]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    W = pd.read_parquet(WQ).reset_index(drop=True)
    N, C = len(W), len(CHANNELS)
    print(f"exporting {N} windows × {T} steps × {C} channels "
          f"(~{N*T*C*2/1e9:.2f} GB fp16)", flush=True)

    X = np.lib.format.open_memmap(os.path.join(OUTDIR, "windows_x.npy"),
                                  mode="w+", dtype=np.float16, shape=(N, T, C))
    M = np.lib.format.open_memmap(os.path.join(OUTDIR, "windows_mask.npy"),
                                  mode="w+", dtype=np.uint8, shape=(N, T))
    done = 0
    for (mdl, drv, rt), g in W.groupby(["model", "driver", "route"]):
        p = os.path.join(CACHE, mdl, f"{drv}__{rt}.parquet")
        try:
            df = pd.read_parquet(p, columns=COLS)
        except Exception:
            continue
        arr = df.to_numpy(dtype=np.float32)
        for ridx, row in g.iterrows():
            a, b = int(row.wi0), int(row.wi1) + 1
            seg = arr[a:b]
            n = min(len(seg), T)
            x = np.nan_to_num(seg[:n], nan=0.0, posinf=0.0, neginf=0.0)
            X[ridx, :n] = x.astype(np.float16)
            M[ridx, :n] = 1
            done += 1
        if done % 10000 < len(g):
            print(f"  {done}/{N}", flush=True)
    X.flush(); M.flush()

    meta_cols = ["model", "model_canon", "driver", "route", "span_id", "wi0", "wi1", "t0",
                 "scenario", "v_mean", "thw_median"] + [c for c in W.columns if c.startswith("p_")]
    W[meta_cols].to_parquet(os.path.join(OUTDIR, "windows_meta.parquet"), index=False)
    json.dump({"channels": [c for c, _ in CHANNELS], "tiers": dict(CHANNELS),
               "T": T, "rate_hz": 10}, open(os.path.join(OUTDIR, "channels.json"), "w"), indent=1)
    for f in ["driver_folds.json", "within_vehicle.json", "cross_route.json"]:
        src = os.path.join(SPLITS, f)
        if os.path.exists(src):
            shutil.copy(src, OUTDIR)
    shutil.copy(os.path.join(SPLITS, "matched_context_pairs.parquet"), OUTDIR)

    print(f"windows written: {done}/{N}")
    print("bundle contents:", sorted(os.listdir(OUTDIR)))
    total = sum(os.path.getsize(os.path.join(OUTDIR, f)) for f in os.listdir(OUTDIR))
    print(f"bundle size: {total/1e9:.2f} GB  →  tar: {OUTDIR}.tar.gz")
    # tar (no extra compression benefit on fp16, but single-file upload)
    os.system(f"tar -czf {OUTDIR}.tar.gz -C {os.path.dirname(OUTDIR)} {os.path.basename(OUTDIR)}")
    print(f"tar size: {os.path.getsize(OUTDIR + '.tar.gz')/1e9:.2f} GB")


if __name__ == "__main__":
    main()
