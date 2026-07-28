#!/usr/bin/env python3
"""
Align V-JEPA2 per-segment CLIP embeddings (data/video_emb_vjepa2) to tagged windows.

Each segment-minute holds 15 clip embeddings (clip i covers seconds [4i, 4i+4)).
For each 60 s window: slot j (j=0..14) is the clip covering t0 + 4j → [15, 1024] fp16
(zero-padded + validity count). Counterpart of export_window_video.py (DINOv2 120×768).

Output:
  data/colab_bundle/windows_vidj.npy   [N, 15, 1024] fp16 (~1.9 GB — fits Colab as-is)
  data/colab_bundle/windows_vidj_n.npy [N] uint8 (# valid clips)
Run after embed_video_vjepa2.py completes.
"""
import os
import numpy as np
import pandas as pd

BUNDLE = "./data/colab_bundle"
SEGQ = "./data/segments/segments.parquet"
VID = "./data/video_emb_vjepa2"
CLIP_S, TV, D = 4, 15, 1024


def main():
    meta = pd.read_parquet(os.path.join(BUNDLE, "windows_meta.parquet"))
    seg = pd.read_parquet(SEGQ)
    kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
    N = len(meta)
    V = np.lib.format.open_memmap(os.path.join(BUNDLE, "windows_vidj.npy"),
                                  mode="w+", dtype=np.float16, shape=(N, TV, D))
    VN = np.zeros(N, dtype=np.uint8)
    done = miss = 0
    for (mdl, drv, rt), g in meta.groupby(["model", "driver", "route"]):
        npz_path = os.path.join(VID, mdl, f"{drv}__{rt}.npz")
        if not os.path.exists(npz_path):
            miss += len(g); continue
        try:
            z = np.load(npz_path)
        except Exception:
            miss += len(g); continue
        km = kmin.get((mdl, drv, rt), 0)
        for ridx, row in g.iterrows():
            t0 = float(row.t0)
            out = np.zeros((TV, D), dtype=np.float16)
            n_valid = 0
            for j in range(TV):                      # clip slot j covers t0 + 4j
                t = t0 + j * CLIP_S
                k = km + int(t // 60)
                key = f"seg{k}"
                if key in z.files:
                    arr = z[key]
                    ci = int((t - (k - km) * 60) // CLIP_S)
                    if 0 <= ci < len(arr):
                        out[j] = arr[ci]
                        n_valid += 1
            if n_valid:
                V[ridx] = out
                VN[ridx] = n_valid
                done += 1
        if (done + miss) % 10000 < len(g):
            print(f"  {done+miss}/{N} (aligned {done}, missing {miss})", flush=True)
    V.flush()
    np.save(os.path.join(BUNDLE, "windows_vidj_n.npy"), VN)
    print(f"\naligned {done}/{N} windows ({miss} without V-JEPA2 video)  "
          f"mean valid clips={VN[VN>0].mean():.1f}/15")
    print("wrote windows_vidj.npy + windows_vidj_n.npy")


if __name__ == "__main__":
    main()
