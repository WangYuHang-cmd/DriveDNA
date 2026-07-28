#!/usr/bin/env python3
"""
Align cached per-segment video embeddings (data/video_emb_siglip2, V1 modernization wave) to tagged windows → memmap.

For each window (route, t0..t0+60 s): frame time within seg k is (k − k_min)*60 + j/FPS;
gather the 2 fps frames covering the window → [120, 768] fp16 (zero-padded + validity count).

Output (local, for 5080 MCPP training):
  data/colab_bundle/windows_vids2.npy   [N, 120, 768] fp16 (~11.5 GB memmap)
  data/colab_bundle/windows_vids2_n.npy [N] uint8 (# valid frames)
Also writes a 12-token downsampled copy windows_vids2_12.npy (~1.2 GB) for Colab upload.
"""
import os
import numpy as np
import pandas as pd

BUNDLE = "./data/colab_bundle"
SEGQ = "./data/segments/segments.parquet"
VID = "./data/video_emb_siglip2"
FPS, TV, D = 2, 120, 768


def main():
    meta = pd.read_parquet(os.path.join(BUNDLE, "windows_meta.parquet"))
    seg = pd.read_parquet(SEGQ)
    kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
    N = len(meta)
    V = np.lib.format.open_memmap(os.path.join(BUNDLE, "windows_vids2.npy"),
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
            frames = []
            for j in range(TV):                      # frame j at time t0 + j/FPS
                t = t0 + j / FPS
                k = km + int(t // 60)
                key = f"seg{k}"
                if key in z.files:
                    arr = z[key]
                    fi = int(round((t - (k - km) * 60) * FPS))
                    if 0 <= fi < len(arr):
                        frames.append(arr[fi]); continue
                frames.append(None)
            valid = [f for f in frames if f is not None]
            if valid:
                out = np.zeros((TV, D), dtype=np.float16)
                for j, f in enumerate(frames):
                    if f is not None:
                        out[j] = f
                V[ridx] = out
                VN[ridx] = len(valid)
                done += 1
        if (done + miss) % 10000 < len(g):
            print(f"  {done+miss}/{N} (aligned {done}, missing {miss})", flush=True)
    V.flush()
    np.save(os.path.join(BUNDLE, "windows_vids2_n.npy"), VN)
    # 12-token downsample for Colab (1 frame / 5 s)
    V12 = V[:, ::10][:, :12]
    np.save(os.path.join(BUNDLE, "windows_vids2_12.npy"), np.ascontiguousarray(V12))
    print(f"\naligned {done}/{N} windows ({miss} without video)  "
          f"mean valid frames={VN[VN>0].mean():.0f}/120")
    print("wrote windows_vids2.npy (full) + windows_vids2_12.npy (Colab)")


if __name__ == "__main__":
    main()
