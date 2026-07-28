#!/usr/bin/env python3
"""
D3 — Frozen-ViT forward-video embedding cache over HUMAN-driving segments only.

For every qcamera segment-minute overlapping a human span (from data/segments/segments.parquet),
decode at FPS=2 via ffmpeg and embed with frozen DINOv2 ViT-B/14 (CLS, 768-d, fp16).
Video is a CONTEXT modality: embeddings feed MCPP (S4), T5, and T6 — never driver-ID directly.

    PYTHONPATH= ../openpilot/.venv/bin/python embed_video.py [--limit N]
Output: DriveDNA/data/video_emb/<MODEL>/<driver>__<route>.npz  {seg<k>: [n_frames,768] float16}
Resumable at route level. Runs on the 5080 (data-local).
"""
import os
import sys
import glob
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import torch

SEGQ = "./data/segments/segments.parquet"
DATASET = "../Dataset"
OUTDIR = "./data/video_emb"
FPS = 2
W, H = 526, 330          # qcamera.ts native
SIDE = 224               # DINOv2 input (multiple of 14)
BATCH = 128

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def decode_seg(path):
    """ffmpeg → rgb24 frames at FPS. Returns uint8 array [n, H, W, 3] or None."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=60).stdout
    except Exception:
        return None
    n = len(raw) // (W * H * 3)
    if n == 0:
        return None
    return np.frombuffer(raw[:n * W * H * 3], dtype=np.uint8).reshape(n, H, W, 3)


def needed_segments(spans):
    """Route spans -> sorted list of needed segment indices (k_min + minute bins)."""
    ks = set()
    for _, r in spans.iterrows():
        ks.update(range(int(r.k_min + r.t0 // 60), int(r.k_min + r.t1 // 60) + 1))
    return sorted(ks)


@torch.no_grad()
def embed_frames(model, frames, device):
    x = torch.from_numpy(frames).to(device).permute(0, 3, 1, 2).float().div_(255)
    x = torch.nn.functional.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
    x = (x - MEAN.to(device)) / STD.to(device)
    out = []
    for i in range(0, len(x), BATCH):
        out.append(model(x[i:i + BATCH].half()).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max routes (smoke test)")
    ap.add_argument("--workers", type=int, default=4, help="ffmpeg decode threads")
    args = ap.parse_args()

    seg = pd.read_parquet(SEGQ)
    routes = seg.groupby(["model", "driver", "route"])
    keys = list(routes.groups.keys())
    if args.limit:
        keys = keys[:args.limit]
    print(f"routes with human spans: {len(keys)}", flush=True)

    device = "cuda"
    torch.hub.set_dir("./data/.torchhub")
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").to(device).half().eval()
    print("DINOv2 ViT-B/14 loaded", flush=True)

    done = fail = 0
    t_start = time.time()
    pool = ThreadPoolExecutor(max_workers=args.workers)
    for ri, key in enumerate(keys):
        mdl, drv, rt = key
        outp = os.path.join(OUTDIR, mdl, f"{drv}__{rt}.npz")
        if os.path.exists(outp):
            done += 1
            continue
        os.makedirs(os.path.dirname(outp), exist_ok=True)
        ks = needed_segments(routes.get_group(key))
        paths = [(k, os.path.join(DATASET, mdl, drv, rt, f"{k}--qcamera.ts")) for k in ks]
        paths = [(k, p) for k, p in paths if os.path.exists(p)]
        if not paths:
            fail += 1
            continue
        # prefetch decodes; embed on GPU as they arrive
        futs = {k: pool.submit(decode_seg, p) for k, p in paths}
        store = {}
        for k, _ in paths:
            frames = futs[k].result()
            if frames is None or len(frames) < 2:
                continue
            store[f"seg{k}"] = embed_frames(model, frames, device)
        if store:
            np.savez_compressed(outp + ".tmp.npz", **store)
            os.replace(outp + ".tmp.npz", outp)
            done += 1
        else:
            fail += 1
        if (ri + 1) % 25 == 0:
            el = time.time() - t_start
            rate = (ri + 1) / el
            print(f"  {ri+1}/{len(keys)}  ok={done} fail={fail}  {rate:.2f} route/s  "
                  f"ETA {(len(keys)-ri-1)/rate/60:.0f} min", flush=True)
    print(f"DONE: {done} routes embedded, {fail} skipped/failed in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
