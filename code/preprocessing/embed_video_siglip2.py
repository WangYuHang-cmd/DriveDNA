#!/usr/bin/env python3
"""
V1 — 2025-generation PER-FRAME video embedding cache (SigLIP2) over human segments.

Purpose (model-modernization wave): re-test finding #19's per-frame negative transfer
with a 2025 extractor. SigLIP2-base (Google, Feb 2025) replaces DINOv2 (2023);
DINOv3 was first choice but its HF repo is license-gated on this account.
Same layout as embed_video.py: 2 fps frames → 768-d fp16 per frame.

    PYTHONPATH= .../python embed_video_siglip2.py [--limit N]
Output: DriveDNA/data/video_emb_siglip2/<MODEL>/<driver>__<route>.npz {seg<k>: [n,768]}
Resumable at route level; bounded decode prefetch.
"""
import os
import time
import argparse
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import torch

SEGQ = "./data/segments/segments.parquet"
DATASET = "../Dataset"
OUTDIR = "./data/video_emb_siglip2"
CKPT = "google/siglip2-base-patch16-256"
FPS = 2
W, H = 526, 330
SIDE = 256
BATCH = 128


def decode_seg(path):
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
    ks = set()
    for _, r in spans.iterrows():
        ks.update(range(int(r.k_min + r.t0 // 60), int(r.k_min + r.t1 // 60) + 1))
    return sorted(ks)


@torch.no_grad()
def embed_frames(model, frames, device):
    x = torch.from_numpy(frames.copy()).to(device).permute(0, 3, 1, 2).float().div_(255)
    x = torch.nn.functional.interpolate(x, size=(SIDE, SIDE), mode="bilinear", align_corners=False)
    x = (x - 0.5) / 0.5                                     # SigLIP normalization
    out = []
    for i in range(0, len(x), BATCH):
        h = model(pixel_values=x[i:i + BATCH].half()).pooler_output
        out.append(h.float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()

    seg = pd.read_parquet(SEGQ)
    routes = seg.groupby(["model", "driver", "route"])
    keys = list(routes.groups.keys())
    if args.limit:
        keys = keys[:args.limit]
    print(f"routes with human spans: {len(keys)}", flush=True)

    device = "cuda"
    from transformers import AutoModel
    model = AutoModel.from_pretrained(
        CKPT, cache_dir="./data/.hfhub"
    ).vision_model.to(device).half().eval()
    print(f"SigLIP2 loaded ({CKPT})", flush=True)

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
        store = {}
        pending = []
        pi = 0
        while pi < len(paths) or pending:
            while pi < len(paths) and len(pending) < args.workers:
                k, p = paths[pi]
                pending.append((k, pool.submit(decode_seg, p)))
                pi += 1
            k, fut = pending.pop(0)
            frames = fut.result()
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
    print(f"DONE: {done} routes, {fail} failed in {(time.time()-t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
