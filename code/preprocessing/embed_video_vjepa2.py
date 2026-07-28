#!/usr/bin/env python3
"""
M1 — V-JEPA2 (temporal) forward-video embedding cache over HUMAN-driving segments.

Counterpart of embed_video.py (DINOv2 per-frame): decode each needed qcamera
segment-minute at FPS=4 and embed non-overlapping 16-frame clips (4 s each) with
frozen V-JEPA2 ViT-L (facebook/vjepa2-vitl-fpc64-256, 3D-RoPE handles 16-frame
inputs), mean-pooled to 1024-d fp16. Video stays a CONTEXT modality: these feed
the MCPP video-encoder ablation (per-frame vs temporal), T5, and T6.

    PYTHONPATH= ../openpilot/.venv/bin/python embed_video_vjepa2.py [--limit N]
Output: DriveDNA/data/video_emb_vjepa2/<MODEL>/<driver>__<route>.npz
        {seg<k>: [n_clips, 1024] float16}   (clip i covers seconds [4i, 4i+4) of minute k)
Resumable at route level. Runs on the 5080 (data-local). Bounded decode prefetch
keeps RAM flat on long routes.
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
OUTDIR = "./data/video_emb_vjepa2"
CKPT = "facebook/vjepa2-vitl-fpc64-256"
FPS = 4
CLIP = 16                # frames per clip -> 4 s @ 4 fps
W, H = 526, 330          # qcamera.ts native
SIDE = 256               # V-JEPA2 input
CLIP_BATCH = 8

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)


def decode_seg(path):
    """ffmpeg → rgb24 frames at FPS. Returns uint8 array [n, H, W, 3] or None."""
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS}",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=90).stdout
    except Exception:
        return None
    n = len(raw) // (W * H * 3)
    if n < CLIP:
        return None
    return np.frombuffer(raw[:n * W * H * 3], dtype=np.uint8).reshape(n, H, W, 3)


def needed_segments(spans):
    ks = set()
    for _, r in spans.iterrows():
        ks.update(range(int(r.k_min + r.t0 // 60), int(r.k_min + r.t1 // 60) + 1))
    return sorted(ks)


@torch.no_grad()
def embed_clips(model, frames, device):
    """frames uint8 [n,H,W,3] -> [n//CLIP, 1024] fp16 (mean-pooled tokens per clip)."""
    n_clips = len(frames) // CLIP
    frames = frames[:n_clips * CLIP]
    x = torch.from_numpy(frames.reshape(n_clips, CLIP, H, W, 3))
    out = []
    for i in range(0, n_clips, CLIP_BATCH):
        b = x[i:i + CLIP_BATCH].to(device).permute(0, 1, 4, 2, 3).float().div_(255)
        b = torch.nn.functional.interpolate(
            b.flatten(0, 1), size=(SIDE, SIDE), mode="bilinear", align_corners=False
        ).unflatten(0, (b.shape[0], CLIP))
        b = (b - MEAN.to(device)) / STD.to(device)
        h = model(pixel_values_videos=b.half()).last_hidden_state  # [B, tokens, 1024]
        out.append(h.mean(1).float().cpu())
    return torch.cat(out).numpy().astype(np.float16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max routes (smoke test)")
    ap.add_argument("--workers", type=int, default=3, help="ffmpeg decode threads (bounded prefetch)")
    args = ap.parse_args()

    seg = pd.read_parquet(SEGQ)
    routes = seg.groupby(["model", "driver", "route"])
    keys = list(routes.groups.keys())
    if args.limit:
        keys = keys[:args.limit]
    print(f"routes with human spans: {len(keys)}", flush=True)

    device = "cuda"
    from transformers import VJEPA2Model
    model = VJEPA2Model.from_pretrained(
        CKPT, cache_dir="./data/.hfhub"
    ).to(device).half().eval()
    print(f"V-JEPA2 loaded ({CKPT})", flush=True)

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
        # bounded prefetch: at most `workers` decoded minutes in RAM at once
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
            if frames is None:
                continue
            emb = embed_clips(model, frames, device)
            if len(emb):
                store[f"seg{k}"] = emb
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
