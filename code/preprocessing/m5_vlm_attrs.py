#!/usr/bin/env python3
"""
M5 — VLM structured scene attributes (Qwen2.5-VL-3B, LOCAL 5080 run; was A100 queue A2).

For every unique T4 matched-context pair window (17,848), extract the mid-window
keyframe (t0+30 s) from qcamera and ask the VLM for strict-JSON scene attributes:
  traffic_density · lead_vehicle · intersection · traffic_signal · road_type ·
  time_of_day · weather
Uses: (a) VISUAL validation of matched-context pairs (do matched pairs agree on
attributes?), (b) attribute-conditioned MCPP variant, (c) T6 evidence.

Resumable: appends to results/vlm_attrs.jsonl, skips done windows.
    PYTHONPATH= .../python m5_vlm_attrs.py [--limit N] [--batch 4]
"""
import os
import io
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from PIL import Image

import torch

BASE = "."
DATASET = "../Dataset"
OUTP = f"{BASE}/results/vlm_attrs.jsonl"
CKPT = "Qwen/Qwen2.5-VL-3B-Instruct"  # default; override with --ckpt

PROMPT = (
    "Look at this forward road-camera image. Respond with ONLY a JSON object, no other text:\n"
    '{"traffic_density": "none|light|moderate|heavy", "lead_vehicle": "yes|no", '
    '"intersection": "yes|no", "traffic_signal": "none|red|yellow|green", '
    '"road_type": "highway|urban|suburban|rural", "time_of_day": "day|dusk_dawn|night", '
    '"weather": "clear|overcast|rain|snow"}'
)


def grab_frame(args):
    """(path, offset_s) -> PIL.Image or None (mid-window keyframe via ffmpeg)."""
    path, off = args
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{off:.1f}", "-i", path,
           "-frames:v", "1", "-f", "image2", "-c:v", "mjpeg", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        return Image.open(io.BytesIO(raw)).convert("RGB") if raw else None
    except Exception:
        return None


def parse_json(text):
    try:
        a, b = text.find("{"), text.rfind("}")
        return json.loads(text[a:b + 1]) if a >= 0 <= b else None
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--out", default=OUTP)
    args = ap.parse_args()
    outp = args.out
    dev = "cuda"

    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    pairs = pd.read_parquet(f"{BASE}/data/splits/matched_context_pairs.parquet")
    seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
    kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
    need = np.unique(np.concatenate([pairs.win_a.to_numpy(), pairs.win_b.to_numpy()]))
    done = set()
    if os.path.exists(outp):
        with open(outp) as f:
            done = {json.loads(l)["gi"] for l in f if l.strip()}
    todo = [int(g) for g in need if int(g) not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"T4 pair windows: {len(need)}  done: {len(done)}  todo: {len(todo)}", flush=True)

    from transformers import AutoModelForImageTextToText, AutoProcessor
    model = AutoModelForImageTextToText.from_pretrained(
        args.ckpt, dtype=torch.bfloat16, cache_dir=f"{BASE}/data/.hfhub").to(dev).eval()
    proc = AutoProcessor.from_pretrained(args.ckpt, cache_dir=f"{BASE}/data/.hfhub",
                                         min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28,
                                         padding_side="left")
    print(f"VLM loaded: {args.ckpt}", flush=True)

    def frame_job(gi):
        r = W.iloc[gi]
        km = kmin.get((r.model, r.driver, r.route))
        if km is None:
            return gi, None
        t = float(r.t0) + 30.0
        k = km + int(t // 60)
        path = os.path.join(DATASET, r.model, r.driver, r.route, f"{k}--qcamera.ts")
        if not os.path.exists(path):
            return gi, None
        return gi, grab_frame((path, t % 60))

    pool = ThreadPoolExecutor(max_workers=4)
    fout = open(outp, "a")
    n_ok = n_fail = 0
    import time
    t0 = time.time()
    B = args.batch
    for ci in range(0, len(todo), B):
        chunk = todo[ci:ci + B]
        frames = list(pool.map(frame_job, chunk))
        frames = [(gi, im) for gi, im in frames if im is not None]
        n_fail += len(chunk) - len(frames)
        if not frames:
            continue
        msgs = [[{"role": "user", "content": [{"type": "image", "image": im},
                                              {"type": "text", "text": PROMPT}]}]
                for _, im in frames]
        texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
        inputs = proc(text=texts, images=[im for _, im in frames],
                      return_tensors="pt", padding=True).to(dev)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=96, do_sample=False)
        outs = proc.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
        for (gi, _), txt in zip(frames, outs):
            attrs = parse_json(txt)
            if attrs:
                fout.write(json.dumps({"gi": int(gi), **attrs}) + "\n")
                n_ok += 1
            else:
                n_fail += 1
        fout.flush()
        if (ci // B) % 50 == 0:
            el = time.time() - t0
            rate = (ci + B) / max(el, 1)
            print(f"  {ci+B}/{len(todo)}  ok={n_ok} fail={n_fail}  {rate:.1f} win/s  "
                  f"ETA {(len(todo)-ci-B)/max(rate,0.01)/60:.0f} min", flush=True)
    print(f"DONE: ok={n_ok} fail={n_fail} → {outp}", flush=True)


if __name__ == "__main__":
    main()
