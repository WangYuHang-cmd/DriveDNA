#!/usr/bin/env python3
"""
V2 — VLM-generation audit: Qwen3-VL-4B (Oct 2025) vs Qwen2.5-VL-3B attributes.

Re-annotate a random subsample of already-annotated T4 windows with Qwen3-VL and
report per-attribute agreement. Decision rule: full 17.8k re-run ONLY if agreement is
materially low (< ~0.75 on matched-relevant attributes); else appendix note
"attributes robust across VLM generations".

    PYTHONPATH= .../python m5_qwen3_audit.py [--n 1500]
Output: results/vlm_qwen3_audit.json (+ per-window jsonl for reuse)
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

import torch

import sys
sys.path.insert(0, "./code/preprocessing")
from m5_vlm_attrs import grab_frame, parse_json, PROMPT  # noqa: E402

BASE = "."
DATASET = "../Dataset"
CKPT = "Qwen/Qwen3-VL-4B-Instruct"
KEYS = ["traffic_density", "lead_vehicle", "intersection", "traffic_signal",
        "road_type", "time_of_day", "weather"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()
    dev = "cuda"
    rng = np.random.default_rng(20260710)

    old = {}
    with open(f"{BASE}/results/vlm_attrs.jsonl") as f:
        for l in f:
            d = json.loads(l)
            old[d.pop("gi")] = d
    sample = rng.choice(sorted(old), min(args.n, len(old)), replace=False)
    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
    kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
    print(f"audit sample: {len(sample)} windows", flush=True)

    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        CKPT, dtype=torch.bfloat16, cache_dir=f"{BASE}/data/.hfhub").to(dev).eval()
    proc = AutoProcessor.from_pretrained(CKPT, cache_dir=f"{BASE}/data/.hfhub",
                                         min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28)
    print("Qwen3-VL-4B loaded", flush=True)

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
    new = {}
    fout = open(f"{BASE}/results/vlm_attrs_qwen3_audit.jsonl", "w")
    B = args.batch
    todo = [int(g) for g in sample]
    for ci in range(0, len(todo), B):
        frames = [(gi, im) for gi, im in pool.map(frame_job, todo[ci:ci + B]) if im is not None]
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
                new[gi] = attrs
                fout.write(json.dumps({"gi": int(gi), **attrs}) + "\n")
        if (ci // B) % 50 == 0:
            print(f"  {ci+B}/{len(todo)}  ok={len(new)}", flush=True)
    fout.close()

    both = [gi for gi in new if gi in old]
    out = {"n": len(both)}
    print(f"\n== Qwen3-VL-4B vs Qwen2.5-VL-3B agreement (n={len(both)}) ==")
    for k in KEYS:
        a = np.mean([old[gi].get(k) == new[gi].get(k) for gi in both])
        out[k] = float(a)
        print(f"  {k:18} {a:.3f}")
    json.dump(out, open(f"{BASE}/results/vlm_qwen3_audit.json", "w"), indent=1)
    print("saved results/vlm_qwen3_audit.json")


if __name__ == "__main__":
    main()
