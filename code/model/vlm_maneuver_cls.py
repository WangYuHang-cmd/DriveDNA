#!/usr/bin/env python3
"""
Zero-shot VLM 6-way maneuver classification, evaluated on HUMAN-VERIFIED events
from the audited maneuver layer (the first use of the completed annotation as an
evaluation set).

Eval set: 300 verified (yes-labeled) events per class, fixed seed.  Four frames
are drawn from each event's audit clip; the VLM answers a single letter (a-f);
the score is the argmax over first-token letter logits.

    PYTHONPATH= .../python vlm_maneuver_cls.py --model qwen3-vl-4b
Outputs results/vlm_maneuver_{model}.json (accuracy, macro-F1, confusion).
"""
import os, sys, json, argparse, subprocess
import numpy as np
import pandas as pd
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

BASE = "."
OUT = f"{BASE}/results/maneuver_audit"
FRDIR = f"{BASE}/data/llm_bundle/mframes"
SEED = 20260709
PER_CLASS = 300
MODELS = {"qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
          "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct"}
CLASSES = ["decel", "accel", "turn", "curve", "car_following", "lane_change"]
LETTER = dict(zip(CLASSES, "abcdef"))
PROMPT = ("These four images are consecutive forward-camera frames (about 8 seconds) "
          "from a car driven by a human. Which maneuver does the ego vehicle perform?\n"
          "(a) deceleration / braking\n(b) acceleration / speeding up\n"
          "(c) turning at an intersection\n(d) cornering on a curved road\n"
          "(e) car following (steady behind a lead vehicle)\n(f) lane change\n"
          "Answer with exactly one letter (a-f).")
FILES = {"decel": "decelp", "accel": "accelv6p", "turn": "turnv2p",
         "curve": "curvep", "car_following": "car_followingp"}


def load_verified():
    man = pd.read_csv(f"{OUT}/manifest.csv")
    cur = {(r.cls, r.eid): int(r.gi) for r in man.itertuples()}
    ver = defaultdict(dict)
    for fn in sorted(os.listdir(f"{OUT}/answers")):
        if not fn.endswith(".json"):
            continue
        cls = next((c for c, p in FILES.items() if fn.startswith(p)), None)
        if cls is None and (fn.startswith("lane_changev2p") or fn.startswith("lanemergedp")):
            cls = "lane_change"
        if cls is None:
            continue
        for x in json.load(open(f"{OUT}/answers/{fn}")):
            if x.get("answer") == "yes" and cur.get((cls, x["eid"])) == x["gi"]:
                ver[cls][x["eid"]] = x["gi"]
    return ver


def grab(eid, cls):
    """4 frames from the event's audit clip."""
    outs = [f"{FRDIR}/{cls}_e{eid}_{i}.jpg" for i in range(4)]
    if all(os.path.exists(o) and os.path.getsize(o) > 2000 for o in outs):
        return outs
    clip = f"{OUT}/clips/e{eid}.mp4"
    if not os.path.exists(clip):
        return None
    for i, ts in enumerate((1.0, 3.5, 6.0, 8.5)):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{ts}", "-i", clip,
                        "-frames:v", "1", "-q:v", "4", "-vf", "scale=640:-1", outs[i]],
                       capture_output=True, timeout=60)
    return outs if all(os.path.exists(o) and os.path.getsize(o) > 2000 for o in outs) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-vl-4b", choices=list(MODELS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(FRDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ver = load_verified()
    items = []
    for cls in CLASSES:
        eids = sorted(ver[cls])
        print(f"{cls}: {len(eids)} verified", flush=True)
        pick = rng.choice(eids, min(PER_CLASS, len(eids)), replace=False)
        items += [(cls, int(e)) for e in pick]
    if args.limit:
        items = items[:args.limit]
    print(f"eval items: {len(items)}", flush=True)

    with ThreadPoolExecutor(max_workers=6) as pool:
        frames = list(pool.map(lambda t: grab(t[1], t[0]), items))
    ok = [i for i, f in enumerate(frames) if f]
    print(f"frames ready: {len(ok)}/{len(items)}", flush=True)

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(MODELS[args.model], cache_dir=f"{BASE}/data/.hfhub",
                                         min_pixels=256*28*28, max_pixels=512*28*28,
                                         padding_side="left")
    model = AutoModelForImageTextToText.from_pretrained(
        MODELS[args.model], dtype=torch.bfloat16, cache_dir=f"{BASE}/data/.hfhub").to("cuda").eval()
    tok = proc.tokenizer
    lids = {c: [tok.encode(s, add_special_tokens=False)[0]
                for s in (LETTER[c], LETTER[c].upper(), " " + LETTER[c], "(" + LETTER[c])]
            for c in CLASSES}

    preds, gts = [], []
    B = 2
    with torch.no_grad():
        for bi in range(0, len(ok), B):
            idxs = ok[bi:bi + B]
            imgs = [[Image.open(p) for p in frames[i]] for i in idxs]
            msgs = [[{"role": "user", "content": [{"type": "image"}]*4 +
                      [{"type": "text", "text": PROMPT}]}] for _ in idxs]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to("cuda")
            try:
                out = model(**enc, logits_to_keep=1)
            except TypeError:
                out = model(**enc)
            lg = out.logits[:, -1, :].float()
            for row, i in enumerate(idxs):
                sc = {c: float(torch.logsumexp(lg[row, ids], -1)) for c, ids in lids.items()}
                preds.append(max(sc, key=sc.get))
                gts.append(items[i][0])
            if (bi // B) % 100 == 0:
                print(f"  {bi}/{len(ok)}", flush=True)

    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
    acc = accuracy_score(gts, preds)
    f1 = f1_score(gts, preds, average="macro")
    cm = confusion_matrix(gts, preds, labels=CLASSES).tolist()
    per_class = {c: float(np.mean([p == c for p, g in zip(preds, gts) if g == c])) for c in CLASSES}
    res = {"model": args.model, "n": len(gts), "accuracy": float(acc),
           "macro_f1": float(f1), "per_class_recall": per_class,
           "confusion(rows=gt)": cm, "classes": CLASSES}
    print(json.dumps(res, indent=1), flush=True)
    if not args.limit:
        json.dump(res, open(f"{BASE}/results/vlm_maneuver_{args.model}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
