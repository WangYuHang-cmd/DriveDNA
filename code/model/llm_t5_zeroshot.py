#!/usr/bin/env python3
"""
Zero-shot LLM / VLM rows for T5 event forecasting on the fixed anchor subsample.

Text mode: serialize the 5-s history (10 Hz, five Tier-A channels) and ask
"will a hard-braking or sharp-steering event begin within the next 5 s?";
score = p(yes) from the first generated token's logits over {yes, no}.
VLM mode: same question with three forward-camera frames (t-4, t-2, t0)
plus the serialized CAN text.

    PYTHONPATH= .../python llm_t5_zeroshot.py --model qwen3-4b
    PYTHONPATH= .../python llm_t5_zeroshot.py --model qwen3-vl-4b
Outputs results/llm_t5_{model}.json + per-anchor scores npz.
Frames cached under data/llm_bundle/frames/ (shared with the Colab bundle).
"""
import os, sys, json, argparse, subprocess
import numpy as np
import pandas as pd
import torch
from concurrent.futures import ThreadPoolExecutor

BASE = "."
DATASET = "../Dataset"
BUNDLE = f"{BASE}/data/colab_bundle"
FRAMES = f"{BASE}/data/llm_bundle/frames"
L_H = 50

TEXT_MODELS = {"qwen3-4b": "Qwen/Qwen3-4B", "qwen3-1.7b": "Qwen/Qwen3-1.7B",
               "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
               "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct"}
VLM_MODELS = {"qwen3-vl-4b": "Qwen/Qwen3-VL-4B-Instruct",
              "qwen2.5-vl-3b": "Qwen/Qwen2.5-VL-3B-Instruct"}

QUESTION = ("You observe 5 seconds of driving sensor data sampled at 10 Hz "
            "(columns: t_s, speed_mps, accel_mps2, curvature_x1000, yaw_dps, headway_s; "
            "headway 9.9 means no lead vehicle).\n{can}\n"
            "Question: will a hard-braking or sharp-steering event BEGIN within the "
            "next 5 seconds? Answer with exactly one word: yes or no.")
VQUESTION = ("These three images are forward-camera frames from a car at t=-4s, t=-2s, "
             "and now (t=0). The last 5 seconds of sensor data (10 Hz; columns: t_s, "
             "speed_mps, accel_mps2, curvature_x1000, yaw_dps, headway_s; headway 9.9 "
             "means no lead):\n{can}\n"
             "Question: will a hard-braking or sharp-steering event BEGIN within the "
             "next 5 seconds? Answer with exactly one word: yes or no.")


def serialize(w, ix, a):
    seg = w[a - L_H:a]
    v = seg[:, ix["vEgo"]]; ac = seg[:, ix["aEgo"]]
    k = seg[:, ix["actual_curvature"]] * 1e3
    y = seg[:, ix["yaw_rate"]]; dr = seg[:, ix["leadOne_dRel"]]
    thw = np.where((dr > 0) & (v > 2), dr / np.maximum(v, 2), 9.9)
    thw = np.clip(np.nan_to_num(thw, nan=9.9), 0, 9.9)
    rows = []
    for t in range(L_H):
        rows.append(f"{(t-L_H)/10.0:.1f},{v[t]:.1f},{np.nan_to_num(ac[t]):.2f},"
                    f"{np.nan_to_num(k[t]):.2f},{np.nan_to_num(y[t]):.2f},{thw[t]:.1f}")
    return "\n".join(rows)


def frame_path(r, dt):
    return f"{FRAMES}/g{r.gi}_a{r.a}_{dt}.jpg"


def grab_frames(sub, kmin):
    os.makedirs(FRAMES, exist_ok=True)
    def one(args):
        r, dt = args
        outp = frame_path(r, dt)
        if os.path.exists(outp) and os.path.getsize(outp) > 2000:
            return True
        km = kmin.get((r.model, r.driver, r.route))
        if km is None:
            return False
        tt = r.t_abs + dt
        if tt < 0:
            tt = 0.0
        p1 = os.path.join(DATASET, r.model, r.driver, r.route, f"{km + int(tt // 60)}--qcamera.ts")
        if not os.path.exists(p1):
            return False
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{tt % 60:.2f}", "-i", p1,
                        "-frames:v", "1", "-q:v", "4", "-vf", "scale=640:-1", outp],
                       capture_output=True, timeout=60)
        return os.path.exists(outp) and os.path.getsize(outp) > 2000
    jobs = [(r, dt) for r in sub.itertuples() for dt in (-4, -2, 0)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        oks = list(pool.map(one, jobs))
    ok_by_row = np.array(oks).reshape(-1, 3).all(1)
    print(f"frames ready for {ok_by_row.sum()}/{len(sub)} anchors", flush=True)
    return ok_by_row


def yes_no_ids(tok):
    ids_yes, ids_no = set(), set()
    for s in ("yes", "Yes", " yes", " Yes", "YES"):
        t = tok.encode(s, add_special_tokens=False)
        if len(t) >= 1:
            ids_yes.add(t[0])
    for s in ("no", "No", " no", " No", "NO"):
        t = tok.encode(s, add_special_tokens=False)
        if len(t) >= 1:
            ids_no.add(t[0])
    return sorted(ids_yes), sorted(ids_no)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(TEXT_MODELS) + list(VLM_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    dev = "cuda"
    is_vlm = args.model in VLM_MODELS

    X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
    ch = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
    ix = {c: i for i, c in enumerate(ch)}
    sub = pd.read_parquet(f"{BASE}/data/segments/t5_llm_subsample.parquet")
    if args.limit:
        sub = sub.iloc[:args.limit].copy()
    print(f"{args.model}: {len(sub)} anchors (vlm={is_vlm})", flush=True)

    if is_vlm:
        seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
        kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
        ok = grab_frames(sub, kmin)
        sub = sub[ok].reset_index(drop=True)
        from transformers import AutoProcessor, AutoModelForImageTextToText
        proc = AutoProcessor.from_pretrained(VLM_MODELS[args.model], cache_dir=f"{BASE}/data/.hfhub",
                                             min_pixels=256*28*28, max_pixels=512*28*28,
                                             padding_side="left")
        model = AutoModelForImageTextToText.from_pretrained(
            VLM_MODELS[args.model], dtype=torch.bfloat16,
            cache_dir=f"{BASE}/data/.hfhub").to(dev).eval()
        tok = proc.tokenizer
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(TEXT_MODELS[args.model])
        model = AutoModelForCausalLM.from_pretrained(
            TEXT_MODELS[args.model], dtype=torch.bfloat16).to(dev).eval()
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        proc = None
    ids_yes, ids_no = yes_no_ids(tok)
    print("loaded; yes/no token ids:", ids_yes, ids_no, flush=True)

    from PIL import Image
    scores, keep = [], []
    B = 2 if is_vlm else 4
    with torch.no_grad():
        for i in range(0, len(sub), B):
            rows = list(sub.iloc[i:i + B].itertuples())
            cans = [serialize(X[int(r.gi)].astype(np.float32), ix, int(r.a)) for r in rows]
            if is_vlm:
                msgs, imgs = [], []
                for r, can in zip(rows, cans):
                    ims = [Image.open(frame_path(r, dt)) for dt in (-4, -2, 0)]
                    imgs.append(ims)
                    msgs.append([{"role": "user", "content":
                                  [{"type": "image"}]*3 + [{"type": "text",
                                   "text": VQUESTION.format(can=can)}]}])
                texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in msgs]
                enc = proc(text=texts, images=imgs, return_tensors="pt", padding=True).to(dev)
            else:
                msgs = [[{"role": "user", "content": QUESTION.format(can=can)}] for can in cans]
                texts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
                         if "qwen3" in args.model else
                         tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
                         for m in msgs]
                enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(dev)
            try:
                out = model(**enc, logits_to_keep=1)     # keep only final-position logits
            except TypeError:
                out = model(**enc)
            logits = out.logits[:, -1, :].float()
            py = torch.logsumexp(logits[:, ids_yes], dim=-1)
            pn = torch.logsumexp(logits[:, ids_no], dim=-1)
            s = torch.sigmoid(py - pn).cpu().numpy()
            scores.extend(s.tolist()); keep.extend([r.Index for r in rows])
            if (i // B) % 40 == 0:
                print(f"  {i}/{len(sub)}", flush=True)
    scores = np.array(scores)
    y = sub.loc[keep].y.to_numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    res = {"auroc": float(roc_auc_score(y, scores)),
           "ap": float(average_precision_score(y, scores)),
           "n": int(len(y)), "pos_rate": float(y.mean()),
           "mean_score_pos": float(scores[y == 1].mean()),
           "mean_score_neg": float(scores[y == 0].mean())}
    print(f"{args.model} zero-shot:", res, flush=True)
    if not args.limit:
        json.dump(res, open(f"{BASE}/results/llm_t5_{args.model}.json", "w"), indent=1)
        np.savez(f"{BASE}/results/llm_t5_{args.model}_scores.npz", scores=scores, y=y)
        print("saved results")


if __name__ == "__main__":
    main()
