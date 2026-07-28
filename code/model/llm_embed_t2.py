#!/usr/bin/env python3
"""
LLM-as-text-encoder rows for T2 (zero-shot, no training).

Each 60-s window is serialized to a compact 1-Hz CSV of five Tier-A channels
(speed, accel, curvature*1e3, yaw rate, THW) and passed through a frozen
decoder LLM; the attention-masked mean of the last hidden layer is the window
embedding, evaluated with harness.enrollment_protocol on the SAME unseen-driver
population as every other Table-2 row (val + few-shot folds).

Extends the foundation-model finding of Sec. 6.2 (MOMENT-1) from time-series
to language models: does generic pretraining encode driver identity? Zero-shot.

    PYTHONPATH= .../python llm_embed_t2.py --model qwen3-4b [--smoke]
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")
SEED = 20260709

MODELS = {
    "qwen3-4b": "Qwen/Qwen3-4B",
    "qwen3-1.7b": "Qwen/Qwen3-1.7B",
    "llama-3.2-3b": "meta-llama/Llama-3.2-3B-Instruct",
    "qwen2.5-7b": "Qwen/Qwen2.5-7B-Instruct",
}
HEADER = ("Driving sensor log, 1 Hz, 60 s.\n"
          "t,speed_mps,accel_mps2,curv_1e3,yaw_dps,headway_s\n")


def serialize(w, ix):
    v = w[:, ix["vEgo"]]; a = w[:, ix["aEgo"]]
    k = w[:, ix["actual_curvature"]] * 1e3
    y = w[:, ix["yaw_rate"]]; dr = w[:, ix["leadOne_dRel"]]
    thw = np.where((dr > 0) & (v > 2), dr / np.maximum(v, 2), np.nan)
    rows = [HEADER.rstrip("\n")]
    for t in range(0, 600, 10):
        sl = slice(t, t + 10)
        tv = np.nanmean(v[sl]); ta = np.nanmean(a[sl])
        tk = np.nanmean(k[sl]); ty = np.nanmean(y[sl])
        th = np.nanmean(thw[sl])
        th = 9.9 if not np.isfinite(th) or th > 9.9 else th
        rows.append(f"{t//10},{tv:.1f},{ta:.2f},{np.nan_to_num(tk):.2f},"
                    f"{np.nan_to_num(ty):.2f},{th:.1f}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3-4b", choices=list(MODELS))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    dev = "cuda"

    X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
    ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
    ix = {c: i for i, c in enumerate(ch)}
    meta = pd.read_parquet(f"{BASE}/data/colab_bundle/windows_meta.parquet")
    folds = json.load(open(f"{BASE}/data/colab_bundle/driver_folds.json"))
    ev_idx = np.flatnonzero(
        meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
    if args.smoke:
        ev_idx = ev_idx[:200]
    print(f"model={args.model} eval windows={len(ev_idx)}", flush=True)

    from transformers import AutoTokenizer, AutoModel
    ckpt = MODELS[args.model]
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModel.from_pretrained(ckpt, dtype=torch.bfloat16).to(dev).eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print("loaded", ckpt, flush=True)

    Z = []
    with torch.no_grad():
        for i in range(0, len(ev_idx), args.batch):
            sl = np.sort(ev_idx[i:i + args.batch])
            texts = [serialize(X[int(g)].astype(np.float32), ix) for g in sl]
            enc = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=1024).to(dev)
            out = model(**enc)
            h = out.last_hidden_state.float()                       # [B, T, D]
            m = enc["attention_mask"].unsqueeze(-1).float()
            emb = (h * m).sum(1) / m.sum(1).clamp(min=1)
            # restore original order within batch
            order = np.argsort(np.argsort(ev_idx[i:i + args.batch]))
            Z.append(emb.cpu().numpy()[order])
            if (i // args.batch) % 50 == 0:
                print(f"  {i}/{len(ev_idx)}", flush=True)
    Z = np.concatenate(Z)
    print("embeddings:", Z.shape, flush=True)
    np.savez(f"{BASE}/results/llm_t2_{args.model}.npz", Z=Z, idx=ev_idx)

    me = meta.iloc[ev_idx]
    from harness import enrollment_protocol
    res = enrollment_protocol(Z, me.driver.to_numpy(), me.route.to_numpy())
    print(f"\n== {args.model} zero-shot text-encoder on T2 (UNSEEN drivers) ==")
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    if not args.smoke:
        json.dump({str(k): v for k, v in res.items()},
                  open(f"{BASE}/results/llm_t2_{args.model}.json", "w"), indent=1)
        print("saved", f"results/llm_t2_{args.model}.json")


if __name__ == "__main__":
    main()
