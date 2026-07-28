#!/usr/bin/env python3
"""
Package the Colab bundle for the large-model rows (Qwen2.5-VL-7B zero-shot,
Qwen3-8B LoRA-SFT) of the foundation-model baseline appendix.

Produces data/llm_bundle/:
  t5_train.jsonl   30k serialized train-fold anchors {id, text, label}
  t5_eval.jsonl    3k fixed subsample {id, text, label, frames[3]}
  frames/          three 640px JPEGs per eval anchor (shared with local runs)
  colab_run.py     self-contained runner (zero-shot VLM + LoRA-SFT text LLM)
  README_COLAB.md  step-by-step instructions
then tars everything to data/llm_bundle.tar.gz.

Run AFTER code/model/llm_t5_zeroshot.py has populated frames/ (it checks).
"""
import os, sys, json, tarfile
import numpy as np
import pandas as pd

BASE = "."
OUT = f"{BASE}/data/llm_bundle"
SEED = 20260709
N_TRAIN_POS, N_TRAIN_NEG = 10000, 20000

sys.path.insert(0, f"{BASE}/code/model")
from llm_t5_zeroshot import serialize, frame_path            # same serialization
from llm_t5_data import build_with_pos

os.makedirs(OUT, exist_ok=True)
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c: i for i, c in enumerate(ch)}
meta = pd.read_parquet(f"{BASE}/data/colab_bundle/windows_meta.parquet")
folds = json.load(open(f"{BASE}/data/colab_bundle/driver_folds.json"))

# ---- train JSONL ----
rng = np.random.default_rng(SEED)
tr_i = np.flatnonzero(meta.driver.isin(set(folds["train"])).to_numpy())
tr = build_with_pos(tr_i, X, ch.index("aEgo"), ch.index("actual_curvature"), rng)
pos = tr[tr.y == 1].sample(min(N_TRAIN_POS, (tr.y == 1).sum()), random_state=SEED)
neg = tr[tr.y == 0].sample(min(N_TRAIN_NEG, (tr.y == 0).sum()), random_state=SEED)
trs = pd.concat([pos, neg]).sample(frac=1, random_state=SEED).reset_index(drop=True)
with open(f"{OUT}/t5_train.jsonl", "w") as f:
    for i, r in enumerate(trs.itertuples()):
        f.write(json.dumps({"id": f"tr{i}", "label": int(r.y),
                            "text": serialize(X[int(r.gi)].astype(np.float32), ix, int(r.a))}) + "\n")
print(f"t5_train.jsonl: {len(trs)} ({trs.y.mean()*100:.0f}% pos)", flush=True)

# ---- eval JSONL (fixed subsample; frames must exist) ----
sub = pd.read_parquet(f"{BASE}/data/segments/t5_llm_subsample.parquet")
n_fr = 0
with open(f"{OUT}/t5_eval.jsonl", "w") as f:
    for i, r in enumerate(sub.itertuples()):
        fr = [os.path.relpath(frame_path(r, dt), OUT) for dt in (-4, -2, 0)]
        ok = all(os.path.exists(os.path.join(OUT, p)) for p in fr)
        n_fr += ok
        f.write(json.dumps({"id": f"ev{i}", "label": int(r.y),
                            "text": serialize(X[int(r.gi)].astype(np.float32), ix, int(r.a)),
                            "frames": fr if ok else None}) + "\n")
print(f"t5_eval.jsonl: {len(sub)} anchors, frames complete for {n_fr}", flush=True)

# ---- tar ----
with tarfile.open(f"{BASE}/data/llm_bundle.tar.gz", "w:gz") as t:
    t.add(OUT, arcname="llm_bundle")
sz = os.path.getsize(f"{BASE}/data/llm_bundle.tar.gz") / 1e9
print(f"bundle: data/llm_bundle.tar.gz ({sz:.2f} GB)", flush=True)
