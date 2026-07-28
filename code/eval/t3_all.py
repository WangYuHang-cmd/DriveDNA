#!/usr/bin/env python3
"""
T3 completion: evaluate EVERY representation on the 14,868 matched-context pairs
(same cosine-verification protocol as t4_eval.py, which produced the published
descriptor .550 and SupCon .812 rows).

Rows: descriptors, S1 SupCon (sanity), W1 CI-PatchTST / iTransformer / ArcFace,
W2 masked-SSL / JEPA-SSL probes, M4 CLIP-aligned CAN, MOMENT-1 (zero-shot),
Qwen3-4B text embedding (zero-shot), video-only (DINOv2 mean-pooled, diagnostic).

    PYTHONPATH= .../python t3_all.py            # all cheap rows
    PYTHONPATH= .../python t3_all.py --moment --llm   # add the slow zero-shot rows
Writes results/t3_all.json (auroc/eer per representation).
"""
import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")
BUNDLE = f"{BASE}/data/colab_bundle"
CK = f"{BASE}/experiments/checkpoints"

from harness import verification_metrics, STATS
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

pairs = pd.read_parquet(f"{BASE}/data/splits/matched_context_pairs.parquet")
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
need = np.unique(np.concatenate([pairs.win_a.to_numpy(), pairs.win_b.to_numpy()]))
y = pairs.same_driver.to_numpy()
ia, ib = pairs.win_a.to_numpy(), pairs.win_b.to_numpy()
print(f"pairs={len(pairs)} unique windows={len(need)}", flush=True)
results = {}


def pair_auroc_from_Z(Z, name):
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    s = (Zn[ia] * Zn[ib]).sum(1)
    m = verification_metrics(s, y)
    results[name] = m
    print(f"{name:34s} AUROC {m['auroc']:.3f}  EER {m['eer']:.3f}", flush=True)


def embed_torch(enc, mu, sd, dim=128, bs=512):
    Z = np.zeros((len(W), dim), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(need), bs):
            idx = need[i:i + bs]
            a = (X[idx].astype(np.float32) - mu) / sd
            Z[idx] = enc(torch.from_numpy(np.nan_to_num(a)).cuda()).cpu().numpy()
    return Z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moment", action="store_true")
    ap.add_argument("--llm", action="store_true")
    args = ap.parse_args()

    # ---- descriptors (anchor, -L2) ----
    D = W[STATS].to_numpy(float)
    D = StandardScaler().fit_transform(SimpleImputer(strategy="median").fit_transform(D))
    s = -np.linalg.norm(D[ia] - D[ib], axis=1)
    results["descriptors"] = verification_metrics(s, y)
    print(f"{'descriptors':34s} AUROC {results['descriptors']['auroc']:.3f}", flush=True)

    # ---- S1 SupCon (sanity vs published .812) ----
    from s1_supcon import Encoder
    ck = torch.load(f"{CK}/s1_supcon.pt", map_location="cuda", weights_only=False)
    enc = Encoder(c_in=X.shape[2]).cuda().eval(); enc.load_state_dict(ck["model"])
    pair_auroc_from_Z(embed_torch(enc, ck["mu"], ck["sd"]), "s1_supcon (sanity)")

    # ---- W1 backbones ----
    from w1_backbones import CIPatchTST, ITransformer
    for tag, cls in (("w1_ci", CIPatchTST), ("w1_itr", ITransformer), ("w1_arcface", Encoder)):
        try:
            ck = torch.load(f"{CK}/{tag}.pt", map_location="cuda", weights_only=False)
            m = cls(c_in=X.shape[2]).cuda().eval()
            m.load_state_dict(ck["model"])
            pair_auroc_from_Z(embed_torch(m, ck["mu"], ck["sd"]), tag)
        except Exception as e:
            print(f"{tag}: SKIP ({type(e).__name__}: {e})", flush=True)

    # ---- W2 SSL probes ----
    from w2_ssl import Trunk, PoolHead
    for tag in ("w2_masked", "w2_jepa"):
        try:
            ck = torch.load(f"{CK}/{tag}.pt", map_location="cuda", weights_only=False)
            trunk = Trunk(c_in=X.shape[2]).cuda().eval(); trunk.load_state_dict(ck["trunk"])
            head = PoolHead().cuda().eval(); head.load_state_dict(ck["head"])
            enc = lambda x: head(trunk(x))
            pair_auroc_from_Z(embed_torch(enc, ck["mu"], ck["sd"]), tag)
        except Exception as e:
            print(f"{tag}: SKIP ({type(e).__name__}: {e})", flush=True)

    # ---- M4 CLIP-aligned CAN ----
    try:
        ck = torch.load(f"{CK}/m4_clip.pt", map_location="cuda", weights_only=False)
        enc = Encoder(c_in=X.shape[2]).cuda().eval(); enc.load_state_dict(ck["can"])
        pair_auroc_from_Z(embed_torch(enc, ck["mu"], ck["sd"]), "m4_clip_can")
    except Exception as e:
        print(f"m4_clip: SKIP ({e})", flush=True)

    # ---- video-only (DINOv2 mean-pooled tokens; diagnostic) ----
    try:
        V = np.load(f"{BUNDLE}/windows_vid12.npy", mmap_mode="r")
        Z = np.zeros((len(W), V.shape[2]), dtype=np.float32)
        for i in range(0, len(need), 512):
            idx = need[i:i + 512]
            Z[idx] = np.asarray(V[idx]).mean(1)
        pair_auroc_from_Z(Z, "video_only_dinov2")
    except Exception as e:
        print(f"video: SKIP ({e})", flush=True)

    # ---- MOMENT-1 zero-shot ----
    if args.moment:
        from momentfm import MOMENTPipeline
        mom = MOMENTPipeline.from_pretrained("AutonLab/MOMENT-1-large",
                                             model_kwargs={"task_name": "embedding"},
                                             cache_dir=f"{BASE}/data/.hfhub")
        mom.init(); mom = mom.cuda().bfloat16().eval()
        samp = X[np.sort(np.random.default_rng(0).choice(len(W), 3000, replace=False))].astype(np.float32)
        mu = samp.reshape(-1, X.shape[2]).mean(0); sd = samp.reshape(-1, X.shape[2]).std(0) + 1e-3
        Z = np.zeros((len(W), 1024), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(need), 16):
                idx = need[i:i + 16]
                w = ((X[idx].astype(np.float32) - mu) / sd)[:, -512:]
                x = torch.from_numpy(w).permute(0, 2, 1).cuda().bfloat16()
                Z[idx] = mom(x_enc=x).embeddings.float().cpu().numpy()
                if (i // 16) % 200 == 0:
                    print(f"  moment {i}/{len(need)}", flush=True)
        pair_auroc_from_Z(Z, "moment1_zeroshot")

    # ---- Qwen3-4B text embedding zero-shot ----
    if args.llm:
        from llm_embed_t2 import serialize, MODELS
        from transformers import AutoTokenizer, AutoModel
        ch = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
        ix2 = {c: i for i, c in enumerate(ch)}
        tok = AutoTokenizer.from_pretrained(MODELS["qwen3-4b"])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lm = AutoModel.from_pretrained(MODELS["qwen3-4b"], dtype=torch.bfloat16).cuda().eval()
        Z = np.zeros((len(W), 2560), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(need), 16):
                idx = np.sort(need[i:i + 16])
                texts = [serialize(X[int(g)].astype(np.float32), ix2) for g in idx]
                enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                          max_length=1024).to("cuda")
                out = lm(**enc)
                h = out.last_hidden_state.float()
                m = enc["attention_mask"].unsqueeze(-1).float()
                Z[idx] = ((h * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy()[:, :2560]
                if (i // 16) % 100 == 0:
                    print(f"  llm {i}/{len(need)}", flush=True)
        pair_auroc_from_Z(Z, "qwen3_4b_text_zeroshot")

    json.dump(results, open(f"{BASE}/results/t3_all.json", "w"), indent=1)
    print("saved results/t3_all.json", flush=True)


if __name__ == "__main__":
    main()
