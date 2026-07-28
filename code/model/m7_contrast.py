#!/usr/bin/env python3
"""
M7b — leakage-contrast completion: same driver/route/vehicle linear probes on
(a) the video-only SupCon embedding and (b) the CAN S1 embedding, SAME eval windows.

Completes the M7 story: video re-ID strength is carried by place/vehicle channels
(route/model probes) far more than CAN's, so treating video as an identity modality
would reward geographic/vehicle fingerprinting → video stays a context modality.
"""
import os
import json
import numpy as np
import pandas as pd
import torch

import sys
BASE = "."
sys.path.insert(0, f"{BASE}/code/eval")
sys.path.insert(0, f"{BASE}/code/model")
from s1_supcon import Encoder  # noqa: E402
from m7_video_probe import VidPool, linear_probe_acc  # noqa: E402

SEED = 20260709
BUNDLE = f"{BASE}/data/colab_bundle"


def main():
    rng = np.random.default_rng(SEED)
    dev = "cuda"
    X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
    V = np.load(f"{BUNDLE}/windows_vid.npy", mmap_mode="r")
    VN = np.load(f"{BUNDLE}/windows_vid_n.npy")
    meta = pd.read_parquet(f"{BUNDLE}/windows_meta.parquet")
    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    folds = json.load(open(f"{BUNDLE}/driver_folds.json"))
    ev = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy() & (VN > 0))
    me = meta.iloc[ev]
    vmodel = W["model"].to_numpy()[ev] if "model" in W.columns else me.get("model", pd.Series()).to_numpy()

    # CAN S1 embedding
    ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location=dev, weights_only=False)
    enc = Encoder(c_in=X.shape[2]).to(dev).eval(); enc.load_state_dict(ck["model"])
    Zc = []
    with torch.no_grad():
        for i in range(0, len(ev), 512):
            sl = ev[i:i + 512]
            a = (X[np.sort(sl)].astype(np.float32) - ck["mu"]) / ck["sd"]
            Zc.append(enc(torch.from_numpy(a[np.argsort(np.argsort(sl))]).to(dev)).cpu().numpy())
    Zc = np.concatenate(Zc)

    # video SupCon embedding — retrain quickly? no: recompute via saved probe if exists,
    # else re-derive from m7 checkpoint. m7 did not save the probe; retrain is wasteful —
    # instead reuse zero-shot mean-pool AND the trained probe path if present.
    Zv = None
    ckp = f"{BASE}/experiments/checkpoints/m7_vidpool.pt"
    if os.path.exists(ckp):
        vp = VidPool().to(dev).eval()
        vp.load_state_dict(torch.load(ckp, map_location=dev, weights_only=False)["model"])
        Zv = []
        with torch.no_grad():
            for i in range(0, len(ev), 512):
                sl = ev[i:i + 512]
                srt = np.sort(sl); inv = np.argsort(np.argsort(sl))
                v = torch.from_numpy(V[srt].astype(np.float32)[inv]).to(dev)
                n = torch.from_numpy(VN[srt][inv].astype(np.int64)).to(dev)
                Zv.append(vp(v, n).cpu().numpy())
        Zv = np.concatenate(Zv)
    else:  # fall back to zero-shot mean-pool video embedding
        Zv = np.zeros((len(ev), 768), np.float32)
        for j in range(0, len(ev), 1024):
            sl = ev[j:j + 1024]
            srt = np.sort(sl); inv = np.argsort(np.argsort(sl))
            v = V[srt].astype(np.float32)[inv]
            n = VN[srt][inv].astype(int)
            for i in range(len(sl)):
                Zv[j + i] = v[i, :max(n[i], 1)].mean(0)
        print("(m7_vidpool.pt not found — using zero-shot mean-pool video embedding)")

    out = {}
    print(f"eval windows: {len(ev)}")
    for name, Z in [("CAN-S1", Zc), ("video", Zv)]:
        row = {}
        for lab_name, lab in [("driver", me.driver.to_numpy()),
                              ("route", me.route.to_numpy()),
                              ("vehicle", vmodel)]:
            if lab is None or len(lab) != len(Z):
                continue
            acc, ch, ncls = linear_probe_acc(Z, lab, rng)
            row[lab_name] = {"acc": acc, "chance": ch, "n": ncls, "x": acc / ch}
            print(f"  {name:8} {lab_name:8} acc {acc:.3f}  chance {ch:.4f} ({ncls})  = {acc/ch:.1f}x")
        out[name] = row
    json.dump(out, open(f"{BASE}/results/m7_contrast.json", "w"), indent=1)
    print("saved results/m7_contrast.json")


if __name__ == "__main__":
    main()
