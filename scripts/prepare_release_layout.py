#!/usr/bin/env python3
"""Build the working layout that the DriveDNA scripts expect from the public Hugging Face release.

The scripts in code/ were written against an internal layout (data/segments/windows.parquet,
data/colab_bundle/, data/splits/, data/cache/<model>/<driver>__<drive>.parquet, data/video_emb*/...)
and against the column names `driver`, `route`, `model`. The public release uses `driver`, `drive`,
`model_canon` and a flatter tree. This script creates that layout under the current directory:
tables are copied with alias columns added (`route` = `drive`, `model` = `model_canon`); large
arrays, embeddings and checkpoints are symlinked.

    hf download HenryYHW/DriveDNA        --repo-type dataset --local-dir ./hf/DriveDNA
    hf download HenryYHW/DriveDNA-models --repo-type model   --local-dir ./hf/DriveDNA-models
    python scripts/prepare_release_layout.py --data ./hf/DriveDNA --models ./hf/DriveDNA-models

Run from the repository root; afterwards e.g. `python code/eval/t4_eval.py` works unchanged.
Raw openpilot logs (../Dataset) are not released; preprocessing scripts that need them are documented in README.md.
"""
import argparse
import glob
import os
import shutil

import pandas as pd


def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        os.remove(dst)
    os.symlink(os.path.abspath(src), dst)


def table_with_aliases(src, dst):
    df = pd.read_parquet(src)
    if "drive" in df.columns and "route" not in df.columns:
        df["route"] = df["drive"]
    if "model_canon" in df.columns and "model" not in df.columns:
        df["model"] = df["model_canon"]
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    df.to_parquet(dst, index=False)
    return len(df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="local copy of the HenryYHW/DriveDNA dataset repo")
    ap.add_argument("--models", default=None, help="local copy of the HenryYHW/DriveDNA-models repo (optional)")
    ap.add_argument("--root", default=".", help="repository root to populate (default: current directory)")
    a = ap.parse_args()
    D, R = a.data, a.root
    log = []

    # tables the scripts read from data/segments/ and data/splits/
    for src, dst in [("data/windows.parquet", "data/segments/windows.parquet"),
                     ("index/human_segments.parquet", "data/segments/segments.parquet"),
                     ("data/maneuver_events.parquet", "data/segments/maneuver_events.parquet"),
                     ("data/t5_llm_subsample.parquet", "data/segments/t5_llm_subsample.parquet"),
                     ("data/lane_changes_verified.parquet", "data/segments/lane_changes_verified.parquet"),
                     ("splits/matched_condition_pairs.parquet", "data/splits/matched_context_pairs.parquet")]:
        if os.path.exists(f"{D}/{src}"):
            n = table_with_aliases(f"{D}/{src}", f"{R}/{dst}"); log.append(f"{dst}: {n:,} rows (from {src})")
    for f in glob.glob(f"{D}/splits/*.json"):
        link(f, f"{R}/data/splits/{os.path.basename(f)}"); log.append(f"data/splits/{os.path.basename(f)}")

    # the "colab bundle": window tensors, channel list, video features and the driver folds
    for f in glob.glob(f"{D}/features/*"):
        link(f, f"{R}/data/colab_bundle/{os.path.basename(f)}")
    link(f"{D}/splits/driver_folds.json", f"{R}/data/colab_bundle/driver_folds.json")
    log.append(f"data/colab_bundle/: {len(os.listdir(f'{R}/data/colab_bundle'))} files")

    # per-drive 10 Hz tables and video embeddings, keyed <driver>__<drive> under the vehicle model
    idx = pd.read_parquet(f"{D}/index/drives.parquet")
    n_cache = n_emb = 0
    fam_dirs = {"dinov2": "video_emb", "dinov3": "video_emb_dinov3", "siglip2": "video_emb_siglip2", "vjepa2": "video_emb_vjepa2"}
    for r in idx.itertuples():
        src = f"{D}/raw_signals/{r.driver}/{r.drive}.parquet"
        if os.path.exists(src):
            link(src, f"{R}/data/cache/{r.model_canon}/{r.driver}__{r.drive}.parquet"); n_cache += 1
        for fam, d in fam_dirs.items():
            e = f"{D}/embeddings/{fam}/{r.driver}/{r.drive}.npz"
            if os.path.exists(e):
                link(e, f"{R}/data/{d}/{r.model_canon}/{r.driver}__{r.drive}.npz"); n_emb += 1
    log.append(f"data/cache/: {n_cache} tables; data/video_emb*/: {n_emb} embedding files")

    for f in glob.glob(f"{D}/data/vlm_attrs*.jsonl"):
        link(f, f"{R}/results/{os.path.basename(f)}"); log.append(f"results/{os.path.basename(f)}")

    if a.models:
        n = 0
        for f in glob.glob(f"{a.models}/*.pt"):
            link(f, f"{R}/experiments/checkpoints/{os.path.basename(f)}"); n += 1
        log.append(f"experiments/checkpoints/: {n} checkpoints")
    os.makedirs(f"{R}/experiments/logs", exist_ok=True)
    print("\n".join(log))


if __name__ == "__main__":
    main()
