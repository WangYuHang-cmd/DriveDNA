#!/usr/bin/env python3
"""
D1 — Human-driving segment extraction (the benchmark's atomic units).

From each route's 10 Hz signal parquet, extract the maximal contiguous runs where
`is_human == 1` (i.e. cs_enabled==0 AND cruiseState_enabled==0 — ONLY these two flags),
debounced by merging gaps < 2 s, keeping runs >= 30 s. Records k_min (the route's minimum
log-segment index) so span times map to `<seg>--qcamera.ts` files: seg = k_min + floor(t/60).

    ../openpilot/.venv/bin/python extract_segments.py
Output: DriveDNA/data/segments/segments.parquet  (one row per human-driving span)
"""
import os
import re
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_merge import canon  # noqa: E402

CACHE = "./data/cache"
DATASET = "../Dataset"
OUT = "./data/segments/segments.parquet"
RATE = 10.0
MIN_RUN_S = 30.0
MAX_GAP_S = 2.0


def runs_from_mask(mask, max_gap, min_len):
    """Maximal contiguous True runs; gaps < max_gap merged; runs < min_len dropped. Returns [(i0,i1)] inclusive."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    brk = np.flatnonzero(np.diff(idx) > max_gap)
    starts = np.r_[idx[0], idx[brk + 1]]
    ends = np.r_[idx[brk], idx[-1]]
    return [(int(s), int(e)) for s, e in zip(starts, ends) if (e - s + 1) >= min_len]


def kmin_for_route(model, driver, route):
    """Minimum log-segment index in the raw route dir (grid t=0 anchor)."""
    rd = os.path.join(DATASET, model, driver, route)
    ks = []
    for pat in ("*--qlog", "*--qlog.bz2", "*--qlog.zst", "*--rlog", "*--rlog.zst", "*--rlog.bz2"):
        for p in glob.glob(os.path.join(rd, pat)):
            m = re.match(r"(\d+)--", os.path.basename(p))
            if m:
                ks.append(int(m.group(1)))
    return min(ks) if ks else 0


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    paths = sorted(glob.glob(os.path.join(CACHE, "*", "*.parquet")))
    print(f"extracting human-driving spans from {len(paths)} routes ...", flush=True)
    rows = []
    for i, p in enumerate(paths):
        model = os.path.basename(os.path.dirname(p))
        base = os.path.basename(p)[:-8]
        driver, route = base.split("__", 1)
        try:
            df = pd.read_parquet(p, columns=["time_s", "vEgo", "is_human"])
        except Exception:
            continue
        mask = df["is_human"].to_numpy() > 0.5
        spans = runs_from_mask(mask, int(MAX_GAP_S * RATE), int(MIN_RUN_S * RATE))
        if not spans:
            continue
        kmin = kmin_for_route(model, driver, route)
        t = df["time_s"].to_numpy()
        v = df["vEgo"].to_numpy()
        for sid, (i0, i1) in enumerate(spans):
            vv = v[i0:i1 + 1]
            rows.append({
                "model": model, "model_canon": canon(model), "driver": driver, "route": route,
                "span_id": sid, "i0": i0, "i1": i1,
                "t0": float(t[i0]), "t1": float(t[i1]),
                "dur_s": float(t[i1] - t[i0]),
                "moving_frac": float(np.mean(vv > 2.0)),
                "k_min": kmin,
            })
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(paths)}  spans so far: {len(rows)}", flush=True)
    seg = pd.DataFrame(rows)
    seg.to_parquet(OUT, index=False)
    tot_h = seg["dur_s"].sum() / 3600
    mov_h = (seg["dur_s"] * seg["moving_frac"]).sum() / 3600
    print(f"\nwrote {OUT}")
    print(f"spans={len(seg)}  routes={seg.route.nunique()}  drivers={seg.driver.nunique()}  "
          f"models={seg.model_canon.nunique()}")
    print(f"human hours={tot_h:.0f}  human+moving hours={mov_h:.0f}  "
          f"median span={seg.dur_s.median():.0f}s  p90={seg.dur_s.quantile(.9):.0f}s")


if __name__ == "__main__":
    main()
