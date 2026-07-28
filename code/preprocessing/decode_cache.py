#!/usr/bin/env python3
"""
Batch-decode selected routes of one vehicle model to cached parquet (committed signals @20Hz).
Used to build the substrate for the within-vehicle identifiability sanity check.

    PYTHONPATH=../openpilot \
      ../openpilot/.venv/bin/python decode_cache.py \
      --model RIVIAN_R1_GEN1 --drivers d1,d2,... --max-routes 6 --max-segments 15

Writes: DriveDNA/data/cache/<model>/<driver>__<route>.parquet
"""
import os
import glob
import argparse
import pandas as pd
from rlog_extract import resample_route, find_segment_logs

DATASET = "../Dataset"
CACHE = "./data/cache"


def routes_with_rlog(model, driver):
    base = os.path.join(DATASET, model, driver)
    out = []
    for r in sorted(glob.glob(os.path.join(base, "*"))):
        if os.path.isdir(r) and find_segment_logs(r):
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--drivers", required=True, help="comma-separated dongle ids")
    ap.add_argument("--max-routes", type=int, default=6)
    ap.add_argument("--max-segments", type=int, default=15)
    ap.add_argument("--rate", type=float, default=20.0)
    args = ap.parse_args()

    outdir = os.path.join(CACHE, args.model)
    os.makedirs(outdir, exist_ok=True)
    drivers = args.drivers.split(",")
    manifest = []
    for drv in drivers:
        routes = routes_with_rlog(args.model, drv)[: args.max_routes]
        print(f"[{drv}] {len(routes)} routes")
        for r in routes:
            rid = os.path.basename(r)
            outp = os.path.join(outdir, f"{drv}__{rid}.parquet")
            if os.path.exists(outp):
                print(f"  cached {rid}"); manifest.append((drv, rid, "cached")); continue
            try:
                df = resample_route(r, rate_hz=args.rate, max_segments=args.max_segments)
                df.to_parquet(outp, index=False)
                hum = df["is_human"].mean() * 100
                print(f"  ok {rid}: {len(df)} rows, human {hum:.0f}%")
                manifest.append((drv, rid, "ok"))
            except Exception as e:
                print(f"  FAIL {rid}: {e}")
                manifest.append((drv, rid, f"fail:{e}"))
    pd.DataFrame(manifest, columns=["driver", "route", "status"]).to_csv(
        os.path.join(outdir, "_manifest.csv"), index=False)
    print("done ->", outdir)


if __name__ == "__main__":
    main()
