#!/usr/bin/env python3
"""
Full-corpus parallel decode: every route with rlogs -> committed-signal parquet @20Hz.
Resumable (skips existing outputs), RAM-aware worker pool. This is the sprint's Day-0 long pole.

    PYTHONPATH=../openpilot \
      ../openpilot/.venv/bin/python decode_corpus.py --workers 6

Writes: DriveDNA/data/cache/<model>/<driver>__<route>.parquet
Progress log: DriveDNA/experiments/logs/decode_corpus_progress.csv
"""
import os
import sys
import glob
import time
import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.filterwarnings("ignore")

DATASET = "../Dataset"
CACHE = "./data/cache"
LOG = "./experiments/logs/decode_corpus_progress.csv"


def list_all_routes(models=None):
    """Every (model, driver, route_dir) that has >=1 log segment (rlog OR qlog) — full corpus."""
    routes = []
    model_glob = models if models else ["*"]
    seen = set()
    logpats = ("*--rlog", "*--rlog.zst", "*--rlog.bz2", "*--qlog", "*--qlog.bz2", "*--qlog.zst")
    for mg in model_glob:
        for mp in sorted(glob.glob(os.path.join(DATASET, mg))):
            if not os.path.isdir(mp):
                continue
            model = os.path.basename(mp)
            for dp in sorted(glob.glob(os.path.join(mp, "*"))):
                if not os.path.isdir(dp):
                    continue
                driver = os.path.basename(dp)
                for rp in sorted(glob.glob(os.path.join(dp, "*"))):
                    if not os.path.isdir(rp):
                        continue
                    key = (model, driver, os.path.basename(rp))
                    if key in seen:
                        continue
                    if any(glob.glob(os.path.join(rp, pat)) for pat in logpats):
                        seen.add(key)
                        routes.append((model, driver, rp))
    return routes


def _decode_one(args):
    model, driver, route_dir, rate, max_segments = args
    from rlog_extract import resample_route, find_segment_logs
    rid = os.path.basename(route_dir)
    outdir = os.path.join(CACHE, model)
    os.makedirs(outdir, exist_ok=True)
    outp = os.path.join(outdir, f"{driver}__{rid}.parquet")
    if os.path.exists(outp):
        return (model, driver, rid, "cached", 0, -1)
    # log-type composition for V6 (fraction of segments sourced from rlog)
    logs = find_segment_logs(route_dir)
    rlog_frac = round(sum("--rlog" in p for p in logs) / max(len(logs), 1), 3)
    t0 = time.time()
    try:
        df = resample_route(route_dir, rate_hz=rate, max_segments=max_segments)
        if len(df) < 200:
            return (model, driver, rid, "too_short", round(time.time() - t0, 1), rlog_frac)
        df.to_parquet(outp, index=False)
        return (model, driver, rid, f"ok:{len(df)}rows", round(time.time() - t0, 1), rlog_frac)
    except Exception as e:
        return (model, driver, rid, f"fail:{type(e).__name__}", round(time.time() - t0, 1), rlog_frac)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6, help="parallel processes (RAM-aware; ~1GB each)")
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--max-segments", type=int, default=None, help="cap segments/route (None=all)")
    ap.add_argument("--models", nargs="*", default=None, help="restrict to these model dir names")
    args = ap.parse_args()

    routes = list_all_routes(models=args.models)
    print(f"found {len(routes)} routes with any log; workers={args.workers} rate={args.rate} "
          f"max_seg={args.max_segments}", flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    tasks = [(m, d, r, args.rate, args.max_segments) for (m, d, r) in routes]

    done = ok = fail = cached = 0
    t0 = time.time()
    with open(LOG, "w") as lg:
        lg.write("model,driver,route,status,sec,rlog_frac\n")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_decode_one, t) for t in tasks]
            for fut in as_completed(futs):
                m, d, rid, status, sec, rlog_frac = fut.result()
                lg.write(f"{m},{d},{rid},{status},{sec},{rlog_frac}\n"); lg.flush()
                done += 1
                if status.startswith("ok"):
                    ok += 1
                elif status == "cached":
                    cached += 1
                elif status.startswith("fail") or status == "too_short":
                    fail += 1
                if done % 50 == 0 or done == len(tasks):
                    el = time.time() - t0
                    rate = done / el if el else 0
                    eta = (len(tasks) - done) / rate / 60 if rate else 0
                    print(f"  {done}/{len(tasks)}  ok={ok} cached={cached} fail={fail}  "
                          f"{rate:.1f} route/s  ETA {eta:.0f} min", flush=True)
    print(f"DONE: {ok} decoded, {cached} cached, {fail} failed/short in {(time.time()-t0)/60:.1f} min", flush=True)
    print(f"total parquet in cache: {len(glob.glob(os.path.join(CACHE, '*', '*.parquet')))}", flush=True)


if __name__ == "__main__":
    main()
