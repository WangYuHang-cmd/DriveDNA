#!/usr/bin/env python3
"""
S2 — Per-model signal-availability + human-driving audit over the decoded parquet cache.

Drives task eligibility: which routes/models qualify for car-following (radar), lane-keeping
(lane offsets), longitudinal, and lateral input/path features. A signal is "valid" on a route
if it is present (not all-NaN) AND not constant (std>eps) over human+moving frames.

    ../openpilot/.venv/bin/python availability_audit.py
Writes: DriveDNA/data/availability_audit.csv (+ printed per-model summary)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_merge import canon  # noqa: E402

CACHE = "./data/cache"
OUT = "./data/availability_audit.csv"
RATE = 10.0
MIN_SPEED = 2.0
EPS = 1e-6

# signal -> columns that must be valid for that capability
CAPS = {
    "lateral_input": ["steeringAngleDeg"],
    "lateral_path": ["actual_curvature"],
    "yaw_sensed": ["yaw_rate"],
    "longitudinal": ["aEgo"],
    "car_following": ["leadOne_dRel"],
    "lane_keeping": ["laneLeft_y"],
    "brake_pedal": ["brake"],
    "gas_pedal": ["gas"],
}


def route_row(p):
    model = os.path.basename(os.path.dirname(p))
    base = os.path.basename(p)[:-8]
    driver, route = base.split("__", 1)
    df = pd.read_parquet(p)
    hm = df[(df["is_human"] > 0.5) & (df["vEgo"] > MIN_SPEED)]
    row = {"model": model, "model_canon": canon(model), "driver": driver, "route": route,
           "rows": len(df), "human_move_rows": len(hm),
           "human_move_min": round(len(hm) / RATE / 60.0, 2),
           "human_frac": round((df["is_human"] > 0.5).mean(), 3)}
    for cap, cols in CAPS.items():
        ok = True
        for c in cols:
            if c not in hm.columns or len(hm) == 0:
                ok = False; break
            v = hm[c].to_numpy()
            if not np.isfinite(v).any() or np.nanstd(v) < EPS:
                ok = False; break
        row[cap] = int(ok)
    # radar presence rate (fraction of human frames with a lead)
    row["lead_rate"] = round(float((df.get("leadOne_status", pd.Series([np.nan])) > 0.5).mean()), 3)
    return row


def main():
    paths = sorted(glob.glob(os.path.join(CACHE, "*", "*.parquet")))
    print(f"auditing {len(paths)} decoded routes ...", flush=True)
    rows = []
    for i, p in enumerate(paths):
        try:
            rows.append(route_row(p))
        except Exception as e:
            print(f"  skip {os.path.basename(p)}: {e}")
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(paths)}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}\n")

    # corpus-level summary
    print(f"=== CORPUS: {df.model.nunique()} models, {df.driver.nunique()} drivers, "
          f"{len(df)} routes, {df.human_move_min.sum()/60:.0f} h human+moving ===")
    caps = list(CAPS) + ["lead_rate"]
    print("\n=== per-model coverage (route count + %routes with each capability) ===")
    g = df.groupby("model")
    summ = g.agg(drivers=("driver", "nunique"), routes=("route", "count"),
                 human_h=("human_move_min", lambda s: round(s.sum() / 60, 1)))
    for cap in CAPS:
        summ[cap] = (g[cap].mean() * 100).round(0).astype(int)
    summ = summ.sort_values("routes", ascending=False)
    with pd.option_context("display.max_rows", 60, "display.width", 200):
        print(summ.to_string())

    # capability totals (routes qualifying corpus-wide)
    print("\n=== corpus capability coverage (%% of routes) ===")
    for cap in CAPS:
        print(f"  {cap:16} {df[cap].mean()*100:5.1f}%  ({int(df[cap].sum())} routes)")


if __name__ == "__main__":
    main()
