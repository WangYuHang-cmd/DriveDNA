#!/usr/bin/env python3
"""
S3 — Curate the benchmark subset and freeze OFFICIAL splits from the availability audit.

Reads data/availability_audit.csv (per-route metadata + capability flags) and emits split
manifests under data/splits/. Splits are deterministic (seeded) so the benchmark is reproducible.

Tasks the splits support (see DriveDNA/Plan.md):
  * within_vehicle   — driver-ID within a fixed vehicle model (leakage-free identity)
  * cross_route      — held-out trips of the same drivers (verification)
  * cross_vehicle    — drivers who span >=2 vehicle models (transfer case study)
  * driver_folds     — driver-disjoint train/val/test + a held-out FEW-SHOT driver set
  * human_vs_adas    — per-route human vs ADAS minutes (secondary analysis)

    ../openpilot/.venv/bin/python curate_splits.py
"""
import os
import json
import random
import numpy as np
import pandas as pd

AUD = "./data/availability_audit.csv"
OUTDIR = "./data/splits"

# curation thresholds
MIN_ROUTE_HUMAN_MIN = 3.0       # a route is usable if >=3 min human+moving driving
MIN_DRIVER_ROUTES = 2           # need >=2 routes for any cross-trip split
MIN_DRIVER_HUMAN_MIN = 5.0      # total human driving per driver
WITHIN_VEHICLE_MIN_DRIVERS = 4  # a model qualifies for within-vehicle driver-ID
FEWSHOT_HELDOUT_FRAC = 0.15     # fraction of eligible drivers reserved for few-shot eval
SEED = 20260705


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(AUD)
    n0 = len(df)

    # ---- route-level curation ----
    df["usable"] = df["human_move_min"] >= MIN_ROUTE_HUMAN_MIN
    cur = df[df["usable"]].copy()
    # driver-level aggregates on usable routes
    dstat = cur.groupby("driver").agg(routes=("route", "count"),
                                      human_min=("human_move_min", "sum"),
                                      models=("model", "nunique")).reset_index()
    good_drivers = set(dstat[(dstat.human_min >= MIN_DRIVER_HUMAN_MIN)].driver)
    cur = cur[cur.driver.isin(good_drivers)].copy()
    cur.to_csv(os.path.join(OUTDIR, "corpus.csv"), index=False)

    # ---- within-vehicle: CONSOLIDATED models with >= MIN_DRIVERS multi-route drivers ----
    within = {}
    for model, g in cur.groupby("model_canon"):
        dv = g.groupby("driver").route.count()
        multi = dv[dv >= MIN_DRIVER_ROUTES].index.tolist()
        if len(multi) >= WITHIN_VEHICLE_MIN_DRIVERS:
            within[model] = {
                "drivers": sorted(multi),
                "n_drivers": len(multi),
                "routes": g[g.driver.isin(multi)][["driver", "route"]].values.tolist(),
            }

    # ---- cross-route: drivers with >=3 routes (hold out ~1/3 of their routes as test) ----
    rng = random.Random(SEED)
    cross_route = {}
    for drv, g in cur.groupby("driver"):
        routes = sorted(g.route.tolist())
        if len(routes) >= 3:
            rng.shuffle(routes)
            k = max(1, len(routes) // 3)
            cross_route[drv] = {"train": sorted(routes[k:]), "test": sorted(routes[:k])}

    # ---- cross-vehicle: drivers spanning >=2 CONSOLIDATED models (usable routes) ----
    cross_vehicle = {}
    for drv, g in cur.groupby("driver"):
        by_model = {m: sorted(gg.route.tolist()) for m, gg in g.groupby("model_canon")}
        if len(by_model) >= 2:
            cross_vehicle[drv] = by_model

    # ---- driver-disjoint folds + held-out few-shot drivers ----
    eligible = sorted(good_drivers)
    rng2 = random.Random(SEED + 1)
    rng2.shuffle(eligible)
    n_few = int(len(eligible) * FEWSHOT_HELDOUT_FRAC)
    fewshot = sorted(eligible[:n_few])
    rest = eligible[n_few:]
    n_test = int(len(rest) * 0.15)
    n_val = int(len(rest) * 0.15)
    folds = {"few_shot_heldout": fewshot,
             "test": sorted(rest[:n_test]),
             "val": sorted(rest[n_test:n_test + n_val]),
             "train": sorted(rest[n_test + n_val:])}

    # ---- human vs ADAS per route (from audit human_frac) ----
    df[["model", "driver", "route", "human_frac", "human_move_min"]].to_csv(
        os.path.join(OUTDIR, "human_adas.csv"), index=False)

    # write manifests
    for name, obj in [("within_vehicle", within), ("cross_route", cross_route),
                      ("cross_vehicle", cross_vehicle), ("driver_folds", folds)]:
        with open(os.path.join(OUTDIR, f"{name}.json"), "w") as f:
            json.dump(obj, f, indent=1)

    # ---- report ----
    print(f"=== CURATION (seed {SEED}) ===")
    print(f"routes: {n0} audited -> {len(cur)} usable (>={MIN_ROUTE_HUMAN_MIN}min human)")
    print(f"drivers: {df.driver.nunique()} -> {len(good_drivers)} kept "
          f"(>={MIN_DRIVER_HUMAN_MIN}min total)")
    print(f"total human+moving hours (curated): {cur.human_move_min.sum()/60:.0f} h")
    print(f"\nwithin_vehicle: {len(within)} eligible models "
          f"(>={WITHIN_VEHICLE_MIN_DRIVERS} multi-route drivers each)")
    for m, o in sorted(within.items(), key=lambda kv: -kv[1]['n_drivers'])[:12]:
        print(f"   {m:26} {o['n_drivers']} drivers, {len(o['routes'])} routes")
    print(f"\ncross_route: {len(cross_route)} drivers with held-out trips")
    print(f"cross_vehicle: {len(cross_vehicle)} drivers span >=2 models")
    print(f"driver_folds: train {len(folds['train'])} / val {len(folds['val'])} / "
          f"test {len(folds['test'])} / few-shot-heldout {len(folds['few_shot_heldout'])}")
    print(f"\nwrote manifests -> {OUTDIR}")


if __name__ == "__main__":
    main()
