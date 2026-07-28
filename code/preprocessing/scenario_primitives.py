#!/usr/bin/env python3
"""
D2 — Shared infrastructure: scenario taxonomy + T1 behavioral-primitive weak labels.

Pass 1: slide 60 s windows (stride 30 s) over every human-driving span (segments.parquet),
compute per-window statistics, and assign a rule-derived SCENARIO bucket
(stop_go > curve > car_following > high_speed > urban > free — deterministic priority).
Pass 2: within each scenario bucket, assign T1 primitive weak labels by percentile
thresholds (high = ≥Q80 | scenario, low = ≤Q20 | scenario, middle 60% = 0/neutral) —
NOT fixed absolute cuts, and explicitly NOT ground-truth style. Steering primitives are
dual-versioned (raw steering-rate vs curvature-rate) to feed the vehicle-leakage diagnostic.

Used by: T1 (labels) · T4 (matching variables) · scenario-normalized metrics · S2 normalization.

    ../openpilot/.venv/bin/python scenario_primitives.py
Output: DriveDNA/data/segments/windows.parquet
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CACHE = "./data/cache"
SEGQ = "./data/segments/segments.parquet"
OUT = "./data/segments/windows.parquet"
RATE = 10.0
WIN = int(60 * RATE)
STRIDE = int(30 * RATE)

COLS = ["time_s", "vEgo", "aEgo", "steeringAngleDeg", "steeringRateDeg", "actual_curvature",
        "leadOne_status", "leadOne_dRel", "leadOne_vRel", "laneLeft_y", "laneRight_y",
        "brakePressed", "gasPressed"]


def window_stats(w):
    """Compact per-window statistics used for scenario bucketing + primitive labels."""
    v = w["vEgo"].to_numpy(); a = w["aEgo"].to_numpy()
    curv = w["actual_curvature"].to_numpy()
    jerk = np.diff(a) * RATE
    lead = w["leadOne_status"].to_numpy() > 0.5
    moving = v > 2.0
    s = {
        "v_mean": np.nanmean(v), "stop_frac": float(np.mean(v < 1.0)),
        "decel_p05": np.nanpercentile(a, 5), "accel_p95": np.nanpercentile(a, 95),
        "jerk_rms": float(np.sqrt(np.nanmean(jerk ** 2))) if len(jerk) else np.nan,
        "curv_p95": np.nanpercentile(np.abs(curv), 95),
        "curv_rate_rms": float(np.sqrt(np.nanmean((np.diff(curv) * RATE) ** 2))),
        "steer_rate_rms": float(np.sqrt(np.nanmean(w["steeringRateDeg"].to_numpy() ** 2))),
        "lead_frac": float(lead.mean()),
        "brake_rate": float(np.nanmean(w["brakePressed"].to_numpy())),
    }
    m = lead & moving
    if m.sum() > 10:
        thw = w["leadOne_dRel"].to_numpy()[m] / np.maximum(v[m], 0.5)
        s["thw_median"] = float(np.nanmedian(thw))
    else:
        s["thw_median"] = np.nan
    ll, lr = w["laneLeft_y"].to_numpy(), w["laneRight_y"].to_numpy()
    if np.isfinite(ll).sum() > 10:
        s["lane_sdlp"] = float(np.nanstd((ll + lr) / 2.0))
    else:
        s["lane_sdlp"] = np.nan
    return s


def scenario_of(s):
    """Deterministic priority rules → one bucket per window."""
    if s["stop_frac"] > 0.10:
        return "stop_go"
    if s["curv_p95"] > 0.01:
        return "curve"
    if s["lead_frac"] > 0.5 and np.isfinite(s["thw_median"]) and s["thw_median"] < 3.5:
        return "car_following"
    if s["v_mean"] > 25.0:
        return "high_speed"
    if s["v_mean"] < 15.0:
        return "urban"
    return "free"


# primitive -> (stat, direction): +1 label when stat ≥ Q80 (or ≤ Q20 if direction 'low')
PRIMS = {
    "close_following":   ("thw_median", "low"),      # low THW = close (within car_following mainly)
    "large_headway":     ("thw_median", "high"),
    "hard_braking":      ("decel_p05", "low"),        # more negative = harder
    "high_jerk":         ("jerk_rms", "high"),
    "sharp_steer_raw":   ("steer_rate_rms", "high"),  # raw input version (vehicle-dependent)
    "sharp_steer_path":  ("curv_rate_rms", "high"),   # curvature-normalized version
    "lane_correction":   ("lane_sdlp", "high"),
    "curve_entry_decel": ("decel_p05", "low"),        # evaluated within curve bucket only
}


def main():
    seg = pd.read_parquet(SEGQ)
    groups = seg.groupby(["model", "model_canon", "driver", "route"])
    print(f"windowing {len(groups)} routes / {len(seg)} spans ...", flush=True)
    rows = []
    for gi, ((mdl, mc, drv, rt), spans) in enumerate(groups):
        p = os.path.join(CACHE, mdl, f"{drv}__{rt}.parquet")
        try:
            df = pd.read_parquet(p, columns=COLS)
        except Exception:
            continue
        for _, sp in spans.iterrows():
            i0, i1 = int(sp.i0), int(sp.i1)
            n = i1 - i0 + 1
            starts = [i0] if n < WIN else list(range(i0, i1 - WIN + 2, STRIDE))
            for ws in starts:
                we = min(ws + WIN, i1 + 1)
                if we - ws < int(30 * RATE):
                    continue
                w = df.iloc[ws:we]
                if np.nanmean(w["vEgo"].to_numpy() > 2.0) < 0.5:   # require mostly-moving
                    continue
                s = window_stats(w)
                s.update({"model": mdl, "model_canon": mc, "driver": drv, "route": rt,
                          "span_id": int(sp.span_id), "wi0": ws, "wi1": we - 1,
                          "t0": float(df["time_s"].iloc[ws]),
                          "scenario": scenario_of(s)})
                rows.append(s)
        if (gi + 1) % 500 == 0:
            print(f"  {gi+1}/{len(groups)}  windows: {len(rows)}", flush=True)
    W = pd.DataFrame(rows)
    print(f"windows: {len(W)}", flush=True)

    # ---- pass 2: scenario-conditioned percentile labels {-1,0,+1}, NaN when stat missing ----
    for prim, (stat, direction) in PRIMS.items():
        lab = np.full(len(W), np.nan)
        for sc, g in W.groupby("scenario"):
            if prim == "curve_entry_decel" and sc != "curve":
                continue
            if prim in ("close_following", "large_headway") and sc != "car_following":
                continue
            x = g[stat].to_numpy()
            ok = np.isfinite(x)
            if ok.sum() < 50:
                continue
            q20, q80 = np.nanpercentile(x[ok], 20), np.nanpercentile(x[ok], 80)
            v = np.zeros(len(g))
            if direction == "high":
                v[x >= q80] = 1; v[x <= q20] = -1
            else:
                v[x <= q20] = 1; v[x >= q80] = -1
            v[~ok] = np.nan
            lab[g.index.to_numpy()] = v
        W[f"p_{prim}"] = lab

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    W.to_parquet(OUT, index=False)
    print(f"\nwrote {OUT}")
    print("=== scenario distribution ===")
    print(W.scenario.value_counts().to_string())
    print("\n=== primitive label coverage (+1 / -1 counts) ===")
    for prim in PRIMS:
        c = W[f"p_{prim}"]
        print(f"  {prim:18} +1:{int((c==1).sum()):>6}  -1:{int((c==-1).sum()):>6}  "
              f"neutral:{int((c==0).sum()):>7}  n/a:{int(c.isna().sum()):>7}")
    print(f"\ndrivers={W.driver.nunique()}  routes={W.route.nunique()}  windows={len(W)}")


if __name__ == "__main__":
    main()
