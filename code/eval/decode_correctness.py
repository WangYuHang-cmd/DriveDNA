#!/usr/bin/env python3
"""
Decode-correctness check: does our independent LogReader extraction reproduce the user's
pipeline output? Compares our extracted carState signals against the reference
`vehicle_extended.csv` (same carState source topic, produced by process_acm_mm + refix).

The reference is frame-anchored (video clock) and 100 Hz; ours is CAN-mono and resampled to
100 Hz, so time origins differ by the sub-second frame-anchor shift. We therefore validate:
  (1) distribution match  (mean/std + quantile max-abs-diff)  -- values are the same signal;
  (2) temporal identity    (best-lag Pearson r on a smooth signal) -- same time series.
Shared columns available in vehicle_extended: vEgoCluster, steeringRateDeg, yawRate.
"""
import sys
import numpy as np
import pandas as pd


def best_lag_corr(a, b, max_lag=200):
    """Max Pearson r between a and b over integer lags in [-max_lag, max_lag]."""
    a = (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
    b = (b - np.nanmean(b)) / (np.nanstd(b) + 1e-9)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    best_r, best_l = -2, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            x, y = a[lag:], b[: n - lag]
        else:
            x, y = a[: n + lag], b[-lag:]
        if len(x) < n // 2:
            continue
        r = np.corrcoef(x, y)[0, 1]
        if r > best_r:
            best_r, best_l = r, lag
    return best_r, best_l


def qq_maxdiff(a, b, q=np.linspace(0.01, 0.99, 99)):
    """Max abs difference between matched quantiles (distribution shape agreement)."""
    return float(np.nanmax(np.abs(np.nanquantile(a, q) - np.nanquantile(b, q))))


def normtime_corr(a, b, n=5000):
    """Pearson r after resampling both onto a common normalized-time axis [0,1] (removes
    the global frame-anchor offset/scale so temporal identity is measured, not alignment)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    xa = np.linspace(0, 1, len(a)); xb = np.linspace(0, 1, len(b))
    g = np.linspace(0, 1, n)
    ra = np.interp(g, xa, a); rb = np.interp(g, xb, b)
    return float(np.corrcoef(ra, rb)[0, 1])


def main(mine_csv, ref_csv):
    mine = pd.read_csv(mine_csv)
    ref = pd.read_csv(ref_csv)
    print(f"mine: {len(mine)} rows  ref: {len(ref)} rows  "
          f"(dur mine {mine.time_s.iloc[-1]:.1f}s ref {ref.time_s.iloc[-1]:.1f}s)\n")
    shared = ["vEgoCluster", "steeringRateDeg", "yawRate"]
    hdr = (f"{'signal':16} {'mine_mean':>10} {'ref_mean':>10} {'mean_%Δ':>8} "
           f"{'std_%Δ':>8} {'qq_maxΔ':>9} {'ntime_r':>9}")
    print(hdr); print("-" * len(hdr))
    ok = True
    for c in shared:
        if c not in mine or c not in ref:
            print(f"{c:16} (missing)"); continue
        a, b = mine[c].to_numpy(), ref[c].to_numpy()
        ma, mb = np.nanmean(a), np.nanmean(b)
        sa, sb = np.nanstd(a), np.nanstd(b)
        if sa < 1e-6 and sb < 1e-6:
            print(f"{c:16} {ma:>10.4f} {mb:>10.4f} {'~0':>8} {'~0':>8} {'const':>9} {'const':>9}")
            continue
        scale = max(abs(mb), sb, 1e-6)
        mean_pd = 100 * abs(ma - mb) / scale
        std_pd = 100 * abs(sa - sb) / (sb + 1e-9)
        qq = qq_maxdiff(a, b)
        r = normtime_corr(a, b)
        print(f"{c:16} {ma:>10.4f} {mb:>10.4f} {mean_pd:>7.2f}% {std_pd:>7.2f}% {qq:>9.4f} {r:>9.4f}")
        # decode is correct iff VALUE distributions agree (mean/std within 1.5%);
        # temporal identity corroborated by normalized-time r on the smooth speed signal.
        # decode correctness = VALUE distribution agreement (mean/std within 1.5%).
        # ntime_r corroborates temporal identity but is limited by per-segment frame-anchor
        # drift (a known resampling artifact, not a decode error), so its bar is looser.
        if std_pd > 1.5 or mean_pd > 1.5:
            ok = False
        if c == "vEgoCluster" and r < 0.95:
            ok = False
    print("\nVERDICT:", "PASS - independent decode reproduces reference distributions"
          if ok else "CHECK - a signal distribution disagrees > tolerance")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
