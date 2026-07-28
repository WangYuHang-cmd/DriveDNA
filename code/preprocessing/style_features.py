#!/usr/bin/env python3
"""
Handcrafted driving-style features from a decoded route DataFrame (committed signals @20Hz).

Features are computed per fixed-length WINDOW over HUMAN-controlled, moving frames only
(is_human==1 & vEgo>MIN_SPEED), so that automation dynamics and parking don't contaminate
"style". Each feature maps to an established construct (see DriveDNA/Plan.md citations):
  lateral:   lane-position bias/SDLP (laneLeft_y/laneRight_y), steering std, steering entropy,
             steering reversal rate
  long.:     accel std/percentiles, jerk RMS, hard-brake/accel rates
  following: time-headway (THW) & TTC stats when a lead is present
  pedal:     brake/gas application rates & magnitudes
Speed itself is context (route-dependent), so we downweight raw speed and emphasize *how*
the driver controls the car; we keep a few coarse speed-context features for reference.
"""
import numpy as np
import pandas as pd

RATE = 10.0              # full-corpus unified rate (qlog-base + rlog per-segment)
MIN_SPEED = 2.0          # m/s; ignore standstill/parking
WIN_S = 60.0             # window length (s) -> 600 samples @10Hz


def _steering_entropy(angle, rate_hz=RATE):
    """Boer steering entropy: entropy of the 2nd-difference (prediction-error) distribution."""
    if len(angle) < 10:
        return np.nan
    pred_err = np.diff(angle, n=2)                     # proxy for Boer 3-point prediction error
    if np.allclose(pred_err, 0):
        return 0.0
    alpha = np.nanpercentile(np.abs(pred_err), 90) + 1e-6
    bins = np.array([-np.inf, -5, -2.5, -1, -0.5, 0.5, 1, 2.5, 5, np.inf]) * alpha
    p, _ = np.histogram(pred_err, bins=bins)
    p = p / max(p.sum(), 1)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)) / np.log(len(bins) - 1))


def _reversal_rate(angle, theta=2.0, rate_hz=RATE):
    """Steering reversal rate: direction reversals with gap > theta deg, per minute."""
    if len(angle) < 3:
        return np.nan
    d = np.diff(angle)
    sign = np.sign(d)
    revs = 0
    last = 0
    accum = 0.0
    for s, dd in zip(sign, d):
        accum += dd
        if s != 0 and s != last and abs(accum) > theta:
            revs += 1
            last = s
            accum = 0.0
    return revs / (len(angle) / rate_hz / 60.0 + 1e-9)


# feature -> group tag, so evaluations can slice INPUT vs PATH vs long/following/lane.
# INPUT = driver's raw actuation (vehicle-DEPENDENT, e.g. steerRatio); PATH = realized,
# vehicle-NORMALIZED trajectory (actual_curvature/slip). See DriveDNA/Plan.md.
FEATURE_GROUPS = {}


def _tag(name, group):
    FEATURE_GROUPS[name] = group
    return name


def window_features(w):
    """Compute one feature vector from a window DataFrame (already human+moving)."""
    f = {}
    v = w["vEgo"].to_numpy()
    a = w["aEgo"].to_numpy()
    ang = w["steeringAngleDeg"].to_numpy()
    # --- longitudinal control (LONG) ---
    jerk = np.diff(a) * RATE
    f[_tag("accel_std", "long")] = np.nanstd(a)
    f[_tag("accel_p95", "long")] = np.nanpercentile(a, 95)
    f[_tag("decel_p05", "long")] = np.nanpercentile(a, 5)
    f[_tag("jerk_rms", "long")] = np.sqrt(np.nanmean(jerk ** 2)) if len(jerk) else np.nan
    f[_tag("hard_brake_rate", "long")] = np.nanmean(a < -2.0)
    f[_tag("hard_accel_rate", "long")] = np.nanmean(a > 1.5)
    # --- lateral INPUT: driver steering actuation (vehicle-dependent) ---
    f[_tag("steer_std", "input")] = np.nanstd(ang)
    f[_tag("steer_rate_rms", "input")] = np.sqrt(np.nanmean(w["steeringRateDeg"].to_numpy() ** 2))
    f[_tag("steer_entropy", "input")] = _steering_entropy(ang)
    f[_tag("steer_reversal_rate", "input")] = _reversal_rate(ang)
    f[_tag("steer_touch_rate", "input")] = np.nanmean(w["steeringPressed"].to_numpy())
    # --- lateral PATH: realized, vehicle-normalized trajectory (actual_curvature/slip) ---
    if "actual_curvature" in w.columns:
        c = w["actual_curvature"].to_numpy()
        ac = np.abs(c)
        crate = np.diff(c) * RATE
        f[_tag("curv_absmean", "path")] = np.nanmean(ac)
        f[_tag("curv_std", "path")] = np.nanstd(c)
        f[_tag("curv_p95", "path")] = np.nanpercentile(ac, 95)
        f[_tag("curv_rate_rms", "path")] = np.sqrt(np.nanmean(crate ** 2)) if len(crate) else np.nan
        f[_tag("sharp_corner_rate", "path")] = np.nanmean(ac > 0.01)   # tight-curve fraction
        f[_tag("slip_absmean", "path")] = np.nanmean(np.abs(w["slip"].to_numpy()))
        f[_tag("slip_std", "path")] = np.nanstd(w["slip"].to_numpy())
    # lane placement (bias + SDLP); center offset = mean of the two edge distances (LANE, path-like)
    ll, lr = w["laneLeft_y"].to_numpy(), w["laneRight_y"].to_numpy()
    if np.isfinite(ll).any():
        center = (ll + lr) / 2.0
        f[_tag("lane_bias", "lane")] = np.nanmean(center)
        f[_tag("lane_sdlp", "lane")] = np.nanstd(center)
        f[_tag("lane_width", "lane")] = np.nanmean(lr - ll)
    else:
        f[_tag("lane_bias", "lane")] = f[_tag("lane_sdlp", "lane")] = f[_tag("lane_width", "lane")] = np.nan
    # --- car-following (only frames with a valid lead) (FOLLOW) ---
    lead = w["leadOne_status"].to_numpy() > 0.5
    moving = v > MIN_SPEED
    m = lead & moving
    if m.sum() > 5:
        thw = w["leadOne_dRel"].to_numpy()[m] / np.maximum(v[m], 0.5)
        vrel = w["leadOne_vRel"].to_numpy()[m]
        drel = w["leadOne_dRel"].to_numpy()[m]
        closing = vrel < -0.5
        ttc = drel[closing] / (-vrel[closing]) if closing.any() else np.array([np.nan])
        f[_tag("thw_median", "follow")] = np.nanmedian(thw)
        f[_tag("thw_p25", "follow")] = np.nanpercentile(thw, 25)
        f[_tag("close_follow_rate", "follow")] = np.nanmean(thw < 1.0)
        f[_tag("ttc_min", "follow")] = np.nanmin(ttc) if np.isfinite(ttc).any() else np.nan
        f[_tag("lead_present_rate", "follow")] = lead.mean()
    else:
        for k in ["thw_median", "thw_p25", "close_follow_rate", "ttc_min"]:
            f[_tag(k, "follow")] = np.nan
        f[_tag("lead_present_rate", "follow")] = lead.mean()
    # --- pedal behaviour (INPUT: driver actuation) ---
    f[_tag("brake_rate", "input")] = np.nanmean(w["brakePressed"].to_numpy())
    f[_tag("gas_rate", "input")] = np.nanmean(w["gasPressed"].to_numpy())
    f[_tag("brake_mag_p95", "input")] = np.nanpercentile(w["brake"].to_numpy(), 95)
    # --- coarse speed context (SPEED; mostly context, kept for reference) ---
    f[_tag("speed_mean", "speed")] = np.nanmean(v)
    f[_tag("speed_std", "speed")] = np.nanstd(v)
    return f


def route_windows(df, win_s=WIN_S, min_frac=0.5):
    """Yield feature dicts for consecutive windows of human+moving driving in a route."""
    hm = df[(df["is_human"] > 0.5) & (df["vEgo"] > MIN_SPEED)].reset_index(drop=True)
    n = int(win_s * RATE)
    feats = []
    for start in range(0, len(hm) - n + 1, n):
        w = hm.iloc[start:start + n]
        if len(w) < n * min_frac:
            continue
        feats.append(window_features(w))
    return feats


FEATURE_COLS = None  # set on first use to keep a stable column order
