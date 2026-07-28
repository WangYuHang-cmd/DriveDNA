#!/usr/bin/env python3
"""
Self-contained openpilot rlog -> committed-signal extractor for DriveDNA.

Reads the raw per-segment capnp logs of a route (``<seg>--rlog`` / ``.bz2`` / ``.zst``)
with openpilot's LogReader and resamples the DriveDNA "committed signals" onto a uniform
time grid. This is the general decoder we use for the whole corpus (the user's own
``process_acm_mm.py`` groups by driver-camera segments; we need every route, camera or not).

Run under the openpilot venv:
    PYTHONPATH=../openpilot \
      ../openpilot/.venv/bin/python rlog_extract.py <route_dir>

Signal provenance (confirmed against process_acm_mm.py field dicts):
  carState:         vEgo, aEgo, steeringAngleDeg, steeringPressed, gas, gasPressed,
                    brake, brakePressed, cruiseState.enabled, yawRate, steeringRateDeg,
                    vEgoCluster, leftBlinker, rightBlinker
  controlsState:    enabled            -> cs_enabled (openpilot ADAS engaged)
  radarState:       leadOne.{status,dRel,vLead,vRel}   (NaN on radar-less cars)
  drivingModelData: laneLineMeta.{leftY,rightY}        (lane lateral offsets, NaN if absent)
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd

# --- enable zstd-compressed rlog streaming (mirror process_acm_mm.py monkey-patch) ---
try:  # pragma: no cover - environment dependent
    import zstandard as _zstd  # noqa: F401
except Exception:
    _zstd = None

from openpilot.tools.lib.logreader import LogReader  # noqa: E402

# out_col, topic, dotpath (within topic), kind: 'f' float(interp) | 'b' bool/int(hold)
# dotpath supports integer parts as list indices, e.g. "angularVelocityCalibrated.value.2".
FIELDS = [
    ("vEgo",                "carState",         "vEgo",                    "f"),
    ("aEgo",                "carState",         "aEgo",                    "f"),
    ("steeringAngleDeg",    "carState",         "steeringAngleDeg",        "f"),
    ("steeringRateDeg",     "carState",         "steeringRateDeg",         "f"),
    ("steeringPressed",     "carState",         "steeringPressed",         "b"),
    ("gas",                 "carState",         "gas",                     "f"),
    ("gasPressed",          "carState",         "gasPressed",              "b"),
    ("brake",               "carState",         "brake",                   "f"),
    ("brakePressed",        "carState",         "brakePressed",            "b"),
    ("yawRate",             "carState",         "yawRate",                 "f"),  # 0 on most cars; kept for audit
    ("vEgoCluster",         "carState",         "vEgoCluster",             "f"),
    ("leftBlinker",         "carState",         "leftBlinker",             "b"),
    ("rightBlinker",        "carState",         "rightBlinker",            "b"),
    ("cruiseState_enabled", "carState",         "cruiseState.enabled",     "b"),
    ("cs_enabled",          "controlsState",    "enabled",                 "b"),
    # --- lateral INPUT vs realized PATH ---
    ("actual_curvature",    "controlsState",    "curvature",               "f"),  # realized path curvature (1/m)
    ("_yaw_lp",             "livePose",         "angularVelocityDevice.z", "f"),  # localizer yaw rate (rad/s)
    ("_yaw_llk",            "liveLocationKalman", "angularVelocityCalibrated.value.2", "f"),  # Tesla fallback
    ("leadOne_status",      "radarState",       "leadOne.status",          "b"),
    ("leadOne_dRel",        "radarState",       "leadOne.dRel",            "f"),
    ("leadOne_vLead",       "radarState",       "leadOne.vLead",           "f"),
    ("leadOne_vRel",        "radarState",       "leadOne.vRel",            "f"),
    ("laneLeft_y",          "drivingModelData", "laneLineMeta.leftY",      "f"),
    ("laneRight_y",         "drivingModelData", "laneLineMeta.rightY",     "f"),
]
NEEDED_TOPICS = sorted({t for _, t, _, _ in FIELDS})


def _seg_index(path):
    """Sort key: real segment index from a '<seg>--rlog...' filename."""
    base = os.path.basename(path)
    m = re.match(r"(\d+)--", base)
    return int(m.group(1)) if m else 1 << 30


def find_segment_logs(route_dir):
    """Segment log paths in index order, PER-SEGMENT QLOG-PREFERRED-else-rlog (uniform-qlog; V6 fix).
    Native-10Hz qlog everywhere → single source, no log-type sampling confound. rlog only fills qlog gaps.
    Every source is resampled to the same 10 Hz grid downstream."""
    rlog, qlog = {}, {}
    for pat in ("*--rlog", "*--rlog.zst", "*--rlog.bz2"):
        for p in glob.glob(os.path.join(route_dir, pat)):
            if os.path.getsize(p) > 0:
                rlog.setdefault(_seg_index(p), p)
    for pat in ("*--qlog", "*--qlog.bz2", "*--qlog.zst"):
        for p in glob.glob(os.path.join(route_dir, pat)):
            if os.path.getsize(p) > 0:
                qlog.setdefault(_seg_index(p), p)
    return [qlog.get(i, rlog.get(i)) for i in sorted(set(rlog) | set(qlog))]


def _deep_get(obj, dotpath):
    for part in dotpath.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def read_route_raw(route_dir, verbose=False, max_segments=None):
    """Read raw (logMonoTime, value) samples per output column across all segments of a route."""
    logs = find_segment_logs(route_dir)
    if not logs:
        raise FileNotFoundError(f"no rlog/qlog segments in {route_dir}")
    if max_segments is not None:
        logs = logs[:max_segments]
    # per-topic list of (mono, msgreader); we pull all fields of a topic in one pass
    times = {col: [] for col, *_ in FIELDS}
    vals = {col: [] for col, *_ in FIELDS}
    topic_fields = {}
    for col, topic, dot, kind in FIELDS:
        topic_fields.setdefault(topic, []).append((col, dot, kind))
    for seg in logs:
        try:
            lr = LogReader(seg, only_union_types=True, sort_by_time=True)
        except Exception as e:  # skip a corrupt segment, keep the route
            if verbose:
                print(f"  WARN skip {os.path.basename(seg)}: {e}")
            continue
        for m in lr:
            w = m.which()
            if w not in topic_fields:
                continue
            mono = int(m.logMonoTime)
            payload = getattr(m, w)
            for col, dot, kind in topic_fields[w]:
                try:
                    v = _deep_get(payload, dot)
                except Exception:
                    continue
                times[col].append(mono)
                vals[col].append(float(v) if kind == "f" else int(bool(v)))
    return times, vals, logs


def resample_route(route_dir, rate_hz=20, verbose=False, max_segments=None):
    """Return a DataFrame of committed signals on a uniform grid (time_s relative, starts at 0)."""
    times, vals, logs = read_route_raw(route_dir, verbose=verbose, max_segments=max_segments)
    # reference clock = carState span (vEgo present on every car)
    ref = np.asarray(times["vEgo"], dtype=np.int64)
    if ref.size < 2:
        raise ValueError(f"no carState samples in {route_dir}")
    t0, t1 = ref.min(), ref.max()
    step = int(1e9 / rate_hz)
    grid = np.arange(t0, t1 + step // 2, step, dtype=np.int64)
    out = {"time_s": (grid - t0) / 1e9}
    for col, topic, dot, kind in FIELDS:
        tt = np.asarray(times[col], dtype=np.int64)
        vv = np.asarray(vals[col], dtype=np.float64)
        if tt.size == 0:
            out[col] = np.full(grid.shape, np.nan)      # topic absent on this car
            continue
        order = np.argsort(tt)
        tt, vv = tt[order], vv[order]
        if kind == "f":
            out[col] = np.interp(grid, tt, vv)
        else:  # zero-order hold
            idx = np.searchsorted(tt, grid, side="right") - 1
            idx = np.clip(idx, 0, len(vv) - 1)
            out[col] = vv[idx]
    df = pd.DataFrame(out)
    # localizer yaw rate: livePose (most cars) else liveLocationKalman (Tesla); then derived curvature
    yaw = df["_yaw_lp"].to_numpy()
    if not np.isfinite(yaw).any():
        yaw = df["_yaw_llk"].to_numpy()
    df["yaw_rate"] = yaw
    v = df["vEgo"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        cm = np.where(v > 3.0, yaw / v, np.nan)          # independent, sensed realized curvature
    df["curv_measured"] = cm
    df["slip"] = df["actual_curvature"].to_numpy() - cm   # steering-implied minus sensed = slip/limit proxy
    df.drop(columns=["_yaw_lp", "_yaw_llk"], inplace=True)
    # human-control mask: neither openpilot nor OEM ACC engaged
    df["is_human"] = ((df["cs_enabled"] < 0.5) & (df["cruiseState_enabled"] < 0.5)).astype(int)
    if verbose:
        print(f"  {os.path.basename(route_dir)}: {len(logs)} segs, {len(df)} rows @ {rate_hz}Hz, "
              f"human {df['is_human'].mean()*100:.1f}%")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("route_dir")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--out", default=None, help="optional CSV output path")
    args = ap.parse_args()
    df = resample_route(args.route_dir, rate_hz=args.rate, verbose=True)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(df.describe().T[["count", "mean", "std", "min", "max"]])
    if args.out:
        df.to_csv(args.out, index=False, float_format="%.6f")
        print("wrote", args.out)
