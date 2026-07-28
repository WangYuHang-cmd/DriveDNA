#!/usr/bin/env python3
"""
V1 correctness gate for `actual_curvature`.

`actual_curvature` (controlsState.curvature) is derived from steeringAngle via the bicycle
model; `curv_measured` (localizer yaw_rate / vEgo) is an INDEPENDENT sensed estimate. If the
two agree during human+moving driving, `actual_curvature` is a valid realized-curvature signal.
Reports per-model Pearson r, slope, and bias on human (is_human) frames with vEgo>3 m/s.

    PYTHONPATH=../openpilot \
      ../openpilot/.venv/bin/python curvature_correctness.py \
      RIVIAN_R1_GEN1 HONDA_CIVIC TESLA_AP3_MODEL_3 HYUNDAI_IONIQ_5 FORD_F_150_MK14
"""
import os
import sys
import glob
import numpy as np

sys.path.insert(0, "./code/preprocessing")
from rlog_extract import resample_route, find_segment_logs  # noqa: E402

DATASET = "../Dataset"


def first_human_route(model, min_human_rows=1200, max_try=8, max_segments=8):
    """Find a route with enough human+moving driving to test curvature agreement."""
    for drv in sorted(glob.glob(os.path.join(DATASET, model, "*"))):
        if not os.path.isdir(drv):
            continue
        for r in sorted(glob.glob(os.path.join(drv, "*")))[:max_try]:
            if not (os.path.isdir(r) and find_segment_logs(r)):
                continue
            try:
                df = resample_route(r, rate_hz=20, max_segments=max_segments)
            except Exception:
                continue
            m = (df.is_human > 0.5) & (df.vEgo > 3.0) & np.isfinite(df.curv_measured)
            if m.sum() >= min_human_rows:
                return r, df[m]
    return None, None


def main(models):
    hdr = f"{'model':22} {'n':>6} {'corr':>7} {'slope':>7} {'bias':>10} {'std_act':>9} {'std_meas':>9}"
    print(hdr); print("-" * len(hdr))
    ok = True
    for model in models:
        r, d = first_human_route(model)
        if d is None:
            print(f"{model:22} (no usable human route found)"); continue
        a = d.actual_curvature.to_numpy()
        b = d.curv_measured.to_numpy()
        good = np.isfinite(a) & np.isfinite(b)
        a, b = a[good], b[good]
        corr = np.corrcoef(a, b)[0, 1]
        slope = np.polyfit(b, a, 1)[0]
        bias = np.mean(a - b)
        print(f"{model:22} {len(a):>6} {corr:>7.3f} {slope:>7.3f} {bias:>10.5f} "
              f"{np.std(a):>9.5f} {np.std(b):>9.5f}")
        if corr < 0.7:
            ok = False
    print("\nVERDICT:", "PASS - actual_curvature agrees with independent sensed curv_measured"
          if ok else "CHECK - a model shows low curvature agreement")


if __name__ == "__main__":
    models = sys.argv[1:] or ["RIVIAN_R1_GEN1", "HONDA_CIVIC", "TESLA_AP3_MODEL_3",
                              "HYUNDAI_IONIQ_5", "FORD_F_150_MK14"]
    main(models)
