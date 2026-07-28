#!/usr/bin/env python3
"""
V3 — Vehicle-leakage probe (the signal-level demonstration of the paper thesis).

Claim: the driver's raw steering INPUT carries the vehicle's steerRatio signature, so it
predicts the VEHICLE MODEL; the realized PATH curvature is vehicle-normalized, so it does
not. We pool windows from multiple models, label each by MODEL, and train a model-classifier
from INPUT(steering) features vs PATH(curvature) features. Driver-grouped CV (GroupKFold on
driver) forces the classifier to use the vehicle signal, not memorize drivers.

Expected: INPUT >> PATH at predicting the vehicle => raw steering leaks vehicle identity while
curvature is vehicle-invariant. This motivates C2's adversarial vehicle head and C3's leakage
ablation.

    ../openpilot/.venv/bin/python leakage_probe.py [MODEL1 MODEL2 ...]
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "./code/preprocessing")
from style_features import route_windows, FEATURE_GROUPS  # noqa: E402

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from model_merge import canon  # noqa: E402

CACHE = "./data/cache"


def build(models):
    """models = CONSOLIDATED model names; gathers all raw fingerprint dirs that map to them."""
    want = {canon(m) for m in models}
    rows, model_lab, drv_lab = [], [], []
    for raw in sorted(os.listdir(CACHE)):
        c = canon(raw)
        if c not in want:
            continue
        for p in sorted(glob.glob(os.path.join(CACHE, raw, "*.parquet"))):
            driver = os.path.basename(p).split("__", 1)[0]
            df = pd.read_parquet(p)
            for wf in route_windows(df):
                rows.append(wf); model_lab.append(c); drv_lab.append(driver)
    return pd.DataFrame(rows), np.array(model_lab), np.array(drv_lab)


def _pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      class_weight="balanced", random_state=0, n_jobs=-1)),
    ])


def run(X, y, groups, cols, n_splits):
    Xv = X[cols].to_numpy(float)
    yp = cross_val_predict(_pipe(), Xv, y, groups=groups, cv=GroupKFold(n_splits), method="predict")
    return accuracy_score(y, yp), balanced_accuracy_score(y, yp)


def main(models):
    X, ymodel, ydriver = build(models)
    cols = list(X.columns)
    g = lambda name: [c for c in cols if FEATURE_GROUPS.get(c) == name]
    present = sorted(set(models))
    n_models = len(present)
    chance = 1.0 / n_models
    n_splits = min(5, len(set(ydriver)))
    print(f"models={n_models} {present}")
    print(f"windows={len(X)}  drivers={len(set(ydriver))}  chance={chance:.3f}  "
          f"(driver-grouped {n_splits}-fold CV; target = VEHICLE MODEL)")
    print(f"windows per model:")
    print(pd.Series(ymodel).value_counts().to_string())

    # steering-ONLY input (isolates the steerRatio mechanism; excludes pedals whose
    # decode-availability itself differs by model and would be a confound)
    steer_only = ["steer_std", "steer_rate_rms", "steer_entropy", "steer_reversal_rate", "steer_touch_rate"]
    steer_only = [c for c in steer_only if c in cols]
    subsets = {
        "STEERING-only (input)": steer_only,
        "PATH (curvature+slip)": g("path"),
        "LONG (accel/jerk)": g("long"),
        "ALL": cols,
    }
    print(f"\n{'feature group':26} {'#feat':>5} {'model_acc':>10} {'bal_acc':>8} {'lift':>6}")
    res = {}
    for name, cc in subsets.items():
        if not cc:
            print(f"{name:26} (no features)"); continue
        acc, bacc = run(X, ymodel, ydriver, cc, n_splits)
        res[name] = acc
        print(f"{name:26} {len(cc):>5} {acc:>10.3f} {bacc:>8.3f} {acc/chance:>5.1f}x")

    if "STEERING-only (input)" in res and "PATH (curvature+slip)" in res:
        gap = res["STEERING-only (input)"] - res["PATH (curvature+slip)"]
        print(f"\nSTEERING − PATH vehicle-prediction gap: {gap:+.3f}")
        print("VERDICT:", "LEAKAGE CONFIRMED — steering predicts vehicle >> curvature (steering leaks steerRatio)"
              if gap > 0.05 else "INCONCLUSIVE — gap small; need more models/data")


if __name__ == "__main__":
    ms = sys.argv[1:] or [d for d in os.listdir(CACHE)
                          if os.path.isdir(os.path.join(CACHE, d)) and glob.glob(os.path.join(CACHE, d, "*.parquet"))]
    main(ms)
