#!/usr/bin/env python3
"""
V4 — Cross-vehicle driver-style transfer (small-n case study).

For drivers who appear in >=2 vehicle models (cached), test whether a driver-classifier trained
on vehicle A recognises the SAME drivers in vehicle B — comparing STEERING(input) vs PATH
(curvature) features. Hypothesis: PATH transfers across vehicles (vehicle-normalized style) while
STEERING collapses (it encodes the vehicle's steerRatio, which changes between A and B).

Honest limits: depends on how many drivers span >=2 models with enough human windows (typically
few) -> reported as a directional case study, chance = 1/(#shared drivers).

    ../openpilot/.venv/bin/python cross_vehicle_transfer.py
"""
import os
import sys
import glob
import itertools
import numpy as np
import pandas as pd

sys.path.insert(0, "./code/preprocessing")
from style_features import route_windows, FEATURE_GROUPS  # noqa: E402

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

CACHE = "./data/cache"
STEER = ["steer_std", "steer_rate_rms", "steer_entropy", "steer_reversal_rate", "steer_touch_rate"]


def load_all():
    rows, mdl, drv = [], [], []
    for m in sorted(os.listdir(CACHE)):
        d = os.path.join(CACHE, m)
        if not os.path.isdir(d):
            continue
        for p in sorted(glob.glob(os.path.join(d, "*.parquet"))):
            driver = os.path.basename(p).split("__", 1)[0]
            df = pd.read_parquet(p)
            for wf in route_windows(df):
                rows.append(wf); mdl.append(m); drv.append(driver)
    return pd.DataFrame(rows), np.array(mdl), np.array(drv)


def _clf(Xtr, ytr, Xte, cols):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                   class_weight="balanced", random_state=0, n_jobs=-1))])
    pipe.fit(Xtr[cols].to_numpy(float), ytr)
    return pipe.predict(Xte[cols].to_numpy(float))


def main():
    X, model, driver = load_all()
    cols = list(X.columns)
    path = [c for c in cols if FEATURE_GROUPS.get(c) == "path"]
    # find model pairs sharing >=2 drivers with >=3 windows each in both
    dm = pd.DataFrame({"model": model, "driver": driver})
    def shared(a, b):
        da = {d for d in set(driver[model == a]) if (model[(driver == d)] == a).sum() >= 3}
        db = {d for d in set(driver[model == b]) if (model[(driver == d)] == b).sum() >= 3}
        return sorted(da & db)
    models = sorted(set(model))
    any_run = False
    for a, b in itertools.combinations(models, 2):
        sh = shared(a, b)
        if len(sh) < 2:
            continue
        any_run = True
        chance = 1.0 / len(sh)
        print(f"\n=== {a}  <->  {b}   shared drivers: {len(sh)} (chance {chance:.2f}) ===")
        maskA = np.isin(driver, sh) & (model == a)
        maskB = np.isin(driver, sh) & (model == b)
        for name, cc in [("STEERING", STEER), ("PATH", path), ("ALL", cols)]:
            # train on A, test on B, and vice-versa; average
            accs = []
            for mtr, mte in [(maskA, maskB), (maskB, maskA)]:
                pred = _clf(X[mtr], driver[mtr], X[mte], cc)
                accs.append(accuracy_score(driver[mte], pred))
            print(f"  {name:9} ({len(cc)} feat)  A->B/B->A acc = {accs[0]:.2f}/{accs[1]:.2f}  "
                  f"mean {np.mean(accs):.2f}  lift {np.mean(accs)/chance:.1f}x")
    if not any_run:
        print("No model pair shares >=2 drivers with enough windows in cache — decode more overlap first.")


if __name__ == "__main__":
    main()
