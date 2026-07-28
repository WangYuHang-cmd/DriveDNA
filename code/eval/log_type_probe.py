#!/usr/bin/env python3
"""
V6 — Log-type leakage probe (safeguard for the mixed qlog/rlog @10 Hz corpus).

Threat: if rlog-availability correlates with driver identity, a qlog(10 Hz) vs rlog(100->10 Hz)
sampling artifact in high-freq features could spuriously inflate driver-ID. This probe checks:
  (1) DRIVER↔log-type correlation — is rlog-availability driver-specific? (per-driver rlog fraction)
  (2) can style features PREDICT log-type (rlog vs qlog)? Driver-grouped CV; must be ≈chance.
If (2) is ≈chance, the mixed corpus is safe. If it leaks, add log-type to the adversarial head
or fall back to uniform-qlog.

    ../openpilot/.venv/bin/python log_type_probe.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "./code/preprocessing")
from style_features import route_windows  # noqa: E402

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

CACHE = "./data/cache"
PROG = "./experiments/logs/decode_corpus_progress.csv"


def main():
    prog = pd.read_csv(PROG)
    # route -> rlog fraction (key by driver__route to match parquet names)
    prog["key"] = prog.driver.astype(str) + "__" + prog.route.astype(str)
    frac = dict(zip(prog.key, prog.rlog_frac))

    rows, y_log, y_drv = [], [], []
    per_driver_frac = {}
    for p in sorted(glob.glob(os.path.join(CACHE, "*", "*.parquet"))):
        base = os.path.basename(p)[:-8]
        driver = base.split("__", 1)[0]
        rf = frac.get(base, np.nan)
        if not np.isfinite(rf):
            continue
        per_driver_frac.setdefault(driver, []).append(rf)
        lab = "rlog" if rf >= 0.5 else "qlog"
        df = pd.read_parquet(p)
        for wf in route_windows(df):
            rows.append(wf); y_log.append(lab); y_drv.append(driver)

    X = pd.DataFrame(rows)
    y_log = np.array(y_log); y_drv = np.array(y_drv)
    n_rlog = (y_log == "rlog").sum(); n_qlog = (y_log == "qlog").sum()
    print(f"windows={len(X)}  rlog={n_rlog}  qlog={n_qlog}  drivers={len(set(y_drv))}")

    # (1) DRIVER↔log-type correlation: how many drivers are >90% one log type?
    pdf = pd.Series({d: np.mean(v) for d, v in per_driver_frac.items()})
    pure = ((pdf > 0.9) | (pdf < 0.1)).mean()
    print(f"\n(1) driver-level rlog fraction: {pure*100:.0f}% of drivers are >90% one log-type "
          f"(mean {pdf.mean():.2f}); the more 'pure' drivers, the bigger the confound risk.")

    # (2) predict log-type from features, driver-grouped CV
    if n_rlog < 20 or n_qlog < 20:
        print("\n(2) too few of one class to probe reliably."); return
    n_splits = min(5, len(set(y_drv)))
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                   class_weight="balanced", random_state=0, n_jobs=-1))])
    yp = cross_val_predict(pipe, X.to_numpy(float), y_log, groups=y_drv,
                           cv=GroupKFold(n_splits), method="predict")
    acc = accuracy_score(y_log, yp); bacc = balanced_accuracy_score(y_log, yp)
    print(f"\n(2) predict log-type from style features (driver-grouped CV): "
          f"acc={acc:.3f}  balanced_acc={bacc:.3f}  (chance 0.500)")
    print("VERDICT:", "SAFE — features are log-type-agnostic (≈chance)"
          if bacc < 0.60 else
          "LEAKS — control log-type (adversarial head) or fall back to uniform-qlog")


if __name__ == "__main__":
    main()
