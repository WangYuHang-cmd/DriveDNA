#!/usr/bin/env python3
"""
Within-vehicle-model identifiability sanity check (the premise test for DriveDNA).

Question: holding the VEHICLE MODEL fixed (so any signal must be driver *style*, not vehicle),
do handcrafted style features separate drivers on HELD-OUT trips?

Protocol:
  * one vehicle model; cached routes -> 60s windows of human+moving driving -> style features
  * route-grouped CV (GroupKFold on route id): a route is never in train and test together,
    so we measure cross-trip generalisation, not within-trip memorisation
  * leak-free pipeline (impute+scale+classifier fit on train folds only)
  * report window-level & route-level (majority-vote) driver-ID accuracy vs chance (1/n_drivers)
  * plus unsupervised route-embedding leave-one-out nearest-neighbour same-driver retrieval

    ../openpilot/.venv/bin/python identifiability.py RIVIAN_R1_GEN1
"""
import os
import sys
import glob
import numpy as np
import pandas as pd

sys.path.insert(0, "./code/preprocessing")
from style_features import route_windows, FEATURE_GROUPS  # noqa: E402
from model_merge import canon  # noqa: E402

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import accuracy_score, balanced_accuracy_score

CACHE = "./data/cache"


def build_dataset(model):
    """model = CONSOLIDATED model name; gathers all raw fingerprint dirs that map to it."""
    want = canon(model)
    rows, meta = [], []
    for raw in sorted(os.listdir(CACHE)):
        if canon(raw) != want:
            continue
        for p in sorted(glob.glob(os.path.join(CACHE, raw, "*.parquet"))):
            base = os.path.basename(p)[:-8]
            driver, route = base.split("__", 1)
            df = pd.read_parquet(p)
            for wf in route_windows(df):
                rows.append(wf)
                meta.append((driver, route))
    X = pd.DataFrame(rows)
    meta = pd.DataFrame(meta, columns=["driver", "route"])
    return X, meta


def _pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                      class_weight="balanced", random_state=0, n_jobs=-1)),
    ])


def _cv(X, y, groups, cols, n_splits):
    """Route-grouped CV window+route accuracy for a feature subset."""
    Xv = X[cols].to_numpy(dtype=float)
    yp = cross_val_predict(_pipe(), Xv, y, groups=groups, cv=GroupKFold(n_splits), method="predict")
    win = accuracy_score(y, yp)
    dfp = pd.DataFrame({"route": groups, "true": y, "pred": yp})
    rt = [(g.true.iloc[0], g.pred.mode().iloc[0]) for _, g in dfp.groupby("route")]
    rt = np.array(rt)
    route = (rt[:, 0] == rt[:, 1]).mean()
    return win, route, yp


def _eligible_drivers(meta, min_routes=2, min_windows=8):
    """Prefer the frozen within-vehicle split; else keep drivers with enough routes+windows."""
    rpd = meta.groupby("driver").route.nunique()
    wpd = meta.driver.value_counts()
    return sorted(d for d in meta.driver.unique() if rpd[d] >= min_routes and wpd[d] >= min_windows)


def main(model):
    X, meta = build_dataset(model)
    feat_cols = list(X.columns)
    # restrict to drivers with enough data so route-grouped CV is well-posed
    keep = _eligible_drivers(meta)
    mask = meta.driver.isin(keep).to_numpy()
    X, meta = X[mask].reset_index(drop=True), meta[mask].reset_index(drop=True)
    drivers = sorted(meta.driver.unique())
    n_drv = len(drivers)
    if n_drv < 2:
        print(f"model={model}: <2 eligible drivers — skip"); return
    chance = 1.0 / n_drv
    print(f"model={model}  eligible_drivers={n_drv}  routes={meta.route.nunique()}  "
          f"windows={len(X)}  features={len(feat_cols)}  chance={chance:.3f}")
    print("windows per driver:")
    print(meta.driver.value_counts().to_string())

    y = meta.driver.to_numpy()
    groups = meta.route.to_numpy()
    Xv = X.to_numpy(dtype=float)
    min_routes_per_driver = meta.groupby("driver").route.nunique().min()
    n_splits = int(max(2, min(5, meta.route.nunique(), min_routes_per_driver)))

    # ---- feature-GROUP comparison (INPUT vs PATH vs long/follow/lane) ----
    groups_of = {}
    for c in feat_cols:
        groups_of.setdefault(FEATURE_GROUPS.get(c, "other"), []).append(c)
    subsets = {
        "ALL": feat_cols,
        "INPUT (steering+pedal)": groups_of.get("input", []),
        "PATH (curvature+slip)": groups_of.get("path", []),
        "LATERAL input+path": groups_of.get("input", []) + groups_of.get("path", []),
        "LONG (accel/jerk)": groups_of.get("long", []),
        "FOLLOW (THW/TTC)": groups_of.get("follow", []),
        "LANE (SDLP/bias)": groups_of.get("lane", []),
        "no-speed": [c for c in feat_cols if FEATURE_GROUPS.get(c) != "speed"],
    }
    print(f"\n== FEATURE-GROUP identifiability (route-grouped CV, chance {chance:.3f}) ==")
    print(f"{'group':26} {'#feat':>5} {'win_acc':>8} {'route_acc':>9} {'lift':>6}")
    for name, cols in subsets.items():
        if not cols:
            print(f"{name:26} {'0':>5}  (no features)"); continue
        win, route, _ = _cv(X, y, groups, cols, n_splits)
        print(f"{name:26} {len(cols):>5} {win:>8.3f} {route:>9.3f} {route/chance:>5.1f}x")

    # ---- detailed analysis on ALL features ----
    gkf = GroupKFold(n_splits=n_splits)
    y_pred = cross_val_predict(_pipe(), Xv, y, groups=groups, cv=gkf, method="predict")

    win_acc = accuracy_score(y, y_pred)
    win_bacc = balanced_accuracy_score(y, y_pred)
    print(f"\n== WINDOW-LEVEL (route-grouped CV) ==")
    print(f"accuracy         {win_acc:.3f}   (chance {chance:.3f}, lift {win_acc/chance:.1f}x)")
    print(f"balanced acc     {win_bacc:.3f}")

    # route-level: majority vote of a route's window predictions
    dfp = pd.DataFrame({"route": groups, "true": y, "pred": y_pred})
    route_true, route_pred = [], []
    for r, g in dfp.groupby("route"):
        route_true.append(g.true.iloc[0])
        route_pred.append(g.pred.mode().iloc[0])
    route_acc = accuracy_score(route_true, route_pred)
    print(f"\n== ROUTE-LEVEL (majority vote) ==")
    print(f"accuracy         {route_acc:.3f}   (chance {chance:.3f}, lift {route_acc/chance:.1f}x)  "
          f"[{sum(np.array(route_true)==np.array(route_pred))}/{len(route_true)} routes]")

    # unsupervised: route embedding = mean window feature; leave-one-out NN same-driver retrieval
    Xr = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(Xv), columns=feat_cols)
    Xr["route"] = groups; Xr["driver"] = y
    emb = Xr.groupby("route").agg({**{c: "mean" for c in feat_cols}, "driver": "first"})
    E = StandardScaler().fit_transform(emb[feat_cols].to_numpy())
    D = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1)
    np.fill_diagonal(D, np.inf)
    nn = D.argmin(axis=1)
    drv = emb.driver.to_numpy()
    nn_acc = np.mean(drv[nn] == drv)
    print(f"\n== ROUTE-EMBEDDING NN RETRIEVAL (leave-one-out) ==")
    print(f"top-1 same-driver {nn_acc:.3f}   (chance ~{chance:.3f}, lift {nn_acc/chance:.1f}x)")

    # per-driver route-level recall
    print(f"\n== per-driver route-level recall ==")
    rt = pd.DataFrame({"true": route_true, "pred": route_pred})
    for d in drivers:
        sub = rt[rt.true == d]
        print(f"  {d[:12]}  {sum(sub.true==sub.pred)}/{len(sub)}")

    # feature-importance snapshot (fit on all, for interpretation only)
    pipe = _pipe()
    pipe.fit(Xv, y)
    imp = pd.Series(pipe.named_steps["rf"].feature_importances_, index=feat_cols).sort_values(ascending=False)
    print(f"\n== top style features (RF importance) ==")
    print(imp.head(10).to_string())

    print("\nVERDICT:", "PREMISE HOLDS — style is identifiable within-vehicle on held-out trips"
          if route_acc > 2 * chance else "WEAK — identifiability near chance; revisit features/data")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "RIVIAN_R1_GEN1")
