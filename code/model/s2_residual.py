#!/usr/bin/env python3
"""
S2 — ResidualStyle baseline (anti-shortcut; must-land #3) + utility–leakage comparison.

Operationalizes: style = what remains after conditioning on context & vehicle.
Population model: per behavior stat y, GBM ŷ = f(scenario, speed, curvature, lead context,
vehicle params/model). Driver representation = window residual vector r = y − ŷ.
Compare RAW descriptors vs RESIDUALS on: (a) T2 enrollment protocol (utility, unseen-driver
few-shot) and (b) vehicle-leakage probe (invariance) → the first two points of the
utility–leakage Pareto (must-land #7).

    nice .../python s2_residual.py
"""
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "./code/eval")
from harness import enrollment_protocol, leakage_probe  # noqa: E402

WQ = "./data/segments/windows.parquet"
SEED = 20260709

BEHAV = ["decel_p05", "accel_p95", "jerk_rms", "steer_rate_rms", "curv_rate_rms",
         "thw_median", "lane_sdlp", "brake_rate"]          # behavior stats (style-bearing)
CONTEXT = ["v_mean", "stop_frac", "curv_p95", "lead_frac"]  # context covariates


def main():
    W = pd.read_parquet(WQ).reset_index(drop=True)
    # context design matrix: numeric context + scenario onehot + vehicle-model onehot (top models)
    Xc = W[CONTEXT].copy()
    Xc = pd.concat([Xc, pd.get_dummies(W.scenario, prefix="sc")], axis=1)
    top_models = W.model_canon.value_counts().head(30).index
    mc = W.model_canon.where(W.model_canon.isin(top_models), "OTHER")
    Xc = pd.concat([Xc, pd.get_dummies(mc, prefix="veh")], axis=1)
    Xc = Xc.astype(np.float32).to_numpy()

    # population models + residuals (out-of-fold not critical for a representation; use full fit
    # but with monotone-free shallow GBMs to avoid memorizing drivers via context quirks)
    RES = np.zeros((len(W), len(BEHAV)), dtype=np.float32)
    RAW = np.zeros_like(RES)
    for j, stat in enumerate(BEHAV):
        y = W[stat].to_numpy(dtype=np.float32)
        ok = np.isfinite(y)
        gbm = HistGradientBoostingRegressor(max_depth=4, max_iter=150, random_state=SEED)
        gbm.fit(Xc[ok], y[ok])
        pred = gbm.predict(Xc).astype(np.float32)
        r = y - pred
        r[~ok] = np.nan
        RES[:, j] = r
        RAW[:, j] = y
        print(f"  fit {stat:16} R2(train)={gbm.score(Xc[ok], y[ok]):.3f}", flush=True)

    def prep(A):
        A = SimpleImputer(strategy="median").fit_transform(A)
        return StandardScaler().fit_transform(A)

    ok_drv = (W.groupby("driver").route.transform("nunique") >= 2) & \
             (W.groupby("driver").driver.transform("size") >= 10)
    m = ok_drv.to_numpy()
    drv, rts = W.driver.to_numpy()[m], W.route.to_numpy()[m]
    veh = W.model_canon.to_numpy()[m]

    print(f"\nprotocol windows={m.sum()}  drivers={pd.Series(drv).nunique()}")
    out = {}
    for name, A in [("RAW descriptors", prep(RAW)[m]), ("RESIDUAL style", prep(RES)[m])]:
        res = enrollment_protocol(A, drv, rts)
        lk = leakage_probe(A, veh, drv)
        k = max(res)
        out[name] = (res[k], lk)
        print(f"\n== {name} ==")
        print(f"  T2 @{k}min: top1={res[k]['top1']:.3f} top5={res[k]['top5']:.3f} "
              f"AUROC={res[k]['auroc']:.3f} EER={res[k]['eer']:.3f} ({res[k]['n_drivers']} drivers)")
        print(f"  vehicle-leakage: bal_acc={lk['bal_acc']:.3f} (chance {lk['chance']:.3f})")

    r_raw, l_raw = out["RAW descriptors"]
    r_res, l_res = out["RESIDUAL style"]
    print("\n== UTILITY–LEAKAGE (Pareto points) ==")
    print(f"  RAW:      utility(top1)={r_raw['top1']:.3f}  leakage={l_raw['bal_acc']:.3f}")
    print(f"  RESIDUAL: utility(top1)={r_res['top1']:.3f}  leakage={l_res['bal_acc']:.3f}")
    print("  → residualization should cut leakage; utility drop indicates how much apparent"
          " 'style' was context/vehicle shortcut.")


if __name__ == "__main__":
    main()
