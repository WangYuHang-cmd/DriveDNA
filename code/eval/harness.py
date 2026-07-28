#!/usr/bin/env python3
"""
DriveDNA evaluation harness — the shared metrics library for all tasks/baselines.

Implements the plan's style-sensitive metric suite:
  * verification: AUROC, EER                       (T2, T4)
  * retrieval: top-1/top-5 via driver prototypes   (T2)
  * enrollment curves: 1/3/5/10-min support        (T2)
  * distribution distances: MMD(RBF), KL(hist), Wasserstein — scenario-normalized (T3)
  * Personalization Gain: PG = D_generic − D_personalized                        (T3)
  * leakage probes: balanced-acc of nuisance prediction from embeddings + Pareto (diagnostics)

Deterministic (seeded). Self-test (`python harness.py`) runs the T2 protocol on handcrafted
window statistics from windows.parquet — which IS the interpretable descriptor-audit anchor.
"""
import os
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from scipy.stats import wasserstein_distance

SEED = 20260709
RATE = 10.0
WIN_MIN = 1.0  # one 60 s window ≈ 1 minute of support


# ---------- verification ----------
def verification_metrics(scores, labels):
    """scores: similarity (higher = same); labels: 1 same / 0 different."""
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return {"auroc": float(auroc), "eer": float((fpr[i] + fnr[i]) / 2)}


def cosine(a, b):
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-9)
    return (a * b).sum(-1)


# ---------- T2 protocol: few-shot enrollment + retrieval + verification ----------
def enrollment_protocol(emb, drivers, routes, k_minutes=(1, 3, 5, 10), min_query=5, seed=SEED):
    """
    emb [N,D] window embeddings; drivers/routes [N]. For each driver with ≥2 routes:
    support = windows from a random half of their routes (capped at k minutes),
    query = windows from the held-out routes (route-disjoint). Prototype = mean(support).
    Returns per-k: top-1/top-5 retrieval over all enrolled prototypes + verification AUROC/EER.
    """
    rng = np.random.default_rng(seed)
    emb = np.asarray(emb, dtype=np.float64)
    df = pd.DataFrame({"driver": drivers, "route": routes, "i": np.arange(len(drivers))})
    out = {}
    for k in k_minutes:
        n_sup = max(1, int(round(k / WIN_MIN)))
        protos, proto_drv = [], []
        queries = []  # (driver, idx array)
        for d, g in df.groupby("driver"):
            rts = g.route.unique()
            if len(rts) < 2:
                continue
            rng.shuffle(rts)
            sup_r = set(rts[: max(1, len(rts) // 2)])
            sup_i = g[g.route.isin(sup_r)].i.to_numpy()
            qry_i = g[~g.route.isin(sup_r)].i.to_numpy()
            if len(sup_i) < 1 or len(qry_i) < min_query:
                continue
            sup_i = rng.choice(sup_i, min(n_sup, len(sup_i)), replace=False)
            protos.append(emb[sup_i].mean(0))
            proto_drv.append(d)
            queries.append((d, qry_i))
        if len(protos) < 5:
            continue
        P = np.stack(protos)
        Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
        proto_drv = np.array(proto_drv)
        top1 = top5 = nq = 0
        scores, labels = [], []
        for d, qi in queries:
            Q = emb[qi]
            Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
            S = Qn @ Pn.T                     # [nq, n_drivers]
            rank = np.argsort(-S, axis=1)
            true_col = int(np.where(proto_drv == d)[0][0])
            top1 += (rank[:, 0] == true_col).sum()
            top5 += (rank[:, :5] == true_col).any(1).sum()
            nq += len(qi)
            scores += list(S[:, true_col])                 # genuine
            imp = rng.integers(0, len(proto_drv), len(qi)) # impostor
            imp[imp == true_col] = (true_col + 1) % len(proto_drv)
            scores += list(S[np.arange(len(qi)), imp])
            labels += [1] * len(qi) + [0] * len(qi)
        v = verification_metrics(np.array(scores), np.array(labels))
        out[k] = {"n_drivers": len(protos), "n_queries": int(nq),
                  "top1": float(top1 / nq), "top5": float(top5 / nq), **v}
    return out


# ---------- distribution distances (scenario-normalized) ----------
def mmd_rbf(x, y, sigma=None):
    x = np.asarray(x, float).reshape(-1, 1); y = np.asarray(y, float).reshape(-1, 1)
    x = x[np.isfinite(x[:, 0])]; y = y[np.isfinite(y[:, 0])]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    z = np.concatenate([x, y])
    if sigma is None:
        d = np.abs(z - z.T); sigma = np.median(d[d > 0]) + 1e-9
    k = lambda a, b: np.exp(-np.square(a - b.T) / (2 * sigma ** 2))
    return float(k(x, x).mean() + k(y, y).mean() - 2 * k(x, y).mean())


def kl_hist(x, y, bins=30):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    lo, hi = np.nanpercentile(np.concatenate([x, y]), [0.5, 99.5])
    px, _ = np.histogram(x, bins=bins, range=(lo, hi)); py, _ = np.histogram(y, bins=bins, range=(lo, hi))
    px = (px + 1e-6) / (px.sum() + bins * 1e-6); py = (py + 1e-6) / (py.sum() + bins * 1e-6)
    return float(np.sum(px * np.log(px / py)))


def w1(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < 5 or len(y) < 5:
        return np.nan
    return float(wasserstein_distance(x, y))


def style_distance(pred, true, scenarios, features, metric="w1"):
    """Scenario-normalized style-distribution distance, averaged over scenario buckets & features.
    pred/true: DataFrames with `features` columns; scenarios: bucket per row (shared index)."""
    fn = {"w1": w1, "mmd": mmd_rbf, "kl": kl_hist}[metric]
    vals = []
    for sc in pd.unique(scenarios):
        m = scenarios == sc
        if m.sum() < 20:
            continue
        for f in features:
            vals.append(fn(pred.loc[m, f], true.loc[m, f]))
    return float(np.nanmean(vals))


def personalization_gain(d_generic, d_personalized):
    return float(d_generic - d_personalized)


# ---------- leakage probes ----------
def leakage_probe(emb, nuisance, groups, n_splits=5, seed=SEED):
    """Balanced-acc of predicting a nuisance label from embeddings (group-CV by driver).
    ≈chance ⇒ invariant; ≫chance ⇒ leaks."""
    y = pd.Series(nuisance).astype(str).to_numpy()
    classes = np.unique(y)
    if len(classes) < 2:
        return {"bal_acc": np.nan, "chance": np.nan}
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                     ("rf", RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                   class_weight="balanced",
                                                   random_state=seed, n_jobs=-1))])
    n_splits = min(n_splits, len(np.unique(groups)))
    yp = cross_val_predict(pipe, np.asarray(emb, float), y, groups=groups,
                           cv=GroupKFold(n_splits), method="predict")
    return {"bal_acc": float(balanced_accuracy_score(y, yp)), "chance": float(1 / len(classes))}


# ---------- self-test = descriptor-audit anchor on T2 ----------
STATS = ["v_mean", "stop_frac", "decel_p05", "accel_p95", "jerk_rms", "curv_p95",
         "curv_rate_rms", "steer_rate_rms", "lead_frac", "brake_rate", "thw_median", "lane_sdlp"]

if __name__ == "__main__":
    W = pd.read_parquet("./data/segments/windows.parquet")
    # descriptor embedding = imputed+standardized window stats
    X = W[STATS].to_numpy(float)
    X = SimpleImputer(strategy="median").fit_transform(X)
    X = StandardScaler().fit_transform(X)
    # keep drivers with >=2 routes and >=10 windows for a meaningful protocol
    ok = W.groupby("driver").route.transform("nunique") >= 2
    ok &= W.groupby("driver").driver.transform("size") >= 10
    Xs, Ws = X[ok.to_numpy()], W[ok.to_numpy()]
    print(f"descriptor-audit T2 protocol: {Ws.driver.nunique()} drivers, {len(Ws)} windows")
    res = enrollment_protocol(Xs, Ws.driver.to_numpy(), Ws.route.to_numpy())
    print(f"{'k(min)':>7} {'drivers':>8} {'top1':>7} {'top5':>7} {'AUROC':>7} {'EER':>7}")
    for k, r in res.items():
        print(f"{k:>7} {r['n_drivers']:>8} {r['top1']:>7.3f} {r['top5']:>7.3f} "
              f"{r['auroc']:>7.3f} {r['eer']:>7.3f}")
    chance = 1.0 / res[max(res)]["n_drivers"]
    print(f"(retrieval chance ≈ {chance:.4f})")
    # leakage probe of the descriptor embedding
    lk = leakage_probe(Xs, Ws.model_canon.to_numpy(), Ws.driver.to_numpy())
    print(f"vehicle-leakage of descriptors: bal_acc={lk['bal_acc']:.3f} (chance {lk['chance']:.3f})")
