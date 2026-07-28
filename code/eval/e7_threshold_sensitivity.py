#!/usr/bin/env python3
"""
E7: T1-primitive threshold-sensitivity check backing the Appendix C claim.

Relabels every window under three symmetric threshold pairs -- Q80/Q20
(production), Q75/Q25, Q85/Q15 -- using the same scenario-conditioned rule as
scenario_primitives.py, then re-runs the frozen S3 personalization evaluation
once (k=5) and stratifies the SAME per-query errors under each label set.

Reports: per-primitive label change rates vs Q80/Q20, stratified PG per
threshold pair, and whether every stratified-PG sign is preserved.
Writes results/e7_threshold_sensitivity.json.  CPU-safe.
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
from s3_personalized import SupportEncoder, Predictor, PredictorTF, L_HIST, L_FUT, TGT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 20260709
PAIRS = {"Q80/Q20": (80, 20), "Q75/Q25": (75, 25), "Q85/Q15": (85, 15)}
PRIMS = {  # primitive -> (window statistic, direction of the "high" label)
    "close_following": ("thw_median", "low"),
    "large_headway": ("thw_median", "high"),
    "hard_braking": ("decel_p05", "low"),
    "high_jerk": ("jerk_rms", "high"),
    "sharp_steer_raw": ("steer_rate_rms", "high"),
    "sharp_steer_path": ("curv_rate_rms", "high"),
    "lane_correction": ("lane_sdlp", "high"),
    "curve_entry_decel": ("decel_p05", "low"),
}

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
tgt_ix = [ch.index(c) for c in TGT]
folds = json.load(open(f"{BASE}/data/splits/driver_folds.json"))
ev_idx = np.flatnonzero(W.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())


def relabel(hi, lo):
    """scenario-conditioned percentile labels {-1,0,+1}, replicating scenario_primitives.py"""
    out = {}
    for prim, (stat, direction) in PRIMS.items():
        lab = np.full(len(W), np.nan)
        for sc, g in W.groupby("scenario"):
            if prim == "curve_entry_decel" and sc != "curve":
                continue
            if prim in ("close_following", "large_headway") and sc != "car_following":
                continue
            x = g[stat].to_numpy(); ok = np.isfinite(x)
            if ok.sum() < 50:
                continue
            qlo, qhi = np.nanpercentile(x[ok], lo), np.nanpercentile(x[ok], hi)
            v = np.zeros(len(g))
            if direction == "high":
                v[x >= qhi] = 1; v[x <= qlo] = -1
            else:
                v[x <= qlo] = 1; v[x >= qhi] = -1
            v[~ok] = np.nan
            lab[g.index.to_numpy()] = v
        out[prim] = lab
    return out


labels = {name: relabel(hi, lo) for name, (hi, lo) in PAIRS.items()}
# sanity: Q80/Q20 must reproduce the released p_* columns
for prim in PRIMS:
    a, b = labels["Q80/Q20"][prim], W[f"p_{prim}"].to_numpy()
    m = np.isfinite(a) & np.isfinite(b)
    assert (a[m] == b[m]).mean() > 0.999, f"reproduction mismatch on {prim}"
print("Q80/Q20 relabeling reproduces released labels: OK", flush=True)

change = {}
for name in ("Q75/Q25", "Q85/Q15"):
    rows = {}
    for prim in PRIMS:
        a, b = labels["Q80/Q20"][prim], labels[name][prim]
        m = np.isfinite(a) & np.isfinite(b) & ((a != 0) | (b != 0))
        rows[prim] = float((a[m] != b[m]).mean())
    change[name] = rows
    print(f"label-change rate vs Q80/Q20 [{name}]: "
          f"mean {np.mean(list(rows.values())):.3f}", flush=True)

# ---------- one frozen S3 eval pass (k=5), capturing per-query errors ----------
ck = torch.load(f"{BASE}/experiments/checkpoints/s3_pred.pt", map_location=DEV, weights_only=False)
mu, sd = ck["mu"], ck["sd"]
norm = lambda a: np.nan_to_num((a.astype(np.float32) - mu) / sd)
sup = SupportEncoder(X.shape[2]).to(DEV); sup.load_state_dict(ck["sup_enc"]); sup.eval()
try:
    pred = Predictor(X.shape[2], len(TGT)).to(DEV); pred.load_state_dict(ck["pred"])
except RuntimeError:
    pred = PredictorTF(X.shape[2], len(TGT)).to(DEV); pred.load_state_dict(ck["pred"])
    print("using PredictorTF checkpoint", flush=True)
pred.eval()
torch.set_num_threads(os.cpu_count())
rng = np.random.default_rng(SEED)
me = W.iloc[ev_idx].reset_index(drop=True)
rec = []
with torch.no_grad():
    for d, g in me.groupby("driver"):
        rts = g.route.unique()
        if len(rts) < 2 or len(g) < 8:
            continue
        rng.shuffle(rts)
        sup_r = set(rts[: max(1, len(rts) // 2)])
        sup_pool = ev_idx[g.index[g.route.isin(sup_r)].to_numpy()]
        qry = g.index[~g.route.isin(sup_r)].to_numpy()
        if len(sup_pool) < 1 or len(qry) < 4:
            continue
        qidx = ev_idx[qry]
        w = norm(X[np.sort(qidx)])[np.argsort(np.argsort(qidx))]
        anchors = rng.integers(L_HIST, 600 - L_FUT, len(qidx))
        xs = torch.from_numpy(np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])).to(DEV)
        ys = np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])
        e_g = pred(xs, torch.zeros(len(qidx), 64, device=DEV)).cpu().numpy()
        pick = np.sort(rng.choice(sup_pool, min(5, len(sup_pool)), replace=False))
        z = sup(torch.from_numpy(norm(X[pick])).to(DEV)).mean(0, keepdim=True)
        e_p = pred(xs, z.expand(len(qidx), -1)).cpu().numpy()
        rg = np.sqrt(((e_g - ys) ** 2).mean((1, 2)))
        rp = np.sqrt(((e_p - ys) ** 2).mean((1, 2)))
        for gi, a_, b_ in zip(qidx, rg, rp):
            rec.append((int(gi), float(a_), float(b_)))
R = pd.DataFrame(rec, columns=["gi", "rg", "rp"]).set_index("gi")
overall = (R.rg.mean() - R.rp.mean()) / R.rg.mean() * 100
print(f"eval queries: {len(R)} | overall PG(k=5) {overall:+.2f}%", flush=True)

strat = {}
for name in PAIRS:
    rows = {}
    for prim in PRIMS:
        lab = labels[name][prim][R.index.to_numpy()]
        m = lab == 1
        if m.sum() >= 30:
            gsum, psum = R.rg.to_numpy()[m].mean(), R.rp.to_numpy()[m].mean()
            rows[prim] = {"n": int(m.sum()), "pg_pct": float((gsum - psum) / gsum * 100)}
    strat[name] = rows

print(f"\n{'primitive':20} " + "  ".join(f"{n:>14}" for n in PAIRS))
signs_ok = True
for prim in PRIMS:
    vals = [strat[n].get(prim) for n in PAIRS]
    line = f"{prim:20} " + "  ".join(f"{v['pg_pct']:+7.2f}% n={v['n']:<5}" if v else f"{'--':>14}" for v in vals)
    print(line, flush=True)
    present = [v["pg_pct"] for v in vals if v]
    if len(present) == len(PAIRS) and len({p > 0 for p in present}) > 1:
        signs_ok = False
print("\nAll stratified-PG signs preserved across threshold pairs:", signs_ok)

json.dump({"pairs": {k: list(v) for k, v in PAIRS.items()}, "label_change_vs_Q80Q20": change,
           "stratified_pg": strat, "overall_pg_pct_k5": float(overall),
           "signs_preserved": bool(signs_ok), "n_queries": int(len(R))},
          open(f"{BASE}/results/e7_threshold_sensitivity.json", "w"), indent=1)
print("saved results/e7_threshold_sensitivity.json", flush=True)
