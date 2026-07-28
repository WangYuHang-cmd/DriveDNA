#!/usr/bin/env python3
"""
E3 + E4 (appendix):
  E3 — distribution distances (MMD / KL / Wasserstein-1) between predicted and
       observed future-behavior statistics, generic vs personalized CVAE,
       scenario-normalized via harness.style_distance.
  E4 — per-driver personalization gain: Delta-NLL and Delta-RMSE per driver,
       share of drivers with positive gain, cluster (driver-level) bootstrap CI.

Replicates the frozen w5_cvae eval protocol (rng 20260709, k=5 support,
route-disjoint) over the three seed checkpoints. CPU-safe.
Writes results/e3_e4_distribution.json.
"""
import sys, os, json
import numpy as np
import pandas as pd
import torch

BASE = "."
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")
from harness import style_distance
from w5_cvae import CVAEPredictor, SupportEncoder, TGT, L_HIST, L_FUT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
BUNDLE = f"{BASE}/data/colab_bundle"
CKPTS = ["w5_cvae.pt", "w5_cvae_fe10.pt", "w5_cvae_fe11.pt"]

X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
meta = pd.read_parquet(f"{BUNDLE}/windows_meta.parquet")
ch = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
tgt_ix = [ch.index(c) for c in TGT]
folds = json.load(open(f"{BUNDLE}/driver_folds.json"))
ev_idx = np.flatnonzero(meta.driver.isin(set(folds["val"]) | set(folds["few_shot_heldout"])).to_numpy())
me = meta.iloc[ev_idx].reset_index(drop=True)


def feats(y):
    """[B, L_FUT, 2] -> per-anchor behavior statistics (normalized units)."""
    a, k = y[:, :, 0], y[:, :, 1]
    return pd.DataFrame({"a_mean": a.mean(1), "a_p05": np.percentile(a, 5, axis=1),
                         "k_absmean": np.abs(k).mean(1)})


def run_ckpt(name):
    ck = torch.load(f"{BASE}/experiments/checkpoints/{name}", map_location=DEV, weights_only=False)
    mu_, sd_ = ck["mu"], ck["sd"]
    norm = lambda a: (a.astype(np.float32) - mu_) / sd_
    sup = SupportEncoder(X.shape[2]).to(DEV); sup.load_state_dict(ck["sup"]); sup.eval()
    net = CVAEPredictor(X.shape[2], len(TGT)).to(DEV); net.load_state_dict(ck["net"]); net.eval()
    rng_eval = np.random.default_rng(20260709)
    per_driver, pools = [], {"g": [], "p": [], "true": [], "scen": []}
    with torch.no_grad():
        for d, g in me.groupby("driver"):
            rts = g.route.unique()
            if len(rts) < 2 or len(g) < 8:
                continue
            rng_eval.shuffle(rts)
            sup_r = set(rts[: max(1, len(rts) // 2)])
            sup_pool = ev_idx[g.index[g.route.isin(sup_r)].to_numpy()]
            qry = ev_idx[g.index[~g.route.isin(sup_r)].to_numpy()]
            qry_scen = g.scenario[~g.route.isin(sup_r)].to_numpy()
            if len(sup_pool) < 1 or len(qry) < 4:
                continue
            w = norm(X[np.sort(qry)])[np.argsort(np.argsort(qry))]
            anchors = rng_eval.integers(L_HIST, 600 - L_FUT, len(qry))
            xs = torch.from_numpy(np.stack([w[i, a - L_HIST:a] for i, a in enumerate(anchors)])).to(DEV)
            ys = torch.from_numpy(np.stack([w[i, a:a + L_FUT][:, tgt_ix] for i, a in enumerate(anchors)])).to(DEV)
            pick = np.sort(rng_eval.choice(sup_pool, min(5, len(sup_pool)), replace=False))
            zd = sup(torch.from_numpy(norm(X[pick])).to(DEV)).mean(0, keepdim=True).expand(len(qry), -1)
            row = {"driver": d, "n_qry": len(qry)}
            for tag, z in [("g", torch.zeros_like(zd)), ("p", zd)]:
                nll, rmse = net.eval_nll_rmse(xs, z, ys)
                row[f"nll_{tag}"], row[f"rmse_{tag}"] = nll, float(np.mean(rmse))
                # one predictive sample per anchor for the distribution pools
                h = net.ctx(xs, z)
                mp, lp = net.prior(h).chunk(2, -1); lp = lp.clamp(-6, 3)
                zl = mp + torch.randn_like(mp) * (0.5 * lp).exp()
                muy, logv = net.dec(torch.cat([h, zl], -1)).chunk(2, -1)
                ysamp = (muy + torch.randn_like(muy) * (0.5 * logv.clamp(-3, 3)).exp())
                pools[tag].append(ysamp.reshape(len(qry), L_FUT, len(TGT)).cpu().numpy())
            pools["true"].append(ys.cpu().numpy()); pools["scen"].append(qry_scen)
            per_driver.append(row)
    pd_df = pd.DataFrame(per_driver)
    scen = pd.Series(np.concatenate(pools["scen"]))
    Ft = feats(np.concatenate(pools["true"]))
    dist = {}
    for tag in ("g", "p"):
        Fp = feats(np.concatenate(pools[tag]))
        dist[tag] = {m: style_distance(Fp, Ft, scen, list(Ft.columns), metric=m)
                     for m in ("mmd", "kl", "w1")}
    return pd_df, dist


all_pd, all_dist = [], []
for name in CKPTS:
    pd_df, dist = run_ckpt(name)
    pd_df["ckpt"] = name
    all_pd.append(pd_df); all_dist.append(dist)
    dn = pd_df.nll_g - pd_df.nll_p
    dr = (pd_df.rmse_g - pd_df.rmse_p) / pd_df.rmse_g * 100
    print(f"{name}: drivers {len(pd_df)}  dNLL mean {dn.mean():+.2f} (pos {100*(dn>0).mean():.0f}%)"
          f"  dRMSE% {dr.mean():+.2f} (pos {100*(dr>0).mean():.0f}%)"
          f"  W1 g {dist['g']['w1']:.4f} p {dist['p']['w1']:.4f}", flush=True)

P = pd.concat(all_pd)
rng = np.random.default_rng(0)


def boot(series_by_ckpt):
    """driver-level bootstrap of the across-seed mean."""
    means = []
    for _ in range(2000):
        vals = []
        for dfc in series_by_ckpt:
            vals.append(dfc.sample(len(dfc), replace=True, random_state=int(rng.integers(1e9))).mean())
        means.append(np.mean(vals))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


dnll_by = [df.nll_g - df.nll_p for df in all_pd]
drmse_by = [(df.rmse_g - df.rmse_p) / df.rmse_g * 100 for df in all_pd]
out = {
    "per_driver_gain": {
        "dNLL_mean": float(np.mean([s.mean() for s in dnll_by])),
        "dNLL_ci95_driver_bootstrap": boot(dnll_by),
        "dNLL_share_positive": float(np.mean([(s > 0).mean() for s in dnll_by])),
        "dRMSEpct_mean": float(np.mean([s.mean() for s in drmse_by])),
        "dRMSEpct_ci95_driver_bootstrap": boot(drmse_by),
        "dRMSEpct_share_positive": float(np.mean([(s > 0).mean() for s in drmse_by])),
        "n_drivers": int(len(all_pd[0])), "n_seeds": len(CKPTS),
    },
    "distribution_distances": {
        m: {"generic": float(np.mean([d["g"][m] for d in all_dist])),
            "generic_sd": float(np.std([d["g"][m] for d in all_dist])),
            "personalized": float(np.mean([d["p"][m] for d in all_dist])),
            "personalized_sd": float(np.std([d["p"][m] for d in all_dist]))}
        for m in ("mmd", "kl", "w1")
    },
    "features": ["a_mean", "a_p05", "k_absmean"], "units": "normalized (z-scored) targets",
    "per_driver_records": P.to_dict(orient="records"),
}
print(json.dumps({k: out[k] for k in ("per_driver_gain", "distribution_distances")}, indent=1), flush=True)
json.dump(out, open(f"{BASE}/results/e3_e4_distribution.json", "w"), indent=1)
print("saved results/e3_e4_distribution.json", flush=True)
