#!/usr/bin/env python3
"""Appendix figure: t-SNE map of the SupCon driver-style embedding space.

Panel (a): all 428 drivers, hue assigned by the angular position of each
driver's centroid on the 2-D map (nearby clusters get distinguishable hues);
representative drivers (k-means over driver centroids + the multi-vehicle
driver) are outlined and labeled.
Panel (b): the same projection colored by the six driving scenarios --
clusters align with drivers, not scenarios.

CPU-safe. Caches embeddings to figs_making/emb_cache_s1.npz.
"""
import sys, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BASE = "."
HERE = f"{BASE}/figs_making"
sys.path.insert(0, f"{BASE}/code/model")
SEED = 20260709
CAP = 60          # windows per driver on the map
INK, GRAY = "#3d4451", "#8a93a5"
plt.rcParams.update({"font.family": "DejaVu Sans"})

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)

# ---------- 1. embed all windows (cached) ----------
cache = f"{HERE}/emb_cache_s1.npz"
if os.path.exists(cache):
    Z = np.load(cache)["Z"]
    print("loaded cached embeddings", Z.shape, flush=True)
else:
    import torch
    from s1_supcon import Encoder
    X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
    ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location="cpu", weights_only=False)
    enc = Encoder(c_in=X.shape[2]).eval(); enc.load_state_dict(ck["model"])
    mu, sd = ck["mu"], ck["sd"]
    torch.set_num_threads(os.cpu_count())
    Z = np.zeros((len(W), 128), np.float32)
    with torch.no_grad():
        for i in range(0, len(W), 512):
            a = np.nan_to_num((X[i:i + 512].astype(np.float32) - mu) / sd)
            Z[i:i + 512] = enc(torch.from_numpy(a)).numpy()
            if (i // 512) % 20 == 0:
                print(f"  embed {i}/{len(W)}", flush=True)
    np.savez_compressed(cache, Z=Z)
    print("cached", cache, flush=True)
Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)

# ---------- 2. per-driver cap + t-SNE ----------
rng = np.random.default_rng(SEED)
sel = np.concatenate([rng.permutation(ix)[:CAP] for _, ix in W.groupby("driver").indices.items()])
sel = np.sort(sel)
print(f"map points: {len(sel)} from {W.driver.nunique()} drivers", flush=True)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
xy_cache = f"{HERE}/emb_tsne_xy.npz"
if os.path.exists(xy_cache):
    XY = np.load(xy_cache)["XY"]
    print("loaded cached t-SNE", flush=True)
else:
    P50 = PCA(n_components=50, random_state=SEED).fit_transform(Z[sel])
    XY = TSNE(n_components=2, perplexity=40, init="pca", random_state=SEED,
              max_iter=1000, verbose=1).fit_transform(P50)
    np.savez_compressed(xy_cache, XY=XY)
    print("tsne done", flush=True)

sub = W.iloc[sel].reset_index(drop=True)
sub["x"], sub["y"] = XY[:, 0], XY[:, 1]

# ---------- 3. hue by centroid angle ----------
cent = sub.groupby("driver")[["x", "y"]].mean()
ang = np.arctan2(cent.y - cent.y.mean(), cent.x - cent.x.mean())
rad = np.hypot(cent.x - cent.x.mean(), cent.y - cent.y.mean())
hue = (ang + np.pi) / (2 * np.pi)
sat = 0.40 + 0.50 * (rad / rad.max())          # inner drivers muted, outer saturated
import colorsys, hashlib
def jit(d):                                     # deterministic per-driver contrast jitter
    h8 = int(hashlib.md5(d.encode()).hexdigest()[:4], 16)
    return (h8 % 3 - 1) * 0.05, (h8 // 3 % 3 - 1) * 0.10
drv_color = {}
for d, h, s in zip(cent.index, hue, sat):
    dh, dv = jit(d)
    drv_color[d] = colorsys.hsv_to_rgb((h + dh) % 1.0, min(1, max(0.25, s)), min(0.95, max(0.55, 0.80 + dv)))
colors = sub.driver.map(drv_color)

# ---------- 4. representative drivers ----------
from sklearn.cluster import KMeans
big = cent[sub.groupby("driver").size().reindex(cent.index) >= 50]
km = KMeans(n_clusters=8, n_init=10, random_state=SEED).fit(big[["x", "y"]])
reps = []
counts = sub.groupby("driver").size()
for c in range(8):
    members = big.index[km.labels_ == c]
    reps.append(counts[members].idxmax())
multi = "<multi-vehicle-driver-id>"   # the raw device id is withheld; see data/release_mapping (private)
if multi in cent.index and multi not in reps:
    reps.append(multi)
meta = W.groupby("driver").agg(model=("model_canon", lambda s: s.mode().iat[0]),
                               n=("driver", "size"))
print("representatives:", reps, flush=True)

# ---------- 5. figure ----------
fig, axes = plt.subplots(1, 2, figsize=(13.6, 6.4), gridspec_kw={"wspace": 0.06})
for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color("#d7dbe2")

axA, axB = axes
axA.scatter(sub.x, sub.y, s=3.2, c=list(colors), alpha=0.5, linewidths=0, rasterized=True)
handles = []
for j, d in enumerate(reps, 1):
    pts = sub[sub.driver == d]
    axA.scatter(pts.x, pts.y, s=7.5, color=drv_color[d], alpha=0.95,
                linewidths=0.35, edgecolors=INK, rasterized=True)
    cx, cy = pts.x.mean(), pts.y.mean()
    axA.annotate(f"D{j}", (cx, cy), fontsize=11, fontweight="bold", color=INK,
                 ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.18", fc="white", ec=INK, lw=1.0, alpha=0.9))
axA.set_title("(a) Colored by driver (all 428 drivers; D1--D9 mark representative drivers)", fontsize=11.5, color=INK)

EV = pd.read_parquet(f"{BASE}/data/segments/maneuver_events.parquet")
EV = EV[EV.gi.isin(set(sel))]
PRIO = ["curve", "turn", "lane_change", "decel", "accel", "car_following"]
MCOL = {"curve": "#8a5fb0", "turn": "#e0a23f", "lane_change": "#c96a5e",
        "decel": "#7189b9", "accel": "#45a49b", "car_following": "#a9c66f", "none": "#c9ced8"}
gi2cls = {}
for cls in reversed(PRIO):               # 后写覆盖 → 最终留下优先级最高的类
    for g in EV[EV.cls == cls].gi.unique():
        gi2cls[g] = cls
sub["mnv"] = [gi2cls.get(g, "none") for g in sel]
for cls in ["none"] + PRIO[::-1]:        # 先画 none,再按稀有类后画置顶
    m_ = sub.mnv == cls
    axB.scatter(sub.x[m_], sub.y[m_], s=3.2, color=MCOL[cls], alpha=0.45 if cls == "none" else 0.6,
                linewidths=0, rasterized=True)
axB.legend(handles=[Line2D([0], [0], marker="o", ls="", ms=7, mfc=MCOL[k], mec="none",
                           label={"car_following": "car following", "lane_change": "lane change",
                                  "none": "no event"}.get(k, k)) for k in PRIO + ["none"]],
           loc="upper left", fontsize=7.6, frameon=True, framealpha=0.9,
           edgecolor="#d7dbe2", handletextpad=0.3, borderpad=0.5)
axB.set_title("(b) Same projection, colored by maneuver annotation", fontsize=11.5, color=INK)
print("maneuver coloring:", sub.mnv.value_counts().to_dict())

fig.savefig(f"{HERE}/embedding_map.png", dpi=200, bbox_inches="tight")
fig.savefig(f"{HERE}/embedding_map.pdf", dpi=200, bbox_inches="tight")
json.dump({"reps": [{"tag": f"D{j}", "driver": d, "model": meta.loc[d, 'model'],
                     "windows": int(meta.loc[d, 'n'])} for j, d in enumerate(reps, 1)],
           "n_points": int(len(sub)), "n_drivers": int(W.driver.nunique()), "cap": CAP},
          open(f"{HERE}/embedding_map_meta.json", "w"), indent=1)
print("saved embedding_map.{png,pdf}", flush=True)
