#!/usr/bin/env python3
"""Extract teaser Panel-A candidate materials: matched-pair frames + traces + preview."""
import numpy as np, pandas as pd, json, os, subprocess
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "."
OUT = f"{BASE}/figs_making"
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c: i for i, c in enumerate(ch)}
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
CA, CB = "#eb6834", "#2a78d6"
PAIRS = [(54290, 53749), (10737, 10787), (12047, 12024), (10786, 10965)]

def curve_entry(kc, v):
    """first sustained curvature rise above 0.006 at speed"""
    kk = np.convolve(np.abs(kc), np.ones(5)/5, "same")
    cond = (kk > 0.006) & (v > 6)
    i, N = 0, len(cond)
    while i < N:
        if cond[i:i+15].all():
            return i
        i += 1
    return None

def extract(gi):
    w = X[gi].astype(np.float32)
    v, a, kc = w[:, ix["vEgo"]], w[:, ix["aEgo"]], w[:, ix["actual_curvature"]]
    e = curve_entry(kc, v)
    if e is None: return None
    lo, hi = max(0, e-80), min(600, e+80)
    return dict(gi=gi, entry=e, lo=lo, hi=hi,
                v=v[lo:hi], a=np.convolve(a, np.ones(3)/3, "same")[lo:hi],
                k=np.abs(np.convolve(kc, np.ones(3)/3, "same"))[lo:hi],
                t=(np.arange(lo, hi)-e)/10.0)

def frame(gi, entry, outp, dt=-0.5):
    r = W.iloc[gi]
    km = kmin.get((r.model, r.driver, r.route))
    tt = float(r.t0) + entry/10.0 + dt
    k = km + int(tt//60)
    p1 = f"../Dataset/{r.model}/{r.driver}/{r.route}/{k}--qcamera.ts"
    if not os.path.exists(p1): return False
    subprocess.run(["ffmpeg","-y","-v","error","-ss",f"{tt%60:.2f}","-i",p1,"-frames:v","1","-q:v","3",outp],
                   capture_output=True, timeout=60)
    return os.path.exists(outp)

for n, (wa, wb) in enumerate(PAIRS, 1):
    d = f"{OUT}/pair{n}"
    os.makedirs(d, exist_ok=True)
    A, B = extract(wa), extract(wb)
    if A is None or B is None:
        print(f"pair{n}: no curve entry, skipped"); continue
    fa = frame(wa, A["entry"], f"{d}/frame_A.jpg")
    fb = frame(wb, B["entry"], f"{d}/frame_B.jpg")
    ra, rb = W.iloc[wa], W.iloc[wb]
    json.dump({"win_a": wa, "win_b": wb, "model": ra.model, "drivers": [ra.driver, rb.driver],
               "entry_a": int(A["entry"]), "entry_b": int(B["entry"]),
               "decel_p05": [float(ra.decel_p05), float(rb.decel_p05)],
               "v_mean": [float(ra.v_mean), float(rb.v_mean)]},
              open(f"{d}/meta.json","w"), indent=1)
    np.savez(f"{d}/traces.npz", **{f"A_{k}": A[k] for k in ("t","v","a","k")},
             **{f"B_{k}": B[k] for k in ("t","v","a","k")})
    # preview: frames on top, aligned sparklines below (x = seconds from curve entry)
    fig, axes = plt.subplots(4, 2, figsize=(7.5, 6.2),
                             gridspec_kw=dict(height_ratios=[2.4,1,1,1], hspace=0.35, wspace=0.12))
    for col, (S, lab, colr, fr) in enumerate([(A,"Driver A",CA,fa),(B,"Driver B",CB,fb)]):
        ax = axes[0][col]
        if fr:
            ax.imshow(plt.imread(f"{d}/frame_{'A' if col==0 else 'B'}.jpg"))
        ax.set_title(f"{lab}  ({W.iloc[S['gi']].model})", fontsize=9, color=colr)
        ax.axis("off")
        for row, sig, name in ((1,"v","v  (m/s)"), (2,"a","a  (m/s$^2$)"), (3,"k","|κ|  (1/m)")):
            ax = axes[row][col]
            ax.plot(S["t"], S[sig], color=colr, lw=1.6)
            ax.axvline(0, color="#888", ls="--", lw=0.8)
            ax.set_xlim(-8, 8)
            for s in ("top","right"): ax.spines[s].set_visible(False)
            ax.tick_params(labelsize=6)
            if col == 0: ax.set_ylabel(name, fontsize=7)
            if row < 3: ax.set_xticklabels([])
        axes[3][col].set_xlabel("s from curve entry", fontsize=7)
    # unify y-lims across columns
    for row, sig in ((1,"v"),(2,"a"),(3,"k")):
        ymin = min(axes[row][c].get_ylim()[0] for c in (0,1))
        ymax = max(axes[row][c].get_ylim()[1] for c in (0,1))
        for c in (0,1): axes[row][c].set_ylim(ymin, ymax)
    fig.savefig(f"{d}/preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"pair{n}: {ra.model} decel {ra.decel_p05:.1f} vs {rb.decel_p05:.1f}, frames {fa}/{fb} -> {d}/preview.png")
