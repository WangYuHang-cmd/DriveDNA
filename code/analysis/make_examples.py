#!/usr/bin/env python3
"""
Multimodal example package for DriveDNA — human inspection / paper figures / supplement.

Extracts REAL examples from existing artifacts (no re-running of main experiments):
tasks T1–T6 (2–3 each) + supporting figures for findings A–H. Anonymized driver IDs
(salted-hash map). Outputs:
  results/examples/<id>/   frames montage + CAN plot + meta.json
  figures/examples/        finding figures
  results/examples/manifest.csv + REPORT.md
"""
import os
import sys
import json
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BASE = "."
DATASET = "../Dataset"
BUNDLE = f"{BASE}/data/colab_bundle"
RES = f"{BASE}/results/examples"
FIG = f"{BASE}/figures/examples"
sys.path.insert(0, f"{BASE}/code/model")
sys.path.insert(0, f"{BASE}/code/eval")

BLUE, AMBER, GRAY = "#3B82F6", "#D97706", "#6B7280"   # validated palette; gray = GT/reference only
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})
SEED = 20260709
rng = np.random.default_rng(SEED)

# ---------- shared data ----------
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
SEG = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
PAIRS = pd.read_parquet(f"{BASE}/data/splits/matched_context_pairs.parquet")
META = pd.read_parquet(f"{BUNDLE}/windows_meta.parquet")
X = np.load(f"{BUNDLE}/windows_x.npy", mmap_mode="r")
CH = json.load(open(f"{BUNDLE}/channels.json"))["channels"]
ANON = json.load(open(f"{BASE}/data/.anon/dongle_hash_map.json"))
KMIN = {(r.model, r.driver, r.route): int(r.k_min) for r in SEG.itertuples()}
FOLDS = json.load(open(f"{BUNDLE}/driver_folds.json"))
anon = lambda d: ANON.get(d, "drv_" + str(abs(hash(d)) % 10 ** 8))

manifest = []


def grab_frame(model, driver, route, t_route):
    km = KMIN.get((model, driver, route), 0)
    k = km + int(t_route // 60)
    f = os.path.join(DATASET, model, driver, route, f"{k}--qcamera.ts")
    if not os.path.exists(f):
        return None
    off = t_route - (k - km) * 60
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{max(0,off):.2f}", "-i", f,
           "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=30).stdout
        if len(raw) < 526 * 330 * 3:
            return None
        return np.frombuffer(raw[:526 * 330 * 3], np.uint8).reshape(330, 526, 3)
    except Exception:
        return None


def montage(frames, path, labels=None):
    frames = [f for f in frames if f is not None]
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    im = Image.new("RGB", (w * len(frames) + 4 * (len(frames) - 1), h), "white")
    for i, f in enumerate(frames):
        im.paste(Image.fromarray(f), (i * (w + 4), 0))
    im.save(path, quality=88)
    return True


def can_plot(row, path, title="", signals=("vEgo", "aEgo", "steeringAngleDeg", "actual_curvature"),
             extra=None, span=None):
    """Stacked per-signal subplots (one axis each — no dual axes) over the window."""
    mdl, drv, rt = row.model, row.driver, row.route
    df = pd.read_parquet(f"{BASE}/data/cache/{mdl}/{drv}__{rt}.parquet")
    a, b = int(row.wi0), int(row.wi1) + 1
    if span:
        a, b = span
    t = df["time_s"].to_numpy()[a:b] - df["time_s"].to_numpy()[a]
    sigs = list(signals)
    if extra:
        sigs += extra
    fig, axes = plt.subplots(len(sigs), 1, figsize=(7, 1.15 * len(sigs)), sharex=True)
    for ax, s in zip(np.atleast_1d(axes), sigs):
        if s == "THW":
            v = df["leadOne_dRel"].to_numpy()[a:b] / np.maximum(df["vEgo"].to_numpy()[a:b], .5)
            v = np.where(df["leadOne_status"].to_numpy()[a:b] > .5, v, np.nan)
        else:
            v = df[s].to_numpy()[a:b]
        ax.plot(t, v, color=BLUE, lw=1.4)
        ax.set_ylabel(s, fontsize=8, rotation=0, ha="right", va="center")
        if s == "brakePressed":
            ax.fill_between(t, 0, v, color=AMBER, alpha=.4, lw=0)
    np.atleast_1d(axes)[-1].set_xlabel("time in window (s)")
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_example(ex_id, task, row, why, tag, files, extra_meta=None):
    d = f"{RES}/{ex_id}"
    os.makedirs(d, exist_ok=True)
    meta = {"example_id": ex_id, "task_or_finding": task,
            "driver_anon": anon(row.driver), "route": str(row.route)[:10] + "…",
            "vehicle": row.model_canon, "scenario": row.scenario,
            "time_range_s": f"{row.t0:.0f}–{row.t0+60:.0f}",
            "signals": "vEgo,aEgo,steering,curvature,THW,pedals",
            "why_selected": why, "readiness": tag}
    if extra_meta:
        meta.update(extra_meta)
    json.dump(meta, open(f"{d}/meta.json", "w"), indent=1)
    manifest.append(meta | {"files": ";".join(files)})


def frames_for(row, times_rel):
    return [grab_frame(row.model, row.driver, row.route, row.t0 + tr) for tr in times_rel]


# =============================== T1 primitives ===============================
def t1_examples():
    picks = [("p_hard_braking", "stop_go", "hard braking in stop-and-go"),
             ("p_sharp_steer_path", "curve", "sharp curvature-normalized steering in a curve"),
             ("p_close_following", "car_following", "close following (low THW)")]
    for i, (prim, sc, why) in enumerate(picks):
        cand = W[(W[prim] == 1) & (W.scenario == sc) & W.model_canon.ne("UNKNOWN")]
        if not len(cand):
            continue
        row = cand.sample(1, random_state=SEED + i).iloc[0]
        ex = f"T1_{i+1}_{prim[2:]}"
        d = f"{RES}/{ex}"; os.makedirs(d, exist_ok=True)
        montage(frames_for(row, [5, 30, 55]), f"{d}/frames.jpg")
        can_plot(row, f"{d}/can.png", f"T1 {prim} · scenario={sc} · driver={anon(row.driver)[:10]}",
                 extra=["THW", "brakePressed"])
        save_example(ex, f"T1:{prim}", row, why, "paper-ready",
                     ["frames.jpg", "can.png"], {"primitive": prim, "label": "+1"})


# ====================== T2 enrollment / identifiability ======================
def t2_examples():
    import torch
    from s1_supcon import Encoder
    ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location="cuda", weights_only=False)
    enc = Encoder(c_in=X.shape[2]).cuda().eval(); enc.load_state_dict(ck["model"])
    mu, sd = ck["mu"], ck["sd"]

    def emb(idx):
        import torch as T
        a = (X[np.sort(idx)].astype(np.float32) - mu) / sd
        a = a[np.argsort(np.argsort(idx))]
        with T.no_grad():
            z = enc(T.from_numpy(a).cuda()).cpu().numpy()
        return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)

    ev = set(FOLDS["val"]) | set(FOLDS["few_shot_heldout"])
    me = META[META.driver.isin(ev)]
    # drivers with >=2 routes and >=12 windows
    good = [d for d, g in me.groupby("driver") if g.route.nunique() >= 2 and len(g) >= 12]
    ds = rng.choice(good, min(6, len(good)), replace=False)
    protos, pd_ids = [], []
    for d in ds:
        g = me[me.driver == d]
        r0 = g.route.unique()[0]
        sup = g[g.route == r0].index.to_numpy()[:5]
        protos.append(emb(sup).mean(0)); pd_ids.append(d)
    P = np.stack(protos); P /= np.linalg.norm(P, axis=1, keepdims=True) + 1e-9
    for i, d in enumerate(pd_ids[:2]):        # two positive examples
        g = me[me.driver == d]
        r1 = g.route.unique()[1]
        q = g[g.route == r1].index.to_numpy()[:1]
        zq = emb(q)[0]
        sims = P @ zq
        correct = int(np.argmax(sims) == i)
        row = W.loc[q[0]]
        ex = f"T2_{i+1}_same_driver"
        dd = f"{RES}/{ex}"; os.makedirs(dd, exist_ok=True)
        montage(frames_for(row, [10, 50]), f"{dd}/query_frames.jpg")
        srow = W.loc[me[(me.driver == d)].index[0]]
        montage(frames_for(srow, [10, 50]), f"{dd}/support_frames.jpg")
        fig, ax = plt.subplots(figsize=(5, 2.4))
        order = np.argsort(-sims)
        ax.bar(range(len(sims)), sims[order],
               color=[BLUE if pd_ids[j] == d else GRAY for j in order], width=.6)
        ax.set_xticks(range(len(sims)))
        ax.set_xticklabels([anon(pd_ids[j])[:8] for j in order], rotation=45, fontsize=7)
        ax.set_ylabel("cosine to query"); ax.set_title(f"query→prototypes (own driver blue) · correct={bool(correct)}")
        fig.tight_layout(); fig.savefig(f"{dd}/similarity.png", dpi=130); plt.close(fig)
        can_plot(row, f"{dd}/query_can.png", "query window (different route than support)")
        save_example(ex, "T2:enrollment", row,
                     "same driver, support/query from different routes; retrieval vs 6 prototypes",
                     "paper-ready", ["similarity.png", "query_frames.jpg", "support_frames.jpg"],
                     {"retrieval_correct": bool(correct), "own_sim": float(sims[i]),
                      "best_other": float(np.max(np.delete(sims, i)))})
    # contrast: two different drivers, same scenario+model
    key = me
    grp = key.groupby(["model_canon", "scenario"]).driver.nunique()
    mk = grp[grp >= 2].index[0]
    sub = key[(key.model_canon == mk[0]) & (key.scenario == mk[1])]
    d1, d2 = sub.driver.unique()[:2]
    i1, i2 = sub[sub.driver == d1].index[0], sub[sub.driver == d2].index[0]
    z = emb(np.array([i1, i2]))
    row = W.loc[i1]
    ex = "T2_3_contrast_diff_driver"
    dd = f"{RES}/{ex}"; os.makedirs(dd, exist_ok=True)
    montage([grab_frame(*W.loc[i1][["model", "driver", "route"]], W.loc[i1].t0 + 30),
             grab_frame(*W.loc[i2][["model", "driver", "route"]], W.loc[i2].t0 + 30)], f"{dd}/frames.jpg")
    save_example(ex, "T2:contrast", row,
                 f"different drivers, same vehicle({mk[0]})+scenario({mk[1]}): cos={float(z[0]@z[1]):.2f}",
                 "appendix-only", ["frames.jpg"], {"cross_driver_cos": float(z[0] @ z[1])})


# ========================= T3 personalized prediction =========================
def t3_examples():
    import torch
    from s3_personalized import SupportEncoder, Predictor, L_HIST, L_FUT, TGT
    ck = torch.load(f"{BASE}/experiments/checkpoints/s3_pred.pt", map_location="cuda", weights_only=False)
    sup = SupportEncoder(X.shape[2]).cuda().eval(); sup.load_state_dict(ck["sup_enc"])
    net = Predictor(X.shape[2], len(TGT)).cuda().eval(); net.load_state_dict(ck["pred"])
    mu, sd = ck["mu"], ck["sd"]
    tgt_ix = [CH.index(c) for c in TGT]
    norm = lambda a: (a.astype(np.float32) - mu) / sd
    ev = set(FOLDS["val"]) | set(FOLDS["few_shot_heldout"])
    me = META[META.driver.isin(ev)]
    cases = []
    with torch.no_grad():
        for d, g in list(me.groupby("driver"))[:40]:
            if g.route.nunique() < 2 or len(g) < 10:
                continue
            rts = g.route.unique()
            supi = g[g.route == rts[0]].index.to_numpy()[:5]
            qryi = g[g.route == rts[1]].index.to_numpy()[:6]
            z = torch.from_numpy(norm(X[np.sort(supi)])).cuda()
            zd = sup(z).mean(0, keepdim=True)
            for qi in qryi:
                w = norm(X[qi])
                a = 300
                xs = torch.from_numpy(w[None, a - L_HIST:a]).cuda()
                y = w[a:a + L_FUT][:, tgt_ix]
                eg = net(xs, torch.zeros(1, 64).cuda())[0].cpu().numpy()
                ep = net(xs, zd)[0].cpu().numpy()
                rg = float(np.sqrt(((eg - y) ** 2).mean()))
                rp = float(np.sqrt(((ep - y) ** 2).mean()))
                cases.append((rg - rp, qi, supi[0], y, eg, ep, a))
    cases.sort(key=lambda c: -c[0])
    picks = [(cases[0], "largest personalization improvement", "paper-ready", "T3_1_best"),
             (cases[1], "second-best improvement", "appendix-only", "T3_2_good"),
             (cases[len(cases) // 2], "median ≈ no improvement (honest failure case)", "paper-ready", "T3_3_neutral")]
    for (pg, qi, si, y, eg, ep, a), why, tag, ex in picks:
        row = W.loc[qi]
        dd = f"{RES}/{ex}"; os.makedirs(dd, exist_ok=True)
        tt = np.arange(L_FUT) / 10.0
        fig, axes = plt.subplots(2, 1, figsize=(6, 3.6), sharex=True)
        for j, (ax, name) in enumerate(zip(axes, TGT)):
            ax.plot(tt, y[:, j], color=GRAY, lw=1.6, label="ground truth")
            ax.plot(tt, eg[:, j], color=AMBER, lw=1.3, ls="--", label="generic")
            ax.plot(tt, ep[:, j], color=BLUE, lw=1.3, label="personalized")
            ax.set_ylabel(name, fontsize=8)
        axes[0].legend(fontsize=7, ncol=3, frameon=False)
        axes[1].set_xlabel("future time (s)")
        fig.suptitle(f"T3 {why} · ΔRMSE={pg:+.3f} (std units)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{dd}/prediction.png", dpi=130); plt.close(fig)
        montage(frames_for(row, [a / 10 - 5, a / 10]), f"{dd}/frames.jpg")
        can_plot(row, f"{dd}/can.png", "query window")
        save_example(ex, "T3:personalized-pred", row, why, tag,
                     ["prediction.png", "frames.jpg", "can.png"],
                     {"delta_rmse_std": round(pg, 4)})


# ======================= T4 matched-context comparison =======================
def t4_examples():
    import torch
    from s1_supcon import Encoder
    ck = torch.load(f"{BASE}/experiments/checkpoints/s1_supcon.pt", map_location="cuda", weights_only=False)
    enc = Encoder(c_in=X.shape[2]).cuda().eval(); enc.load_state_dict(ck["model"])
    mu, sd = ck["mu"], ck["sd"]

    def emb2(i, j):
        a = (X[np.sort([i, j])].astype(np.float32) - mu) / sd
        a = a[np.argsort(np.argsort([i, j]))]
        with torch.no_grad():
            z = enc(torch.from_numpy(a).cuda()).cpu().numpy()
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-9
        return float(z[0] @ z[1])

    def valid(row):
        df = pd.read_parquet(f"{BASE}/data/cache/{row.model}/{row.driver}__{row.route}.parquet",
                             columns=["aEgo"])
        v = df["aEgo"].to_numpy()[int(row.wi0):int(row.wi1)+1]
        return np.isfinite(v).mean() > .9 and np.nanstd(v) > 0.05

    cf = PAIRS[PAIRS.key.str.startswith("car_following")]
    for lab, tag, ex in [(1, "paper-ready", "T4_1_same_driver_pair"),
                         (0, "paper-ready", "T4_2_diff_driver_pair"),
                         (0, "appendix-only", "T4_3_diff_driver_pair2")]:
        pool = cf[cf.same_driver == lab].sample(frac=1, random_state=SEED + lab + len(ex))
        p = ra = rb = None
        for _, cand in pool.head(40).iterrows():
            ca, cb = W.loc[cand.win_a], W.loc[cand.win_b]
            if valid(ca) and valid(cb):
                p, ra, rb = cand, ca, cb
                break
        if p is None:
            continue
        cs = emb2(p.win_a, p.win_b)
        dd = f"{RES}/{ex}"; os.makedirs(dd, exist_ok=True)
        montage([grab_frame(ra.model, ra.driver, ra.route, ra.t0 + 30),
                 grab_frame(rb.model, rb.driver, rb.route, rb.t0 + 30)], f"{dd}/frames_pair.jpg")
        # time-series overlay (same matched context)
        fig, axes = plt.subplots(2, 1, figsize=(6, 3.4), sharex=True)
        for row, col, nm in [(ra, BLUE, "window A"), (rb, AMBER, "window B")]:
            df = pd.read_parquet(f"{BASE}/data/cache/{row.model}/{row.driver}__{row.route}.parquet")
            a, b = int(row.wi0), int(row.wi1) + 1
            t = np.arange(b - a) / 10.0
            axes[0].plot(t, df["aEgo"].to_numpy()[a:b], color=col, lw=1.2, label=nm)
            thw = df["leadOne_dRel"].to_numpy()[a:b] / np.maximum(df["vEgo"].to_numpy()[a:b], .5)
            axes[1].plot(t, np.where(df["leadOne_status"].to_numpy()[a:b] > .5, thw, np.nan),
                         color=col, lw=1.2)
        axes[0].set_ylabel("aEgo"); axes[1].set_ylabel("THW"); axes[1].set_xlabel("time (s)")
        axes[0].legend(fontsize=7, frameon=False)
        same = "SAME driver" if lab else "DIFFERENT drivers"
        fig.suptitle(f"T4 matched context ({p.key.split('|')[0]}, {p.key.split('|')[3]}) · {same} · S1 cos={cs:.2f}", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{dd}/overlay.png", dpi=130); plt.close(fig)
        save_example(ex, "T4:matched-context", ra,
                     f"{same} in matched context (key={p.key}); embedding similarity {cs:.2f}",
                     tag, ["overlay.png", "frames_pair.jpg"],
                     {"same_driver": bool(lab), "s1_cosine": round(cs, 3),
                      "driver_b_anon": anon(rb.driver), "vehicle_b": rb.model_canon})


# ============================ T5/T6 event candidates ============================
def t5_t6_examples():
    cand = W[(W.p_hard_braking == 1) & W.scenario.isin(["car_following", "stop_go"])]
    rows = cand.sample(3, random_state=SEED)
    for i, (_, row) in enumerate(rows.iterrows()):
        df = pd.read_parquet(f"{BASE}/data/cache/{row.model}/{row.driver}__{row.route}.parquet")
        a, b = int(row.wi0), int(row.wi1) + 1
        acc = df["aEgo"].to_numpy()[a:b]
        onset_i = next((j for j, v in enumerate(acc) if v < -2.0), None)
        if onset_i is None or onset_i < 60:
            continue
        t_on = row.t0 + onset_i / 10.0
        span = (a + onset_i - 50, min(a + onset_i + 30, b))
        # T5 candidate
        ex = f"T5_{i+1}_hard_brake_candidate"
        dd = f"{RES}/{ex}"; os.makedirs(dd, exist_ok=True)
        df2 = df.iloc[span[0]:span[1]]
        t = (df2["time_s"] - df2["time_s"].iloc[0] - 5.0).to_numpy()
        fig, axes = plt.subplots(3, 1, figsize=(6.5, 4.2), sharex=True)
        axes[0].plot(t, df2["vEgo"], color=BLUE, lw=1.4); axes[0].set_ylabel("vEgo")
        axes[1].plot(t, df2["aEgo"], color=BLUE, lw=1.4); axes[1].set_ylabel("aEgo")
        axes[1].axhline(-2, color=AMBER, ls=":", lw=1)
        thw = df2["leadOne_dRel"] / np.maximum(df2["vEgo"], .5)
        axes[2].plot(t, np.where(df2["leadOne_status"] > .5, thw, np.nan), color=BLUE, lw=1.4)
        axes[2].set_ylabel("THW"); axes[2].set_xlabel("time rel. onset (s)")
        for ax in axes:
            ax.axvline(0, color=GRAY, ls="--", lw=1)
            ax.axvspan(-5, 0, color=BLUE, alpha=.06)
        axes[0].set_title("T5 candidate: hard-brake onset (input=[−5,0] shaded, forecast target onset@0)", fontsize=9)
        fig.tight_layout(); fig.savefig(f"{dd}/event.png", dpi=130); plt.close(fig)
        montage([grab_frame(row.model, row.driver, row.route, t_on + dt) for dt in (-4, -1.5, 0, 2)],
                f"{dd}/frames.jpg")
        save_example(ex, "T5:event-forecast", row,
                     "hard-brake onset with ≥5 s clean pre-history; taxonomy not yet frozen",
                     "sanity-check only (candidate)", ["event.png", "frames.jpg"],
                     {"event": "hard_braking", "onset_rel_s": round(onset_i / 10.0, 1)})
        # T6 candidate (explanation) for the first two events
        if i < 2:
            vrel = df2["leadOne_vRel"].to_numpy()
            lead = (df2["leadOne_status"].to_numpy() > .5)
            pre = slice(0, 50)
            if lead[pre].mean() > .5 and np.nanmean(vrel[pre]) < -0.3:
                expl = "lead vehicle decelerating (leadOne_vRel<0 with falling THW) → reactive braking"
            elif row.scenario == "curve":
                expl = "curve entry deceleration"
            else:
                expl = "no strong external evidence → possibly driver-conservative braking"
            ex6 = f"T6_{i+1}_explanation_candidate"
            dd6 = f"{RES}/{ex6}"; os.makedirs(dd6, exist_ok=True)
            montage([grab_frame(row.model, row.driver, row.route, t_on + dt) for dt in (-5, -2, 0, 2)],
                    f"{dd6}/evidence_frames.jpg")
            json.dump({"window": "[-5s,+3s]", "event": "hard_braking",
                       "CAN_evidence": {"mean_leadVRel_pre": float(np.nanmean(vrel[pre])),
                                        "lead_present_pre": float(lead[pre].mean())},
                       "explanation_draft": expl},
                      open(f"{dd6}/explanation.json", "w"), indent=1)
            save_example(ex6, "T6:explanation", row, f"rule-based evidence draft: {expl}",
                         "sanity-check only (candidate)", ["evidence_frames.jpg", "explanation.json"],
                         {"explanation": expl})


# ============================ Findings A–H figures ============================
def finding_figures():
    os.makedirs(FIG, exist_ok=True)

    def bars(name, labels, vals, ylabel, title, colors=None, note=""):
        fig, ax = plt.subplots(figsize=(4.6, 2.6))
        ax.bar(labels, vals, color=colors or BLUE, width=.55)
        for i, v in enumerate(vals):
            ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel(ylabel); ax.set_title(title + ("\n" + note if note else ""), fontsize=9)
        fig.tight_layout(); fig.savefig(f"{FIG}/{name}.png", dpi=140); plt.close(fig)

    # A: style exists → reuse T2 example; plus enrollment curve
    bars("A_style_exists_enrollment", ["1min", "3min", "5min", "10min"],
         [0.889, 0.923, 0.931, 0.931], "verification AUROC (unseen drivers)",
         "A · Style exists: few-shot verification rises with enrollment")
    # B: confounded by context+vehicle (GBM R²)
    bars("B_context_explains", ["brake", "SDLP", "THW", "steer", "accel", "curv", "jerk"],
         [0.60, 0.59, 0.48, 0.44, 0.46, 0.36, 0.29], "population-model R²",
         "B · Context+vehicle explain 29–60% of behavior variance", colors=AMBER)
    # C: residual survives
    bars("C_residual_survives", ["RAW", "RESIDUAL"], [7, 4], "top-1 lift over chance (×)",
         "C · Identifiability survives conditioning", colors=[AMBER, BLUE],
         note="drop 7×→4× = context-shortcut share; residual 4× = stable driver style")
    # D: steering leaks vehicle
    bars("D_input_vs_path_leakage", ["raw steering", "realized curvature"], [2.3, 1.0],
         "vehicle-prediction lift (×chance)", "D · Steering leaks vehicle; curvature is agnostic",
         colors=[AMBER, BLUE])
    # E: identity ≠ prediction
    bars("E_identity_vs_prediction", ["joint task-aligned", "frozen S1 (re-ID optimal)"],
         [1.1, -0.2], "PG (%)", "E · Identity-discriminative ≠ prediction-useful", colors=[BLUE, AMBER])
    # F: PG vs horizon
    bars("F_pg_vs_horizon", ["1 s", "3 s", "5 s"], [2.3, 0.9, 0.8], "PG (%)",
         "F · Personalization gain decays with horizon")
    # H: complementary
    bars("H_mcpp_ablation", ["+video", "+driver", "+both"], [0.2, 0.3, 0.4],
         "ΔRMSE vs CAN-only (%)", "H · Video (context) and driver (style) are complementary")
    # G: distributional (MDN NLL vs RMSE PG)
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    ax.bar(["RMSE PG (%)", "NLL PG (nats×10)"], [0.0, 11.6], color=[GRAY, BLUE], width=.5)
    ax.set_title("G · Style is distributional: personalization sharpens\nthe distribution (+1.16 nats), not the mean", fontsize=9)
    for i, v in enumerate([0.0, 11.6]):
        ax.text(i, v, f"{v:g}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{FIG}/G_distributional.png", dpi=140); plt.close(fig)


def main():
    os.makedirs(RES, exist_ok=True); os.makedirs(FIG, exist_ok=True)
    print("T1 ..."); t1_examples()
    print("T2 ..."); t2_examples()
    print("T3 ..."); t3_examples()
    print("T4 ..."); t4_examples()
    print("T5/T6 ..."); t5_t6_examples()
    print("findings ..."); finding_figures()
    mdf = pd.DataFrame(manifest)
    mdf.to_csv(f"{RES}/manifest.csv", index=False)
    print(f"\nexamples: {len(mdf)}  → {RES}/manifest.csv")
    print(mdf[["example_id", "task_or_finding", "vehicle", "scenario", "readiness"]].to_string(index=False))


if __name__ == "__main__":
    main()
