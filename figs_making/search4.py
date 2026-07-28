import numpy as np, pandas as pd, json, os, subprocess
BASE = "."
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c:i for i,c in enumerate(ch)}
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
kmin = {(r.model,r.driver,r.route):int(r.k_min) for r in seg.itertuples()}
ev = pd.read_parquet(f"{BASE}/data/segments/maneuver_events.parquet")
cv = ev[ev.cls=="curve"]

feats = []
for r in cv.itertuples():
    gi, o = int(r.gi), int(r.onset)
    if o < 80 or o > 500: continue
    w = X[gi].astype(np.float32)
    a = w[:,ix["aEgo"]]; v = w[:,ix["vEgo"]]; kc = np.abs(w[:,ix["actual_curvature"]])
    dr = w[:,ix["leadOne_dRel"]]
    if np.abs(a).max() > 9 or np.abs(a).std() < 0.05: continue
    seg_dr = dr[o-40:o+40]
    if ((seg_dr > 0) & (seg_dr < 60)).any(): continue      # lead-free only
    kpk = np.convolve(kc, np.ones(5)/5,"same")[o:o+40].max()
    feats.append((gi, o, kpk, a[o-40:o].mean(), v[o-40]))
F = pd.DataFrame(feats, columns=["gi","o","kpk","pre_a","v0"])
F["model"] = W.iloc[F.gi].model_canon.values
F["driver"] = W.iloc[F.gi].driver.values
F = F[(F.v0 > 12) & (F.kpk > 0.008) & (F.kpk < 0.03)].reset_index(drop=True)
print("lead-free sane curve events:", len(F), flush=True)

pairs = []
for model, g in F.groupby("model"):
    g = g.reset_index()
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            a1, a2 = g.iloc[i], g.iloc[j]
            if a1.driver == a2.driver: continue
            if abs(a1.kpk - a2.kpk) > 0.2*max(a1.kpk, a2.kpk): continue
            if abs(a1.v0 - a2.v0) > 3: continue
            con = abs(a1.pre_a - a2.pre_a)
            if con > 1.2:
                pairs.append((con, model, int(a1["index"]), int(a2["index"])))
pairs.sort(reverse=True)
print("high-contrast lead-free pairs:", len(pairs), flush=True)

CACHE = {}
def thumb(idx):
    if idx in CACHE: return CACHE[idx]
    r = F.iloc[idx]; wr = W.iloc[int(r.gi)]
    km = kmin.get((wr.model, wr.driver, wr.route))
    out = None
    if km is not None:
        tt = float(wr.t0) + r.o/10.0
        p1 = f"../Dataset/{wr.model}/{wr.driver}/{wr.route}/{km+int(tt//60)}--qcamera.ts"
        if os.path.exists(p1):
            rr = subprocess.run(["ffmpeg","-v","error","-ss",f"{tt%60:.2f}","-i",p1,"-frames:v","1",
                                 "-vf","scale=48:27","-f","rawvideo","-pix_fmt","rgb24","-"],
                                capture_output=True, timeout=30)
            if rr.stdout and len(rr.stdout) == 48*27*3:
                out = np.frombuffer(rr.stdout, np.uint8).reshape(27,48,3).astype(np.float32)
    CACHE[idx] = out
    return out

def scene_sim(fa, fb):
    """correlation of joint color histograms"""
    ha = np.histogramdd(fa.reshape(-1,3), bins=(6,6,6), range=((0,255),)*3)[0].ravel()
    hb = np.histogramdd(fb.reshape(-1,3), bins=(6,6,6), range=((0,255),)*3)[0].ravel()
    ha, hb = ha/ha.sum(), hb/hb.sum()
    return float(np.minimum(ha, hb).sum())     # histogram intersection 0..1

keep = []
for con, model, i, j in pairs:
    fa, fb = thumb(i), thumb(j)
    if fa is None or fb is None: continue
    if fa.mean() < 60 or fb.mean() < 60: continue          # daytime both
    sim = scene_sim(fa, fb)
    keep.append((sim, con, model, i, j))
    if len(keep) >= 40: break
keep.sort(reverse=True)                                     # most scene-similar first
print("bright pairs evaluated:", len(keep), flush=True)
out = []
for sim, con, model, i, j in keep[:8]:
    a1, a2 = F.iloc[i], F.iloc[j]
    print(f"  sim {sim:.2f} contrast {con:.2f} {model} | pre_a {a1.pre_a:+.2f}/{a2.pre_a:+.2f} kpk {a1.kpk:.3f}/{a2.kpk:.3f} v0 {a1.v0:.0f}/{a2.v0:.0f}")
    out.append({"model": model, "ga": int(a1.gi), "oa": int(a1.o), "gb": int(a2.gi), "ob": int(a2.o), "sim": float(sim), "con": float(con)})
json.dump(out, open(f"{BASE}/figs_making/candidates4.json","w"))
