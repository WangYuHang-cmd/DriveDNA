import numpy as np, pandas as pd, json, os, subprocess
BASE = "."
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c: i for i, c in enumerate(ch)}
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
p = pd.read_parquet(f"{BASE}/data/splits/matched_context_pairs.parquet")
dp = p[(p.same_driver==0) & (p.key.str.startswith("curve"))]

def entry_of(gi):
    w = X[gi].astype(np.float32)
    v, a, kc = w[:, ix["vEgo"]], w[:, ix["aEgo"]], np.abs(w[:, ix["actual_curvature"]])
    if np.abs(a).max() > 9 or np.abs(a).std() < 0.05:   # dead or corrupt accel
        return None
    kk = np.convolve(kc, np.ones(5)/5, "same")
    cond = (kk > 0.006) & (v > 6)
    for i in range(80, 520):
        if cond[i:i+15].all():
            return i
    return None

def video_ok(gi, e):
    r = W.iloc[gi]
    km = kmin.get((r.model, r.driver, r.route))
    if km is None: return None
    tt = float(r.t0) + e/10.0
    pth = f"../Dataset/{r.model}/{r.driver}/{r.route}/{km+int(tt//60)}--qcamera.ts"
    return (pth, tt%60) if os.path.exists(pth) else None

cands = []
seen = set()
sc = dp.copy()
sc["contrast"] = [abs(W.iloc[r.win_a].decel_p05 - W.iloc[r.win_b].decel_p05) for r in dp.itertuples()]
sc = sc[(sc.contrast > 1.2)]
# sanity on precomputed stats first
ok = lambda g: -9 < W.iloc[g].decel_p05 <= 0.5 and W.iloc[g].jerk_rms > 0.15 and W.iloc[g].v_mean > 8
sc = sc[[ok(r.win_a) and ok(r.win_b) for r in sc.itertuples()]]
sc = sc.sort_values("contrast", ascending=False)
print("sane contrast pairs:", len(sc))
for r in sc.itertuples():
    key = tuple(sorted((r.win_a, r.win_b)))
    if key in seen: continue
    ea, eb = entry_of(r.win_a), entry_of(r.win_b)
    if ea is None or eb is None: continue
    va, vb = video_ok(r.win_a, ea), video_ok(r.win_b, eb)
    if va is None or vb is None: continue
    # daytime check: extract tiny frame, mean brightness
    bright = []
    for (pth, off) in (va, vb):
        rr = subprocess.run(["ffmpeg","-v","error","-ss",f"{off:.2f}","-i",pth,"-frames:v","1",
                             "-vf","scale=32:18","-f","rawvideo","-pix_fmt","gray","-"],
                            capture_output=True, timeout=30)
        bright.append(np.frombuffer(rr.stdout, np.uint8).mean() if rr.stdout else 0)
    if min(bright) < 60: continue
    seen.add(key)
    cands.append((int(r.win_a), int(r.win_b), r.key, float(r.contrast), ea, eb, bright))
    A, B = W.iloc[r.win_a], W.iloc[r.win_b]
    print(f"KEEP {r.win_a}/{r.win_b} {r.key} contrast {r.contrast:.1f} decel {A.decel_p05:.1f}/{B.decel_p05:.1f} v {A.v_mean:.0f}/{B.v_mean:.0f} bright {bright[0]:.0f}/{bright[1]:.0f}")
    if len(cands) >= 5: break
json.dump([{"wa":c[0],"wb":c[1],"key":c[2],"ea":c[4],"eb":c[5]} for c in cands],
          open(f"{BASE}/figs_making/candidates2.json","w"))
