import numpy as np, pandas as pd, json, os, subprocess
BASE = "."
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c:i for i,c in enumerate(ch)}
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
ev = pd.read_parquet(f"{BASE}/data/segments/maneuver_events.parquet")
cv = ev[ev.cls=="curve"].reset_index(drop=True)
# event features
feats = []
for r in cv.itertuples():
    gi, o = int(r.gi), int(r.onset)
    if o < 60 or o > 500: continue
    w = X[gi].astype(np.float32)
    a = w[:, ix["aEgo"]]; v = w[:, ix["vEgo"]]; kc = np.abs(w[:, ix["actual_curvature"]])
    if np.abs(a).max() > 9 or np.abs(a).std() < 0.05: continue
    kpk = np.convolve(kc, np.ones(5)/5, "same")[o:o+40].max()
    pre_a = a[o-40:o].mean()          # mean accel 4s before entry
    v0 = v[o-40]                      # speed 4s before
    ventry = v[o]
    feats.append((gi, o, kpk, pre_a, v0, ventry))
F = pd.DataFrame(feats, columns=["gi","o","kpk","pre_a","v0","ventry"])
F["model"] = W.iloc[F.gi].model_canon.values
F["driver"] = W.iloc[F.gi].driver.values
F = F[(F.v0 > 12) & (F.kpk > 0.008) & (F.kpk < 0.03)]
print("usable curve events:", len(F))
# pair search: same model, kpk within 20%, v0 within 3, different driver, contrast in pre_a
best = []
for model, g in F.groupby("model"):
    g = g.reset_index(drop=True)
    for i in range(len(g)):
        for j in range(i+1, len(g)):
            a1, a2 = g.iloc[i], g.iloc[j]
            if a1.driver == a2.driver: continue
            if abs(a1.kpk - a2.kpk) > 0.2*max(a1.kpk, a2.kpk): continue
            if abs(a1.v0 - a2.v0) > 3: continue
            contrast = abs(a1.pre_a - a2.pre_a)
            if contrast > 0.8:
                best.append((contrast, model, int(a1.gi), int(a1.o), int(a2.gi), int(a2.o),
                             float(a1.pre_a), float(a2.pre_a), float(a1.kpk), float(a2.kpk), float(a1.v0), float(a2.v0)))
best.sort(reverse=True)
print("event pairs with contrast>0.8:", len(best))
for b in best[:12]:
    print(f"  {b[1]:28s} contrast {b[0]:.2f} | pre_a {b[6]:+.2f} vs {b[7]:+.2f} | kpk {b[8]:.3f}/{b[9]:.3f} | v0 {b[10]:.0f}/{b[11]:.0f}")
json.dump([{"model":b[1],"ga":b[2],"oa":b[3],"gb":b[4],"ob":b[5]} for b in best[:12]],
          open(f"{BASE}/figs_making/candidates3.json","w"))
