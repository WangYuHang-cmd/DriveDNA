#!/usr/bin/env python3
"""
T1 human-audit package: one self-contained HTML for auditing annotation labels.

Samples 8 primitives x {high, low} x 15 = 240 labeled windows (seed-fixed).
For each: mid-window video keyframe + relevant signal traces + yes/no/unsure radio.
"Export JSON" downloads the answers; code/analysis/summarize_audit.py turns them
into the Appendix-B agreement table.

    PYTHONPATH= .../python make_audit_package.py
Output: results/audit/t1_audit.html  (+ manifest results/audit/audit_manifest.csv)
"""
import os
import io
import json
import base64
import subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "."
DATASET = "../Dataset"
SEED = 20260709
PER_CELL = 15
BLUE, GRAY = "#3B82F6", "#6B7280"

PRIMS = {
    "p_close_following":   (["leadOne_dRel", "vEgo"],
        {"high": "Is the driver following the lead vehicle closely (small gap)?",
         "low":  "Is the driver keeping a clearly comfortable gap (no tailgating)?"}),
    "p_large_headway":     (["leadOne_dRel", "vEgo"],
        {"high": "Is the driver keeping an unusually large gap to the lead vehicle?",
         "low":  "Is the gap to the lead clearly NOT large (normal or small)?"}),
    "p_hard_braking":      (["aEgo", "vEgo"],
        {"high": "Does the window contain hard braking?",
         "low":  "Is all braking in this window gentle (no hard braking)?"}),
    "p_high_jerk":         (["aEgo", "vEgo"],
        {"high": "Is the longitudinal motion jerky (abrupt accel changes)?",
         "low":  "Is the longitudinal motion smooth (no abrupt accel changes)?"}),
    "p_sharp_steer_raw":   (["steeringAngleDeg", "vEgo"],
        {"high": "Does the window contain sharp steering input?",
         "low":  "Is the steering smooth throughout (no sharp inputs)?"}),
    "p_sharp_steer_path":  (["actual_curvature", "vEgo"],
        {"high": "Does the vehicle's path contain sharp curvature changes?",
         "low":  "Is the vehicle's path smooth (no sharp curvature changes)?"}),
    "p_lane_correction":   (["laneLeft_y", "laneRight_y"],
        {"high": "Does the driver make noticeable in-lane corrections?",
         "low":  "Is the lane keeping steady (no noticeable corrections)?"}),
    "p_curve_entry_decel": (["actual_curvature", "aEgo"],
        {"high": "Does the driver slow down noticeably when entering the curve?",
         "low":  "Does the driver enter the curve without notable slowing?"}),
}


def make_window_clip(rowdir, k, t0_in_seg, outpath):
    """Full 60-s window clip; windows usually span segments k and k+1 (TS concat)."""
    if os.path.exists(outpath) and os.path.getsize(outpath) > 10000:
        return True
    p1 = os.path.join(rowdir, f"{k}--qcamera.ts")
    p2 = os.path.join(rowdir, f"{k+1}--qcamera.ts")
    if not os.path.exists(p1):
        return False
    src = f"concat:{p1}|{p2}" if (t0_in_seg > 0 and os.path.exists(p2)) else p1
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{t0_in_seg:.1f}", "-i", src,
           "-t", "60", "-c:v", "libx264", "-preset", "veryfast", "-crf", "30",
           "-an", "-movflags", "+faststart", outpath]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
        return os.path.exists(outpath) and os.path.getsize(outpath) > 10000
    except Exception:
        return False


def key_moment(w, prim, ch_index):
    """Second (0-60) of the most label-relevant moment in the window."""
    import numpy as _np
    def col(c):
        return w[:, ch_index[c]].astype(float)
    if prim in ("p_close_following", "p_large_headway"):
        thw = col("leadOne_dRel") / _np.maximum(col("vEgo"), 1.0)
        thw[~_np.isfinite(thw)] = _np.inf
        thw[col("leadOne_dRel") <= 0] = _np.inf
        i = int(_np.argmin(thw)) if _np.isfinite(thw).any() else 300
    elif prim == "p_hard_braking":
        i = int(_np.nanargmin(col("aEgo")))
    elif prim == "p_high_jerk":
        i = int(_np.nanargmax(_np.abs(_np.diff(col("aEgo"), prepend=col("aEgo")[0]))))
    elif prim == "p_sharp_steer_raw":
        d = _np.abs(_np.diff(col("steeringAngleDeg"), prepend=col("steeringAngleDeg")[0]))
        i = int(_np.nanargmax(d))
    elif prim == "p_sharp_steer_path":
        d = _np.abs(_np.diff(col("actual_curvature"), prepend=col("actual_curvature")[0]))
        i = int(_np.nanargmax(d))
    elif prim == "p_lane_correction":
        d = _np.abs(_np.diff(col("laneLeft_y"), prepend=col("laneLeft_y")[0]))
        d[~_np.isfinite(d)] = 0
        i = int(_np.argmax(d))
    else:  # curve_entry_decel: curve apex
        i = int(_np.nanargmax(_np.abs(col("actual_curvature"))))
    return i / 10.0


def plot_b64(w, chans, ch_index, tmark=None):
    fig, axes = plt.subplots(len(chans), 1, figsize=(5.4, 1.15 * len(chans)), sharex=True)
    if len(chans) == 1:
        axes = [axes]
    t = np.arange(w.shape[0]) / 10.0
    for ax, c in zip(axes, chans):
        ax.plot(t, w[:, ch_index[c]], color=BLUE, lw=0.9)
        if tmark is not None:
            ax.axvline(tmark, color="#D97706", lw=1.2, ls="--")
        ax.set_ylabel(c, fontsize=6.5, color=GRAY)
        ax.tick_params(labelsize=6)
        ax.spines[["top", "right"]].set_visible(False)
    axes[-1].set_xlabel("seconds", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    rng = np.random.default_rng(SEED)
    W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
    X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
    ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
    ch_index = {c: i for i, c in enumerate(ch)}
    seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
    kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}

    os.makedirs(f"{BASE}/results/audit/clips", exist_ok=True)
    from concurrent.futures import ThreadPoolExecutor
    pool_ex = ThreadPoolExecutor(max_workers=4)
    items, manifest, jobs = [], [], []
    for prim, (chans, questions) in PRIMS.items():
        for lab, name in [(1, "high"), (-1, "low")]:
            pool = np.flatnonzero((W[prim] == lab).to_numpy())
            pick = rng.choice(pool, min(PER_CELL, len(pool)), replace=False)
            for gi in pick:
                r = W.iloc[gi]
                km = kmin.get((r.model, r.driver, r.route))
                if km is None:
                    continue
                t0 = float(r.t0)
                k = km + int(t0 // 60)
                rowdir = os.path.join(DATASET, r.model, r.driver, r.route)
                clipname = f"clips/w{int(gi)}.mp4"
                jobs.append(pool_ex.submit(make_window_clip, rowdir, k, t0 % 60,
                                           f"{BASE}/results/audit/{clipname}"))
                wv = X[gi].astype(np.float32)
                tm = key_moment(wv, prim, ch_index)
                plot = plot_b64(wv, chans, ch_index, tmark=tm)
                items.append({"gi": int(gi), "prim": prim, "label": name,
                              "q": questions[name], "clip": clipname, "plot": plot,
                              "tm": tm})
                manifest.append({"gi": int(gi), "prim": prim, "label": name})
        print(f"{prim}: queued", flush=True)
    ok = sum(j.result() for j in jobs)
    print(f"clips encoded: {ok}/{len(jobs)}", flush=True)

    rng.shuffle(items)   # blind the auditor to any ordering pattern
    os.makedirs(f"{BASE}/results/audit", exist_ok=True)
    pd.DataFrame(manifest).to_csv(f"{BASE}/results/audit/audit_manifest.csv", index=False)

    cards = []
    for i, it in enumerate(items):
        img = (f'<video src="{it["clip"]}" controls muted preload="none"></video>'
               if it["clip"] else "<p><em>no video</em></p>")
        cards.append(f"""
<div class="card" id="c{i}">
  <div class="head"><b>#{i+1} / {len(items)}</b>
    <span class="tag">{it['prim'].replace('p_','')} = {it['label']}</span></div>
  <div class="media">{img}
    <img src="data:image/png;base64,{it['plot']}" alt="signals"></div>
  <p class="q">{it['q']} <span class="km">key moment &asymp; {it['tm']:.0f}s (dashed line)</span></p>
  <div class="opts" data-gi="{it['gi']}" data-prim="{it['prim']}" data-label="{it['label']}">
    <label><input type="radio" name="a{i}" value="yes"> label is correct</label>
    <label><input type="radio" name="a{i}" value="no"> label is wrong</label>
    <label><input type="radio" name="a{i}" value="unsure"> unsure</label>
  </div>
</div>""")

    html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>DriveDNA T1 label audit ({len(items)} windows)</title>
<style>
body{{font:15px/1.5 system-ui;margin:0;background:#F7F8FA;color:#1A2233}}
.top{{position:sticky;top:0;background:#fff;border-bottom:1px solid #D9DEE7;padding:.6rem 4vw;
     display:flex;justify-content:space-between;align-items:center;z-index:5}}
.wrap{{max-width:760px;margin:0 auto;padding:1rem 4vw 4rem}}
.card{{background:#fff;border:1px solid #D9DEE7;margin:1rem 0;padding:1rem}}
.head{{display:flex;justify-content:space-between;margin-bottom:.5rem}}
.tag{{font-family:ui-monospace,monospace;font-size:.8rem;background:#EAF1FE;color:#2563EB;padding:.1rem .5rem}}
.media img,.media video{{max-width:100%;display:block;margin:.35rem 0;border:1px solid #EEE}}
.q{{font-weight:600}}
.km{{font-family:ui-monospace,monospace;font-size:.8rem;color:#B45309;font-weight:400;margin-left:.6rem}}
.opts label{{margin-right:1.2rem}}
button{{font:inherit;padding:.45rem 1rem;background:#2563EB;color:#fff;border:0;cursor:pointer}}
#done{{font-family:ui-monospace,monospace}}
</style></head><body>
<div class="top"><span><b>DriveDNA T1 label audit</b> — for each window: does the label match what you see?</span>
<span><span id="done">0</span>/{len(items)} <button onclick="exp()">Export JSON</button></span></div>
<div class="wrap">{''.join(cards)}</div>
<script>
function count(){{document.getElementById('done').textContent=
  document.querySelectorAll('input:checked').length;}}
document.addEventListener('change',count);
function exp(){{
  const out=[];
  document.querySelectorAll('.opts').forEach(o=>{{
    const c=o.querySelector('input:checked');
    out.push({{gi:+o.dataset.gi,prim:o.dataset.prim,label:o.dataset.label,
              answer:c?c.value:null}});
  }});
  const b=new Blob([JSON.stringify(out,null,1)],{{type:'application/json'}});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download='t1_audit_answers.json';a.click();
}}
</script></body></html>"""
    out = f"{BASE}/results/audit/t1_audit.html"
    open(out, "w").write(html)
    print(f"wrote {out}  ({len(items)} items, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
