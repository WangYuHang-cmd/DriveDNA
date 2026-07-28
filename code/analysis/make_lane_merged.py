#!/usr/bin/env python3
"""Merged lane-change annotation pages (user spec 2026-07-22).

Unanswered lane candidates whose 10-s clips overlap by >=5 s (consecutive
absolute-time gap <= 5 s within the same drive) are chain-merged into ONE
watch unit spanning [first onset - 5 s, last onset + 5 s].  One keypress
labels every member event; the page expands the answer to per-eid records
on save, so all downstream tooling (gi validation, stats) works unchanged.

Already-answered eids are untouched and excluded.  Groups whose merged span
needs a missing minute file fall back to singleton cases reusing the
existing per-event clips.

Output: results/maneuver_audit/lane_merged_p{n}.html + mclips/ +
        lane_merged_index.html + lane_merged_manifest.csv
"""
import os, json, subprocess
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASE = "."
DATASET = "../Dataset"
OUT = f"{BASE}/results/maneuver_audit"
PER_PAGE = 200
GAP = 5.0           # max consecutive onset gap (s) for merging
PAD = 5.0           # clip padding around first/last onset

man = pd.read_csv(f"{OUT}/manifest.csv")
lane = man[man.cls == "lane_change"].sort_values("eid").reset_index(drop=True)
cur = {r.eid: int(r.gi) for r in lane.itertuples()}
answered = set()
for fn in os.listdir(f"{OUT}/answers"):
    if fn.startswith(("lane_changev2p", "lane_mergedp")) and fn.endswith(".json"):
        for x in json.load(open(f"{OUT}/answers/{fn}")):
            if x.get("answer") and cur.get(x["eid"]) == x["gi"]:
                answered.add(x["eid"])

W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
lane["model"] = W.iloc[lane.gi].model.values
lane["driver"] = W.iloc[lane.gi].driver.values
lane["route"] = W.iloc[lane.gi].route.values
lane["t_abs"] = W.iloc[lane.gi].t0.values + lane.onset.values / 10.0
lane["has_clip"] = [os.path.exists(f"{OUT}/clips/e{e}.mp4") for e in lane.eid]
un = lane[(~lane.eid.isin(answered)) & lane.has_clip].copy()
print(f"answered {len(answered)} | unanswered with video {len(un)}", flush=True)

# ---- chain-merge ----
groups = []
for (d, r), g in un.groupby(["driver", "route"], sort=True):
    g = g.sort_values("t_abs")
    rows = list(g.itertuples())
    curg = [rows[0]]
    for x in rows[1:]:
        if x.t_abs - curg[-1].t_abs <= GAP:
            curg.append(x)
        else:
            groups.append(curg); curg = [x]
    groups.append(curg)
print(f"merged cases: {len(groups)}", flush=True)

os.makedirs(f"{OUT}/mclips", exist_ok=True)

def files_ok(g):
    r0 = g[0]
    km = kmin.get((r0.model, r0.driver, r0.route))
    if km is None:
        return None
    start = max(0.0, g[0].t_abs - PAD)
    end = g[-1].t_abs + PAD
    ks = list(range(km + int(start // 60), km + int(end // 60) + 1))
    paths = [os.path.join(DATASET, r0.model, r0.driver, r0.route, f"{k}--qcamera.ts") for k in ks]
    if not all(os.path.exists(p) for p in paths):
        return None
    return paths, start % 60, end - start

def encode(idx, g):
    outp = f"{OUT}/mclips/m{idx}.mp4"
    if os.path.exists(outp) and os.path.getsize(outp) > 5000:
        return True
    fo = files_ok(g)
    if fo is None:
        return False
    paths, off, dur = fo
    src = paths[0] if len(paths) == 1 else "concat:" + "|".join(paths)
    try:
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", f"{off:.1f}", "-i", src,
                        "-t", f"{dur:.1f}", "-c:v", "libx264", "-preset", "veryfast",
                        "-crf", "30", "-an", "-movflags", "+faststart", outp],
                       capture_output=True, timeout=180)
        return os.path.exists(outp) and os.path.getsize(outp) > 5000
    except Exception:
        return False

# cases: (clip_relpath, members) — multi groups get merged clips; failures/singles use e{eid}.mp4
cases = []
multi = [(i, g) for i, g in enumerate(groups) if len(g) > 1]
print(f"encoding {len(multi)} merged clips ...", flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:
    oks = list(pool.map(lambda t: encode(t[0], t[1]), multi))
okmap = {i: ok for (i, g), ok in zip(multi, oks)}
n_fail = sum(1 for v in okmap.values() if not v)
print(f"merged clips ok {sum(okmap.values())}/{len(multi)} (fallback singles for {n_fail})", flush=True)
for i, g in enumerate(groups):
    if len(g) == 1 or not okmap.get(i, False):
        for x in g:
            cases.append((f"clips/e{x.eid}.mp4", [x]))
    else:
        cases.append((f"mclips/m{i}.mp4", list(g)))
print(f"final cases: {len(cases)}", flush=True)

STYLE = """<style>body{font:16px/1.5 system-ui;margin:0;background:#F7F8FA;color:#1A2233}
.top{position:sticky;top:0;background:#fff;border-bottom:1px solid #D9DEE7;padding:.6rem 3vw;display:flex;justify-content:space-between;align-items:center;z-index:5}
.wrap{max-width:780px;margin:0 auto;padding:1rem 3vw 2rem}
.card{background:#fff;border:1px solid #D9DEE7;padding:1rem}
video{width:100%;display:block;margin:.3rem 0;border:1px solid #EEE;background:#000}
.q{font-weight:650;font-size:1.1rem}
.tag{font-family:ui-monospace,monospace;font-size:.8rem;background:#EAF1FE;color:#2563EB;padding:.05rem .45rem}
.keys{display:flex;gap:.6rem;margin-top:.6rem;flex-wrap:wrap}
.key{border:1px solid #D9DEE7;background:#fff;padding:.45rem 1rem;cursor:pointer;font:inherit}
.key b{font-family:ui-monospace,monospace;background:#EAF1FE;color:#2563EB;padding:0 .35rem;margin-right:.35rem}
.key.sel{border-color:#2563EB;background:#EAF1FE}
.hint{color:#6B7280;font-size:.82rem;margin-top:.6rem}
.mono{font-family:ui-monospace,monospace}
button.exp{font:inherit;padding:.4rem .9rem;background:#2563EB;color:#fff;border:0;cursor:pointer}</style>"""
SCRIPT = """<script>
const items=ITEMS; const PF=PFILL; const KEY='mans_'+document.title.replace(/ /g,'_');
let ans=Object.assign({},PF,JSON.parse(localStorage.getItem(KEY)||'{}')); let cur=0; let rate=5;
while(cur<items.length&&ans[items[cur].mid]!==undefined)cur++;
const $=id=>document.getElementById(id);
function render(){
  const it=items[cur]||items[items.length-1];
  $('pos').textContent=(cur+1)+'/'+items.length;
  $('nans').textContent=Object.keys(ans).length;
  $('tag').textContent=it.n+' event'+(it.n>1?'s':'')+' merged · '+it.len+'s';
  $('q').textContent=it.q;
  const v=$('vid');
  if(it.clip){v.style.display='';v.src=it.clip;v.playbackRate=rate;v.play().catch(()=>{});}
  else{v.style.display='none';v.removeAttribute('src');}
  $('rate').textContent=rate+'x';
  ['yes','no','unsure'].forEach(k=>$('k_'+k).classList.toggle('sel',ans[it.mid]===k));
}
function save(){const out=[];
  items.forEach(it=>{const a=ans[it.mid]??null;
    it.eids.forEach((e,j)=>out.push({eid:e,gi:it.gis[j],cls:'lane_change',answer:a}));});
  fetch('/save',{method:'POST',body:JSON.stringify({page:document.title,answers:out})})
    .then(()=>{$('sv').textContent='saved \\u2713';$('sv').style.color='#15803D';})
    .catch(()=>{$('sv').textContent='offline (browser-saved)';$('sv').style.color='#B45309';});}
function mark(val){const it=items[cur];if(!it)return;ans[it.mid]=val;
  localStorage.setItem(KEY,JSON.stringify(ans));save();next();}
function next(){if(cur<items.length-1){cur++;render();}else{render();}}
function prev(){if(cur>0){cur--;render();}}
function cycleRate(){const rs=[1,1.5,2,3,4,5];rate=rs[(rs.indexOf(rate)+1)%rs.length];
  $('vid').playbackRate=rate;$('rate').textContent=rate+'x';}
function exp(){const out=[];
  items.forEach(it=>{const a=ans[it.mid]??null;
    it.eids.forEach((e,j)=>out.push({eid:e,gi:it.gis[j],cls:'lane_change',answer:a}));});
  const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});
  const a=document.createElement('a');a.href=URL.createObjectURL(b);
  a.download=document.title.replace(/ /g,'_')+'_answers.json';a.click();}
addEventListener('keydown',e=>{
  if(e.key==='1'||e.key==='y')mark('yes');
  else if(e.key==='2'||e.key==='n')mark('no');
  else if(e.key==='3'||e.key==='u')mark('unsure');
  else if(e.key==='s'||e.key==='ArrowRight')next();
  else if(e.key==='ArrowLeft')prev();
  else if(e.key==='f')cycleRate();
  else if(e.key==='r'){$('vid').currentTime=0;$('vid').play();}
  else if(e.key===' '){e.preventDefault();const v=$('vid');v.paused?v.play():v.pause();}
  else if(e.key==='e')exp();
});
addEventListener('DOMContentLoaded',render);
</script>"""

Q = "Does a lane change occur anywhere in this clip?"
pages = int(np.ceil(len(cases) / PER_PAGE))
index = [f"<h1>Merged lane-change annotation ({len(cases)} cases, {pages} pages)</h1>"
         "<p>One answer labels every merged event. Keys: <b>1/y</b>=yes · <b>2/n</b>=no · "
         "<b>3/u</b>=unsure · <b>s/→</b>=skip · <b>←</b>=back · <b>f</b>=speed · "
         "<b>r</b>=replay · <b>space</b>=pause. Default 4x.</p><ul>"]
manifest_rows = []
for p in range(pages):
    chunk = cases[p*PER_PAGE:(p+1)*PER_PAGE]
    items = []
    for ci, (clip, members) in enumerate(chunk):
        mid = f"m{p*PER_PAGE+ci}"
        length = round(members[-1].t_abs - members[0].t_abs + 2*PAD)
        items.append({"mid": mid, "eids": [int(x.eid) for x in members],
                      "gis": [int(x.gi) for x in members], "n": len(members),
                      "len": length, "q": Q, "clip": clip})
        for x in members:
            manifest_rows.append({"mid": mid, "eid": int(x.eid), "gi": int(x.gi),
                                  "page": f"lane_merged_p{p+1}.html"})
    fn = f"lane_merged_p{p+1}.html"
    shell = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>lane merged p{p+1}</title>{STYLE}</head><body>
<div class='top'><span><b>lane merged</b> — page {p+1}/{pages} · <span class='mono' id='pos'></span> · answered <span class='mono' id='nans'></span></span>
<span><span class='mono' id='sv' style='margin-right:.8rem'></span><span class='mono' id='rate'>5x</span> <button class='exp' onclick='exp()'>Export (e)</button></span></div>
<div class='wrap'><div class='card'>
<span class='tag' id='tag'></span>
<video id='vid' muted playsinline></video>
<p class='q' id='q'></p>
<div class='keys'>
<button class='key' id='k_yes' onclick="mark('yes')"><b>1</b>yes</button>
<button class='key' id='k_no' onclick="mark('no')"><b>2</b>no</button>
<button class='key' id='k_unsure' onclick="mark('unsure')"><b>3</b>unsure</button>
<button class='key' onclick='next()'><b>s</b>skip</button>
<button class='key' onclick='prev()'><b>←</b>back</button>
<button class='key' onclick='cycleRate()'><b>f</b>speed</button>
</div>
<p class='hint'>One answer labels all merged events in the clip. Autosaves to disk on every keypress.</p>
</div></div>
{SCRIPT.replace("ITEMS", json.dumps(items)).replace("PFILL", "{}")}
</body></html>"""
    open(f"{OUT}/{fn}", "w").write(shell)
    index.append(f"<li><a href='{fn}'>page {p+1} ({len(chunk)} cases)</a></li>")
index.append("</ul>")
open(f"{OUT}/lane_merged_index.html", "w").write(
    "<!DOCTYPE html><html><head><meta charset='utf-8'><title>lane merged</title>"
    + STYLE + "</head><body><div class='wrap'>" + "".join(index) + "</div></body></html>")
pd.DataFrame(manifest_rows).to_csv(f"{OUT}/lane_merged_manifest.csv", index=False)
print(f"pages: {pages} -> {OUT}/lane_merged_index.html", flush=True)
