#!/usr/bin/env python3
"""
Maneuver-annotation package: paginated HTMLs for reviewing detected events.

lane_change: ALL score>=2 candidates (~1.7k). Other classes: 1,000 sampled each.
Each item: 10-s clip centered on the event onset + yes/no/unsure. 200 items per page.
Answers export per page; merge with summarize_audit.py (accepts multiple files).

Output: results/maneuver_audit/{cls}_p{n}.html + clips/ + index.html
"""
import os, json, subprocess
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

BASE = "."
DATASET = "../Dataset"
OUT = f"{BASE}/results/maneuver_audit"
SEED, PER_CLASS, PER_PAGE = 20260717, 1000, 200
QU = {"decel": "Does a distinct deceleration episode begin here?",
      "accel": "Does a distinct acceleration episode begin here?",
      "turn": "Is this an intersection turn (left/right at a junction)?",
      "curve": "Is this road-geometry cornering (curved road, no junction)?",
      "car_following": "Is the ego car in a car-following state for the WHOLE clip (same lead vehicle ahead)?",
      "lane_change": "Is this a lane change?"}
# seconds of video BEFORE the event onset. accel onset = velocity trough, so a
# small pre-roll keeps the preceding deceleration OUT of the clip (user spec).
PRE = {"accel": 0.5, "car_following": 0.5}
PRE_DEFAULT = 5.0
HINT = {"accel": "acceleration starts right at the beginning of the clip",
        "car_following": "the ENTIRE clip should show following behind the same lead"}
HINT_DEFAULT = "event at ~5 s into the clip"
# version suffix in the page TITLE and FILENAME (busts stale browser cache/localStorage
# after a rule change; answered items are carried over via slot-preserving sampling + prefill)
VER = {"accel": " v6", "turn": " v2", "lane_change": " v2"}
RATE0 = {"lane_change": 4}                    # default playback speed per class
RATE0_DEFAULT = 1.5

ev = pd.read_parquet(f"{BASE}/data/segments/maneuver_events.parquet")
W = pd.read_parquet(f"{BASE}/data/segments/windows.parquet").reset_index(drop=True)
seg = pd.read_parquet(f"{BASE}/data/segments/segments.parquet")
kmin = {(r.model, r.driver, r.route): int(r.k_min) for r in seg.itertuples()}
prev_man = pd.read_csv(f"{OUT}/manifest.csv") if os.path.exists(f"{OUT}/manifest.csv") else None
sel, replaced_slots = [], []                 # replaced_slots: (cls, slot) whose event changed
for ci, cls in enumerate(QU):
    rng = np.random.default_rng(SEED + ci)   # per-class stream: decel (ci=0) = original sample
    g = ev[ev.cls == cls]
    if cls == "lane_change":
        # FULL candidate set. Slot-stable carry-append (2026-07-19): candidates
        # already in the manifest keep their slots (eids + encoded clips reused);
        # genuinely-new candidates are appended after them, strongest first.
        gg = g.reset_index(drop=True)
        if prev_man is not None and (prev_man.cls == cls).sum() > 0:
            prev = prev_man[prev_man.cls == cls][["gi", "onset"]].reset_index(drop=True)
            pos = {(int(r.gi), int(r.onset)): j for j, r in enumerate(gg.itertuples())}
            kept = [pos[k] for r in prev.itertuples()
                    if (k := (int(r.gi), int(r.onset))) in pos]
            used = set(kept)
            rest = gg.iloc[[j for j in range(len(gg)) if j not in used]]
            rest = rest.sort_values(["score", "gi", "onset"], ascending=[False, True, True])
            g = pd.concat([gg.iloc[kept], rest])
            print(f"lane_change carry-append: kept {len(kept)} slots, appended {len(rest)}", flush=True)
        else:
            g = gg.sort_values(["score", "gi", "onset"], ascending=[False, True, True])
    elif prev_man is not None and (prev_man.cls == cls).sum() == PER_CLASS and len(g) > PER_CLASS:
        # slot-preserving carry (2026-07-17): keep previously sampled events that survived
        # rule changes in their ORIGINAL slots (annotation eids stay valid); refill
        # dropped slots with fresh events from the new pool. Classes with unchanged
        # rules carry 100% and stay byte-identical.
        gg = g.reset_index(drop=True)
        prev = prev_man[prev_man.cls == cls][["gi", "onset"]].reset_index(drop=True)
        pos = {(int(r.gi), int(r.onset)): j for j, r in enumerate(gg.itertuples())}
        kept = [pos.get((int(r.gi), int(r.onset))) for r in prev.itertuples()]
        used = {j for j in kept if j is not None}
        pool = [j for j in range(len(gg)) if j not in used]
        rng2 = np.random.default_rng(SEED + ci + 777)
        refill = iter(np.array(pool)[rng2.choice(len(pool), sum(j is None for j in kept), replace=False)])
        rows_new = []
        for slot, j in enumerate(kept):
            if j is None:
                j = int(next(refill)); replaced_slots.append((cls, slot))
            rows_new.append(gg.iloc[j])
        g = pd.DataFrame(rows_new)
        nrep = sum(1 for c, s in replaced_slots if c == cls)
        print(f"{cls} carry: kept {PER_CLASS - nrep}, replaced {nrep} slots", flush=True)
    elif len(g) > PER_CLASS:
        g = g.iloc[rng.choice(len(g), PER_CLASS, replace=False)]
    sel.append(g)
sel = pd.concat(sel).reset_index(drop=True)
print(sel.cls.value_counts().to_string(), flush=True)

# eids of replaced slots (position of the class's first row + slot); their stale
# clips and stale answers must not be reused
replaced_eids = set()
for cls2, slot in replaced_slots:
    c0 = int(np.flatnonzero((sel.cls == cls2).to_numpy())[0])
    replaced_eids.add(c0 + slot)
for e in replaced_eids:
    p = f"{OUT}/clips/e{e}.mp4"
    if os.path.exists(p):
        os.remove(p)

# carry existing on-disk answers (by eid) into the pages as PREFILL.
# An answer is carried ONLY if its recorded gi matches the event currently in
# that slot — answers made on stale cached pages for since-replaced events are
# dropped automatically.
PREV_ANS = {}
if os.path.isdir(f"{OUT}/answers"):
    for fn in sorted(os.listdir(f"{OUT}/answers")):
        if fn.endswith(".json"):
            try:
                for x in json.load(open(f"{OUT}/answers/{fn}")):
                    if x.get("answer"):
                        PREV_ANS[int(x["eid"])] = (x["answer"], int(x.get("gi", -1)))
            except Exception:
                pass
PREV_ANS = {e: v[0] for e, v in PREV_ANS.items()
            if e not in replaced_eids and e < len(sel) and v[1] == int(sel.iloc[e].gi)}
print(f"prefill answers carried: {len(PREV_ANS)}", flush=True)

os.makedirs(f"{OUT}/clips", exist_ok=True)

def clip(row, outp):
    if os.path.exists(outp) and os.path.getsize(outp) > 5000:
        return True
    r = W.iloc[int(row.gi)]
    km = kmin.get((r.model, r.driver, r.route))
    if km is None:
        return False
    t_abs = float(r.t0) + row.onset / 10.0
    k = km + int(t_abs // 60)
    start = max(0.0, (t_abs % 60) - PRE.get(row.cls, PRE_DEFAULT))
    p1 = os.path.join(DATASET, r.model, r.driver, r.route, f"{k}--qcamera.ts")
    p2 = os.path.join(DATASET, r.model, r.driver, r.route, f"{k+1}--qcamera.ts")
    if not os.path.exists(p1):
        return False
    src = f"concat:{p1}|{p2}" if (t_abs % 60) > 50 and os.path.exists(p2) else p1
    cmd = ["ffmpeg", "-y", "-v", "error", "-ss", f"{start:.1f}", "-i", src, "-t", "10",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "30", "-an",
           "-movflags", "+faststart", outp]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
        return os.path.exists(outp) and os.path.getsize(outp) > 5000
    except Exception:
        return False

pool = ThreadPoolExecutor(max_workers=6)
names = [f"clips/e{i}.mp4" for i in range(len(sel))]
oks = list(pool.map(lambda t: clip(sel.iloc[t], f"{OUT}/{names[t]}"), range(len(sel))))
print(f"clips: {sum(oks)}/{len(sel)}", flush=True)

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
let ans=Object.assign({},PF,JSON.parse(localStorage.getItem(KEY)||'{}')); let cur=0; let rate=RINIT;
while(cur<items.length&&ans[items[cur].eid]!==undefined)cur++;
const $=id=>document.getElementById(id);
function render(){
  const it=items[cur]||items[items.length-1];
  $('pos').textContent=(cur+1)+'/'+items.length;
  $('nans').textContent=Object.keys(ans).length;
  $('tag').textContent=it.cls+(it.sub?' ('+it.sub+')':'');
  $('q').textContent=it.q;
  const v=$('vid');
  if(it.clip){v.style.display='';v.src=it.clip;v.playbackRate=rate;v.play().catch(()=>{});}
  else{v.style.display='none';v.removeAttribute('src');}
  $('rate').textContent=rate+'x';
  ['yes','no','unsure'].forEach(k=>$('k_'+k).classList.toggle('sel',ans[it.eid]===k));
}
function save(){const out=items.map(it=>({eid:it.eid,gi:it.gi,cls:it.cls,answer:ans[it.eid]??null}));
  fetch('/save',{method:'POST',body:JSON.stringify({page:document.title,answers:out})})
    .then(()=>{$('sv').textContent='saved ✓';$('sv').style.color='#15803D';})
    .catch(()=>{$('sv').textContent='offline (browser-saved)';$('sv').style.color='#B45309';});}
function mark(val){const it=items[cur];if(!it)return;ans[it.eid]=val;
  localStorage.setItem(KEY,JSON.stringify(ans));save();next();}
function next(){if(cur<items.length-1){cur++;render();}else{render();}}
function prev(){if(cur>0){cur--;render();}}
function cycleRate(){const rs=[1,1.5,2,3,4];rate=rs[(rs.indexOf(rate)+1)%rs.length];
  $('vid').playbackRate=rate;$('rate').textContent=rate+'x';}
function exp(){const out=items.map(it=>({eid:it.eid,gi:it.gi,cls:it.cls,answer:ans[it.eid]??null}));
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

index = ["<h1>DriveDNA maneuver audit (rapid keyboard mode)</h1>"
         "<p>Keys: <b>1/y</b>=yes · <b>2/n</b>=no · <b>3/u</b>=unsure · <b>s/→</b>=skip · <b>←</b>=back · "
         "<b>f</b>=speed · <b>r</b>=replay · <b>space</b>=pause · <b>e</b>=export. Answers autosave in the browser.</p><ul>"]
manifest = []
for cls in QU:
    g = sel[sel.cls == cls]
    idxs = g.index.to_numpy()
    pages = int(np.ceil(len(idxs) / PER_PAGE))
    for p in range(pages):
        fn = f"{cls}{VER.get(cls, '').strip()}_p{p+1}.html"
        chunk = idxs[p*PER_PAGE:(p+1)*PER_PAGE]
        items = []
        for i in chunk:
            row = sel.iloc[i]
            manifest.append({"eid": int(i), "gi": int(row.gi), "cls": cls, "onset": int(row.onset),
                             "page": fn})
            if not oks[i]:
                continue                       # no source video -> not reviewable, skip
            items.append({"eid": int(i), "gi": int(row.gi), "cls": cls,
                          "sub": row.subtype, "q": QU[cls],
                          "clip": names[i]})
        prefill = {int(i): PREV_ANS[int(i)] for i in chunk if int(i) in PREV_ANS}
        rate0 = RATE0.get(cls, RATE0_DEFAULT)
        shell = f"""<!DOCTYPE html><html><head><meta charset='utf-8'><title>{cls}{VER.get(cls, "")} p{p+1}</title>{STYLE}</head><body>
<div class='top'><span><b>{cls}</b> — page {p+1}/{pages} · <span class='mono' id='pos'></span> · answered <span class='mono' id='nans'></span></span>
<span><span class='mono' id='sv' style='margin-right:.8rem'></span><span class='mono' id='rate'>{rate0}x</span> <button class='exp' onclick='exp()'>Export (e)</button></span></div>
<div class='wrap'><div class='card'>
<span class='tag' id='tag'></span>
<p style='color:#6B7280;font-size:.85rem;margin:.3rem 0'>{HINT.get(cls, HINT_DEFAULT)}</p>
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
<p class='hint'>Autosaves locally; press <b>e</b> to export when the page is done. Refresh-safe.</p>
</div></div>
{SCRIPT.replace("ITEMS", json.dumps(items)).replace("PFILL", json.dumps(prefill)).replace("RINIT", str(rate0))}
</body></html>"""
        open(f"{OUT}/{fn}", "w").write(shell)
        index.append(f"<li><a href='{fn}'>{cls} page {p+1} ({len(chunk)} items)</a></li>")
index.append("</ul>")
open(f"{OUT}/index.html", "w").write("<!DOCTYPE html><html><head><meta charset='utf-8'>"
                                     f"<title>maneuver audit</title>{STYLE}</head><body><div class='wrap'>"
                                     + "".join(index) + "</div></body></html>")
pd.DataFrame(manifest).to_csv(f"{OUT}/manifest.csv", index=False)
print(f"pages written to {OUT}/index.html  ({len(sel)} items)", flush=True)
