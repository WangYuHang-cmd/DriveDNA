#!/usr/bin/env python3
"""
Detect maneuver-onset events (6 classes + cut-in subtype) over all windows and save
them to data/segments/maneuver_events.parquet (gi, cls, onset, score, subtype).

Rules are precision-first (user spec 2026-07-17):
  turn  = steering angle + steering rate together;  curve = road curvature;
  decel/accel = simple episodes; follow_start = far/no-lead -> sustained close lead
  (cutin subtype = appears already close / dRel drop); lane_change = LOOSE candidates
  (3 rules, deduped, curvature-constrained, scored) for full human review.
"""
import numpy as np, pandas as pd, json

BASE = "."
X = np.load(f"{BASE}/data/colab_bundle/windows_x.npy", mmap_mode="r")
ch = json.load(open(f"{BASE}/data/colab_bundle/channels.json"))["channels"]
ix = {c: i for i, c in enumerate(ch)}


def runs(cond):
    out, i, N = [], 0, len(cond)
    while i < N:
        if cond[i]:
            j = i
            while j < N and cond[j]:
                j += 1
            out.append((i, j)); i = j
        else:
            i += 1
    return out


def accel_events(v, a, win=95, min_gain=3.0, rise_min=50, dec_max=30,
                 a_thresh=1.0, a_dur=5, smooth=5):
    """Acceleration events, clip-window rule (user spec 2026-07-17 v5).
    onset = start of a speed rise; judged over the visible 10-s clip span
    [onset, onset + 9.5 s]:
      (1) distinct acceleration: smoothed aEgo >= a_thresh for >= a_dur steps;
      (2) rising speed (>= +0.1 m/s^2) for >= rise_min steps (5 s) in the span;
      (3) decelerating speed (<= -0.3 m/s^2) for <= dec_max steps (3 s);
      (4) net speed gain over the span >= min_gain;
      (5) NO reversing in the clip: signed v never < -0.2, and no
          moved-then-stopped pattern (smoothed v exceeds 1.0 then returns
          below 0.3) — on unsigned-vEgo platforms a back-out-then-drive
          maneuver shows up as exactly this pattern.
    A hit advances the scan by `win` (no overlapping events); onsets too close
    to the window end to judge the full span are skipped."""
    kk = np.ones(smooth) / smooth
    vs = np.convolve(v, kk, mode="same")
    asm = np.convolve(a, kk, mode="same")
    dv = np.diff(vs)
    riseC = np.concatenate(([0], np.cumsum(dv > 0.01)))
    decC = np.concatenate(([0], np.cumsum(dv < -0.03)))
    hc = np.concatenate(([0], np.cumsum(asm >= a_thresh)))
    qend = np.zeros(len(vs), bool)                 # step where a >=a_dur run of high-a ends
    qend[a_dur - 1:] = (hc[a_dur:] - hc[:-a_dur]) == a_dur
    qC = np.concatenate(([0], np.cumsum(qend)))
    out, i, N = [], 0, len(vs)
    while i < N - win:
        if vs[i + 1] > vs[i]:                      # candidate rise start
            gain = vs[i:i + win + 1].max() - vs[i]
            rise = riseC[i + win] - riseC[i]
            dec = decC[i + win] - decC[i]
            okA = qC[i + win + 1] - qC[i + a_dur - 1] > 0
            if gain >= min_gain and rise >= rise_min and dec <= dec_max and okA:
                lo = max(0, i - 5)                 # include the 0.5-s clip pre-roll
                s = vs[lo:i + win + 1]
                mv = np.where(s > 1.0)[0]
                if v[lo:i + win + 1].min() > -0.2 and not (len(mv) and (s[mv[0]:] < 0.3).any()):
                    out.append((i, float(gain)))
                    i += win
                    continue
        i += 1
    return out


def has_reversing(v, lo, hi, sa=None, kk=np.ones(5) / 5):
    """Reversing signature inside a clip span [lo, hi): signed v goes negative, or
    (unsigned-vEgo platforms) a move -> full stop -> move-again shuffle. When `sa`
    is given, the stop must ALSO happen with the wheels cranked (|angle| > 45 deg):
    a three-point turn / driveway shuffle stops mid-steer, while a normal
    queue-at-the-light-then-turn stops with the wheels straight and is kept."""
    seg = v[lo:hi]
    if seg.min() < -0.2:
        return True
    s = np.convolve(seg, kk, mode="same")
    m1 = np.where(s > 1.0)[0]
    if not len(m1):
        return False
    st = np.where(s[m1[0]:] < 0.3)[0]
    if not len(st):
        return False
    if not (s[m1[0] + st[0]:] > 1.0).any():
        return False
    if sa is None:
        return True
    stop0 = m1[0] + st[0]                      # stopped phase: until speed picks back up
    stop_end = stop0
    while stop_end < len(s) and s[stop_end] < 0.5:
        stop_end += 1
    return bool(np.nanmax(np.abs(sa[lo + stop0: lo + stop_end])) > 45)


rows = []
N = X.shape[0]
for gi in range(N):
    w = X[gi].astype(np.float32)
    a = w[:, ix["aEgo"]]; v = w[:, ix["vEgo"]]
    kcur = np.abs(w[:, ix["actual_curvature"]])
    sa = w[:, ix["steeringAngleDeg"]]; sr = np.abs(w[:, ix["steeringRateDeg"]])
    dr = w[:, ix["leadOne_dRel"]]; ll = w[:, ix["laneLeft_y"]]
    cond = a < -1.2
    for i, j in runs(cond):
        if j - i >= 8 and i >= 20 and not cond[max(0, i-20):i].any():
            rows.append((gi, "decel", i, 1.0, ""))
    for onset, gain in accel_events(v, a):
        if onset >= 5:
            rows.append((gi, "accel", onset, gain, ""))
    if np.isfinite(sa).all() and np.nanstd(sa) > 0.1:
        cond = (np.abs(sa) > 60) & (v > 1.5) & (v < 10)
        for i, j in runs(cond):
            if (j - i >= 15 and i >= 20 and sr[max(0, i-5):i+5].max() > 40
                    and not cond[max(0, i-20):i].any()
                    and not has_reversing(v, max(0, i - 50), min(len(v), i + 51), sa)):
                rows.append((gi, "turn", i, 1.0, ""))
        sa_ok = True
    else:
        sa_ok = False
    cond = (kcur > 0.005) & (kcur < 0.03) & (v >= 10)
    for i, j in runs(cond):
        if j - i >= 25 and i >= 30 and not cond[max(0, i-30):i].any():
            rows.append((gi, "curve", i, 1.0, ""))
    # car_following STATE (user spec 2026-07-18): not the onset transition but the
    # complete following behavior. Following frames: lead present, THW < 5 s,
    # moving (v > 2). Frames break the state at (a) radar-target discontinuities
    # (dRel jump >50% AND >5 m), (b) sustained turning (|sa|>60 deg or realized
    # curvature >0.02 for >=0.5 s). Each maximal clean run is cut into
    # non-overlapping 10-s chunks; every chunk is one event whose FULL clip shows
    # stable following. score = mean THW over the chunk; the first chunk of a run
    # is tagged "cutin" when the state began with a cut-in.
    thw = np.where((dr > 0) & (v > 2), dr / np.maximum(v, 2), np.inf)
    ok = thw < 5.0
    prev_dr = np.concatenate(([dr[0]], dr[:-1]))
    dd = np.abs(dr - prev_dr)
    ok &= ~((dd > 5.0) & (dd > 0.5 * np.maximum(prev_dr, 1e-3)))
    turnf = kcur > 0.02
    if sa_ok:
        turnf |= np.abs(sa) > 60
    for i2, j2 in runs(turnf):
        if j2 - i2 >= 5:
            ok[i2:j2] = False
    for i, j in runs(ok):
        for t in range(i, j - 99, 100):
            sub = ""
            if t == i and (thw[i:i+5].min() < 1.5 or (i > 0 and dr[i-1] > 0 and dr[i] < 0.7 * dr[i-1])):
                sub = "cutin"
            rows.append((gi, "car_following", t, float(np.mean(thw[t:t+100])), sub))
    # lane-change candidates v1 (user decision 2026-07-19: keep the FULL loose set,
    # every candidate is human-reviewed): 3 loose rules -> dedupe within 3 s ->
    # score = #rules hit. One hard gate (user spec): the whole visible clip
    # (onset +-5 s) stays above 6.7 m/s (15 mph) — this also removes any clip
    # containing a stop or reversing, since those pass through low speed.
    # v3 expansion (user spec 2026-07-19): looser thresholds + RIGHT-lane-line
    # mirror rules (covers vehicles/spans where the left line is not tracked).
    rl = w[:, ix["laneRight_y"]]
    fin = np.isfinite(ll); finr = np.isfinite(rl)
    hits = []                                   # (step, rule)
    dll = np.abs(np.diff(ll, prepend=ll[0]))
    drl = np.abs(np.diff(rl, prepend=rl[0]))
    for i, j in runs(fin & (dll > 0.6)):
        hits.append((i, "jump"))
    for i, j in runs(finr & (drl > 0.6)):
        hits.append((i, "jumpR"))
    lls = np.where(fin, ll, np.nan)
    rls = np.where(finr, rl, np.nan)
    for t in range(0, 570, 10):                 # monotone drift on ~straight road
        for sig, name in ((lls, "drift"), (rls, "driftR")):
            seg = sig[t:t+30]
            if (np.isfinite(seg[0]) and np.isfinite(seg[-1]) and abs(seg[-1]-seg[0]) > 1.0
                    and np.nanmax(kcur[t:t+30]) < 0.004):
                hits.append((t + 15, name))
    if sa_ok:
        cond = (sr > 20) & (v > 10) & (kcur < 0.005)
        for i, j in runs(cond):
            if j - i >= 4:
                hits.append((i, "sshape"))
    hits.sort()
    merged = []                                 # dedupe within 30 steps
    for step, rule in hits:
        if merged and step - merged[-1][0] <= 30:
            merged[-1] = (merged[-1][0], merged[-1][1] | {rule})
        else:
            merged.append((step, {rule}))
    for step, rules_hit in merged:
        if v[max(0, step - 50):step + 51].min() <= 6.7:
            continue
        rows.append((gi, "lane_change", int(step), float(len(rules_hit)), "+".join(sorted(rules_hit))))
    if gi % 20000 == 0:
        print(f"  {gi}/{N} rows={len(rows)}", flush=True)

df = pd.DataFrame(rows, columns=["gi", "cls", "onset", "score", "subtype"])
df.to_parquet(f"{BASE}/data/segments/maneuver_events.parquet")
print(df.cls.value_counts().to_string())
print("lane_change by score:", df[df.cls == "lane_change"].score.value_counts().to_dict())
print(f"saved {len(df)} events")
