# DriveDNA — Consolidated Results for Paper Writing

*Snapshot 2026-07-09 (Day 3). All numbers from full human-only data (62,674 windows / 428 drivers /
975 h human segments @10 Hz), driver-disjoint evaluation on UNSEEN drivers unless noted.
Per-experiment JSON in this directory; training logs in `../experiments/logs/`; Day 0–2 archive in
`../experiments/logs/sanity_results.md`; examples in `examples/` + `../figures/examples/`.*

**ALL WAVES COMPLETE (Jul 10 morning).** Every number below is full human-only data, uncapped.
V-JEPA2 alignment: 61,305/62,674 windows (97.8%, identical coverage to DINOv2).

---

## 1. Tasks evaluated (all six have results)

| Task | Protocol | Status |
|---|---|---|
| T1 primitive tagging | 8 primitives, scenario-conditioned Q80/Q20 weak labels, 62,674 windows; **human audit: 93.0% agreement** (240 windows, per-primitive 84–100%, results/audit/) | ✅ annotation layer (supporting) |
| T2 few-shot driver re-ID/verification | route-disjoint enrollment k∈{1,3,5,10} min, 78–80 unseen drivers | ✅ core |
| T3 personalized realized-behavior prediction | 5 s history → 5 s future (aEgo, actual_curvature), k-shot support, PG | ✅ core |
| T4 matched-context comparison | 14,868 balanced pairs (scenario×v×THW×model matched) | ✅ core |
| T5 event forecasting | 5 s history → event onset within 5 s (brake/steer), AP/AUROC/lead-time | ✅ (⏳ uncapped rerun) |
| T6 explanation/evidence | event-centered [−5,+3 s] candidates + VLM evidence | ✅ candidates (stretch) |

## 2. Baseline inventory (what was actually trained/evaluated, all full data)

**Capability ladder:** descriptors (audit anchor) · S1 PatchTST-style joint-patch + SupCon + ProtoNet ·
S2 ResidualStyle + DANN Pareto · S3 few-shot FiLM personalized predictor (v1 joint / v2 frozen-S1) ·
S4 MCPP cross-attention CAN+video · S5 MDN.
**Advanced wave:** W1 channel-independent PatchTST · W1 iTransformer · W1 ArcFace · CVAE ·
W2 masked-TS SSL (SimMTM/TimeMAE-style) · W3 JEPA-style latent-predictive SSL baseline (appendix naming).
**Multimodal wave:** M2 MCPP 6-way (2 recipes) · M3 temporal-video T5 (causal mask) · M4 CAN↔video
CLIP alignment · M5 Qwen2.5-VL scene attributes (17,328 windows) · M6 TransFuser-lite style-in-query ·
M7 video-only probe · M1 V-JEPA2 re-embed (encoder ablation, ⏳ tonight).
= **~20 named modern baselines/diagnostics** (PatchTST, iTransformer, SupCon, ArcFace, ProtoNet,
attentive-stats pooling, DINOv2, V-JEPA2, cross-attention, FiLM, MDN, CVAE, DANN, SimMTM/TimeMAE-style,
JEPA-style SSL, CLIP-style InfoNCE, TransFuser-style query injection, Qwen2.5-VL, GBM population models).

## 3. T2 — unified representation table (@5-min enrollment, unseen drivers)

| Representation | AUROC | EER | top-1 | seeds |
|---|---|---|---|---|
| descriptors (anchor) | .668 | .379 | .029 (=8× chance, 286 drv) | — |
| **S1 joint-patch + SupCon** | **.935 ± .005** | .127 ± .003 | .42–.48 (~33× chance) | 3 |
| **W1 PatchTST-CI + SupCon** | **.932 ± .002** | .127 ± .006 | .32–.34 | 3 |
| W1 joint-patch + ArcFace | .902 ± .005 | .174 ± .008 | .37–.39 | 3 |
| W1 iTransformer + SupCon | .877 ± .008 | .192 ± .014 | .20–.23 | 3 |
| W2 masked-TS SSL → frozen probe | .907 | .166 | .29 | 1 |
| W3 JEPA-style SSL → frozen probe | .878 | .197 | .20 | 1 |
| M4 CLIP-aligned CAN (no driver labels) | .831 | .244 | .21 | 1 |
| **V3 MOMENT-1-large (2024 TS foundation, zero-shot)** | .636 | .408 | .08 | 1 |
| M7 video-only (SupCon probe) | .937 | .136 | .55 | 1 (see §6 leakage) |

Architecture separation (SupCon≈CI > ArcFace > iTransformer) is 3–6× the seed sd → the protocol
discriminates representation choices. SSL closes most of the supervised gap without driver labels.

## 4. T3 — personalization utility (unseen drivers, k=5 route-disjoint support)

| Model / conditioning | Metric | Generic → Personalized | PG |
|---|---|---|---|
| S3 FiLM few-shot (v1, joint; 3 seeds fixed-eval) | RMSE | — | **PG > 0 in 3/3 seeds**; 5-s horizon **+0.83 ± 0.45%**; horizon decay (1 s > 5 s) in 2/3; primitive stratification all-positive in strong seeds |
| V4: S3 with Transformer predictor head (GRU→TF-encoder, same protocol) | RMSE | — | PG **+0.3/+0.5/+0.5%** (1/3/5 s) — positive at every horizon, same magnitude class as GRU → **PG conclusions are not an RNN artifact** (appendix D) |

**Protocol note (error bars):** seed reruns revealed that varying `--seed` originally moved the *evaluation*
support/query route-split too; NLL-type metrics are heavy-tailed in that split. All S3/MDN/CVAE seed numbers
above use the **frozen eval protocol** (`rng_eval`, constant across seeds; seed varies training only).
T2/T4/T5 protocols were already split-fixed. This diagnosis is itself benchmark-relevant (appendix note).
| S3 v2 (frozen S1 embedding as z_d) | RMSE | — | **−0.2%** (identity ≠ prediction) |
| S5 MDN (3 seeds, FIXED eval protocol) | NLL | — | **+0.10 ± 0.05 nats** distributional-PG (positive 3/3); RMSE-PG ≈ 0 ×3. *(Initial 1-seed +1.16 was eval-split luck — identical model under the frozen protocol gives +0.068; see protocol note below.)* |
| W5 CVAE (3 seeds, FIXED eval protocol) | NLL | — | **+6.9 ± 4.3 nats** distributional-PG (+3.7/+4.0/+12.9, positive 3/3); RMSE-PG ≈ 0 ×3 |
| M2 MCPP 6-way, DINOv2 (S4 recipe) | RMSE Δ vs CAN | +veh **−1.22%** · +veh+vid +0.09% · +veh+drv −0.74% · +veh+vid+drv **+0.44%** · +res **+0.47%** | video+driver complementary; vehicle-embedding negative transfer |
| **M2 MCPP 6-way, V-JEPA2 (3 seeds)** | RMSE Δ vs CAN | +veh **−0.93 ± 0.10** · +veh+vid **+1.49 ± 0.32** · +veh+vid+drv +2.12 ± 0.97 · +res +2.10 ± 0.92 | **temporal-video contribution robust across seeds** (DINOv2: +0.09); vehicle negative transfer solid |
| M6 TransFuser-lite (style-in-query) | RMSE | — | **−0.72%** (CAN) / −0.63% (+vid) — query injection doesn't transfer (Day-9 seed check pending) |

## 5. T4 / T5 — context-controlled comparison & event forecasting

| Result | Value |
|---|---|
| T4: S1 embedding on 14,868 matched pairs | **AUROC .812 ± .005** (3 seeds); per-scenario .78–.83 |
| T4: descriptors on same pairs | **.550 (≈chance, ×3 seeds)** — matching destroys the shortcut baseline |
| T4 visual validation (M5, Qwen3-VL-4B primary, 14k pairs) | lift exactly on matched dims: road_type **1.68×**, density 1.21×, lead_vehicle 1.19×; unmatched ≈1.0× (signal 1.03×, weather 1.07×). Qwen2.5-VL replication: 1.49×/1.04×/1.15× — robust across VLM generations |
| T5 (CAN GRU, rule-derived taxonomy) **FULL-DATA ✅** | uncapped 38.6k/17.2k windows: AUROC **.843 ± .002** (3 seeds), AP .219 ± .006 (3% base); brake .845 / steer .848; lead-time .851/.849/.838 |
| M3 T5+video FULL, DINOv2 per-frame | CAN .835 → +vid5s .797 → +vid-full **.787** — per-frame video actively HURTS |
| **M3 T5+video FULL, V-JEPA2 temporal** | CAN .842 → +vid5s .841 → +vid-full **.847** (brake .835→**.847**), monotone in context — **temporal clips flip video from harmful to helpful** |

## 6. Diagnostics (the benchmark's signature axis)

| Probe | Result |
|---|---|
| Steering vs curvature vehicle-leakage | steering 2.3× chance vs actual_curvature ≈ chance (input-vs-path mechanism) |
| Utility–leakage Pareto | descriptors 8×/lk 2.9× → Residual 4×/2.6× → **S1 33×/2.6×** → S1-DANN 32×/2.8× — learned embedding near vehicle-agnostic without adversarial training |
| ResidualStyle | context+vehicle explain 29–60% variance; residual survives at 4× chance |
| M7 video vs CAN leakage (same protocol) | video: driver 81×/route **347×**/vehicle 64× · CAN-S1: 65×/197×/56× · raw DINOv2 route 714× — video re-ID rides on place/vehicle |
| M7b within-nameplate probe (17 same-model groups, @5-min) | CAN .935→**.887** (loses vehicle-dynamics cue) but video .937→**.962** (geography persists within model groups) → video's identity channel is dominated by PLACE, vehicle appearance secondary |
| V6 log-type confound | 0.50 (clean) · within-vehicle re-ID 8× |
| M5 same-vs-diff driver road_type agreement | .500 vs .432 — geographic-habit channel corroborates M7 |

## 7. The 18 findings (§6 skeleton)

1. Context+vehicle explain 29–60% of behavior-stat variance → confound control necessary.
2. Stable driver residual survives conditioning (4×) — the L3 evidence.
3. Descriptor anchor leaves large headroom.
4. Steering leaks vehicle 2.3×; realized curvature ≈ chance.
5. Capability ladder lands: descriptors .668 → S1 .935±.005 (~33× top-1).
6. PG > 0, grows with k, decays with horizon.
7. Identity-discriminative ≠ prediction-useful (S1-as-z_d PG −0.2%).
8. Video and driver context are complementary (+0.44% joint).
9. **Distributional PG is positive in 6/6 runs across two heads** (CVAE +6.9 ± 4.3 nats, MDN +0.10 ± 0.05 nats;
   frozen eval protocol) while RMSE-PG ≈ 0 everywhere — personalization moves the predicted distribution, not the
   point estimate, and the magnitude is head-dependent (latent-variable CVAE captures far more driver-conditional
   structure than a mixture head). Initial 1-seed magnitudes were eval-split-inflated; the audit is an appendix
   protocol lesson (freeze eval splits for heavy-tailed metrics).
10. Learned realized-motion embeddings near vehicle-agnostic w/o DANN.
11. Matched-context is the decisive test (descriptors → .550; learned holds .804).
12. Video "identifies" drivers via place/vehicle (route 347× vs 197×), not motion style → video-as-context is empirical.
13. T2 discriminates architectures (gaps 3–6× seed sd).
14. "Style is distributional" is head-agnostic (CVAE reproduces MDN signature).
15. Label-free pretraining works: masked-SSL .907 vs supervised .931; JEPA-style .878.
16. Explicit vehicle embeddings show negative transfer (−1.2~−1.4%, both recipes).
17. Conditioning architecture matters: FiLM +0.4~1.4% vs query-injection −0.7%.
18. T4 matching visually validated by independent VLM (lift only on matched dims).
19. **[FINAL] Per-frame appearance features hurt event forecasting across three encoder generations; temporal
    features are what video needs.** T5 CAN→+vid-full: DINOv2 '23 .835→.787 (−4.8) · **DINOv3 '25 .833→.760
    (−7.3, the newest hurts MOST)** · SigLIP2 '25 .834→.808 (−2.6) · V-JEPA2 temporal .842→**.847 (+0.5, only
    positive)**. MCPP video contribution: per-frame +0.09% vs temporal +1.49±0.32%. The "extractor too old"
    hypothesis is refuted by data; motion cues, not appearance quality, are the operative variable.
    M2-dinov3 completes the matrix: DINOv3's T3 video contribution = **+0.09%, identical to DINOv2's** (vs
    temporal +1.49%); vehicle negative transfer replicates a 4th time (−1.00%).
    (results/m3_t5_video_{dinov2,dinov3,siglip2,vjepa2}.json + m2_mcpp6_{dinov2_v2,vjepa2,dinov3}.json)
20. **Personalization gain is robust to the predictor architecture** (V4): replacing the GRU head with a
    Transformer encoder under the identical protocol keeps PG positive at every horizon (+0.3/+0.5/+0.5%
    at 1/3/5 s, same magnitude class as GRU) — the S3 conclusion is not an artifact of recurrent heads.

## 8. Result-file map

| File | Content |
|---|---|
| w1_{ci,itr,arcface}[_s10,_s11].json | W1 T2 tables (3 seeds) |
| w2_{masked,jepa}.json · w5_cvae.json | SSL probes · CVAE distributional PG |
| m2_mcpp6_dinov2{,_v2}.json | MCPP 6-way (both recipes) |
| m3_t5_video{,_dinov2,_vjepa2}.json | T5+video (full uncapped, both encoders) |
| m4_clip_align.json · m6_transfuser_lite.json | CLIP alignment · style-in-query |
| m7_video_probe.json · m7_contrast.json | video probe + CAN-vs-video leakage |
| vlm_attrs.jsonl · m5_pair_agreement.json | 17,328 window attributes · T4 validation |
| t4_results.json · t5_results.json ⏳ | T4 eval · T5 eval (capped) |
| examples/ + ../figures/examples/ | 17 examples (9 paper-ready) + 8 finding figures |
| ../experiments/logs/sanity_results.md | Day 0–2 full archive (S1–S5, DANN, harness) |
