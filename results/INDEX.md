# Result files

Every JSON here was written by the script in the second column; file names with `_s10` / `_s11` are the additional training seeds. Driver identifiers inside the files are the release pseudonyms (`driver_XXX`); drive identifiers are `driver_XXX__drive_YYY`. Two truncated files from interrupted runs (`w5_cvae.json`, `m2_mcpp6_dinov2.json`) and the large embedding `.npz` files are not included. `CHECKSUMS.sha256` in the repository root lists the sha256 of every file in this folder.

| File(s) | Produced by | Where it appears in the paper |
|---|---|---|
| t4_results.json / t4_results_s10.json / t4_results_s11.json | `code/eval/t4_eval.py [--ckpt experiments/checkpoints/s1_supcon_s10.pt --tag _s10]` | Table 4, Sec. 6.3: PatchTST+SupCon vs. descriptors on the 14,868 condition-matched pairs (three seeds) |
| t3_all.json | `code/eval/t3_all.py [--moment] [--llm]` | Table 4: every representation under condition matching |
| t3_seeds.json | hand-collected seed summary of Table 4 rows (no script) | Table 4 error bars |
| w1_ci*.json, w1_itr*.json, w1_arcface*.json | `code/model/w1_backbones.py --arch {ci,itr,arcface} [--seed 10|11]` | Table 3: CI-PatchTST, iTransformer, ArcFace rows |
| w2_masked.json, w2_jepa.json | `code/model/w2_ssl.py --mode {masked,jepa}` | Table 3: self-supervised rows |
| v3_moment.json | `code/model/v3_moment.py` | Table 3: MOMENT-1 zero-shot row |
| llm_t2_qwen3-4b.json, llm_t2_llama-3.2-3b.json | `code/model/llm_embed_t2.py --model {qwen3-4b,llama-3.2-3b}` | Table 3: LLM text-encoder zero-shot rows (the embedding .npz files are not included) |
| e5_descriptor_pop.json | `code/eval/e5_descriptor_pop.py` | Table 3: classical-descriptor anchor on the same evaluation population |
| m4_clip_align.json | `code/model/m4_clip_align.py` | Table 3: CLIP-aligned CAN (no labels) row; Sec. 6.5 |
| m7_video_probe.json, m7_contrast.json | `code/model/m7_video_probe.py`, `code/model/m7_contrast.py` | Sec. 6.5: video-only re-identification probe and route/vehicle leakage contrast |
| m6_transfuser_lite.json | `code/model/m6_transfuser_lite.py` | Sec. 6.5 / App. K: multimodal conditioning baseline |
| m2_mcpp6_dinov2_v2.json, m2_mcpp6_dinov3.json, m2_mcpp6_vjepa2*.json | `code/model/m2_mcpp6.py --video {dinov2,dinov3,vjepa2} [--tag _v2] [--seed 10|11]` | Sec. 6.5 / App. K: MCPP six-way modality ablation |
| e3_e4_distribution.json | `code/eval/e3_e4_distribution.py` | Sec. 6.4 / App. H: MMD, KL, W1 and per-driver personalization gains (driver ids pseudonymised) |
| t5_results.json | `code/eval/t5_events.py [--seed]` | Event forecasting (optional task): CAN GRU baseline |
| llm_t5_gru_subsample.json | `code/model/llm_t5_data.py` | App. L / Table 12: trained GRU re-scored on the fixed anchor subsample |
| llm_t5_qwen3-4b.json, llm_t5_qwen2.5-vl-3b.json, llm_t5_qwen3-vl-4b.json | `code/model/llm_t5_zeroshot.py --model ...` | App. L / Table 12: zero-shot LLM/VLM event-forecasting rows |
| m3_t5_video*.json | `code/model/m3_t5_video.py --video {dinov2,dinov3,siglip2,vjepa2}` | App. L: event forecasting with video cross-attention (`m3_t5_video.json` is an earlier run without the `--video` suffix) |
| vlm_maneuver_qwen2.5-vl-3b.json, vlm_maneuver_qwen3-vl-4b.json | `code/model/vlm_maneuver_cls.py --model ...` | App. D: zero-shot VLM maneuver classification on verified events |
| e1_cross_vehicle.json | `code/eval/e1_cross_vehicle.py` | App. G: within- vs. cross-vehicle verification (driver ids pseudonymised) |
| e2_missing_channel.json | `code/eval/e2_missing_channel.py` | App. G / Table 9: missing-channel robustness |
| e7_threshold_sensitivity.json | `code/eval/e7_threshold_sensitivity.py` | App. C: primitive-threshold sensitivity (Q75/Q85) |
| m5_pair_agreement.json, m5_pair_agreement_qwen3.json | summaries computed from `code/preprocessing/m5_vlm_attrs.py` outputs (no script) | App. F: VLM scene-attribute agreement of matched vs. random pairs |
| vlm_qwen3_audit.json | `code/preprocessing/m5_qwen3_audit.py` | App. F: cross-generation VLM attribute audit |
| t1_audit_answers.json | browser export of the audit interface (`code/analysis/make_audit_package.py`, `summarize_audit.py`) | App. C: 240-window primitive audit (93.0% agreement) |
| RESULTS.md | hand-written running summary | working notes kept with the results |

## Files in this folder

```
e1_cross_vehicle.json
e2_missing_channel.json
e3_e4_distribution.json
e5_descriptor_pop.json
e7_threshold_sensitivity.json
llm_t2_llama-3.2-3b.json
llm_t2_qwen3-4b.json
llm_t5_gru_subsample.json
llm_t5_qwen2.5-vl-3b.json
llm_t5_qwen3-4b.json
llm_t5_qwen3-vl-4b.json
m2_mcpp6_dinov2_v2.json
m2_mcpp6_dinov3.json
m2_mcpp6_vjepa2.json
m2_mcpp6_vjepa2_s10.json
m2_mcpp6_vjepa2_s11.json
m3_t5_video.json
m3_t5_video_dinov2.json
m3_t5_video_dinov3.json
m3_t5_video_siglip2.json
m3_t5_video_vjepa2.json
m4_clip_align.json
m5_pair_agreement.json
m5_pair_agreement_qwen3.json
m6_transfuser_lite.json
m7_contrast.json
m7_video_probe.json
t1_audit_answers.json
t3_all.json
t3_seeds.json
t4_results.json
t4_results_s10.json
t4_results_s11.json
t5_results.json
v3_moment.json
vlm_maneuver_qwen2.5-vl-3b.json
vlm_maneuver_qwen3-vl-4b.json
vlm_qwen3_audit.json
w1_arcface.json
w1_arcface_s10.json
w1_arcface_s11.json
w1_ci.json
w1_ci_s10.json
w1_ci_s11.json
w1_itr.json
w1_itr_s10.json
w1_itr_s11.json
w2_jepa.json
w2_masked.json
RESULTS.md
```
