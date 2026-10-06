<div align="center">

# 🧬 DriveDNA

### A Large-Scale Multimodal Naturalistic Driving Dataset and Benchmark for Driving Style Identification

[![arXiv](https://img.shields.io/badge/arXiv-2607.23822-b31b1b?logo=arxiv)](https://arxiv.org/abs/2607.23822) [![HF Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Papers-2607.23822-ffcc4d)](https://huggingface.co/papers/2607.23822) [![HF Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-DriveDNA-ffcc4d)](https://huggingface.co/datasets/HenryYHW/DriveDNA) [![HF Models](https://img.shields.io/badge/%F0%9F%A4%97%20Models-DriveDNA--models-ffcc4d)](https://huggingface.co/HenryYHW/DriveDNA-models)
[![License](https://img.shields.io/badge/license-research--only-blue)](#-license)
[![Benchmark](https://img.shields.io/badge/tasks-3%20core%20%2B%202%20optional-45a49b)](#-benchmark-tasks--splits)
[![Baselines](https://img.shields.io/badge/baselines-30%20configurations-7189b9)](#-key-results)

<img src="https://huggingface.co/datasets/HenryYHW/DriveDNA/resolve/main/assets/teaser.png" alt="DriveDNA teaser" width="92%"/>

*Recognizing a driver is not the same as capturing driving style — DriveDNA makes vehicle, route, and driving-condition shortcuts measurable.*

</div>

---

## 📌 TL;DR

**DriveDNA** turns a large, in-the-wild naturalistic driving corpus into a benchmark for **personalized driving style**: representing *who* is driving as distinct from *what* they are driving and *where*. It pairs time-synchronized **CAN telemetry (10 Hz)** and **forward-road video** across hundreds of drivers and vehicle models, retains **only human-controlled driving** (automation-engaged frames are flagged and excluded from the benchmark), and ships the **leakage probes** needed to tell driver-specific signal from vehicle, route and condition shortcuts.

> *Does a model recognize **how a person drives** — or merely **which car they own, which roads they frequent, and which conditions they encounter**?*

This repository holds the **evaluation harness, every baseline trained in the paper's harness, the preprocessing pipeline, the result files behind the paper's tables (coverage listed in `results/INDEX.md`), and the exact environment**. The data live on Hugging Face; the trained checkpoints live in a companion model repository (see [Resources](#-resources)).

## ✨ Highlights

| | |
|---|---|
| 🧑‍✈️ **Drivers** | **460** released drivers with persistent pseudonymous identities (`driver_001` … `driver_460`), consistent across vehicles; the paper's corpus statistics cite 465 decoded drivers |
| 🚗 **Vehicle models** | **115** nameplate-level models across **26 brands** (paper consolidation); driver-sharing counts per cohort are in [Counts and cohorts](#-counts-and-cohorts) |
| 🛣️ **Drives** | **4,121** decoded drives (Mar 2023 – Jul 2026, multi-continent) |
| ⏱️ **Human-controlled driving** | **975 h** total, **581 h** in motion, at 10 Hz with forward video |
| 🪟 **Benchmark windows** | **62,674** tagged 60-s windows from 428 drivers (355 in frozen folds) |
| 🏷️ **Annotations** | 6 driving scenarios · 8 behavioral primitives (93.0% audit agreement) · **276,248 maneuver events** incl. **22,322 individually verified lane changes** |
| 🧪 **Protocol** | Driver-disjoint splits · 3-seed error bars · frozen evaluation manifests · leakage probes |
| 🚘 **Companion set** | **DriveDNA-Controlled**: 14 additional drivers, one physical Honda Civic, largely shared roads (same format, outside every fold) |

## 📡 Modalities & Committed Signals

All streams are decoded from openpilot logs and resampled to a unified **10 Hz** grid:

| Signal | Meaning | Style construct |
|---|---|---|
| `vEgo`, `aEgo` (+ jerk) | speed, longitudinal accel | longitudinal aggressiveness |
| `steeringAngleDeg`, `steeringRateDeg` | **driver steering INPUT** (vehicle-dependent via steer ratio) | steering entropy, reversal rate |
| **`actual_curvature`** | **realized path curvature** (vehicle-normalized) | cornering sharpness, path geometry |
| `yaw_rate` → `curv_measured` | independently-sensed turning | aggressiveness, slip |
| `leadOne_dRel/vRel/status` | lead-vehicle distance & relative speed (radar) | THW, TTC, gap preference |
| `gas`, `brake` (+ pressed) | pedal application (subset of fleet) | pedal dynamics |
| `laneLeft_y`, `laneRight_y` | lane offsets | lane-keeping (SDLP) |

**Key distinction — steering INPUT vs realized PATH.** `steeringAngleDeg` is the raw wheel input and is *vehicle-dependent*; `actual_curvature` is the *vehicle-normalized* realized path. Their gap is a signal-level handle on the "who vs which-car" question at the heart of the benchmark: vehicle-model probes read **2.3× chance from steering angle but ≈chance from realized curvature**.

## 🎯 Benchmark Tasks & Splits

| Task | Input → Output | Metrics |
|---|---|---|
| **Driver re-identification** (core) | k-min support → driver identity | Top-k, AUROC, EER |
| **Personalized behavior prediction** (core) | 5-s history → 1–5-s future motion | RMSE, PG, MMD/KL/W1 |
| **Condition-matched comparison** (core) | matched window pair → same driver? | AUROC, EER |
| Event forecasting (optional) | 5-s history → event in 1–5 s | AP, AUROC, lead time |
| Style explanation (optional) | event window → category + evidence | accuracy (exploratory) |

The main driver-disjoint split is **212 train / 45 val / 45 test**, plus a **53-driver few-shot hold-out** (support and query always from different drives). Additional frozen manifests isolate generalization sources: **within-nameplate** (same model, different drivers), **cross-vehicle** (same driver, different vehicles), **condition-matched pairs** (14,868), and **missing-channel** robustness. All manifests are shipped with the data (`splits/`) and are never re-drawn at evaluation time.

## 📊 Key Results

| Finding | Evidence |
|---|---|
| Learned representations ≫ classical descriptors | AUROC **.935** vs **.707** on unseen drivers |
| Driver signal survives condition matching | **.811 ± .006** on 14,868 matched pairs (descriptors → **.550**, chance) |
| High re-ID ≠ driving style | Video-only probe hits .937 re-ID but predicts **route at 347× chance**; collapses to .675 under matching |
| Recognition ≠ prediction | Best re-ID embedding yields **no** prediction gain (−0.2%); task-aligned FiLM conditioning does (+0.4 to +1.4%) |
| Foundation models need adaptation | Zero-shot LLM/TS/VLM rows land at/below the descriptor level; 1-epoch LoRA lifts Qwen3-8B to .871 on event forecasting |

*30 baseline configurations across five families — representation learning, shortcut robustness, personalization, multimodal modeling, distributional prediction — under one fixed multi-seed protocol.* The JSON behind the reported rows is in [`results/`](results/INDEX.md), which also lists the rows that come from single external runs.

## 🔗 Resources

| Resource | Where | Access |
|---|---|---|
| Code, harness, baselines, results | this repository, tag `v1.1-kdd2027` | research-only license (see below) |
| DriveDNA (signals, features, embeddings, splits, forward video) | [HenryYHW/DriveDNA](https://huggingface.co/datasets/HenryYHW/DriveDNA) | gated, granted on request |
| DriveDNA-Sample (small) | [HenryYHW/DriveDNA-Sample](https://huggingface.co/datasets/HenryYHW/DriveDNA-Sample) | open after accepting the terms (automatic approval) |
| DriveDNA-Controlled (14 drivers, one car, same format) | [HenryYHW/DriveDNA-Controlled](https://huggingface.co/datasets/HenryYHW/DriveDNA-Controlled) | gated, granted on request |
| Trained checkpoints (47 files, 162 MB) | [HenryYHW/DriveDNA-models](https://huggingface.co/HenryYHW/DriveDNA-models) | gated, granted on request |

## 📌 Frozen versions

| Artifact | Identifier |
|---|---|
| this repository | tag `v1.1-kdd2027` |
| HenryYHW/DriveDNA | data files frozen at revision `1f9c67170db726c977686ebe6ca86284f2aed444`; later commits (`b6ac457b` onward) change only the dataset card and LICENSE and remove 50 duplicate `.ts` segments and a stale code copy, so `hf download` of the current head yields the same data files |
| HenryYHW/DriveDNA-Controlled | revision `1d24e058d74ca86965495e0458ac585ec82d3b37` |
| HenryYHW/DriveDNA-models | revision `b0d7ab6713296ca90933eb078c4ab8c3ad821395` |
| checksums | [`CHECKSUMS.sha256`](CHECKSUMS.sha256): sha256 of all 47 checkpoints and of every file in `results/` |

## 🔢 Counts and cohorts

Different numbers on the paper and the public pages referred to different cohorts. With the released nameplate-level consolidation (`code/preprocessing/model_merge.py`, function `canon`) they are:

| Cohort | Drivers | Share a vehicle model with ≥1 other driver | On two or more models |
|---|---:|---:|---:|
| Benchmark window cohort (`data/windows.parquet`, 62,674 windows) | 428 | 392 | 20 (7 admit model-matched negatives, Appendix G) |
| Full release index (`index/drives.parquet`, 4,121 drives) | 460 | 426 | 22 |

The paper's count of 420 drivers sharing a model was computed on the 465-driver decoded corpus before the release index was frozen; the camera-ready version will use the release-index figures with these definitions. Placeholder vehicle labels (`UNKNOWN` in the index, `MOCK` in the windows table) are counted as a model in these figures, as in the paper; excluding them gives 411 / 18 (release index) and 379 / 18 (benchmark cohort), with 117 and 115 real nameplates respectively. The released cross-vehicle manifest (`splits/cross_vehicle.json`) lists the 16 drivers with drive-disjoint data on two or more models. Forward video exists for 3,983 drives (452 drivers) and the four embedding families for 3,891 of them. Both rows are reproducible from the released files:

```python
import pandas as pd, sys; sys.path.insert(0, "code/preprocessing"); from model_merge import canon
idx = pd.read_parquet("hf/DriveDNA/index/drives.parquet"); idx["model"] = idx.model_canon.map(canon)
models = idx.groupby("driver").model.agg(set); per_model = idx.groupby("model").driver.nunique()
print(sum(any(per_model[m] >= 2 for m in ms) for ms in models), (models.apply(len) >= 2).sum())   # 426 22
```

## 🗂️ This Repository

```
DriveDNA/
├── code/
│   ├── preprocessing/   # decode (needs raw logs) → human-driving extraction → scenario/primitive tagging → splits → bundle → video embeddings → matched pairs → maneuvers → VLM attributes
│   ├── eval/            # harness.py (enrollment protocol, metrics, distribution distances, leakage probes), task evaluators, appendix experiments
│   ├── model/           # all baseline configurations (S1–S5, W-, M-, V-wave, LLM/VLM rows)
│   └── analysis/        # annotation / audit tooling, LLM bundle export
├── results/             # result JSONs for the paper's tables (coverage and gaps in INDEX.md)
├── scripts/prepare_release_layout.py   # builds the working layout from the Hugging Face release
├── figs_making/         # paper figure scripts
├── CHECKSUMS.sha256 · ENVIRONMENT.md · requirements.txt · LICENSE
```

Scripts use paths relative to the repository root (`BASE = "."`); run them from the root after preparing the layout below.

## 🧰 Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt            # exact versions used for the paper: ENVIRONMENT.md
```

Python 3.12, PyTorch ≥ 2.4 with CUDA (a single 16 GB GPU is enough for every baseline), `ffmpeg` on `PATH` for the video-embedding and VLM scripts. Decoding raw openpilot logs additionally needs an openpilot checkout on `PYTHONPATH` (details in `ENVIRONMENT.md`); the raw logs are not part of the public release, so the `decode_*` / `rlog_extract.py` steps are documented for completeness only.

## 📦 Getting the data and preparing the layout

```bash
pip install -U huggingface_hub && hf auth login
hf download HenryYHW/DriveDNA        --repo-type dataset --local-dir ./hf/DriveDNA
hf download HenryYHW/DriveDNA-models --repo-type model   --local-dir ./hf/DriveDNA-models
python scripts/prepare_release_layout.py --data ./hf/DriveDNA --models ./hf/DriveDNA-models
sha256sum -c CHECKSUMS.sha256
```

The scripts were written against an internal layout and the column names `driver` / `route` / `model`; the public release uses `driver` / `drive` / `model_canon`. `prepare_release_layout.py` creates the expected tree in place (tables are copied with alias columns; arrays, embeddings and checkpoints are symlinked):

| Script expects | Built from (Hugging Face release) |
|---|---|
| `data/segments/windows.parquet` | `data/windows.parquet` |
| `data/segments/segments.parquet` | `index/human_segments.parquet` |
| `data/segments/{maneuver_events,t5_llm_subsample,lane_changes_verified}.parquet` | `data/*.parquet` |
| `data/splits/*.json`, `data/splits/matched_context_pairs.parquet` | `splits/*.json`, `splits/matched_condition_pairs.parquet` |
| `data/colab_bundle/{windows_x,windows_mask,windows_vid*}.npy`, `channels.json`, `driver_folds.json` | `features/*`, `splits/driver_folds.json` |
| `data/cache/<model>/<driver>__<drive>.parquet` | `raw_signals/<driver>/<drive>.parquet` (model from `index/drives.parquet`) |
| `data/video_emb{,_dinov3,_siglip2,_vjepa2}/<model>/<driver>__<drive>.npz` | `embeddings/{dinov2,dinov3,siglip2,vjepa2}/<driver>/<drive>.npz` |
| `results/vlm_attrs*.jsonl` | `data/vlm_attrs*.jsonl` |
| `experiments/checkpoints/*.pt` | the model repository |

Two inputs are not released and the corresponding scripts will not run: the raw openpilot logs (`../Dataset`, used only by the decoding steps) and the decoding progress log read by `code/eval/log_type_probe.py`.

## 🔁 Reproducing the paper

Run from the repository root with the layout prepared. Training scripts accept `--seed` (the paper reports the default seed and seeds 10 and 11); `--smoke` runs a short sanity pass. Each script writes its JSON into `results/` under the names listed in [`results/INDEX.md`](results/INDEX.md).

```bash
# Table 3 — driver re-identification (5-min enrollment, unseen drivers)
python code/model/s1_supcon.py [--seed 10]                     # PatchTST + SupCon → experiments/checkpoints/s1_supcon[_s10].pt
python code/model/w1_backbones.py --arch ci|itr|arcface [--seed 10]
python code/model/w2_ssl.py --mode masked|jepa
python code/model/v3_moment.py                                 # MOMENT-1 zero-shot
python code/model/llm_embed_t2.py --model qwen3-4b|llama-3.2-3b
python code/model/m4_clip_align.py                             # CLIP-aligned CAN (no labels)
python code/eval/e5_descriptor_pop.py                          # classical-descriptor anchor

# Table 4 — condition-matched comparison (14,868 pairs)
python code/eval/t4_eval.py [--ckpt experiments/checkpoints/s1_supcon_s10.pt --tag _s10]
python code/eval/t3_all.py [--moment] [--llm]                  # every representation under matching

# Sec. 6.4 / App. H, I, M — personalized behavior prediction
python code/model/s3_personalized.py [--arch transformer] [--seed 10]
python code/model/s5_mdn.py [--seed 10]  ;  python code/model/w5_cvae.py [--seed 10]
python code/eval/e3_e4_distribution.py                         # MMD/KL/W1 and per-driver gains

# Sec. 6.5 / 6.6 — leakage, modality and vehicle effects
python code/model/m7_video_probe.py ; python code/model/m7_contrast.py
python code/model/m2_mcpp6.py --video dinov2|dinov3|vjepa2 [--seed 10]
python code/model/m6_transfuser_lite.py ; python code/model/s4_mcpp.py ; python code/model/s1_dann.py
python code/eval/leakage_probe.py ; python code/eval/identifiability.py <MODEL>

# Event forecasting (optional task; App. L / Table 12)
python code/eval/t5_events.py ; python code/model/llm_t5_data.py
python code/model/llm_t5_zeroshot.py --model qwen3-4b|qwen2.5-vl-3b|qwen3-vl-4b
python code/model/m3_t5_video.py --video dinov2|dinov3|siglip2|vjepa2

# Appendices C, D, F, G
python code/eval/e7_threshold_sensitivity.py                   # App. C: primitive thresholds Q75/Q85
python code/model/vlm_maneuver_cls.py --model qwen3-vl-4b      # App. D: zero-shot VLM maneuver classification
python code/preprocessing/m5_vlm_attrs.py ; python code/preprocessing/m5_qwen3_audit.py   # App. F (needs forward video)
python code/eval/e1_cross_vehicle.py ; python code/eval/e2_missing_channel.py            # App. G, Table 9
```

Preprocessing pipeline (for reference; the first two steps need the raw logs): `rlog_extract.py` → `decode_corpus.py` → `model_merge.py` → `availability_audit.py` → `extract_segments.py` → `scenario_primitives.py` → `curate_splits.py` → `export_colab_bundle.py` → `embed_video*.py` → `export_window_video*.py` → `mine_matched_pairs.py` → `detect_maneuvers.py` → `m5_vlm_attrs.py`.

## 🧠 Checkpoints

All 47 checkpoints from the paper's runs (44 used in reported rows, 2 superseded and 1 unused, each flagged on the model card; re-identification encoders for three seeds, personalization predictors, distributional heads, multimodal and event-forecasting models) are in [HenryYHW/DriveDNA-models](https://huggingface.co/HenryYHW/DriveDNA-models), each with its training command and sha256. They are plain PyTorch state dicts with the input normalisation statistics:

```python
import torch, sys
sys.path.insert(0, "code/model"); sys.path.insert(0, "code/eval")
from s1_supcon import Encoder
ck = torch.load("experiments/checkpoints/s1_supcon.pt", map_location="cpu", weights_only=False)
enc = Encoder(c_in=17).eval(); enc.load_state_dict(ck["model"])
x = (windows_x - ck["mu"]) / ck["sd"]          # windows_x: [N, 600, 17] float32 from data/colab_bundle/windows_x.npy
```

## 🔒 Ethics & Privacy

- Collected from community drivers with **informed consent** and compensation; follows source-platform terms.
- Driver and drive identifiers are **sequential pseudonyms** (`driver_001`, `drive_001`); the private mapping is withheld by the authors. VINs, device identifiers, precise timestamps and **GPS coordinates are removed**; no cabin video or audio; faces and plates are blurred in the gated video tier.
- Leakage probes ship *as part of the benchmark* — users are asked to report leakage alongside utility.
- **Prohibited**: re-identification attempts; insurance, employment, or law-enforcement scoring of individuals.
- A takedown contact allows any driver to request removal from future versions.

## 📜 License

The code in this repository is released under the **DriveDNA Research License** ([LICENSE](LICENSE)): non-commercial research use, no re-identification or individual scoring, attribution required. The data and checkpoints are governed by the DriveDNA research license and data-use agreement on Hugging Face. (The first public revision of this repository, 2026-07-27, was released under MIT; that revision keeps its original terms.)

## 📖 Citation

```bibtex
@article{drivedna2026,
  title   = {DriveDNA: A Large-Scale Multimodal Naturalistic Driving Dataset and
             Benchmark for Driving Style Identification},
  author  = {Wang, Yuhang and Li, Lingyao and Zhou, Hao},
  journal = {arXiv preprint arXiv:2607.23822},
  year    = {2026}
}
```

## 🔗 Links & Status

- 📄 **Paper**: [arXiv:2607.23822](https://arxiv.org/abs/2607.23822) · [🤗 Papers page](https://huggingface.co/papers/2607.23822) · KDD 2027 Datasets & Benchmarks (under review)
- 💻 **Code & harness**: this repository (tag `v1.1-kdd2027`)
- 📦 **Data**: [DriveDNA](https://huggingface.co/datasets/HenryYHW/DriveDNA) · [DriveDNA-Sample](https://huggingface.co/datasets/HenryYHW/DriveDNA-Sample) · [DriveDNA-Controlled](https://huggingface.co/datasets/HenryYHW/DriveDNA-Controlled) · 🧠 [DriveDNA-models](https://huggingface.co/HenryYHW/DriveDNA-models)
- ✉️ **Contact**: haozhou1@usf.edu

---

<div align="center"><sub>DriveDNA · University of South Florida &amp; University of Arizona · 2026</sub></div>
