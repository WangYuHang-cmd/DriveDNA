<div align="center">

# 🧬 DriveDNA

### A Large-Scale Multimodal Naturalistic Driving Dataset and Benchmark for Driving Style Identification

[![arXiv](https://img.shields.io/badge/arXiv-2607.23822-b31b1b?logo=arxiv)](https://arxiv.org/abs/2607.23822) [![HF [![HF Paper](https://img.shields.io/badge/%F0%9F%A4%97%20Papers-2607.23822-ffcc4d)](https://huggingface.co/papers/2607.23822)
[![License](https://img.shields.io/badge/license-research--only-blue)](#-ethics--privacy)
[![Benchmark](https://img.shields.io/badge/tasks-3%20core%20%2B%202%20optional-45a49b)](#-benchmark-tasks--splits)
[![Baselines](https://img.shields.io/badge/baselines-30%20configurations-7189b9)](#-key-results)

<img src="https://huggingface.co/datasets/HenryYHW/DriveDNA/resolve/main/assets/teaser.png" alt="DriveDNA teaser" width="92%"/>

*Recognizing a driver is not the same as capturing driving style — DriveDNA makes vehicle, route, and driving-condition shortcuts measurable.*

</div>

---

## 📌 TL;DR

**DriveDNA** turns a large, in-the-wild naturalistic driving corpus into a benchmark for **personalized driving style**: representing *who* is driving as distinct from *what* they are driving and *where*. It pairs time-synchronized **CAN telemetry (10 Hz)** and **forward-road video** across hundreds of drivers and vehicle models, retains **only human-controlled driving** (automation-engaged frames removed), and ships a frozen evaluation protocol whose central question is:

> *Does a model recognize **how a person drives** — or merely **which car they own, which roads they frequent, and which conditions they encounter**?*

**Why it's unique.** Public personalized-style resources are small and hold vehicle/route fixed (e.g., PDB: 12 drivers, one car), while large AV datasets (nuScenes, Waymo) carry no persistent driver identity. DriveDNA is the first public corpus combining **many drivers × many vehicles × multi-session CAN+video**, with clean **human-vs-automation separation** and **explicit confound diagnostics**.

## ✨ Highlights

| | |
|---|---|
| 🧑‍✈️ **Drivers** | **465** persistent, salted-hashed identities, consistent across vehicles |
| 🚗 **Vehicle models** | **115** across **26 brands** — 392 drivers share a model with another driver; 22 drivers appear on 2+ models |
| 🛣️ **Drives** | **4,121** decoded drives (Mar 2023 – Jul 2026, multi-continent) |
| ⏱️ **Human-controlled driving** | **975 h** total, **581 h** in motion, at 10 Hz with forward video |
| 🪟 **Benchmark windows** | **62,674** tagged 60-s windows from 428 drivers (355 in frozen folds) |
| 🏷️ **Annotations** | 6 driving scenarios · 8 behavioral primitives (93.0% audit agreement) · **276,248 maneuver events** incl. **22,322 individually verified lane changes** |
| 🧪 **Protocol** | Driver-disjoint splits · 3-seed error bars · frozen evaluation manifests · leakage probes |

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

The main driver-disjoint split is **212 train / 45 val / 45 test**, plus a **53-driver few-shot hold-out** (support and query always from different drives). Additional frozen manifests isolate generalization sources: **within-nameplate** (same model, different drivers, 24 models), **cross-vehicle** (same driver, different vehicles), **condition-matched pairs** (14,868), and **missing-channel** robustness.

## 📊 Key Results

| Finding | Evidence |
|---|---|
| Learned representations ≫ classical descriptors | AUROC **.935** vs **.707** on unseen drivers |
| Driver signal survives condition matching | **.811 ± .006** on 14,868 matched pairs (descriptors → **.550**, chance) |
| High re-ID ≠ driving style | Video-only probe hits .937 re-ID but predicts **route at 347× chance**; collapses to .675 under matching |
| Recognition ≠ prediction | Best re-ID embedding yields **no** prediction gain (−0.2%); task-aligned FiLM conditioning does (+0.4 to +1.4%) |
| Foundation models need adaptation | Zero-shot LLM/TS/VLM rows land at/below the descriptor level; 1-epoch LoRA lifts Qwen3-8B to .871 on event forecasting |

*30 baseline configurations across five families — representation learning, shortcut robustness, personalization, multimodal modeling, distributional prediction — under one fixed multi-seed protocol.*

## 🗂️ This Repository

```
DriveDNA/
├── code/
│   ├── preprocessing/   # decode, human-driving extraction, scenario/primitive tagging, splits
│   ├── eval/            # harness.py (protocol + metrics + leakage probes), diagnostics
│   ├── model/           # all 30 baseline configurations (S1–S5, W-, M-, V-wave)
│   └── analysis/        # audits, examples, release tooling
├── figs_making/         # paper figure scripts
└── README.md
```

**Data lives on Hugging Face** (this repo is code-only): https://huggingface.co/datasets/HenryYHW/DriveDNA
Paths in scripts are relative to a `DriveDNA/` working root with `data/` from the HF release.

## 📦 What's Released

| Tier | Contents |
|---|---|
| **Public** (this repo) | De-identified 10 Hz signal tables · frozen video embeddings (DINOv2/DINOv3/SigLIP2/V-JEPA 2) · all split manifests · VLM scene attributes · evaluation harness · baseline training code |
| **Gated** (DUA) | Raw forward video (faces/plates blurred), research use only |

> The public tier alone reproduces **every number in the paper**.

Planned public-tier layout:

```
DriveDNA/
├── data/
│   ├── segments/windows.parquet        # 62,674 windows: driver, model, scenario, primitives, stats
│   ├── segments/windows_x.npy          # [62674, 600, 17] 10 Hz CAN windows
│   ├── segments/maneuver_events.parquet# 276,248 events (6 classes, verified lane changes flagged)
│   ├── splits/                         # driver_folds / within_vehicle / cross_vehicle / matched pairs
│   └── embeddings/                     # frozen DINOv2 / DINOv3 / SigLIP2 / V-JEPA 2 features
├── code/
│   ├── eval/harness.py                 # enrollment protocol, metrics, distribution distances, leakage probes
│   └── model/                          # all 30 baseline configurations
└── README.md
```

## 🔒 Ethics & Privacy

- Collected from community drivers with **informed consent** and compensation; follows source-platform terms.
- Driver identifiers are **salted hashes**; VINs, device identifiers, and **GPS coordinates removed**; no cabin video/audio; faces and plates blurred in the gated video tier.
- Leakage probes ship *as part of the benchmark* — users are asked to report leakage alongside utility.
- **Prohibited**: re-identification attempts; insurance, employment, or law-enforcement scoring of individuals.
- A takedown contact allows any driver to request removal from future versions.

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
- 💻 **Code & harness**: included in this repository
- 📦 **Data files**: uploading in stages — signal tables and manifests first, embeddings next
- ✉️ **Contact**: haozhou1@usf.edu

---

<div align="center"><sub>DriveDNA · University of South Florida &amp; University of Arizona · 2026</sub></div>
