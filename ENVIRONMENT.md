# Environment used for the paper

All results in the paper and in `results/` were produced with this environment.

| Component | Version |
|---|---|
| OS / CPU | Ubuntu 24.04 (Linux 7.0), x86_64, glibc 2.39 |
| GPU | 1x NVIDIA GeForce RTX 5080 (16 GB), CUDA 12.8, cuDNN 9.19 |
| Python | 3.12.3 |
| numpy / pandas / pyarrow | 2.5.2 / 2.2.2 / 17.0.0 |
| scipy / scikit-learn | 1.14.0 / 1.5.2 |
| torch / torchvision | 2.12.0.dev20260226+cu128 / 0.26.0.dev20260227+cu128 (PyTorch nightly, CUDA 12.8) |
| transformers / accelerate / huggingface-hub / tokenizers / safetensors | 5.13.0 / 1.14.0 / 1.27.0 / 0.22.2 / 0.8.0 |
| momentfm | 0.1.4 |
| Pillow | 11.2.1 |
| ffmpeg (CLI, video embedding and VLM scripts) | 6.1.1 |
| zstandard / pycapnp (raw-log decoding) | 0.23.0 / 2.0.0 |
| openpilot (raw-log decoding only) | checkout `b644555a1` (2024-08-20), imported as `openpilot.tools.lib.logreader` |

Raw openpilot logs are not part of the public release; `code/preprocessing/rlog_extract.py`, `decode_corpus.py`, `decode_cache.py` and `eval/curvature_correctness.py` are included for completeness and require the raw logs plus an openpilot checkout on `PYTHONPATH`.

Pretrained weights downloaded at run time (Hugging Face Hub / torch.hub): `facebook/dinov2` (torch.hub), `facebook/dinov3-vitb16-pretrain-lvd1689m` (gated), `google/siglip2-base-patch16-256`, `facebook/vjepa2-vitl-fpc64-256`, `AutonLab/MOMENT-1-large`, `Qwen/Qwen3-1.7B`, `Qwen/Qwen3-4B`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-VL-3B-Instruct`, `Qwen/Qwen3-VL-4B-Instruct`, `meta-llama/Llama-3.2-3B-Instruct` (gated).

Seeds: training scripts accept `--seed` (default seed plus seeds 10 and 11 give the three-seed error bars; checkpoints and result files carry the `_s10` / `_s11` suffix). Evaluation manifests are fixed and independent of the seed.
