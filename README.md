# ReferGaussian: Referring Segmentation in 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](#environment-setup)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-orange.svg)](#environment-setup)

[Project Page](https://trump0412.github.io/ReferGaussian/) | [arXiv (Coming Soon)](https://arxiv.org/abs/XXXX.XXXXX) | [Citation](#citation) | [Dataset (HuggingFace)](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

</div>

**Authors:** Bangpu Chen, Yaxuan Li, Shirui Peng, Xiangtian Si, Chu Liuxin, Xitong Cao, Hongbo Jin, Jiayu Ding

---

**ReferGaussian** is a unified framework for Referring Segmentation in 4D Gaussian Splatting (R4DGS): grounding natural-language queries in dynamic 4D scenes without retraining scene representations.

<p align="center">
  <img src="docs/assets/Fig3.png" width="90%" alt="ReferGaussian framework overview"/>
  <br>
  <em>Dynamic reconstruction builds the 4D Gaussian scene. A Qwen-based Refer-Planner drives static segmentation, semantic assignment (EntityBank), and training-free spatiotemporal grounding.</em>
</p>

## Results

### R4D-Bench-QA — joint referring + reconstruction

| Method | Acc ↑ | vIoU ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---|---|---|---|---|
| Segment then Splat | 55.6 | 28.4 | 20.3208 | 0.7027 | 0.3971 |
| 4D LangSplat | 58.4 | 32.1 | 20.3208 | 0.7027 | 0.3971 |
| ReferGaussian (Ours) | **76.5** | **34.4** | **20.4159** | **0.7069** | **0.3942** |

### Generalization — 4D LangSplat HyperNeRF split

| Method | Acc ↑ | vIoU ↑ |
|---|---|---|
| LangSplat | 54.27 | 24.13 |
| Deformable CLIP | 65.01 | 45.37 |
| Non-Status Field | 84.58 | 62.00 |
| 4D LangSplat | 88.86 | 66.14 |
| ReferGaussian (Ours) | **91.62** | **66.48** |

### Module ablation — R4D-Bench-QA

| Variant | Acc ↑ | vIoU ↑ |
|---|---|---|
| 4DGS reconstruction (no HyperGS) | 62.9 | 31.5 |
| w/o Stage 1 static segmentation | 48.6 | 17.2 |
| w/o Stage 2 semantic assignment | 62.9 | 29.8 |
| w/o Stage 3 spatio-temporal reasoning | 36.0 | 26.1 |
| ReferGaussian (full) | **76.5** | **34.4** |

### Reconstruction — keyboard scene (appendix)

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ | Train time ↓ | FPS ↑ | Storage (MB) ↓ |
|---|---|---|---|---|---|---|
| 4D Gaussian Splatting | 27.3584 | 0.8571 | 0.2920 | **927 s** | 5.75 | **1214** |
| ReferGaussian (Ours) | **28.4051** | **0.8867** | **0.2072** | 1023 s | **7.09** | 1267 |

### Qualitative results

<p align="center">
  <img src="docs/assets/Fig5_n.png" width="90%" alt="Qualitative comparison"/>
  <br>
  <em>Temporal-state and exclusion queries on R4D-Bench-QA. Rows: RGB, ground truth, ReferGaussian, Segment then Splat, 4D LangSplat.</em>
</p>

<p align="center">
  <img src="docs/assets/appen_fig1.png" width="90%" alt="Additional qualitative results"/>
  <br>
  <em>Additional results across multi-target, reasoning-intensive, and zero-target queries.</em>
</p>

---

## Environment Setup

**Requirements:** CUDA 12.1, Miniconda

```bash
git clone https://github.com/Trump0412/ReferGaussian.git
cd ReferGaussian

# Fetch external dependencies (4DGaussians + Grounded-SAM-2)
bash scripts/bootstrap_external.sh

# Main environment (training / rendering / evaluation)
bash scripts/setup_baseline_env.sh cuda121

# Semantic pipeline (Grounded-SAM2)
bash scripts/setup_grounded_sam2.sh
```

By default, environments and caches are created under `~/.cache/refergaussian/`:
- `~/.cache/refergaussian/conda-envs`
- `~/.cache/refergaussian/conda-pkgs`
- `~/.cache/refergaussian/pip`

Override with `GS4D_ENV_ROOT`, `GS4D_CONDA_PKGS_DIRS`, and `GS4D_PIP_CACHE_DIR`.

> **Note on COLMAP:** `prepare_hypernerf.sh` requires COLMAP to generate the initial point cloud for each scene. Install it with `apt install colmap` or from [colmap.github.io](https://colmap.github.io/install.html) before running data preparation.

## Model Weights

### Qwen3-VL-8B-Instruct (Refer-Planner)

The Refer-Planner uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct). Download before referring evaluation:

```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='models/Qwen3-VL-8B-Instruct')
"
```

Default path is `models/Qwen3-VL-8B-Instruct/` (relative to repo root). Override with:

```bash
export REFERGAUSSIAN_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
# Backward compatibility (legacy name also supported):
# export HYPERGAUSSIAN_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

### SAM2 and Grounding DINO (Grounded-SAM2 pipeline)

Downloaded automatically during `bash scripts/setup_grounded_sam2.sh`:
- `facebook/sam2-hiera-large`
- `IDEA-Research/grounding-dino-base`

If needed, set `HF_ENDPOINT=https://hf-mirror.com` before setup.

## Dataset Setup

### HyperNeRF

Download scenes used in the paper from the [HyperNeRF release page](https://github.com/google/hypernerf/releases/tag/v0.1):

```bash
bash scripts/prepare_hypernerf.sh
```

Prepare a single scene:

```bash
bash scripts/prepare_hypernerf.sh misc keyboard
```

Register a local scene:

```bash
bash scripts/prepare_local_hypernerf_scene.sh /path/to/scene <group> <scene>
```

### 4DLangSplat annotations

```bash
bash scripts/download_4dlangsplat_annotations.sh
```

Downloads to `data/benchmarks/4dlangsplat/HyperNeRF-Annotation/`.

### R4D-Bench-QA

```bash
bash scripts/download_r4d_bench_qa.sh
```

Downloads to `data/benchmarks/r4d_bench_qa/`.

Dataset link: [https://huggingface.co/datasets/LiYacheng/r4d-bench-qa](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

## Training

```bash
# Example: keyboard scene
bash scripts/train.sh hypernerf misc/keyboard
```

Output is written to `runs/refergaussian/hypernerf/keyboard/`.

## Evaluation

### Reconstruction metrics

```bash
bash scripts/eval.sh hypernerf misc/keyboard
```

Writes PSNR / SSIM / LPIPS to `runs/refergaussian/hypernerf/keyboard/metrics.log`.

### Referring evaluation — 4DLangSplat public protocol

Run per scene (example: `split-cookie`):

```bash
SCENE=split-cookie
GROUP=misc
RUN_DIR=runs/refergaussian/hypernerf/${SCENE}
DATASET_DIR=data/hypernerf/${GROUP}/${SCENE}
PROTOCOL_JSON=data/benchmarks/4dlangsplat/HyperNeRF-Annotation/${SCENE}/protocol.json
ANNOT_DIR=data/benchmarks/4dlangsplat/HyperNeRF-Annotation/${SCENE}

bash scripts/run_public_query_protocol.sh \
  "${PROTOCOL_JSON}" \
  "${RUN_DIR}" \
  "${DATASET_DIR}"

source scripts/common.sh
gs_python scripts/evaluate_public_query_protocol.py \
  --protocol-json "${PROTOCOL_JSON}" \
  --annotation-dir "${ANNOT_DIR}" \
  --dataset-dir "${DATASET_DIR}" \
  --query-root "${RUN_DIR}/entitybank/query_guided" \
  --output-json reports/public_eval_${SCENE}.json
```

Reproducible profile switch:

```bash
# Public release default behavior
export QUERY_EVAL_PROFILE=default

# Current vIoU-repair profile used by the latest reproducibility runbook
export QUERY_EVAL_PROFILE=public_time_shape_v4_recall
```

The active profile and its effective fusion parameters are written into each
`final_query_render_sourcebg/validation.json` as `eval_profile` and `fusion_options`.
This makes public reruns auditable and easy to compare.

For paper-style batched reruns, prefer the manifest-based runner so query ids, output roots,
and evaluator maps stay aligned:

```bash
OUT=reports/public_time_shape_v4_recall
source scripts/common.sh
gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --query-set time_sensitive \
  --profile public_time_shape_v4_recall \
  --gpus 0 1 2

gs_python scripts/preflight_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --gpu 0 1 2 \
  --create-output-root

gs_python scripts/run_query_batch_two_gpu.py \
  --manifest "${OUT}/manifest.jsonl" \
  --profile public_time_shape_v4_recall \
  --gpu 0 1 2 \
  --force-rerun \
  --timeout 10800
```

### Referring evaluation — R4D-Bench-QA

```bash
# Optional: profile switch for reproducible ablation
# export QUERY_EVAL_PROFILE=default
# export QUERY_EVAL_PROFILE=public_time_shape_v4_recall

# 1) Build a manifest with official query ids.
source scripts/common.sh
RUN_ROOT=reports/r4d_bench_public_time_shape_v4
gs_python scripts/build_r4d_query_manifest.py \
  --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
  --scenes cut_lemon split_cookie torchchocolate coffee_martini \
  --output "${RUN_ROOT}/manifest.jsonl" \
  --output-root "${RUN_ROOT}/query_root" \
  --gpus 0 1 2

gs_python scripts/preflight_query_batch.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --gpu 0 1 2 \
  --create-output-root

# 2) Run the query pipeline.
gs_python scripts/run_query_batch_two_gpu.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --profile public_time_shape_v4_recall \
  --gpu 0 1 2 \
  --force-rerun \
  --timeout 10800

# 3) Re-evaluate from saved query outputs (no rerun of model inference).
gs_python scripts/evaluate_ours_benchmark.py \
  --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
  --query-root-map "${RUN_ROOT}/query_root_map.json" \
  --dataset-dir-map "${RUN_ROOT}/dataset_dir_map.json" \
  --output-json "${RUN_ROOT}/official_eval.json" \
  --output-md "${RUN_ROOT}/official_eval.md" \
  --skip-missing
```

Output files:
- `${RUN_ROOT}/query_root_map.json`
- `${RUN_ROOT}/dataset_dir_map.json`
- `${RUN_ROOT}/official_eval.json` (per-query Acc/vIoU/tIoU + summary)
- `${RUN_ROOT}/official_eval.md`

Metric rule used in this repository:
- Empty-set rule is enabled in evaluator: if GT and prediction are both empty (temporal union = 0), `vIoU = 1.0` and `tIoU = 1.0`.

Important benchmark note:
- Current HF `LiYacheng/r4d-bench-qa` artifacts provide dense GT for 36-query and 89-query tiers.
- The larger language-only extension set does not always include dense masks, so strict `vIoU`/`tIoU` cannot be computed for those entries without extra GT alignment files.
- Prefer the official query metadata file `R4D-Bench_queries.json` as the source of truth for query text.
  Avoid retyping benchmark queries in ad-hoc shell scripts, because even small wording drift can change the target object set.

Filter the official R4D-Bench query list by scene before running a subset benchmark:

```bash
python scripts/filter_r4d_benchmark_queries.py \
  --input-json data/benchmarks/r4d_bench_qa/evaluation/R4D-Bench_queries.json \
  --output-json reports/r4d_bench_queries_10scene.json \
  --output-md reports/r4d_bench_queries_10scene.md \
  --exclude-scenes americano cut_roasted_beef
```

This produces the 10-scene / 72-query subset used when excluding the weakest reconstruction scenes.

## Repository Layout

```
ReferGaussian/
├── refergaussian/         # core library
│   ├── temporal/          # 4D Gaussian reconstruction and time warp
│   ├── entitybank/        # entity-centric scene memory
│   └── semantics/         # Refer-Planner: query decomposition and grounding
├── scripts/               # training, evaluation, data prep, dependency bootstrap
├── configs/               # scene and benchmark configurations
├── external/              # fetched by scripts/bootstrap_external.sh (not tracked)
├── data/                  # datasets (not tracked)
├── runs/                  # experiment outputs (not tracked)
└── docs/                  # project page
```

## Citation

Replace the placeholder below before final release:
- `XXXX.XXXXX` -> your final arXiv id

```bibtex
@article{refergaussian2026,
  title     = {ReferGaussian: Referring Segmentation in 4D Gaussian Splatting},
  author    = {Bangpu Chen and Yaxuan Li and Shirui Peng and Xiangtian Si and Chu Liuxin and Xitong Cao and Hongbo Jin and Jiayu Ding},
  journal   = {arXiv preprint arXiv:XXXX.XXXXX},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## Acknowledgements

This project builds on [4DGaussians](https://github.com/hustvl/4DGaussians), [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), and [Qwen](https://github.com/QwenLM/Qwen).
