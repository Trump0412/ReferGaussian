# ReferGaussian: Referring Segmentation in 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](#environment-setup)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-orange.svg)](#environment-setup)
[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-8A2BE2.svg)](#citation)

[Project Page](https://trump0412.github.io/ReferGaussian/) | Paper (coming soon) | [Citation](#citation) | [Dataset (HuggingFace)](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

</div>

**Authors:** Bangpu Chen, Yaxuan Li, Shirui Peng, Xiangtian Si, Chu Liuxin, Xitong Cao, Hongbo Jin, Jiayu Ding

Accepted at **ACM Multimedia 2026**.

---

**ReferGaussian** is a unified framework for Referring Segmentation in 4D Gaussian Splatting (R4DGS): grounding natural-language queries in dynamic 4D scenes without retraining scene representations.

<p align="center">
  <img src="docs/assets/Fig3.png" width="90%" alt="ReferGaussian framework overview"/>
  <br>
  <em>Dynamic reconstruction builds the 4D Gaussian scene. A Qwen-based Refer-Planner drives static segmentation, semantic assignment (EntityBank), and training-free spatiotemporal grounding.</em>
</p>

## Results

The following tables are the accepted-paper results under the stated fixed
protocols. They are not partial reruns: a new release reproduction must cover
all expected queries, retain the manifest/evaluator output, and report
overall, non-empty-only, and zero-target outcomes separately.
See [the release reproduction protocol](docs/RELEASE_REPRODUCTION_PROTOCOL.md)
for the fixed scene/query sets and required artifacts.

### Paper results — R4D-Bench-QA referring segmentation

| Method | Acc ↑ | vIoU ↑ |
|---|---:|---:|
| Segment then Splat | 55.6 | 28.4 |
| 4D LangSplat | 58.4 | 32.1 |
| ReferGaussian (Ours) | **76.5** | **34.4** |

### Paper results — reconstruction on the fixed 8-scene protocol

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 4D Gaussian Splatting | 28.312 | 0.8753 | 0.2325 |
| ReferGaussian (Ours) | **28.486** | **0.8777** | **0.2233** |

The reconstruction table uses the fixed 8-scene release protocol. Results where either
method has PSNR below 20 dB are excluded from paper-facing reconstruction comparisons.

### Paper results — 4D LangSplat HyperNeRF split

| Method | Acc ↑ | vIoU ↑ |
|---|---|---|
| LangSplat | 54.27 | 24.13 |
| Deformable CLIP | 65.01 | 45.37 |
| Non-Status Field | 84.58 | 62.00 |
| 4D LangSplat | 88.86 | 66.14 |
| ReferGaussian (Ours) | **91.62** | **66.48** |

### Paper results — module ablation on R4D-Bench-QA

| Variant | Acc ↑ | vIoU ↑ |
|---|---|---|
| 4DGS reconstruction backbone only | 62.9 | 31.5 |
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

## Verification

Before a training or benchmark run, validate the checked-out release and the
benchmark evaluator's empty-query and polygon-mask rules:

```bash
source scripts/common.sh
gs_python scripts/check_release.py
gs_python -m unittest discover -s tests -v
```

By default, environments and caches are created under `~/.cache/refergaussian/`:
- `~/.cache/refergaussian/conda-envs`
- `~/.cache/refergaussian/conda-pkgs`
- `~/.cache/refergaussian/pip`

Override with `GS4D_ENV_ROOT`, `GS4D_CONDA_PKGS_DIRS`, and `GS4D_PIP_CACHE_DIR`.

> **Note on COLMAP:** `prepare_hypernerf.sh` requires COLMAP to generate the initial point cloud for each scene. Install it with `apt install colmap` or from [colmap.github.io](https://colmap.github.io/install.html) before running data preparation.

## Model Weights

### Qwen3-VL-8B-Instruct (Refer-Planner)

The Refer-Planner uses [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct).
The release pins the model snapshot below; download it from the Grounded-SAM2
environment before referring evaluation:

```bash
source scripts/common.sh
export REFERGAUSSIAN_QWEN_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
gsam2_python scripts/download_hf_snapshot.py \
  --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --revision "${REFERGAUSSIAN_QWEN_REVISION}" \
  --local-dir models/Qwen3-VL-8B-Instruct
```

Default path is `models/Qwen3-VL-8B-Instruct/` (relative to repo root); the
download writes `refergaussian_snapshot.json` with the resolved revision.
Override the model location with:

```bash
export REFERGAUSSIAN_QWEN_MODEL=/path/to/Qwen3-VL-8B-Instruct
```

### SAM2 and Grounding DINO (Grounded-SAM2 pipeline)

Downloaded automatically during `bash scripts/setup_grounded_sam2.sh` at the
pinned revisions below:
- `facebook/sam2-hiera-large @ e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`
- `IDEA-Research/grounding-dino-base @ 12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

The default endpoint is `https://huggingface.co`; set `HF_ENDPOINT` only when
an alternative mirror is explicitly required.

## Dataset Setup

### HyperNeRF

Download scenes used in the paper from the [HyperNeRF release page](https://github.com/google/hypernerf/releases/tag/v0.1):

```bash
bash scripts/prepare_hypernerf.sh
```

Prepare a single scene:

```bash
bash scripts/prepare_hypernerf.sh misc keyboard

# Canonical R4D-Bench-QA interpolation scenes
bash scripts/prepare_hypernerf.sh interp cut-lemon1
bash scripts/prepare_hypernerf.sh interp torchocolate
```

Register a local scene:

```bash
bash scripts/prepare_local_hypernerf_scene.sh /path/to/scene <group> <scene>
```

### DyNeRF / Neural 3D Video

The fixed R4D-Bench-QA release additionally uses `coffee_martini`,
`sear_steak`, and `cut_roasted_beef` from Neural 3D Video. Obtain those source
scenes under their original license, then place each scene at
`data/dynerf/<scene>/`. The referring pipeline expects the following layout:

```text
data/dynerf/coffee_martini/
  poses_bounds.npy
  cam00/images/0000.png
  ...
```

Generate the required per-frame camera metadata after placing a scene:

```bash
source scripts/common.sh
gs_python scripts/generate_dynerf_camera_jsons.py \
  --dataset-dir data/dynerf/coffee_martini
```

The command is deterministic and may be repeated safely for each DyNeRF scene.

### 4DLangSplat annotations

```bash
bash scripts/download_4dlangsplat_annotations.sh
```

Downloads the pinned annotation snapshot
`d127a280446206fc97887a304de790a1fe6af5ff` to
`data/benchmarks/4dlangsplat/HyperNeRF-Annotation/`; its manifest records the
resolved revision.
Build a per-scene query protocol from the downloaded temporal annotations
before running the public evaluator:

```bash
source scripts/common.sh
ANN_ROOT=data/benchmarks/4dlangsplat/HyperNeRF-Annotation
gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANN_ROOT}" \
  --scene espresso \
  --output-json "${ANN_ROOT}/espresso/protocol.json"
```

Repeat the command with `americano`, `split-cookie`, or `chickchicken` when
evaluating those scenes. The generated protocol is deterministic from
`video_annotations.json` and is intentionally not tracked with the code.

### R4D-Bench-QA

```bash
bash scripts/download_r4d_bench_qa.sh
```

Downloads the pinned dataset snapshot
`0fe2b3a99a95632ea6d0bd1718723ac24804e49b` to
`data/benchmarks/r4d_bench_qa/`. Override it only with an explicit
`R4D_BENCH_REVISION`; the generated `download_manifest.json` records both the
requested and resolved revisions.

Dataset link: [https://huggingface.co/datasets/LiYacheng/r4d-bench-qa](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

## Training

```bash
# Use one explicit seed for a matched ReferGaussian / 4DGS comparison.
export REFERGAUSSIAN_SEED=6666

# Example: keyboard scene
bash scripts/train.sh hypernerf misc/keyboard

# Matched 4D Gaussian Splatting baseline for the same scene/configuration
bash scripts/train_baseline.sh hypernerf misc/keyboard

# Example: DyNeRF scene after the layout above is prepared
bash scripts/train.sh dynerf coffee_martini
```

Output is written to `runs/refergaussian/hypernerf/keyboard/`.

## Evaluation

### Reconstruction metrics

```bash
bash scripts/eval.sh hypernerf misc/keyboard

# Evaluate the matched 4DGS baseline without enabling ReferGaussian's warp
bash scripts/eval_baseline.sh hypernerf misc/keyboard
```

The two commands write PSNR / SSIM / LPIPS to their respective run directories.
For a fair reconstruction comparison, use the same dataset source, 4DGaussians
scene config, iteration budget, seed, and metric mode for both runs. The wrappers
write `config.yaml`, `results.json`, and `metrics.json` beside each output for
audit.

For iterative debugging only, set `GS_SKIP_FULL_METRICS=1` to run the uniform
subset metric path. Do not compare subset metrics with the full-frame paper
table; omit that variable for any final reconstruction report.

### Referring evaluation — 4DLangSplat public protocol

Run per scene (example: `split-cookie`):

```bash
SCENE=split-cookie
GROUP=misc
RUN_DIR=runs/refergaussian/hypernerf/${SCENE}
DATASET_DIR=data/hypernerf/${GROUP}/${SCENE}
ANNOT_ROOT=data/benchmarks/4dlangsplat/HyperNeRF-Annotation
ANNOT_DIR=${ANNOT_ROOT}/${SCENE}
PROTOCOL_JSON=${ANNOT_DIR}/protocol.json

source scripts/common.sh
gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANNOT_ROOT}" \
  --scene "${SCENE}" \
  --output-json "${PROTOCOL_JSON}"

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
  --category temporal_state_reference \
  --require-complete \
  --output-json reports/public_eval_${SCENE}.json
```

`run_public_query_protocol.sh` defaults to the
`temporal_state_reference` category, so it does not spend compute on the
unscored static-reference rows in the annotation file.

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

The released query path is strictly `mask_supported_lifting`: if multi-frame
lifting cannot form the requested entity, the query is reported as a failure
with its Stage-1 and proposal diagnostics. It is never replaced by an
unconstrained full-scene entity. Query inference also requires the exact
ReferGaussian test renders produced by `scripts/eval.sh`; raw source RGB
frames are not used as a surrogate render.

For paper-style batched reruns, prefer the manifest-based runner so query ids, output roots,
and evaluator maps stay aligned:

```bash
OUT=reports/public_time_shape_v4_recall
source scripts/common.sh
ANN_ROOT=data/benchmarks/4dlangsplat/HyperNeRF-Annotation
PROTOCOL_JSON="${OUT}/public_protocol.json"
gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANN_ROOT}" \
  --output-json "${PROTOCOL_JSON}"

gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --query-set time_sensitive \
  --protocol-json "${PROTOCOL_JSON}" \
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

Evaluate each scene from the same protocol and then aggregate the four
complete scene reports. The aggregate command refuses to turn a partial run
into a final result:

```bash
for SCENE in americano espresso split-cookie chickchicken; do
  case "${SCENE}" in
    chickchicken) GROUP=interp ;;
    *) GROUP=misc ;;
  esac
  gs_python scripts/evaluate_public_query_protocol.py \
    --protocol-json "${PROTOCOL_JSON}" \
    --annotation-dir "${ANN_ROOT}/${SCENE}" \
    --dataset-dir "data/hypernerf/${GROUP}/${SCENE}" \
    --query-root "${OUT}/query_root" \
    --scene "${SCENE}" \
    --category temporal_state_reference \
    --require-complete \
    --output-json "${OUT}/${SCENE}_official_eval.json" \
    --output-md "${OUT}/${SCENE}_official_eval.md"
done

gs_python scripts/aggregate_public_query_evaluations.py \
  --inputs "${OUT}"/*_official_eval.json \
  --expected-queries 9 \
  --require-complete \
  --output-json "${OUT}/official_eval.json" \
  --output-md "${OUT}/official_eval.md"
```

When evaluating a subset such as `time_sensitive`, pass the same manifest to
`evaluate_public_query_protocol.py --query-manifest "${OUT}/manifest.jsonl"`.
This prevents unrequested protocol queries from appearing as missing results.

### Referring evaluation — R4D-Bench-QA

```bash
# Optional: profile switch for reproducible ablation
# export QUERY_EVAL_PROFILE=default
# export QUERY_EVAL_PROFILE=r4d_shape_v4_recall

# 1) Build a manifest with official query ids.
source scripts/common.sh
RUN_ROOT=reports/r4d_bench_public_time_shape_v4
gs_python scripts/build_r4d_query_manifest.py \
  --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
  --scenes coffee_martini sear_steak cut_roasted_beef \
           cut_lemon espresso keyboard split_cookie torchchocolate \
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
  --profile r4d_shape_v4_recall \
  --gpu 0 1 2 \
  --force-rerun \
  --timeout 10800

# 3) Re-evaluate from saved query outputs (no rerun of model inference).
gs_python scripts/evaluate_ours_benchmark.py \
  --benchmark data/benchmarks/r4d_bench_qa/benchmark_all_queries.json \
  --query-root-map "${RUN_ROOT}/query_root_map.json" \
  --dataset-dir-map "${RUN_ROOT}/dataset_dir_map.json" \
  --query-manifest "${RUN_ROOT}/manifest.jsonl" \
  --output-json "${RUN_ROOT}/official_eval.json" \
  --output-md "${RUN_ROOT}/official_eval.md" \
  --require-complete
```

The preflight and query pipeline require `phase: refergaussian`,
`temporal_warp_type: refergaussian`, and `warp_enabled: true` in each run's
`config.yaml`. This prevents accidental evaluation of a baseline or legacy
reconstruction as ReferGaussian.

Output files:
- `${RUN_ROOT}/query_root_map.json`
- `${RUN_ROOT}/dataset_dir_map.json`
- `${RUN_ROOT}/official_eval.json` (per-query Acc/vIoU/tIoU + summary)
- `${RUN_ROOT}/official_eval.md`

Metric rule used in both public and R4D evaluators:
- Empty-set rule is enabled: if GT and prediction are both empty (temporal union = 0), `vIoU = 1.0` and `tIoU = 1.0`. An empty-GT query with any predicted activity scores `0.0` for both values.
- Every evaluator summary also reports the non-empty-query subset and empty-target correctness separately. Report these alongside headline `Acc`/`vIoU`/`tIoU` so background-heavy timelines or zero-target queries cannot conceal ordinary grounding failures.

Important benchmark note:
- Current HF `LiYacheng/r4d-bench-qa` artifacts provide dense GT for 36-query and 89-query tiers.
- The larger language-only extension set does not always include dense masks, so strict `vIoU`/`tIoU` cannot be computed for those entries without extra GT alignment files.
- Prefer the official query metadata file `R4D-Bench_queries.json` as the source of truth for query text.
  Avoid retyping benchmark queries in ad-hoc shell scripts, because even small wording drift can change the target object set.

The paper-facing release protocol is fixed to 8 scenes and 58 English queries:

- DyNeRF: `coffee_martini`, `sear_steak`, `cut_roasted_beef`
- HyperNeRF: `cut_lemon`, `espresso`, `keyboard`, `split_cookie`, `torchchocolate`

Filter the official query metadata to this fixed protocol when preparing a release run:

```bash
source scripts/common.sh
gs_python scripts/filter_r4d_benchmark_queries.py \
  --input-json data/benchmarks/r4d_bench_qa/evaluation/R4D-Bench_queries.json \
  --output-json reports/r4d_bench_queries_8scene.json \
  --output-md reports/r4d_bench_queries_8scene.md \
  --include-scenes coffee_martini sear_steak cut_roasted_beef \
                   cut_lemon espresso keyboard split_cookie torchchocolate
```

This produces the canonical 8-scene / 58-query release set. Scene selection is fixed before evaluation; pass the same manifest to the evaluator so an 89-query source file cannot silently become a partial 58/89 report.

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

Before publishing a release, run:

```bash
source scripts/common.sh
gs_python scripts/check_release.py
```

```bibtex
@inproceedings{chen2026refergaussian,
  title     = {ReferGaussian: Referring Segmentation in 4D Gaussian Splatting},
  author    = {Bangpu Chen and Yaxuan Li and Shirui Peng and Xiangtian Si and Chu Liuxin and Xitong Cao and Hongbo Jin and Jiayu Ding},
  booktitle = {Proceedings of the ACM International Conference on Multimedia},
  year      = {2026}
}
```

## Acknowledgements

This project builds on [4DGaussians](https://github.com/hustvl/4DGaussians), [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2), and [Qwen](https://github.com/QwenLM/Qwen).
