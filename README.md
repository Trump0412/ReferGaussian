# R4DGS: Referring Segmentation in 4D Gaussian Splatting

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10](https://img.shields.io/badge/Python-3.10-brightgreen.svg)](#installation)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-orange.svg)](#installation)
[![ACM MM 2026](https://img.shields.io/badge/ACM%20MM-2026-8A2BE2.svg)](#citation)

[Project Page](https://trump0412.github.io/ReferGaussian/) | [Paper](https://doi.org/10.1145/3767308.3836021) | [Dataset](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa) | [Citation](#citation)

</div>

**Bangpu Chen, Yaxuan Li, Shirui Peng, Xiangtian Si, Liuxin Chu, Xitong Cao, Hongbo Jin, Jiayu Ding**

Accepted at **ACM Multimedia 2026**.

R4DGS studies natural-language referring segmentation in dynamic 4D Gaussian scenes. Our ReferGaussian framework takes a pretrained, frozen 4DGS scene as input and combines a Qwen-based Refer-Planner, Grounded-SAM2 masks, and training-free multi-frame mask-supported lifting. Stage-1 masks supervise and gate Gaussian assignment; final benchmark masks are rendered from the selected Gaussian entity. ReferGaussian does not train or modify the underlying 4DGS representation.

<p align="center">
  <img src="docs/assets/Fig1.png" width="90%" alt="R4DGS task overview"/>
</p>

## Results

The following values are from the accepted paper. Reproduced runs must retain their protocol id, manifest, source hashes, and evaluator output. See [metric definitions](docs/METRICS.md) and [release status](docs/REPRODUCTION_STATUS.md) before comparing a new run with these tables.

### R4D-Bench-QA

| Method | Acc (higher) | vIoU (higher) |
|---|---:|---:|
| Segment then Splat | 55.6 | 28.4 |
| 4D LangSplat | 58.4 | 32.1 |
| **ReferGaussian** | **76.5** | **34.4** |

### 4D LangSplat HyperNeRF split

This table is the three-scene, seven-query time-sensitive protocol.

| Method | Acc (higher) | vIoU (higher) |
|---|---:|---:|
| LangSplat | 54.27 | 24.13 |
| Deformable CLIP | 65.01 | 45.37 |
| Non-Status Field | 84.58 | 62.00 |
| 4D LangSplat | 88.86 | 66.14 |
| **ReferGaussian** | **91.62** | **66.48** |

<p align="center">
  <img src="docs/assets/Fig5_n.png" width="90%" alt="ReferGaussian qualitative results"/>
</p>

## Installation

Requirements: Linux, CUDA 12.1, and Miniconda.

```bash
git clone https://github.com/Trump0412/ReferGaussian.git
cd ReferGaussian

# Pinned 4DGaussians and Grounded-SAM-2 sources.
bash scripts/bootstrap_external.sh

# 4DGS rendering and ReferGaussian evaluation environment.
bash scripts/setup_4dgs_env.sh cuda121

# Grounded-SAM2 and Qwen environment.
GSAM2_INSTALL_EDITABLE=1 bash scripts/setup_grounded_sam2.sh
```

Large files can live outside the repository:

```bash
export GS_DATA_ROOT=/absolute/path/to/refergaussian-data
export GS_RUN_ROOT=/absolute/path/to/refergaussian-runs
source scripts/common.sh
```

Without these variables, the defaults are `data/` and `runs/`.

### Model weights

Grounding DINO and SAM2 are downloaded at pinned revisions by `setup_grounded_sam2.sh`. Download the pinned Refer-Planner snapshot with:

```bash
source scripts/common.sh
export REFERGAUSSIAN_QWEN_REVISION=0c351dd01ed87e9c1b53cbc748cba10e6187ff3b
gsam2_python scripts/download_hf_snapshot.py \
  --repo-id Qwen/Qwen3-VL-8B-Instruct \
  --revision "${REFERGAUSSIAN_QWEN_REVISION}" \
  --local-dir models/Qwen3-VL-8B-Instruct

gsam2_python scripts/check_query_runtime.py \
  --require-qwen \
  --require-pinned-manifest
```

Set `REFERGAUSSIAN_QWEN_MODEL` when the checkpoint is stored elsewhere.

### Verify the checkout

```bash
source scripts/common.sh
gs_python scripts/check_release.py
gs_python -m unittest discover -s tests -v
```

## Data

### HyperNeRF scenes

Install COLMAP, then prepare a downloaded scene or register an existing one:

```bash
bash scripts/prepare_hypernerf.sh misc keyboard
bash scripts/prepare_local_hypernerf_scene.sh /path/to/scene <group> <scene>
```

The Public scenes use `misc/{americano,espresso,split-cookie}` and `interp/chickchicken`.

### DyNeRF scenes

Place each Neural 3D Video scene under `${GS_DATA_ROOT}/dynerf/<scene>/`, preserving `poses_bounds.npy` and `camXX/images/`. Then create deterministic per-frame camera metadata:

```bash
source scripts/common.sh
gs_python scripts/generate_dynerf_camera_jsons.py \
  --dataset-dir "${GS_DATA_ROOT}/dynerf/coffee_martini"
```

### Benchmark annotations

```bash
bash scripts/download_4dlangsplat_annotations.sh
bash scripts/download_r4d_bench_qa.sh
```

Both downloaders pin the source revision and write a manifest beside the downloaded data.

## 4DGS Input

ReferGaussian is an inference method over a frozen 4D Gaussian scene. Scene training is outside this release; the repository does not ship scene checkpoints or rendered benchmark outputs. Prepare the input with the pinned upstream [4DGaussians](https://github.com/hustvl/4DGaussians) code or use a compatible existing 4DGS checkpoint.

Each query-ready run directory must contain the standard model and test-render artifacts:

```text
<run_dir>/
|-- cfg_args
|-- point_cloud/iteration_*/point_cloud.ply
`-- test/ours_*/renders/*.png
```

Register those directories below `${GS_RUN_ROOT}/baseline_4dgs/<dataset>/<scene>` or pass an explicit run root/namespace when building a manifest. `scripts/preflight_query_batch.py` validates the checkpoint contract before inference.

## Referring Inference

All formal runs follow the same order:

1. Build a versioned protocol and query manifest.
2. Run strict preflight.
3. Execute one serial worker per GPU.
4. Evaluate with `--require-complete`.

The released semantic path is `mask_supported_lifting`. Alternative training or scene-specific proposal builders are not included.

### Public time-sensitive protocol

The four-scene extension contains nine queries. Use `paper_public3` and an expected count of seven for the accepted-paper three-scene subset.

```bash
source scripts/common.sh
OUT="reports/public4_time_sensitive_$(date -u +%Y%m%dT%H%M%SZ)"
ANN_ROOT="${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation"

gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANN_ROOT}" \
  --output-json "${OUT}/protocol.json"

gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --query-set time_sensitive \
  --protocol-json "${OUT}/protocol.json" \
  --protocol-id release_public4_extension \
  --annotation-root "${ANN_ROOT}" \
  --profile public_time_boundary_gated_v5_numeric \
  --gpus 0 1 2 3

gs_python scripts/preflight_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_public4_extension \
  --profile public_time_boundary_gated_v5_numeric \
  --strict-release --gpu 0 1 2 3 \
  --require-visible-gpu --create-output-root

gs_python scripts/run_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_public4_extension \
  --profile public_time_boundary_gated_v5_numeric \
  --gpu 0 1 2 3 --force-rerun --strict-release --timeout 3600
```

Evaluate each scene and aggregate the exact nine-query manifest:

```bash
for SCENE_GROUP in americano:misc chickchicken:interp espresso:misc split-cookie:misc; do
  IFS=: read -r SCENE GROUP <<< "${SCENE_GROUP}"
  gs_python scripts/evaluate_public_query_protocol.py \
    --protocol-json "${OUT}/protocol.json" \
    --query-manifest "${OUT}/manifest.jsonl" \
    --annotation-dir "${ANN_ROOT}/${SCENE}" \
    --dataset-dir "${GS_DATA_ROOT}/hypernerf/${GROUP}/${SCENE}" \
    --query-root "${OUT}/query_root" \
    --scene "${SCENE}" \
    --category temporal_state_reference \
    --require-complete \
    --output-json "${OUT}/${SCENE}_eval.json" \
    --output-md "${OUT}/${SCENE}_eval.md"
done

gs_python scripts/aggregate_public_query_evaluations.py \
  --inputs "${OUT}/americano_eval.json" \
           "${OUT}/chickchicken_eval.json" \
           "${OUT}/espresso_eval.json" \
           "${OUT}/split-cookie_eval.json" \
  --expected-queries 9 --require-complete \
  --output-json "${OUT}/official_eval.json" \
  --output-md "${OUT}/official_eval.md"
```

### Public time-agnostic protocol

This separate protocol evaluates all 20 COCO categories on vanilla 4DGS checkpoints: 5 americano, 5 chickchicken, 6 espresso, and 4 split-cookie.

```bash
source scripts/common.sh
OUT="reports/public4_time_agnostic_$(date -u +%Y%m%dT%H%M%SZ)"
ANN_ROOT="${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation"

gs_python scripts/build_4dlangsplat_time_agnostic_protocol.py \
  --annotation-root "${ANN_ROOT}" \
  --output-json "${OUT}/protocol.json"

gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --query-set time_agnostic \
  --protocol-json "${OUT}/protocol.json" \
  --protocol-id release_public4_time_agnostic \
  --annotation-root "${ANN_ROOT}" \
  --profile public_time_agnostic_v1 \
  --run-namespace baseline_4dgs \
  --gpus 0 1 2 3

gs_python scripts/preflight_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_public4_time_agnostic \
  --profile public_time_agnostic_v1 \
  --strict-release --gpu 0 1 2 3 \
  --require-visible-gpu --create-output-root

gs_python scripts/run_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_public4_time_agnostic \
  --profile public_time_agnostic_v1 \
  --gpu 0 1 2 3 --force-rerun --strict-release --timeout 10800

gs_python scripts/evaluate_public_time_agnostic.py \
  --protocol-json "${OUT}/protocol.json" \
  --manifest "${OUT}/manifest.jsonl" \
  --annotation-root "${ANN_ROOT}" \
  --query-root "${OUT}/query_root" \
  --require-complete \
  --output-json "${OUT}/official_eval.json" \
  --output-md "${OUT}/official_eval.md"
```

### R4D-Bench-QA

The downloadable dense release contains 12 scenes and 89 English queries: 36 temporal, 29 multi-target/reasoning, and 24 zero-target/distractor queries.

```bash
source scripts/common.sh
OUT="reports/r4d_dense89_$(date -u +%Y%m%dT%H%M%SZ)"
R4D_ROOT="${GS_DATA_ROOT}/benchmarks/r4d_bench_qa"

gs_python scripts/build_r4d_query_manifest.py \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --query-metadata "${R4D_ROOT}/evaluation/R4D-Bench_queries.json" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --gpus 0 1 2 3

gs_python scripts/preflight_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --strict-release --gpu 0 1 2 3 \
  --require-visible-gpu --create-output-root

gs_python scripts/run_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --gpu 0 1 2 3 --force-rerun --strict-release --timeout 3600

CAMERA_ROOT="${OUT}/source_camera_views"
gs_python scripts/rerender_query_outputs.py \
  --manifest "${OUT}/manifest.jsonl" \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --output-root "${CAMERA_ROOT}" \
  --profile r4d_renderer_consistent \
  --gpu 0 --require-complete

gs_python scripts/evaluate_ours_benchmark.py \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --query-root-map "${CAMERA_ROOT}/query_root_map.json" \
  --dataset-dir-map "${CAMERA_ROOT}/dataset_dir_map.json" \
  --query-manifest "${OUT}/manifest.jsonl" \
  --output-json "${CAMERA_ROOT}/official_eval.json" \
  --output-md "${CAMERA_ROOT}/official_eval.md" \
  --require-complete
```

Zero-target queries score 100 only when both prediction and ground truth are empty. Missing evidence and unresolved selections are rejected by strict evaluation instead of being counted as empty predictions.

## Output Contract

Every query owns one output directory containing its plan, Stage-1 tracks, Gaussian proposal, final selection, rendered masks, validation record, logs, and efficiency summary. A batch additionally writes `batch_provenance.json` and `batch_summary.json`.

Formal evaluators require complete manifests and report non-empty and zero-target subsets separately. Detailed protocol identities and formulas are documented in:

- [Release reproduction protocol](docs/RELEASE_REPRODUCTION_PROTOCOL.md)
- [Metric definitions](docs/METRICS.md)
- [Verification status](docs/REPRODUCTION_STATUS.md)

## Repository Layout

```text
ReferGaussian/
|-- refergaussian/      # EntityBank and training-free semantic grounding
|-- scripts/            # supported setup, inference, and evaluation CLIs
|-- configs/benchmarks/ # immutable protocol registries and English query map
|-- tests/              # release and metric contract tests
|-- docs/               # protocol documentation and project page
`-- external/           # fetched dependencies; not tracked
```

## Citation

```bibtex
@inproceedings{chen2026r4dgs,
  title     = {R4DGS: Referring Segmentation in 4D Gaussian Splatting},
  author    = {Chen, Bangpu and Li, Yaxuan and Peng, Shirui and Si, Xiangtian and Chu, Liuxin and Cao, Xitong and Jin, Hongbo and Ding, Jiayu},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  year      = {2026},
  doi       = {10.1145/3767308.3836021}
}
```

## Acknowledgements

This project builds on [4D Gaussian Splatting](https://github.com/hustvl/4DGaussians), [Grounded-SAM-2](https://github.com/IDEA-Research/Grounded-SAM-2), [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL), and the 4D LangSplat evaluation resources.
