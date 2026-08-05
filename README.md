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

The following paper tables are archival reported values, not a claim that a
new release rerun has already completed. A release reproduction must identify
its scene/query and metric protocol, cover every expected query, retain the
manifest/evaluator output, and report overall, non-empty-only, and zero-target
outcomes separately. [Metric definitions](docs/METRICS.md) documents the
difference between the paper-declared formulas and executable compatibility
fields.
See [the release reproduction protocol](docs/RELEASE_REPRODUCTION_PROTOCOL.md)
and [the release verification status](docs/REPRODUCTION_STATUS.md) for the
fixed scene/query sets, required artifacts, and current verification boundary.

### Paper results — R4D-Bench-QA referring segmentation

| Method | Acc ↑ | vIoU ↑ |
|---|---:|---:|
| Segment then Splat | 55.6 | 28.4 |
| 4D LangSplat | 58.4 | 32.1 |
| ReferGaussian (Ours) | **76.5** | **34.4** |

### Paper results — R4D-Bench reconstruction

| Method | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|
| 4D Gaussian Splatting | 20.3208 | 0.7027 | 0.3971 |
| ReferGaussian (Ours) | **20.4159** | **0.7069** | **0.3942** |

These are the values in the accepted-paper R4D-Bench main table. A later
selected 8-scene historical summary reported 4DGS `28.312 / 0.8753 / 0.2325`
and ReferGaussian `28.486 / 0.8777 / 0.2233`; it is retained as an archival
candidate in the protocol registry, but is not relabeled as the paper table or
as a fresh matched reproduction.

### Paper results — 4D LangSplat HyperNeRF split (3 scenes / 7 queries)

| Method | Acc ↑ | vIoU ↑ |
|---|---|---|
| LangSplat | 54.27 | 24.13 |
| Deformable CLIP | 65.01 | 45.37 |
| Non-Status Field | 84.58 | 62.00 |
| 4D LangSplat | 88.86 | 66.14 |
| ReferGaussian (Ours) | **91.62** | **66.48** |

This accepted-paper table is the seven-query **time-sensitive** result. It is
not a time-agnostic aggregate. The executable compatibility evaluator records
its metric identity explicitly; its full-timeline Acc is not silently
relabeled as the sparse-frame Acc in the 4D LangSplat reference script.

### Paper results — module ablation on R4D-Bench-QA

| Variant | Acc ↑ | vIoU ↑ |
|---|---|---|
| 4DGS reconstruction backbone only | 62.9 | 31.5 |
| w/o Stage 1 static segmentation | 48.6 | 17.2 |
| w/o Stage 2 semantic assignment | 62.9 | 29.8 |
| w/o Stage 3 spatio-temporal reasoning | 36.0 | 26.1 |
| ReferGaussian (full) | **76.5** | **34.4** |

### Reconstruction — keyboard scene (archival appendix)

This row is retained from the accepted-paper appendix. Its historical 4DGS
artifact is not a complete matched-seed release record, so it is not a
fresh-install verification claim. Use the fixed paired protocol below for any
new comparison.

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
# Installs the pinned checkout into its dedicated environment.
GSAM2_INSTALL_EDITABLE=1 bash scripts/setup_grounded_sam2.sh
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

Validate the checkpoint before a long query batch. This is also performed by
the release pipeline before it allocates Stage-1 GPU work:

```bash
gsam2_python scripts/check_query_runtime.py \
  --require-qwen \
  --require-pinned-manifest
```

### SAM2 and Grounding DINO (Grounded-SAM2 pipeline)

The setup script first verifies the local pinned cache and downloads missing
weights only when needed, at the pinned revisions below:
- `facebook/sam2-hiera-large @ e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251`
- `IDEA-Research/grounding-dino-base @ 12bdfa3120f3e7ec7b434d90674b3396eccf88eb`

The default endpoint is `https://huggingface.co`; set `HF_ENDPOINT` only when
an alternative mirror is explicitly required.

After setup, query inference loads these exact pinned snapshots from the local
cache by default and does not issue a runtime Hub request. If a snapshot is
missing, rerun `bash scripts/setup_grounded_sam2.sh`; set
`GSAM2_LOCAL_FILES_ONLY=0` only when an explicit recovery download is intended.
Set `GSAM2_DOWNLOAD_WEIGHTS=0` when setting up an offline machine and you want
the setup command to fail immediately instead of attempting a download.
The setup command installs the pinned checkout editably by default and verifies
that both `sam2` and its optional extension resolve from this repository rather
than another project's Grounded-SAM2 installation.

## Dataset Setup

All preparation scripts use the same storage root as training. The portable
default is `${PWD}/data`; set this before preparation when using mounted
storage:

```bash
export GS_DATA_ROOT=/absolute/path/to/refergaussian-data
source scripts/common.sh
```

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

The 12-scene dense R4D-Bench-QA release additionally uses six Neural 3D Video
scenes. Obtain them under their original license and register them with these
release paths:

| R4D scene id | Local data directory |
| --- | --- |
| `coffee_martini` | `${GS_DATA_ROOT}/dynerf/coffee_martini/` |
| `cook-spinach` | `${GS_DATA_ROOT}/dynerf/cook_spinach/` |
| `cut_roasted_beef` | `${GS_DATA_ROOT}/dynerf/cut_roasted_beef/` |
| `flame_salmon` | `${GS_DATA_ROOT}/dynerf/flame_salmon_1/` |
| `flame_steak` | `${GS_DATA_ROOT}/dynerf/flame_steak/` |
| `sear_steak` | `${GS_DATA_ROOT}/dynerf/sear_steak/` |

Each directory follows the original multi-camera layout, for example:

```text
data/dynerf/coffee_martini/
  poses_bounds.npy
  cam00/images/0000.png
  ...
```

Generate the required per-frame camera metadata after placing every scene:

```bash
source scripts/common.sh
gs_python scripts/generate_dynerf_camera_jsons.py \
  --dataset-dir "${GS_DATA_ROOT}/dynerf/coffee_martini"
```

The command is deterministic and may be repeated safely for each directory in
the table. The differing `flame_salmon` benchmark id and `flame_salmon_1`
source directory are encoded in `scripts/build_r4d_query_manifest.py`.

### 4DLangSplat annotations

```bash
bash scripts/download_4dlangsplat_annotations.sh
```

Downloads the pinned annotation snapshot
`d127a280446206fc97887a304de790a1fe6af5ff` to
`${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation/`; its manifest records the
resolved revision.
Build a per-scene query protocol from the downloaded temporal annotations
before running the public evaluator:

```bash
source scripts/common.sh
ANN_ROOT="${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation"
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
`${GS_DATA_ROOT}/benchmarks/r4d_bench_qa/`. Override it only with an explicit
`R4D_BENCH_REVISION`; the generated `download_manifest.json` records both the
requested and resolved revisions.

Dataset link: [https://huggingface.co/datasets/LiYacheng/r4d-bench-qa](https://huggingface.co/datasets/LiYacheng/r4d-bench-qa)

## Training

By default, prepared data and training outputs live under `data/` and `runs/`
inside the repository. To keep large assets on a mounted volume, set these
generic roots once before running the same commands:

```bash
export GS_DATA_ROOT=/absolute/path/to/refergaussian-data
export GS_RUN_ROOT=/absolute/path/to/refergaussian-runs
```

The defaults remain `data/` and `runs/`; these variables only relocate assets
and do not change a model or query-evaluation setting.

```bash
# Use one explicit seed for a matched ReferGaussian / 4DGS comparison.
# bootstrap_external.sh applies the seed-order patch so this seed is effective.
export REFERGAUSSIAN_SEED=6666
# Both methods default to the release budget below; keep it matched.
export REFERGAUSSIAN_ITERATIONS=14000

# Example: keyboard scene
bash scripts/train.sh hypernerf misc/keyboard

# Matched 4D Gaussian Splatting baseline for the same scene/configuration
bash scripts/train_baseline.sh hypernerf misc/keyboard

# Example: DyNeRF scene after the layout above is prepared
bash scripts/train.sh dynerf coffee_martini
```

Output is written to `runs/refergaussian/hypernerf/keyboard/`.
Both training wrappers explicitly pass the same 14,000-iteration release
budget; they do not inherit an upstream scene config's shorter debug default.
The temporal learning-rate schedule is shared by the Gaussian time primitives
and the learned temporal warp, so the recorded `temporal_lr_*` settings fully
define the temporal optimization schedule.

## Evaluation

### Reconstruction metrics

```bash
bash scripts/eval.sh hypernerf misc/keyboard

# Evaluate the matched 4DGS baseline without enabling ReferGaussian's warp
bash scripts/eval_baseline.sh hypernerf misc/keyboard
```

The two commands stream the complete test-camera split to disk and write PSNR /
SSIM / LPIPS to their respective run directories. Train-camera renders and
preview videos are omitted because they are not metric inputs; this prevents
long sequences from accumulating every rendered tensor in GPU memory without
changing the evaluated frames or metric definitions.
For a fair reconstruction comparison, use the same dataset source, 4DGaussians
scene config, iteration budget, seed, and metric mode for both runs. The wrappers
write `config.yaml`, `results.json`, and `metrics.json` beside each output for
audit.

Run the matching `eval.sh` once before a query batch. The query renderer uses
the resulting ReferGaussian test render grid as its temporal reference and
will refuse to evaluate a run with no such render instead of substituting a
proxy image sequence.

For iterative debugging only, set `GS_SKIP_FULL_METRICS=1` to run the uniform
subset metric path. Do not compare subset metrics with the full-frame paper
table; omit that variable for any final reconstruction report.

For a new matched release comparison, use the frozen runner instead of
launching the two methods by hand:

```bash
source scripts/common.sh
PROTOCOL=release_reconstruction_v2_paper_compat
OUT="reports/${PROTOCOL}_$(date -u +%Y%m%dT%H%M%SZ)"
gs_python scripts/run_matched_reconstruction.py \
  --protocol "${PROTOCOL}" \
  --data-root "${GS_DATA_ROOT}" \
  --output-root "${OUT}" \
  --gpu 0
```

This command requires a clean Git checkout, validates the pinned 4DGaussians
commit, patch-file hashes, exact patched-tree diff, and generated-file hashes,
then trains both methods at the protocol's fixed seed for 14,000
fine iterations, evaluates the same full test-camera set, and writes a
scene-equal aggregate only after all 12 registered scenes complete. It never
filters scenes by PSNR. A `--scenes` subset is emitted only as an incomplete
canary and has no final aggregate.

`release_reconstruction_v2_paper_compat` makes the historical effective seed
(`0`) and constant contextual-warp learning rate explicit while applying the
fixed seed in the correct RNG order. `release_reconstruction_v1` remains frozen
as the seed-6666/shared-schedule August audit baseline. Neither executable
identity is relabeled as the unresolved accepted-paper reconstruction table.
The paper-reported identity, historical selected-8 candidate, conflicting
temporal tube settings, and both executable identities are kept side by side in
[`configs/benchmarks/reconstruction_release_v1.json`](configs/benchmarks/reconstruction_release_v1.json).

The full and subset metric paths cache each LPIPS network once per evaluated
method rather than reconstructing it for every frame. This is a runtime-only
change: PSNR, SSIM, MS-SSIM, LPIPS-vgg, and LPIPS-alex retain their original
per-frame definitions and reduction.

### Referring evaluation — 4DLangSplat public protocol

Run per scene (example: `split-cookie`):

```bash
source scripts/common.sh
SCENE=split-cookie
GROUP=misc
RUN_DIR="${GS_RUN_ROOT}/refergaussian/hypernerf/${SCENE}"
DATASET_DIR="${GS_DATA_ROOT}/hypernerf/${GROUP}/${SCENE}"
ANNOT_ROOT="${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation"
ANNOT_DIR=${ANNOT_ROOT}/${SCENE}
PROTOCOL_JSON=${ANNOT_DIR}/protocol.json

gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANNOT_ROOT}" \
  --scene "${SCENE}" \
  --output-json "${PROTOCOL_JSON}"

# Required for the released time-sensitive compatibility protocol.
export QUERY_EVAL_PROFILE=public_time_boundary_gated_v5

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
`temporal_state_reference` category, so it does not execute time-agnostic
static-category prompts. The sequential helper inherits `QUERY_EVAL_PROFILE`;
set the formal v5 profile explicitly as above. The manifest batch flow below
is required for a complete released compatibility aggregate.

The four-scene annotation snapshot contains 9 time-sensitive state queries.
Its COCO masks also contain 20 scene-local static category prompts with actual
annotations (15 on the three paper scenes). The current builder's four
`static_reference` base-object rows are not the complete time-agnostic split,
and no full time-agnostic result is claimed by this release.

Reproducible profile switch:

```bash
# Exploratory behavior only; do not use this for a paper comparison.
export QUERY_EVAL_PROFILE=default

# Formal release profile: synchronized boundary-gated Gaussian projection.
# Use this explicit profile together with --strict-release.
export QUERY_EVAL_PROFILE=public_time_boundary_gated_v5

# Score-only variant: identical v5 inference and masks, without qualitative exports
export QUERY_EVAL_PROFILE=public_time_boundary_gated_v5_numeric
```

The active profile and its effective fusion parameters are written into each
`final_query_render_sourcebg/validation.json` as `eval_profile` and `fusion_options`.
This makes public reruns auditable and easy to compare.

Stage-1 point prompting defaults to `GSAM2_INFERENCE_SEED=0`. The isolated
Grounded-SAM2 process applies this seed to Python, NumPy, and Torch before
detection or tracking and records it in `grounded_sam2_query_tracks.json`.
Keep the default fixed for formal reruns; changing it defines a different
stochastic Stage-1 run and must be reported explicitly.

The released query path is strictly `mask_supported_lifting`: if multi-frame
lifting cannot form the requested entity, the query is reported as a failure
with its Stage-1 and proposal diagnostics. It is never replaced by an
unconstrained full-scene entity. Query inference also requires the exact
ReferGaussian test renders produced by `scripts/eval.sh`; raw source RGB
frames are not used as a surrogate render.

For the formal boundary-gated profiles, the final entity mask is rendered with
the selected-Gaussian alpha-splat geometry used during multi-frame lifting and
then intersected with a small dilation of the synchronized Stage-1 boundary
mask. This is a geometric gate, not a 2D-mask replacement: every foreground
pixel in the final output must remain supported by the selected Gaussian
projection. The profile rejects stale Stage-1 matches and records boundary
coverage, direct-mask use, and cloud support in `validation.json`.

`public_time_boundary_gated_v5_numeric` keeps the same Stage-1, Qwen,
training-free Gaussian lifting, selected-entity projection, and boundary-gate
contract as v5. It only skips entity-library videos and overlay exports, while
still writing the masks and validation artifacts required by the evaluator.
Annotation diagnostic montages are also omitted in this score-only mode, so
their absence is not treated as an inference failure.

For paper-style batched reruns, prefer the manifest-based runner so query ids, output roots,
and evaluator maps stay aligned:

Each manifest row owns one query output root. The Stage-1 plan and tracks,
Gaussian proposal diagnostics, final render, and validation file are written
under that same root, which makes an interrupted batch directly inspectable and
safe to resume without mixing artifacts from different queries. Before the
first query starts, the runner writes `batch_provenance.json` with manifest,
Git/diff, source, checkpoint, and dataset-metadata hashes.

```bash
OUT="reports/PAPER_PUBLIC3_V5_$(date -u +%Y%m%dT%H%M%SZ)"
source scripts/common.sh
ANN_ROOT="${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation"
PROTOCOL_JSON="${OUT}/public_protocol.json"
gs_python scripts/build_4dlangsplat_query_protocol.py \
  --annotation-root "${ANN_ROOT}" \
  --output-json "${PROTOCOL_JSON}"

gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --query-set time_sensitive \
  --protocol-json "${PROTOCOL_JSON}" \
  --protocol-id paper_public3 \
  --annotation-root "${ANN_ROOT}" \
  --profile public_time_boundary_gated_v5 \
  --gpus 0

gs_python scripts/preflight_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id paper_public3 \
  --profile public_time_boundary_gated_v5 \
  --strict-release \
  --gpu 0 \
  --require-visible-gpu \
  --create-output-root

gs_python scripts/run_query_batch.py \
  --manifest "${OUT}/manifest.jsonl" \
  --protocol-id paper_public3 \
  --profile public_time_boundary_gated_v5 \
  --gpu 0 \
  --force-rerun \
  --strict-release \
  --timeout 3600
```

Evaluate each paper scene from the same manifest and aggregate the three
complete scene reports. The aggregate command refuses to turn a partial run
into a final result:

```bash
for SCENE in americano espresso split-cookie; do
  GROUP=misc
  gs_python scripts/evaluate_public_query_protocol.py \
    --protocol-json "${PROTOCOL_JSON}" \
    --query-manifest "${OUT}/manifest.jsonl" \
    --annotation-dir "${ANN_ROOT}/${SCENE}" \
    --dataset-dir "${GS_DATA_ROOT}/hypernerf/${GROUP}/${SCENE}" \
    --query-root "${OUT}/query_root" \
    --scene "${SCENE}" \
    --category temporal_state_reference \
    --require-complete \
    --output-json "${OUT}/${SCENE}_official_eval.json" \
    --output-md "${OUT}/${SCENE}_official_eval.md"
done

gs_python scripts/aggregate_public_query_evaluations.py \
  --inputs "${OUT}/americano_official_eval.json" \
           "${OUT}/espresso_official_eval.json" \
           "${OUT}/split-cookie_official_eval.json" \
  --expected-queries 7 \
  --require-complete \
  --output-json "${OUT}/official_eval.json" \
  --output-md "${OUT}/official_eval.md"
```

For the separate four-scene extension, replace `paper_public3` with
`release_public4_extension` in the builder, preflight, and runner; include
`chickchicken` in the evaluation loop and use `--expected-queries 9`. Never
merge the 3-scene paper and 4-scene extension under one result label.

The commands above use portable GPU 0. On a verified multi-GPU host, replace
the builder's `--gpus 0` and both consumers' `--gpu 0` with the same list, for
example `--gpus 0 1 2` and `--gpu 0 1 2`. Preflight keeps
`--require-visible-gpu` so a nonexistent assignment fails before execution.

For a canary subset, omit the formal protocol id and pass `--allow-incomplete`
to the builder. Such rows are marked `protocol_complete: false` and are
rejected by `--strict-release`; they must never be reported as a complete
benchmark result.

When evaluating a subset such as `time_sensitive`, pass the same manifest to
`evaluate_public_query_protocol.py --query-manifest "${OUT}/manifest.jsonl"`.
This prevents unrequested protocol queries from appearing as missing results.

To compare a renderer while holding the completed Stage-1 tracks, Qwen
selection, and Gaussian entity fixed, use the re-render utility. It writes a
new evaluator-compatible output root and, by default, exports only numerical
masks and `validation.json` rather than videos or overlays:

```bash
gs_python scripts/rerender_query_outputs.py \
  --manifest "${OUT}/manifest.jsonl" \
  --output-root reports/public_time_boundary_gated_v5_rerender \
  --profile public_time_boundary_gated_v5 \
  --gpu 0 \
  --require-complete
```

The new root includes `query_root_map.json` and `dataset_dir_map.json`, so it
can be evaluated with the same strict evaluator command. This utility never
reruns or changes the original selection and never uses a full-scene fallback.

### Referring evaluation — R4D-Bench-QA

```bash
# 1) Build a manifest with official query ids.
source scripts/common.sh
RUN_ROOT="reports/RELEASE_R4D_DENSE89_RENDERER_CONSISTENT_$(date -u +%Y%m%dT%H%M%SZ)"
R4D_ROOT="${GS_DATA_ROOT}/benchmarks/r4d_bench_qa"
gs_python scripts/build_r4d_query_manifest.py \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --query-metadata "${R4D_ROOT}/evaluation/R4D-Bench_queries.json" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --output "${RUN_ROOT}/manifest.jsonl" \
  --output-root "${RUN_ROOT}/query_root" \
  --gpus 0

gs_python scripts/preflight_query_batch.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --strict-release \
  --gpu 0 \
  --require-visible-gpu \
  --create-output-root

# 2) Run the query pipeline.
gs_python scripts/run_query_batch.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --gpu 0 \
  --force-rerun \
  --strict-release \
  --timeout 3600

# 3) Project the fixed Gaussian entities into the official source-camera
# frames. This reads only published frame ids, never segmentation masks; it
# keeps the Stage-1 tracks and Qwen selection from step 2 unchanged.
CAMERA_ROOT="${RUN_ROOT}/source_camera_views"
gs_python scripts/rerender_query_outputs.py \
  --manifest "${RUN_ROOT}/manifest.jsonl" \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --output-root "${CAMERA_ROOT}" \
  --profile r4d_renderer_consistent \
  --gpu 0 \
  --require-complete

# 4) Evaluate the source-camera projections. The renderer retains the full
# reconstruction test grid for temporal metrics and adds exact official cameras
# for spatial masks.
gs_python scripts/evaluate_ours_benchmark.py \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --query-root-map "${CAMERA_ROOT}/query_root_map.json" \
  --dataset-dir-map "${CAMERA_ROOT}/dataset_dir_map.json" \
  --query-manifest "${RUN_ROOT}/manifest.jsonl" \
  --output-json "${CAMERA_ROOT}/official_eval.json" \
  --output-md "${CAMERA_ROOT}/official_eval.md" \
  --require-complete
```

This example is portable on one GPU. For multi-GPU execution, replace both
`--gpus 0` and `--gpu 0` with the same verified list. An R4D canary instead
uses `--allow-incomplete --scenes <subset>` and cannot be passed to a strict
release run.

The source-camera export is a rendering protocol, not a mask shortcut: each
foreground pixel remains a selected-Gaussian projection, and the formal profile
still requires its synchronized Stage-1 boundary gate. The benchmark utility
extracts only `ground_truth.frames[].frame_id` to identify the published camera
views; it does not inspect or pass any `segmentation` payload into inference.

The released mainline profile is `r4d_renderer_consistent`. It uses the same
fine-stage deformed Gaussian state for multi-frame lifting and final alpha-splat
projection, and fails if that renderer geometry is unavailable. Qwen planning,
semantic assignment, and final entity selection remain enabled for every query,
including multi-target and zero-target cases. The earlier
`release_r4d_dense89` / `r4d_boundary_gated_v5` identity remains in the registry
only to reproduce pre-fix runs that used an analytic trajectory approximation.

The renderer-consistency release gate was checked on one English query from
each R4D category before freezing this protocol. These are canary results, not a
dense-set aggregate: temporal `torchchocolate_q1` reached Acc/vIoU/tIoU
`99.59/62.45/99.28`, multi-target `torchchocolate_q2` reached
`99.59/83.10/99.28`, and zero-target `torchchocolate_q4` reached
`100/100/100`. The first two use all 71 official source-camera masks with no
warning. A final release result still requires all 89 query ids and
`--require-complete`.

The preflight and query pipeline require `phase: refergaussian`,
`temporal_warp_type: refergaussian`, and `warp_enabled: true` in each run's
`config.yaml`. This prevents accidental evaluation of a baseline or legacy
reconstruction as ReferGaussian.

Output files:
- `${RUN_ROOT}/query_root_map.json`
- `${RUN_ROOT}/dataset_dir_map.json`
- `${CAMERA_ROOT}/official_eval.json` (per-query Acc/vIoU/tIoU + summary)
- `${CAMERA_ROOT}/official_eval.md`
- `${CAMERA_ROOT}/rerender_summary.json` (camera-export provenance)

Metric rule used in both public and R4D evaluators:
- Empty-set rule is enabled: if GT and a verified `semantic_empty` prediction are both empty (temporal union = 0), `vIoU = 1.0` and `tIoU = 1.0`. An empty-GT query with any predicted activity scores `0.0` for both values. Phrase misses, missing evidence, and pipeline failures are `unresolved`; strict reports reject them instead of scoring them as empty answers.
- Every evaluator summary also reports the non-empty-query subset and empty-target correctness separately. Report these alongside headline `Acc`/`vIoU`/`tIoU` so background-heavy timelines or zero-target queries cannot conceal ordinary grounding failures.

Important benchmark note:
- Current HF `LiYacheng/r4d-bench-qa` artifacts provide dense GT for 36-query and 89-query tiers.
- The larger language-only extension set does not always include dense masks, so strict `vIoU`/`tIoU` cannot be computed for those entries without extra GT alignment files.
- Prefer the official query metadata file `R4D-Bench_queries.json` as the source of truth for query ids and semantics.
  Legacy releases whose question field is Chinese use the versioned, reviewed English translation map keyed by that official id.
  Avoid retyping benchmark queries in ad-hoc shell scripts, because even small wording drift can change the target object set.

The currently downloadable dense release tier contains 12 scenes and 89
English queries (36 temporal single-target, 29 multi-target/reasoning, and 24
zero-target/distractor). Its source hashes and status are frozen in
`configs/benchmarks/release_protocols.json`. The paper reports 12 scenes and
266 sentence queries, but an executable 266-query dense-mask artifact is not
present in the current release, so the 12/89 run must be labeled separately.

The former 8-scene / 58-query release selection is retained for archival
compatibility only:

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

This produces the archival 8-scene / 58-query set. Pass the same manifest to
the evaluator so an 89-query source file cannot silently become a partial
58/89 report. Do not present this selected subset as the paper's 12-scene
protocol or as the dense 12/89 release result.

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
