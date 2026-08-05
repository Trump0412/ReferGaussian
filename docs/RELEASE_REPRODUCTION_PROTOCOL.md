# Release Reproduction Protocol

This document separates accepted-paper, executable dense-release, extension,
and archival-subset protocols. A partial query set is useful for debugging,
but is not a benchmark result. Exact sizes and source hashes are frozen in
[`configs/benchmarks/release_protocols.json`](../configs/benchmarks/release_protocols.json),
and metric compatibility is defined in [METRICS.md](METRICS.md).

## Reconstruction

Reconstruction identities are frozen in
[`configs/benchmarks/reconstruction_release_v1.json`](../configs/benchmarks/reconstruction_release_v1.json).
The accepted-paper table and the later selected-8 candidate remain reported,
non-executable identities because their exact camera-ready settings and raw
inputs are unresolved. Two executable identities are retained: the August
seed-6666 audit baseline (`release_reconstruction_v1`) and the historical
effective-seed protocol (`release_reconstruction_v2_paper_compat`). Neither is
relabeled as paper reproduction.

Run the complete matched protocol from a clean checkout:

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

The runner validates the source commit, external 4DGaussians commit, patch-file
hashes, the exact resulting patched-tree diff and generated-file hashes,
dataset/config identities, the selected protocol's seed, 14,000 fine iterations, and full
metric mode before launching work. It trains ReferGaussian and the matched 4D
Gaussian Splatting baseline from the same source data and scene configuration.
It verifies identical render filenames and dimensions and reports LPIPS-vgg as
the headline LPIPS metric, with LPIPS-alex retained as a diagnostic.
Evaluation streams the complete test-camera split directly to disk. It skips
train-camera renders and preview-video generation because neither contributes
to reconstruction metrics; this changes memory use and runtime only, not the
evaluated frame set or reductions.

The v2 identity makes seed `0` explicit because the historical training path
reset the RNG to zero after parsing a nominal seed. It also freezes the
historical constant contextual-warp learning rate and tube sigma `0.32`.
The corrected runner applies seed zero directly after backend initialization,
so the effective seed is now visible in provenance rather than accidental.
The v1 identity remains executable with seed `6666`, sigma `0.34`, and its
shared exponential warp schedule; its prior behavior is not changed by v2.

Keep these files for each method:

- `config.yaml`
- `train_meta.json` and `render_meta.json`
- `results.json` and `metrics.json`

Do not remove a completed scene after looking at its score. The runner refuses
to aggregate until all 12 declared scenes and both methods are complete, uses
a scene-equal arithmetic mean, and performs no post-hoc PSNR filtering. A
`--scenes` subset is an explicitly incomplete canary and cannot produce a final
aggregate.

## Public 4DLangSplat Protocols

The accepted-paper protocol evaluates `americano`, `split-cookie`, and
`espresso`, for 7 annotation-derived time-sensitive queries. Build it with:

```bash
gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --protocol-json "${PROTOCOL_JSON}" \
  --protocol-id paper_public3 \
  --annotation-root "${GS_DATA_ROOT}/benchmarks/4dlangsplat/HyperNeRF-Annotation" \
  --profile public_time_boundary_gated_v5 \
  --gpus 0
```

The separate release extension adds `chickchicken`, for 4 scenes and 9
queries. Replace the protocol id with `release_public4_extension`. Both derive
text only from the pinned annotation snapshot's `temporal_state_reference`
rows and verify the registered video/COCO source hashes; they must not be
aggregated under the same protocol name.

These are time-sensitive protocols only. The 4D LangSplat reference
time-agnostic path evaluates every COCO category with a mask: 15 scene-local
prompts on the three paper scenes and 20 on the four-scene extension. The
protocol builder currently exposes only one static base-object row per scene,
so those rows must not be reported as the complete time-agnostic benchmark.
The compatibility evaluator also computes Acc on the full metadata timeline,
whereas the inspected 4D LangSplat reference computes it on sparse annotated
evaluation frames. Keep the accepted-paper table, compatibility results, and
any future reference-exact rerun under separate metric identities.

Run the per-scene evaluator with `--require-complete`, then combine reports
with `scripts/aggregate_public_query_evaluations.py`. Use
`--expected-queries 7` for paper Public-3 or `--expected-queries 9` for the
Public-4 extension.

Before launching a query batch, run `GSAM2_INSTALL_EDITABLE=1 bash
scripts/setup_grounded_sam2.sh` and `gsam2_python
scripts/check_query_runtime.py --require-qwen --require-pinned-manifest`. The pipeline repeats both
runtime provenance checks before Stage-1 so that a missing Qwen checkpoint or
an unrelated editable SAM2 package fails before expensive inference begins.

## R4D-Bench-QA Protocols

The paper reports 12 scenes and 266 sentence queries. The current release does
not contain an executable 266-query dense-mask artifact, so this claim remains
reported provenance rather than a fresh-run target.

The executable dense tier contains 89 English queries across 12 scenes:
`americano`, `coffee_martini`, `cook-spinach`, `cut_lemon`,
`cut_roasted_beef`, `espresso`, `flame_salmon`, `flame_steak`, `keyboard`,
`sear_steak`, `split_cookie`, and `torchchocolate`. It contains 36 temporal
single-target, 29 multi-target/reasoning, and 24 zero-target/distractor
queries. A final dense-tier evaluation must use all 89 official query IDs and
`--require-complete`.

Build the formal dense manifest with the pinned dense GT and metadata files:

```bash
R4D_ROOT="${GS_DATA_ROOT}/benchmarks/r4d_bench_qa"
gs_python scripts/build_r4d_query_manifest.py \
  --benchmark "${R4D_ROOT}/benchmark_all_queries.json" \
  --query-metadata "${R4D_ROOT}/evaluation/R4D-Bench_queries.json" \
  --protocol-id release_r4d_dense89_renderer_consistent \
  --profile r4d_renderer_consistent \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --gpus 0
```

The former selected 58-query subset is archival only:

- DyNeRF: `coffee_martini`, `sear_steak`, `cut_roasted_beef`
- HyperNeRF: `cut_lemon`, `espresso`, `keyboard`, `split_cookie`, `torchchocolate`

It includes 42 non-empty, 16 zero-target, and 20 multi-target queries. It may
be reproduced with its exact manifest for compatibility, but cannot be labeled
as the paper 12-scene protocol or the full dense 12/89 release.

Release query batches must use
`scripts/run_query_batch.py --strict-release --force-rerun --protocol-id
<id>`. Before execution, run `scripts/preflight_query_batch.py` with the same
protocol, profile, and GPU list plus `--strict-release
--require-visible-gpu`. A strict
manifest may set the official query id, query text, run/data/output paths, and
GPU assignment, but may not contain per-query environment overrides.

Formal examples default to GPU 0. A multi-GPU run replaces the builder's
`--gpus 0` and both consumers' `--gpu 0` with the same visible list. Canary
subsets require the builder's `--allow-incomplete`; they are marked incomplete
and cannot pass strict release validation.

After the query pipeline completes, use
`scripts/rerender_query_outputs.py --benchmark <benchmark.json>` before the
official evaluator. This renders the already selected Gaussian entity into the
official frame-id cameras and preserves the reconstruction test grid for
temporal scoring. The utility reads only `ground_truth.frames[].frame_id` from
the benchmark to select cameras; it never reads or passes segmentation masks to
inference. Direct source-camera outputs are preferred by the evaluator over a
legacy nearest-time test-camera match, which is necessary when the camera moves.

Run the matched `scripts/eval.sh` before beginning a query batch. The query
renderer requires that ReferGaussian test render grid as its temporal reference
and deliberately fails when it is missing rather than substituting a proxy
sequence.

Use `public_time_boundary_gated_v5` for the public protocol and
`r4d_renderer_consistent` for R4D-Bench-QA. The score-only
`public_time_boundary_gated_v5_numeric` variant is also formal. These are the
current mainline profiles accepted by strict release mode. They render the
selected Gaussian entity first, then apply only a synchronized dilated Stage-1
boundary gate. The R4D profile additionally requires the same fine-stage
deformed Gaussian geometry during lifting and final projection. A stale
nearest-frame boundary, missing renderer geometry, or a direct 2D-mask output
makes the query run fail after saving its diagnostics; it cannot enter a
release aggregate. The older `release_r4d_dense89` identity remains registered
only for exact reproduction of analytic-geometry diagnostics.

Protocol promotion was gated by three English `torchchocolate` canaries, one
per released category. On the source-camera evaluator, q1 temporal scored
`99.59/62.45/99.28`, q2 multi-target scored `99.59/83.10/99.28`, and q4
zero-target scored `100/100/100` (Acc/vIoU/tIoU). This `3/89` gate is not a
dense benchmark aggregate; the formal result requires all 89 ids.

The numeric profile preserves all inference and scored masks while omitting
videos, overlay frames, and annotation diagnostic montages. Missing qualitative
artifacts therefore cannot turn an otherwise complete score-only query into a
pipeline failure.

The Qwen query planner, semantic assignment, and final entity selection remain
required for every released mainline query, including multi-target and
zero-target cases.

## Reporting Rules

Every final query report must include:

- expected and valid query counts, with complete coverage required;
- spatial-frame coverage: a missing rendered prediction at an annotated GT
  frame is scored as zero IoU and makes `--require-complete` fail;
- source-camera match coverage, including the count of exact official-camera
  masks used for vIoU;
- overall Acc/vIoU/tIoU;
- the evaluator `metric_protocol.id`, explicit `temporal_frame_accuracy`,
  `mean_annotated_frame_iou`, and `annotated_volume_iou` audit fields;
- non-empty-only Acc/vIoU/tIoU;
- zero-target correctness and false-positive count;
- manifest, evaluator output, model/data revision manifests, and the
  ReferGaussian run `config.yaml`.
- `batch_provenance.json`, written before query execution with manifest, Git
  diff, core script, reconstruction checkpoint, and dataset metadata hashes.

The evaluator accepts an empty GT and a verified `semantic_empty` prediction as
a correct zero-target answer. Every selection is labeled `resolved`,
`semantic_empty`, or `unresolved`; phrase misses, missing evidence, and pipeline
failures are unresolved and make `--require-complete` fail. This rule never
permits an empty set of *queries* to become a 100% benchmark, and an empty GT
with predicted activity receives zero vIoU/tIoU.
The compatibility fields called `Acc` and `vIoU` are not relabeled as the
paper's exact-set Acc and exhaustive full-volume vIoU; see `METRICS.md`.

## Release Guards

The published query path requires a run config with:

```yaml
phase: refergaussian
temporal_warp_type: refergaussian
warp_enabled: true
```

It uses training-free `mask_supported_lifting`, requires actual ReferGaussian
test renders, and reports an empty Gaussian projection as a failure. It does
not replace an entity with an all-scene mask, a direct 2D mask, or raw source
RGB frames. Formal profiles keep the final prediction Gaussian-supported and
record Stage-1 boundary coverage for every active selected entity. Numeric-only
profiles may skip qualitative video exports but retain the same planner,
Gaussian entity, final mask, and validation contract. Qwen planning is always
required. Strict mode pins `Qwen3-VL-8B-Instruct` to the source revision in
`scripts/check_query_runtime.py`; an environment revision override is accepted
only by non-strict exploratory runs.

Grounded-SAM2 point prompting is seeded with `GSAM2_INFERENCE_SEED=0` by
default. The backend seeds Python, NumPy, and Torch before inference and writes
the effective value into `grounded_sam2_query_tracks.json`. Formal comparisons
must retain this seed or register the changed Stage-1 identity explicitly.
