# Release Reproduction Protocol

This document separates accepted-paper, executable dense-release, extension,
and archival-subset protocols. A partial query set is useful for debugging,
but is not a benchmark result. Exact sizes and source hashes are frozen in
[`configs/benchmarks/release_protocols.json`](../configs/benchmarks/release_protocols.json),
and metric compatibility is defined in [METRICS.md](METRICS.md).

## Reconstruction

For every compared scene, train ReferGaussian and the matched 4D Gaussian
Splatting baseline from the same source data, scene configuration, iteration
budget, explicit `REFERGAUSSIAN_SEED`, and metric mode. Run `scripts/eval.sh` and
`scripts/eval_baseline.sh` without `GS_SKIP_FULL_METRICS=1` for a final table.

Keep these files for each method:

- `config.yaml`
- `train_meta.json` and `render_meta.json`
- `results.json` and `metrics.json`

Do not remove a completed scene after looking at its score. A quality gate or
invalid-data exclusion must be declared before evaluation and reported with
the excluded scene list. Fresh release comparison uses all scenes declared by
its protocol and reports per-scene rows as well as the aggregate.

## Public 4DLangSplat Protocols

The accepted-paper protocol evaluates `americano`, `split-cookie`, and
`espresso`, for 7 annotation-derived time-sensitive queries. Build it with:

```bash
gs_python scripts/build_public_query_manifest.py \
  --output "${OUT}/manifest.jsonl" \
  --output-root "${OUT}/query_root" \
  --protocol-json "${PROTOCOL_JSON}" \
  --profile public_time_boundary_gated_v5 \
  --scenes americano split-cookie espresso
```

The separate release extension adds `chickchicken`, for 4 scenes and 9
queries. Omit `--scenes` to build that extension. Both derive text only from
the pinned annotation snapshot's `temporal_state_reference` rows; they must
not be aggregated under the same protocol name.

Run the per-scene evaluator with `--require-complete`, then combine reports
with `scripts/aggregate_public_query_evaluations.py`. Use
`--expected-queries 7` for paper Public-3 or `--expected-queries 9` for the
Public-4 extension.

Before launching a query batch, run `GSAM2_INSTALL_EDITABLE=1 bash
scripts/setup_grounded_sam2.sh` and `gsam2_python
scripts/check_query_runtime.py --require-qwen`. The pipeline repeats both
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

The former selected 58-query subset is archival only:

- DyNeRF: `coffee_martini`, `sear_steak`, `cut_roasted_beef`
- HyperNeRF: `cut_lemon`, `espresso`, `keyboard`, `split_cookie`, `torchchocolate`

It includes 42 non-empty, 16 zero-target, and 20 multi-target queries. It may
be reproduced with its exact manifest for compatibility, but cannot be labeled
as the paper 12-scene protocol or the full dense 12/89 release.

Release query batches must use
`scripts/run_query_batch.py --strict-release --force-rerun`. A strict
manifest may set the official query id, query text, run/data/output paths, and
GPU assignment, but may not contain per-query environment overrides.

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
`r4d_boundary_gated_v5` for R4D-Bench-QA. Both profiles render the selected
Gaussian entity first, then apply only a synchronized dilated Stage-1 boundary
gate. A stale nearest-frame boundary or a direct 2D-mask output makes the query
run fail after saving its diagnostics; it cannot enter a release aggregate.

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

The evaluator accepts an empty GT and empty prediction as a correct zero-target
answer. This rule never permits an empty set of *queries* to become a 100%
benchmark, and an empty GT with predicted activity receives zero vIoU/tIoU.
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
Gaussian entity, final mask, and validation contract. Qwen planning is
always required; only the explicitly documented v6 declared-instance contract
can avoid duplicate post-lifting Qwen assignment and selection work.
