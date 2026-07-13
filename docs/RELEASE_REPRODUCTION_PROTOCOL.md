# Release Reproduction Protocol

This document defines the only result protocols that may be compared with the
accepted-paper tables in the README. It is intentionally stricter than an
exploratory run: a partial query set is useful for debugging, but is not a
benchmark result.

## Reconstruction

For every compared scene, train ReferGaussian and the matched 4D Gaussian
Splatting baseline from the same source data, scene configuration, iteration
budget, explicit `REFERGAUSSIAN_SEED`, and metric mode. Run `scripts/eval.sh` and
`scripts/eval_baseline.sh` without `GS_SKIP_FULL_METRICS=1` for a final table.

Keep these files for each method:

- `config.yaml`
- `train_meta.json` and `render_meta.json`
- `results.json` and `metrics.json`

Do not include a scene in a paper-facing reconstruction average if either
method has PSNR below 20 dB.

## Public 4DLangSplat Protocol

The public time-sensitive release evaluates four HyperNeRF scenes:
`americano`, `espresso`, `split-cookie`, and `chickchicken`. Queries are
generated only from the pinned annotation snapshot and only from its
`temporal_state_reference` rows. The resulting fixed protocol has 9 queries.

Run the per-scene evaluator with `--require-complete`, then combine the four
JSON reports with `scripts/aggregate_public_query_evaluations.py
--expected-queries 9 --require-complete`.

Before launching a query batch, run `GSAM2_INSTALL_EDITABLE=1 bash
scripts/setup_grounded_sam2.sh` and `gsam2_python
scripts/check_query_runtime.py --require-qwen`. The pipeline repeats both
runtime provenance checks before Stage-1 so that a missing Qwen checkpoint or
an unrelated editable SAM2 package fails before expensive inference begins.

## R4D-Bench-QA Protocol

The fixed release protocol contains 58 English queries across eight scenes:

- DyNeRF: `coffee_martini`, `sear_steak`, `cut_roasted_beef`
- HyperNeRF: `cut_lemon`, `espresso`, `keyboard`, `split_cookie`, `torchchocolate`

The benchmark includes 42 non-empty queries, 16 zero-target queries, and 20
multi-target queries. Final R4D evaluation must use the official query id
manifest and `scripts/evaluate_ours_benchmark.py --require-complete`.

Release query batches must use
`scripts/run_query_batch_two_gpu.py --strict-release --force-rerun`. A strict
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

Use `public_time_boundary_gated_v5` for the public protocol and
`r4d_boundary_gated_v5` for R4D-Bench-QA. Both profiles render the selected
Gaussian entity first, then apply only a synchronized dilated Stage-1 boundary
gate. A stale nearest-frame boundary or a direct 2D-mask output makes the query
run fail after saving its diagnostics; it cannot enter a release aggregate.

`r4d_multi_instance_boundary_v6` is an opt-in R4D canary profile for a
compositional singular referring expression that yields multiple spatially
distinct same-category detector candidates. It retains those candidates as
separate full-timeline Stage-1 tracks and separately lifted Gaussian entities,
records the selection policy in `grounded_sam2_query_tracks.json`, and keeps
every final mask on the same boundary-gated Gaussian-rendering contract. Run
and report it in a distinct output root; it does not alter the public v5
baseline.

The Qwen query planner remains required in every profile. V6 skips only the
otherwise redundant post-lifting Qwen assignment and selector calls when a
deterministic contract is satisfied: one planned subject, no successor phrase,
an exact matching declared `multi_hypothesis` group with at least two object
ids, and a separately lifted entitybank member for every id. All other v6
queries and all v5 profiles run the ordinary Qwen assignment and selector.

## Reporting Rules

Every final query report must include:

- expected and valid query counts, with complete coverage required;
- spatial-frame coverage: a missing rendered prediction at an annotated GT
  frame is scored as zero IoU and makes `--require-complete` fail;
- source-camera match coverage, including the count of exact official-camera
  masks used for vIoU;
- overall Acc/vIoU/tIoU;
- non-empty-only Acc/vIoU/tIoU;
- zero-target correctness and false-positive count;
- manifest, evaluator output, model/data revision manifests, and the
  ReferGaussian run `config.yaml`.

The evaluator accepts an empty GT and empty prediction as a correct zero-target
answer. This rule never permits an empty set of *queries* to become a 100%
benchmark, and an empty GT with predicted activity receives zero vIoU/tIoU.

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
record Stage-1 boundary coverage for every active selected entity. The R4D
numeric-only profile may skip qualitative video exports. Qwen planning is
always required; only the explicitly documented v6 declared-instance contract
can avoid duplicate post-lifting Qwen assignment and selection work.
