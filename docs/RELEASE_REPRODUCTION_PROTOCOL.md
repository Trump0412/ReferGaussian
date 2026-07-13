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

## R4D-Bench-QA Protocol

The fixed release protocol contains 58 English queries across eight scenes:

- DyNeRF: `coffee_martini`, `sear_steak`, `cut_roasted_beef`
- HyperNeRF: `cut_lemon`, `espresso`, `keyboard`, `split_cookie`, `torchchocolate`

The benchmark includes 42 non-empty queries, 16 zero-target queries, and 20
multi-target queries. Final R4D evaluation must use the official query id
manifest and `scripts/evaluate_ours_benchmark.py --require-complete`.

## Reporting Rules

Every final query report must include:

- expected and valid query counts, with complete coverage required;
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
RGB frames. The R4D numeric-only profile may skip qualitative video exports,
but it does not bypass Qwen planning, semantic assignment, or entity selection.
