# R4DGS Release Reproduction Protocol

This repository releases the ReferGaussian referring-inference and evaluation
pipeline. It treats a standard 4D Gaussian scene as a frozen input and includes
a thin wrapper around the pinned upstream trainer for end-to-end reproduction.

## Scope

The supported path contains:

1. Refer-Planner query decomposition;
2. Grounded-SAM2 multi-frame mask evidence;
3. training-free mask-supported Gaussian entity lifting;
4. EntityBank selection and Gaussian-only mask rendering;
5. strict Public and R4D-Bench-QA evaluators.

The repository intentionally excludes custom reconstruction code, scene-quality
metrics, scene checkpoints, model weights, rendered benchmark outputs,
experiment logs, and ablation runners.

## Frozen 4DGS Input

Use the pinned, unmodified upstream 4DGaussians checkout prepared by:

```bash
bash scripts/bootstrap_external.sh
bash scripts/setup_4dgs_env.sh cuda121
gs_python scripts/train_4dgs.py --dataset hypernerf --scene misc/americano
```

Each query-ready scene run must provide:

```text
<run_dir>/cfg_args
<run_dir>/point_cloud/iteration_*/point_cloud.ply
<run_dir>/test/ours_*/renders/*
```

`scripts/validate_4dgs_run.py` and batch preflight reject missing inputs. The
checkpoint is never optimized by ReferGaussian.

## Protocol Identities

The executable R4D dense tier contains 89 English queries over 12 scenes:

- 36 temporal/dynamic queries;
- 29 multi-target or reasoning queries;
- 24 zero-target or distractor queries.

The accepted paper reports a separate 12-scene, 266-sentence annotation
identity. It is kept separate because the complete 266-sentence executable
artifact is not part of this repository.

Public identities are:

- `paper_public3`: 3 scenes, 7 dynamic queries;
- `release_public4_extension`: 4 scenes, 9 dynamic queries.

All counts, source hashes, category counts, and scene lists are frozen in
`configs/benchmarks/release_protocols.json`.

## Strict Execution

Formal runs must:

1. build a versioned protocol and manifest;
2. run `scripts/preflight_query_batch.py --strict-release`;
3. use one serial worker per GPU;
4. preserve `batch_provenance.json` and every query validation record;
5. evaluate with `--require-complete`;
6. write all new outputs to a new, untracked directory.

Missing queries, unresolved selections, missing masks, or source-camera
mismatches are errors. They cannot be silently skipped when producing an
aggregate.

## Empty-Set Semantics

A zero-target query receives a perfect empty-set score only when both ground
truth and prediction are empty. An empty prediction for non-empty ground truth
scores zero. Missing evidence is an evaluation error, not an empty prediction.

## Asset Policy

Git tracks source code, tests, benchmark protocol metadata, documentation, and
project-page figures only. `.gitignore` excludes `data/`, `models/`, `runs/`,
`reports/`, experiment directories, checkpoints, point clouds, archives, and
videos. `scripts/check_release.py` independently rejects those artifact types
if they are accidentally staged.

Metric formulas and aggregation rules are defined in [METRICS.md](METRICS.md).
