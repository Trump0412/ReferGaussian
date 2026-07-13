# Release Verification Status

This page separates accepted-paper tables from reproducibility evidence that
has been regenerated from the public source tree.

## Release Contract

The release contract is defined by
[RELEASE_REPRODUCTION_PROTOCOL.md](RELEASE_REPRODUCTION_PROTOCOL.md):

- R4D-Bench-QA: 8 fixed scenes and 58 English queries.
- Public 4DLangSplat time-sensitive protocol: 4 scenes and 9 annotation-derived
  queries.
- Final reports require official query ids, `--strict-release --force-rerun`,
  `--require-complete`, model/data revision manifests, and the ReferGaussian
  run configuration.
- A query with missing spatial prediction coverage is scored with zero IoU for
  the missing annotated frame and cannot pass the complete-coverage gate.
- Empty-target scores are reported separately. Only an empty prediction for an
  empty target receives the empty-set score.

## What Is Verified by the Public Source

The release gate and regression suite verify the executable contract:

- no source-RGB, full-scene, direct-2D-mask, sparse-entity, or relaxed-GSAM
  default fallback in the published query path;
- formal v5 outputs are selected-Gaussian projections gated only by synchronized
  Stage-1 boundary neighborhoods; stale boundary matches fail with an auditable
  `validation.json` record rather than contributing a score;
- Qwen planning, semantic assignment, and entity selection remain enabled in
  published profiles;
- Gaussian membership refinement ranks candidates by rendered multi-frame
  overlap rather than proxy evidence alone;
- public and R4D evaluators reject incomplete query or spatial-frame coverage;
- ReferGaussian and the integrated 4DGS control accept the same explicit seed,
  and the external bootstrap restores that seed after backend initialization.

Run these checks before reporting a new result:

```bash
source scripts/common.sh
gs_python scripts/check_release.py
gs_python -m unittest discover -s tests -v
```

## Reporting Boundary

The numerical tables in the README and project page are the accepted-paper
reported values. Historical exploratory artifacts, partial query subsets,
legacy run identities, per-scene environment overrides, and incomplete
baseline records are not release verification evidence and must not be mixed
into a new aggregate.

A fresh result becomes release-verifiable only when its complete manifest,
batch summary, evaluator output, spatial coverage, run configuration, and
model/data revisions are retained together.
