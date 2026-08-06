# R4DGS Reproduction Status

Updated: 2026-08-07

## Release Boundary

ReferGaussian consumes frozen standard upstream 4DGS checkpoints. A thin
wrapper is included to train and test-render those inputs with the pinned
upstream code; no custom scene optimizer, checkpoint, weight file, or benchmark
result bundle is included.

The release gate verifies:

- the camera-ready title and DOI;
- the 12-scene/89-query dense R4D registry;
- English-only released questions;
- standard upstream 4DGS input layout;
- pinned Grounded-SAM2 and Qwen snapshots;
- complete-evaluation and zero-target contracts;
- absence of custom reconstruction, ablation, checkpoint, and experiment-output code.

## Reported Results

The README and project page retain accepted-paper referring results and label
them as paper-reported. They do not contain scene-quality metrics.

The latest complete Public dynamic-query audit covers four scenes and nine
queries with 240/240 annotated masks and no query below 10% vIoU. This audit
preserves its source commit, manifest, evaluator, and checkpoint input identity
outside the Git repository.

The executable R4D registry contains 12 scenes and 89 queries, but a complete
fresh 89-query aggregate is not yet promoted in this release. Small category
canaries must not be reported as the full benchmark mean.

## External Reproduction

Source checkout, environment setup, standard upstream 4DGS training, strict
inference, and evaluation are documented in the README. A fresh run still
requires users to provide:

- prepared scene data;
- a compatible upstream 4DGS checkpoint and test renders;
- pinned Grounded-SAM2, Grounding DINO, SAM2, and Qwen model snapshots;
- Public or R4D-Bench-QA annotations.

No checkpoint or model weight is embedded in the repository. Dataset and model
download scripts write manifests beside external assets rather than adding the
assets to Git.

## Frozen Evaluation Identity

The camera-ready R4D-Bench-QA protocol is the released set of 89 English
sentence-level queries over 12 scenes. The Public protocol is the released set
of 9 English time-sensitive queries over 4 scenes. Both use frame-wise Acc and
time-sensitive vIoU as defined in `METRICS.md`.
