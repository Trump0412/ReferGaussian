# R4DGS Reproduction Status

Updated: 2026-08-06

## Release Boundary

The public release is inference-only. ReferGaussian consumes frozen upstream
4DGS checkpoints. Scene training is out of scope, and no scene optimizer,
checkpoint, weight file, or benchmark result bundle is included.

The release gate verifies:

- the camera-ready title and DOI;
- the 12-scene/89-query dense R4D registry;
- English-only released questions;
- standard upstream 4DGS input layout;
- pinned Grounded-SAM2 and Qwen snapshots;
- complete-evaluation and zero-target contracts;
- absence of scene-training, ablation, checkpoint, and experiment-output code.

## Reported Results

The README and project page retain accepted-paper referring results and label
them as paper-reported. They do not contain scene-quality metrics.

The latest complete Public time-sensitive audit covers four scenes and nine
queries with 240/240 annotated masks and no query below 10% vIoU. The separate
time-agnostic audit covers four scenes and all 20 annotated categories. These
audits preserve their source commit, manifest, evaluator, and checkpoint input
identity outside the Git repository.

The executable R4D registry contains 12 scenes and 89 queries, but a complete
fresh 89-query aggregate is not yet promoted in this release. Small category
canaries must not be reported as the full benchmark mean.

## External Reproduction

Source checkout, environment setup, protocol creation, manifest creation,
strict preflight, inference, and evaluation are documented in the README. A
fresh run still requires users to provide:

- prepared scene data;
- a compatible upstream 4DGS checkpoint and test renders;
- pinned Grounded-SAM2, Grounding DINO, SAM2, and Qwen model snapshots;
- Public or R4D-Bench-QA annotations.

No checkpoint or model weight is embedded in the repository. Dataset and model
download scripts write manifests beside external assets rather than adding the
assets to Git.

## Known Protocol Distinction

The accepted paper describes 12 scenes and 266 sentence-level annotations.
The frame-aligned dense artifact currently released for execution contains 89
official English query ids over the same 12 scenes. These identities must not
be merged or relabeled until the complete 266-query mapping is published.
