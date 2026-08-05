# Release Verification Status

This page separates accepted-paper tables from reproducibility evidence that
has been regenerated from the public source tree.

## Release Contract

The release contract is defined by
[RELEASE_REPRODUCTION_PROTOCOL.md](RELEASE_REPRODUCTION_PROTOCOL.md):

- Paper R4D: reported as 12 scenes / 266 queries; its exact executable dense
  artifact has not been located in the current release.
- Dense R4D release candidate: 12 scenes / 89 English queries.
- Historical R4D subset: 8 scenes / 58 queries, archival and noncanonical.
- Paper Public: 3 scenes / 7 annotation-derived time-sensitive queries.
- Public extension: 4 scenes / 9 annotation-derived time-sensitive queries.
- Public time-agnostic: freshly reproduced for all 20 annotated scene-local
  categories on the four-scene extension, using explicitly identified vanilla
  4DGS checkpoints. The paper three-scene subset contains 15 categories.
- Matched reconstruction release: two separate executable 12-scene identities
  preserve the seed-6666 audit baseline and historical effective-seed behavior;
  their fresh full results are pending and neither is the accepted-paper table.
- Final reports require official query ids, `--strict-release --force-rerun`,
  `--require-complete`, model/data revision manifests, and the ReferGaussian
  run configuration.
- A query with missing spatial prediction coverage is scored with zero IoU for
  the missing annotated frame and cannot pass the complete-coverage gate.
- Empty-target scores are reported separately. Only a verified
  `semantic_empty` prediction for an empty target receives the empty-set score;
  unresolved inference is excluded and fails complete coverage.
- Evaluator outputs identify compatibility metrics explicitly; see
  [METRICS.md](METRICS.md). Paper exact-set Acc and exhaustive full-volume vIoU
  are not inferred when the required identity/mask supervision is unavailable.
- Public compatibility Acc is evaluated on the full metadata timeline. It is
  not presented as bit-identical to the sparse annotated-frame Acc in the
  inspected 4D LangSplat reference evaluator.

## What Is Verified by the Public Source

The release gate and regression suite verify the executable contract:

- no source-RGB, full-scene, direct-2D-mask, sparse-entity, or relaxed-GSAM
  default fallback in the published query path;
- formal v5 outputs are selected-Gaussian projections gated only by synchronized
  Stage-1 boundary neighborhoods; stale boundary matches fail with an auditable
  `validation.json` record rather than contributing a score;
- Qwen planning remains enabled in published profiles. Semantic assignment and
  entity selection also remain enabled except for the opt-in v6 declared
  multi-instance contract, where a fully lifted Stage-1 group deterministically
  fixes the same member set and avoids duplicate post-lifting Qwen calls;
- the query entry point checks the configured Qwen checkpoint before Stage-1
  and verifies that SAM2 resolves from the pinned release checkout, preventing
  a shared-environment package from silently changing a released run;
- Gaussian membership refinement ranks candidates by rendered multi-frame
  overlap rather than proxy evidence alone;
- public and R4D evaluators reject incomplete query or spatial-frame coverage;
- ReferGaussian and the integrated 4DGS control accept the same explicit seed,
  and the external bootstrap restores that seed after backend initialization.
- The contextual warp has its own frozen learning-rate schedule; it is no
  longer implicitly coupled to per-Gaussian temporal-parameter scheduling.
- `scripts/run_matched_reconstruction.py` refuses dirty source trees, mutable
  reconstruction overrides, mismatched render sets, subset metrics, partial
  12-scene aggregates, and post-hoc PSNR filtering.

Run these checks before reporting a new result:

```bash
source scripts/common.sh
gs_python scripts/check_release.py
gs_python -m unittest discover -s tests -v
```

## Isolated Runtime Canaries (2026-07-25)

The following single-query runs were regenerated from an isolated release checkout after
the pinned external dependencies and model checkpoints had been installed.
They are executable smoke evidence, not replacements for either complete
benchmark aggregate.

| Protocol | Query | Acc | vIoU | tIoU | Coverage |
| --- | --- | ---: | ---: | ---: | --- |
| Public 4DLangSplat | `espresso__the_empty_glass_cup` | 100.00 | 64.09 | 100.00 | 36 / 36 spatial frames |
| R4D-Bench-QA | `torchchocolate_q1` | 94.24 | 53.87 | 93.96 | 71 / 71 source-camera spatial frames |

Both runs used the documented strict profiles, official query ids, complete
single-query manifests, synchronized Stage-1 boundary coverage, and a final
Gaussian-projection mask. The R4D canary used the versioned English query text
map keyed by `torchchocolate_q1`; the evaluator still used the benchmark's
official annotation record. Neither run used a direct Stage-1 mask as its
final prediction.

The later renderer-consistent R4D profile was checked on one English query from
each released category before its protocol was frozen:

| Category | Query | Acc | vIoU | tIoU |
| --- | --- | ---: | ---: | ---: |
| temporal | `torchchocolate_q1` | 99.59 | 62.45 | 99.28 |
| multi-target | `torchchocolate_q2` | 99.59 | 83.10 | 99.28 |
| zero-target | `torchchocolate_q4` | 100.00 | 100.00 | 100.00 |

These are category canaries, not a replacement for the complete 12-scene,
89-query dense aggregate.

## Strict Public Scene Smoke (2026-07-26)

A fresh `split-cookie` ReferGaussian reconstruction was trained for the
documented 14,000 fine iterations and evaluated with the public
time-sensitive protocol. The query batch used
`public_time_boundary_gated_v5_numeric` under `--strict-release`; the numeric
suffix only omits qualitative exports and does not change inference or scoring.

| Scene | Queries | Acc | vIoU | tIoU | Spatial coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| `split-cookie` | 2 / 2 | 90.91 | 55.24 | 81.30 | 53 / 53 annotated frames |

Per-query results were `90.72 / 51.19 / 84.33` for the broken-cookie state and
`91.09 / 59.29 / 78.28` for the complete-cookie state (Acc / vIoU / tIoU).
The evaluator reported zero warnings, no missing render masks, no unmapped
annotation masks, and no direct Stage-1 mask output. This is a complete
scene-level smoke for its two released time-sensitive queries, not a
replacement for the nine-query, four-scene public aggregate.

## Complete Public Time-Sensitive Release Run (2026-08-05)

A clean-source run at commit `d4dc5dd5775ffddd0b16ceb341a489a63916cd23`
completed the four-scene extension with the strict numeric v5 profile. The
numeric profile omits qualitative exports only; its inference and masks are
identical to the full v5 profile.

| Scope | Queries | Acc | vIoU | tIoU | Spatial coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| Public-4 time-sensitive | 9 / 9 | 93.65 | 62.51 | 87.60 | 240 / 240 |

The evaluator emitted no warning, every query was resolved, and no query was
below 10% vIoU. This is the four-scene release extension, not the accepted
paper's three-scene/seven-query identity.

## Complete Public Time-Agnostic Release Run (2026-08-05)

Commit `c4d2f43` completed all 20 COCO categories with
`release_public4_time_agnostic`, `public_time_agnostic_v1`, and separately
trained vanilla 4DGS checkpoints. All 20 query processes exited successfully,
the strict evaluator found no missing prediction frame, and the four-GPU batch
wall time was 2,480.1 seconds.

| Scope | Categories | mAcc | mIoU | Pooled all-frame IoU | Coverage |
| --- | ---: | ---: | ---: | ---: | --- |
| Public-4 time-agnostic | 20 / 20 | 94.35 | 42.67 | 42.25 | complete |

Here `mIoU` is the macro category mean of per-frame IoU on category-present
test frames. `mAcc` is the explicitly frozen macro category mean of per-frame
binary pixel accuracy on the same frames. The evaluator also records all-test-
frame mean IoU, foreground recall, pooled IoU, and binary accuracy so the
paper/reference-code frame-domain ambiguity remains visible.

## Current Reconstruction Verification Boundary (2026-08-05)

The accepted-paper reconstruction table remains paper-reported. A fresh
same-seed, same-budget 14k americano canary produced `30.2499 / 0.9015 /
0.1715` for 4DGS and `29.2738 / 0.8866 / 0.2017` for ReferGaussian
(PSNR / SSIM / LPIPS). The executable canary therefore does not verify the
paper-level reconstruction advantage and is not promoted to the headline
table. A complete matched multi-scene rerun is still required.

## Reporting Boundary

The accepted-paper numerical tables and later historical selected-subset tables
have separate identities. Historical exploratory artifacts, partial query
subsets, legacy run identities, per-scene environment overrides, and
incomplete baseline records are not release verification evidence and must not
be mixed into a new aggregate.

A fresh result becomes release-verifiable only when its complete manifest,
batch summary, evaluator output, spatial coverage, run configuration, and
model/data revisions are retained together.
