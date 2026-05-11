# ReferGaussian System Design

## Goal

ReferGaussian should not stop at "better time conditioning".

Its target is a reusable dynamic 4D base with:

- explicit spacetime primitives
- worldline or tube-style motion support
- stable optimization on dynamic scenes
- exportable trajectory-aware entity banks
- a clean path toward clustering and dynamic semantic segmentation

## Where DA3 fits

DA3 is optional.

It is useful as a geometry bootstrap layer, not as the final model.

Recommended role:

1. Use DA3 to predict strong initial geometry when the dataset does not already provide a good point cloud.
2. Convert DA3 outputs into the initial Gaussian or tube bank.
3. Continue optimizing with ReferGaussian so temporal structure, render quality, and downstream clustering remain under our control.

DA3 should not replace the main 4D optimization path because:

- DA3 is a feed-forward geometry model
- ReferGaussian needs scene-specific spacetime refinement
- later dynamic segmentation depends on stable trajectories and entity-level exports

## Current baseline

Today the repository initializes Gaussians from a point cloud and then calls `create_from_pcd(...)`.

Point cloud sources depend on the dataset:

- COLMAP sparse points
- dataset-provided `points3D_downsample2.ply`
- synthetic random initialization for Blender-style D-NeRF scenes when no fused point cloud exists

This means DA3 has the highest value on scenes where the current initialization is weak, especially synthetic or monocular-first cases.

## Target architecture

### Layer 1: Bootstrap

Introduce a pluggable bootstrap interface:

- `random`
- `colmap`
- `dataset_ply`
- `da3`

Expected bootstrap outputs:

- initial point or Gaussian locations
- approximate colors
- optional camera intrinsics / extrinsics
- optional confidence scores

### Layer 2: Spacetime primitive

Move from "3D Gaussian conditioned on time" toward "dynamic spacetime primitive".

The near-term primitive should be a factorized spacetime tube with:

- spatial anchor `x0`
- temporal anchor `t0`
- spatial scale `sx, sy, sz`
- temporal scale `st`
- velocity `v`
- acceleration `a`
- opacity and appearance parameters

Render-time slicing should evaluate:

- tube activity at query time
- centerline drift in world space
- temporal visibility gating

The next upgrade after that is a minimal 4D covariance or worldline-tube primitive.

### Layer 3: Spacetime optimization

Optimization should become tube-aware, not only image-loss-aware.

Core rules:

- densify where residuals and temporal occupancy are both high
- protect short-lived but informative tubes from early pruning
- regularize velocity and acceleration separately
- track visibility over time, not only global opacity

### Layer 4: Spacetime bank

Every trained run should be able to export a reusable spacetime bank.

Minimum artifacts:

- `tube_bank.json`
- `trajectory_samples.npz`
- `cluster_stats.json`
- `entities.json`

The output contract should be compatible with the existing `4dgs4mm`-style entity bank:

- `source_cluster_id`
- `gaussian_ids`
- `keyframes`
- `segments`
- `mode: world_trajectory`

This keeps downstream clustering and semantics decoupled from the renderer internals.

Current implementation status:

- `scripts/export_entitybank.py` can now export `tube_bank.json`, `trajectory_samples.npz`, `cluster_stats.json`, and `entities.json` from trained checkpoints.
- The exporter already works on `stellar_core_full` results and on `stellar_spacetime_quad` pilot results.
- `entities.json` uses a `world_trajectory` decomposition mode so it can serve as the bridge toward a `4dgs4mm`-style semantic pipeline.

### Layer 5: Clustering and dynamic segmentation

Once the tube bank is stable:

1. cluster by trajectory, spatial overlap, and temporal co-visibility
2. build entity tracks
3. attach semantic descriptors or distilled features
4. refine dynamic semantic masks over time

This is where the project should meet the `4dgs4mm` workflow.

## Engineering decisions

### Decision 1

DA3 is recommended but not mandatory.

Use it when:

- no reliable point cloud exists
- initialization is currently random
- camera estimation is weak

Skip it when:

- a high-quality multiview point cloud already exists
- the scene is already well-initialized by COLMAP or dataset geometry

### Decision 2

The mainline should remain optimization-based.

Feed-forward outputs are allowed as initialization, but the repository goal is still a stronger dynamic 4D representation, not only faster single-pass geometry.

### Decision 3

Entity-bank compatibility is a first-class requirement.

The renderer is not the only product. The downstream artifacts for clustering and semantics are equally important.

## Next implementation steps

1. Finish the DA3 bootstrap path so it writes a manifest plus exported Gaussian artifacts on a real scene.
2. Add a converter from DA3 Gaussian output into ReferGaussian initial state.
3. Improve the current `entitybank` clustering from a simple greedy feature pass to occupancy-aware and co-visibility-aware grouping.
4. Add per-entity semantic feature slots so `entities.json` can carry dynamic descriptors instead of empty placeholders.
5. Use that enriched entity bank as the input to dynamic semantic segmentation.
