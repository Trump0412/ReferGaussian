# ReferGaussian Worldtube Design

## Goal

Move ReferGaussian beyond:

- vanilla `4DGS`, where time is mainly a conditioning scalar for deformation
- language or segmentation overlays on top of an unchanged Gaussian primitive

The target is a spacetime-native primitive where each Gaussian carries explicit local support in time and is rendered as a short worldtube rather than a single deformed 3D blob.

## Core idea

Each Gaussian now has explicit temporal state:

- `time_anchor`
- `time_scale`
- `time_velocity`
- `time_acceleration`

The new `stellar_worldtube` branch interprets one Gaussian as a local spacetime tube:

1. At query time `t`, sample multiple local times around the Gaussian's support.
2. Move the Gaussian center along its learned worldline for each sample.
3. Weight each sample by a temporal kernel and visibility gate.
4. Rasterize the set of child Gaussians and accumulate their contribution back to the parent Gaussian for training statistics.

This is different from the earlier weak-tube branch:

- `stellar_tube`
  - adds tube support as an extra covariance correction
  - still renders a single Gaussian per primitive at query time
- `stellar_worldtube`
  - explicitly expands one primitive into multiple time samples
  - approximates a local integral over spacetime support
  - keeps gradients and densification statistics tied to the original parent Gaussian

## Why this is distinct

### Against vanilla 4DGS

- `4DGS` mainly treats time as an input to deformation.
- `ReferGaussian worldtube` treats time as part of the primitive support itself.

### Against 4DLangSplat / SegmentThenSplat / SA4D

- Those methods focus on language grounding, segmentation, or semantic supervision on top of a Gaussian representation.
- They do not redefine the Gaussian primitive into a spacetime support volume.
- ReferGaussian is geometry-first: semantics are attached after a stronger spacetime base exists.

### Against TRASE

- `TRASE` is a strong entity/query backend for clustering, semantic indexing, and query-time mask rendering.
- It is not a new spacetime primitive.
- In ReferGaussian, TRASE is treated as a reusable semantic layer, not the geometric core.

## Current implementation status

- `temporal_worldtube_enabled` adds explicit worldtube integration in the renderer.
- Each parent Gaussian expands into `K` child Gaussians along local time support.
- Child opacity is modulated by temporal gate and kernel weight.
- Child scale can be widened along motion magnitude via `temporal_worldtube_scale_mix`.
- Screen-space gradients still accumulate back to the original Gaussian, so the training loop remains compatible with densification.

## Current limitation

- The worldtube branch is implemented without changing the low-level CUDA rasterizer.
- It is therefore still an approximation of a 4D Gaussian integral, not a full 4D covariance rasterizer.
- Semantic transfer from TRASE to early ReferGaussian HyperNeRF smoke runs is still weak because the current 300-iteration tube checkpoint does not yet produce stable entity-aligned tracks.

## Next step

1. Benchmark `stellar_worldtube` on representative scenes beyond smoke.
2. Replace nearest-time child sampling with a better local quadrature scheme.
3. Add tube-aware densification and pruning.
4. After geometry is stable, attach TRASE-style semantic indexing and query selection on top.
