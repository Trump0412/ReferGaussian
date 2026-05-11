# ReferGaussian Semantic Pipeline

## Thesis

ReferGaussian should not stop at "better reconstruction with a different temporal prior".

Its contribution becomes coherent only if the same spacetime-native primitive also changes:

1. how capacity is allocated in time,
2. how Gaussians are clustered into entities,
3. how semantic prompts are assigned,
4. how query-time segmentation is grounded.

The repository now follows this chain:

`worldtube primitive -> entitybank -> semantic slots/tracks -> semantic priors -> segmentation bootstrap -> TRASE/Qwen semantic assignment`

## Why this is different

This line is deliberately different from several adjacent methods:

- `4DGS`
  - time mainly acts as a conditioning variable for deformation
  - semantics are not part of the primitive design
- `4DLangSplat`
  - language is attached to splats, but the core geometry is not built around explicit local temporal support
- `TRASE`
  - strong on clustering and segmentation transfer, but it is not itself a spacetime-native Gaussian primitive
- `SegmentThenSplat`
  - segmentation leads and splatting follows
- `SA4D`
  - semantics are important, but the primitive is not the same worldtube-style time-support object we are building here

ReferGaussian should instead claim:

- time is part of the primitive support,
- the support statistics drive grouping,
- the grouping structure drives semantic assignment,
- semantic query selection remains grounded in spacetime support rather than pure image heuristics.

## Current implementation

### 1. Spacetime primitive

`stellar_worldtube` already exports:

- temporal support windows
- occupancy mass
- visibility mass
- motion-aligned tube ratio
- sampled world trajectories

These are not just diagnostics. They are the semantic bridge.

### 2. Entity clustering

The entitybank clustering now uses features that include:

- trajectory displacement
- velocity and acceleration
- anchor and temporal scale
- occupancy and visibility
- tube ratio and effective support

This means clustering is no longer only spatial or motion-only. It is support-aware in spacetime.

### 3. Semantic prior heads

`semantic_priors.json` is the new hinge layer.

Each entity now gets three geometry-grounded semantic channels:

- `static_semantics`
  - for persistent appearance and scene-part labeling
- `dynamic_semantics`
  - for moving-object labeling and action-centric descriptions
- `interaction_semantics`
  - for short support windows, contact-like events, or temporally localized manipulation

Each prior is derived from:

- support frame window
- dynamic vs stationary frame ratio
- occupancy / visibility evidence
- worldtube mode and role hints

This is the point where ReferGaussian starts affecting semantics structurally rather than cosmetically.

### 4. Query-time segmentation bootstrap

The segmentation bootstrap now uses:

- `worldtube_support` frame scoring
- dynamic/static slot balancing
- preferred prompt groups per slot
- semantic-head-aware prompt routing

So a frame can receive:

- dynamic prompts for moving phases,
- static prompts for stationary contact/support phases,
- interaction prompts for localized event windows.

This is exactly the mechanism needed for queries like `cut the lemon`, where the important window includes both motion and contact.

## Immediate next work

### Geometry

1. Push `stellar_worldtube` from pilot to stronger cross-scene geometry on `standup` and `cut-lemon1`.
2. Add tube-aware densify/prune decisions based on support occupancy and co-visibility instead of only heuristic motion boosts.
3. Keep worldtube as the primary "true 4D" line, even if `stellar_tube` remains a stronger short-term PSNR baseline on some scenes.

### Semantics

1. Run `semantic_priors` export on all strong checkpoints by default.
2. Use `semantic_head` to separate:
   - static semantic assignment
   - dynamic semantic assignment
   - interaction/event semantic assignment
3. Feed these channels into Qwen or TRASE as separate prompts instead of a single undifferentiated entity caption.

### Segmentation

1. Keep TRASE as the current strong backend for query and mask transfer.
2. Replace pure trajectory matching with worldtube-support overlap and occupancy-aware matching.
3. Promote `semantic_segmentation_bootstrap.json` from a prompt manifest into a trainable dynamic segmentation interface.

## Paper storyline

The clean storyline is:

1. Vanilla 4DGS treats time as an input condition.
2. ReferGaussian turns time into explicit local support in the primitive.
3. This support changes grouping, because entities are defined by worldtube behavior rather than only 3D proximity.
4. This grouped support changes semantics, because labels are assigned separately to static, dynamic, and interaction phases.
5. This makes downstream dynamic semantic segmentation more grounded and more controllable.

That is the continuity we should keep.
