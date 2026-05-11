# ReferGaussian Roadmap

## Current repository status

- Phase A is partially complete:
  - Official 4DGaussians baseline is vendored and runnable.
  - Full-budget training and evaluation have been completed on D-NeRF `mutant` and `standup`.
- Phase B is implemented:
  - `IdentityWarp`, monotonic MLP warp, and density-integral warp exist.
  - The `density` warp is the only surviving chrono branch.
  - It is stable, but it does not outperform baseline on the tested full scenes.
- Phase C has started:
  - a first local stellar metric prototype exists
  - per-Gaussian temporal parameters are now part of `GaussianModel`
  - short smoke training has passed end-to-end
  - full-budget results on D-NeRF `mutant` and `standup` both improve over baseline 4DGS and chrono density
  - an explicit temporal extent / worldline-prior branch (`stellar_spacetime`) now exists in code
  - `stellar_spacetime` smoke passes and its first full `mutant` benchmark improves over `stellar metric v1`
  - `stellar_spacetime` on `standup` partially recovers with conservative gate/drift blending, but still does not beat `stellar_core v1`
  - a first spacetime-aware densify/prune prototype exists, but its `standup` full run regresses and should not be treated as the new default
  - a quadratic worldline scaffold now exists with per-Gaussian acceleration, and its first smoke run passes end-to-end
- Phase D groundwork has started:
  - trained runs can now export `tube_bank.json`, `trajectory_samples.npz`, `cluster_stats.json`, and `entities.json`
  - the first exporter works on both stable `stellar_core` checkpoints and quadratic-worldline pilot checkpoints
  - the current clustering is intentionally simple and should be treated as a bridge layer, not the final semantic decomposition
- Phase E semantic work has started at the bootstrap level:
  - runs can now export `semantic_slots.json`, `semantic_slot_queries.json`, `semantic_tracks.json`, and `semantic_frame_queries.json`
  - runs can now export `semantic_priors.json`, which reorganizes worldtube statistics into static, dynamic, and interaction semantic heads before any VLM assignment
  - HyperNeRF `cut-lemon1` now exports `semantic_segmentation_bootstrap.json` with all test images aligned to nearest spacetime-query frames and top-k slot prompts
  - `cut-lemon1` now also has a `trase_bridge/` import path with copied `TRASE++` query videos and validation for `cuts the lemon`
  - the semantic bootstrap is now worldtube-aware:
    - slots carry temporal support windows, occupancy, visibility, and dynamic/static prompt groups
    - frame queries now separate dynamic and static slot activation by per-frame motion state
    - segmentation bootstrap now selects query frames with a worldtube-support score instead of pure nearest-time lookup
    - segmentation bootstrap now carries per-slot semantic head selection and preferred prompt groups derived from worldtube priors
  - the current semantic layer is still a bootstrap interface, not a trained segmentation model
- A new `stellar_tube` branch now exists and implements a weak spacetime-tube primitive by folding local worldline support into an extra spatial covariance term before rasterization.
- A new `stellar_worldtube` branch now exists and implements explicit worldtube integration:
  - one Gaussian can expand into multiple local time samples at render time
  - gradients are accumulated back to the parent Gaussian
  - this is the current bridge from worldline approximations toward a truer 4D spacetime primitive
- `stellar_worldtube` now also has the first tube-aware optimization path:
  - support-range and drift-ratio regularization
  - tube-aware split/clone along temporal support
  - activity-aware pruning protection for high-ratio tubes
- `stellar_worldtube` now also has a second-generation segment-integral path:
  - segment-centered local time integration instead of pure point-only child samples
  - visibility-aware adaptive support
  - occupancy-aware opacity aggregation
  - explicit occupancy and visibility statistics exported to `entitybank`
- The first representative `mutant ours_2500` pilot with this upgraded path (`stellar_worldtube v5`) beats same-budget baseline and materially narrows the gap to weak `stellar_tube`, while also improving LPIPS over that branch.
- A same-budget `standup` validation shows the branch is not yet universally stronger than baseline, so promotion to mainline still needs cross-scene work.
- A HyperNeRF `cut-lemon1` smoke with subset metrics now beats both baseline and weak `stellar_tube`, while also exporting a richer semantic base (`54` entities / priors).
- On `bouncingballs` with DA3 bootstrap, the conservative weak-tube setting already outperforms the same-init baseline smoke; the first aggressive setting fails badly and remains a rollback case.
- On `mutant`, the same weak-tube setting also beats a same-budget baseline pilot (`27.3190` vs `26.0929` PSNR), so the tube line now has positive signal on both a smoke scene and a representative benchmark scene.

## What the current experiment names mean

- `mutant`
  - A scene from the D-NeRF benchmark.
  - Used as the primary representative scene for Chronometric 4DGS evaluation.
- `standup`
  - Another D-NeRF benchmark scene.
  - Used as the secondary full-training validation scene.
- `baseline 4DGS`
  - The official 4DGaussians pipeline with no learned time warp.
  - Time is still mainly a conditioning variable for deformation.
- `stellar-mlp`
  - Chronometric 4DGS with a global monotonic MLP time warp.
  - This branch was stable enough for smoke runs but weaker than the density version.
- `stellar-density`
  - Chronometric 4DGS with a density-integral time warp.
  - A learned scalar reparameterization `tau = phi(t)` where `phi` is induced by a positive temporal density.
  - This is still not the final ReferGaussian representation.
- `stellar_worldtube`
  - An explicit local spacetime-tube branch where one Gaussian expands into a short support interval in time.
  - The latest `v5` run uses segment-integral sampling and visibility-aware support, not only covariance inflation.

## What is still missing for true ReferGaussian

The current chrono implementation only changes how scalar time enters the existing 4DGS deformation path.

It does **not** yet make time behave like `x, y, z` in the representation itself.

To call the system "true ReferGaussian", we need all of the following:

1. Time must become a first-class geometric variable, not only a warped scalar input.
2. The primitive must carry explicit temporal extent or worldline structure.
3. Densification and pruning must operate in spacetime, not only in 3D plus a time-conditioned deformation.
4. Temporal capacity allocation must be local and content-aware, not only a single global warp.
5. The representation must become a usable base for later clustering, motion grouping, and semantic segmentation.

## Target architecture direction

### Stage C1: Stellar Core

Upgrade from a global time warp to a local spacetime allocation model.

Planned changes:

- Add a learnable temporal metric or temporal density field that can depend on space and motion context.
- Introduce per-Gaussian temporal descriptors:
  - time anchor
  - temporal scale
  - velocity or low-order trajectory parameters
  - optional latent event code
- Replace the current global `tau = phi(t)` with a local mapping such as:
  - `tau = phi(x, t)`
  - or a per-Gaussian temporal density `rho_i(t)`
- Keep the current renderer initially, so the first iteration remains comparable and reversible.

Acceptance criteria:

- Better event allocation than baseline and chrono-warp.
- At least one representative scene shows measurable metric gain or equal quality with better temporal interpretability.
- Ablation can separate gains from global warp vs local temporal structure.

### Stage C2: 4D Gaussian primitive

Promote time into the primitive state itself.

Planned changes:

- Move from "3D Gaussian + deformation conditioned on time" toward "4D Gaussian state in `(x, y, z, t)`".
- Start with a practical approximation:
  - 4D mean `[x, y, z, t]`
  - structured covariance or factorized temporal extent
  - slicing at query time to obtain the active 3D footprint
- Support temporal anisotropy:
  - some primitives become short-lived
  - some become long-lived
  - some become motion-aligned

Current implementation status:

- A first approximation is now wired into the renderer:
  - per-Gaussian temporal extent from `time_anchor` and `time_scale`
  - temporal opacity gating at render time
  - first-order worldline drift from `time_velocity`
- This is still a lightweight approximation, not yet a full 4D covariance or tube primitive.
- `mutant` validates the direction, but `standup` shows that naive explicit temporal slicing is not yet cross-scene robust.
- A conservative gate/drift blend can recover much of the `standup` regression, but it still does not beat `stellar_core v1`.
- A stricter `stellar_worldtube` path now expands each Gaussian into local time samples and trains with tube-aware densification.
- The first `mutant` pilot (`v3`) proved the optimization path is stable but clearly too weak on quality.
- The latest `mutant` pilot (`v5`) reaches `26.2435 / 0.9401 / 0.0634`, which beats same-budget baseline and beats weak `stellar_tube` on LPIPS, but still trails weak tube on PSNR and SSIM.
- A same-budget `standup` validation reaches `25.9450 / 0.9536 / 0.0596`, which is still below baseline on that scene.
- This means `stellar_worldtube` is now a competitive primitive candidate rather than only an experimental scaffold, but it is not yet the universal replacement for baseline or weak `stellar_tube`.

Acceptance criteria:

- Time is encoded in primitive parameters rather than only in the deformation input.
- Rendering at different times can be interpreted as slicing or projecting a spacetime representation.
- Spacetime-aware densification does not destabilize training.

### Stage C3: Quadratic Worldline Primitive

Extend the primitive from first-order motion to a second-order local spacetime trajectory.

Planned changes:

- Add per-Gaussian acceleration to the temporal state.
- Upgrade the render-time worldline model from linear drift to quadratic drift.
- Keep the current rasterizer unchanged so the first benchmark stays directly comparable.
- Track both speed and acceleration in checkpoint summaries and report tables.

Current implementation status:

- `time_acceleration` is now part of `GaussianModel`.
- Checkpoint save/load, densify/prune, and summary tooling all support the new parameter.
- The `stellar` temporal warp context is now resilient to future feature-width changes.
- A short smoke run on D-NeRF `bouncingballs` passes end-to-end with learned non-zero acceleration.
- A same-budget `standup` pilot is mildly positive relative to the first-order blend pilot, so the branch has enough signal to justify a future full benchmark.

Acceptance criteria:

- Representative-scene pilot runs on `mutant` or `standup` remain stable.
- Quadratic drift improves reconstruction quality or temporal interpretability over first-order drift.
- The added motion capacity does not immediately collapse into unstable temporal allocation.

### Stage D: Spacetime Tube / Worldline Gaussian

Build the final ReferGaussian primitive.

Planned changes:

- Represent each dynamic entity as a worldline or spacetime tube instead of repeated deformed 3D positions.
- Parameterize tube centerline with:
  - anchor points
  - low-order spline
  - or velocity-acceleration model
- Parameterize tube thickness separately in spatial and temporal directions.
- Add event-preserving regularization so motion-rich intervals receive more representational budget.

Acceptance criteria:

- The method is clearly no longer a small patch on top of vanilla 4DGS.
- The primitive and optimization logic are genuinely spacetime-native.
- Full-scene benchmarks show either improved quality, improved efficiency, or better controllability.

### Stage E: Semantic extension

Only after the spacetime base is stable:

- cluster tube or Gaussian trajectories into motion groups
- add semantic features or distillation heads
- implement dynamic 4D semantic segmentation
- extend toward open-vocabulary retrieval and event-level reasoning

## Immediate next sprint

### Sprint 1: Enter Phase C safely

1. Create `feature/stellar-metric`.
2. Refactor the current temporal interface so global warp and future local metric modules share one API.
3. Add per-Gaussian temporal parameters to the model state.
4. Implement a first local temporal density module without touching the rasterizer.
5. Compare:
   - baseline 4DGS
   - chrono density
   - local stellar metric v1
6. Run on `mutant` and `standup` first.

### Sprint 2: Spacetime-aware optimization

1. Add spacetime-aware densification and pruning rules.
2. Track temporal occupancy, lifetime, and event-density statistics.
3. Measure whether new temporal parameters actually concentrate capacity on high-dynamics intervals.

Current status:

- Step 1 has a first prototype.
- The first full test on `standup` regressed.
- Conservative render-time blending recovers part of the regression, but not enough to justify promoting the heuristic.
- The next iteration should replace the current heuristic boost with occupancy-aware or visibility-aware temporal statistics before retrying full benchmarks.
- For `stellar_worldtube`, the next iteration should:
  - validate the new segment-integral / visibility-aware formulation on `standup`
  - use the new positive `cut-lemon1` result as the real-scene reference line
  - promote occupancy and visibility cues from diagnostics to entity-level clustering and semantic transfer

### Sprint 3: Primitive upgrade

1. Benchmark the new quadratic worldline primitive on representative scenes.
2. Use the current `stellar_worldtube` branch as the main primitive-upgrade line.
3. If it improves over weak `stellar_tube` across scenes, move toward a stronger tube-style primitive with explicit occupancy-aware integration.
4. If it remains mixed, keep `stellar_tube` as the strongest PSNR-oriented geometry baseline while continuing `stellar_worldtube` as the most 4D-native branch.
5. Keep the rendering path as close as possible to the current pipeline for the first prototype.
6. Only after prototype validation, consider lower-level renderer changes.

### Sprint 4: Entity bank to semantics bridge

1. Keep exporting `entitybank/` from the strongest checkpoints after evaluation.
2. Replace the current greedy clustering with trajectory plus co-visibility statistics.
3. Attach semantic feature placeholders or distilled descriptors at entity level.
4. Upgrade the current `semantic_slots -> semantic_tracks -> semantic_priors -> segmentation_bootstrap` bridge into an actual dynamic segmentation pipeline.

## Engineering rule for the next phase

For now, we should **not** jump straight to semantic segmentation.

The correct order is:

1. make time first-class
2. make the primitive spacetime-native
3. make optimization spacetime-aware
4. export stable entity banks from the learned spacetime representation
5. then add clustering and semantics

Otherwise the semantic layer will sit on a weak geometric base and the whole system will be hard to interpret or improve.
