# ReferGaussian Next Execution Tasks

## Geometry Line

1. Stabilize `stellar_worldtube` across `mutant`, `standup`, and `cut-lemon1`.
2. Keep `stellar_core` as the strongest stable baseline while `worldtube` matures.
3. Push `worldtube` support/visibility tuning on real scenes before larger-scale HyperNeRF runs.
4. Export reusable visual assets from existing runs instead of retraining for every comparison.

## Semantics Line

1. Keep `TRASE` as the external semantic teacher / bridge baseline.
2. Continue improving native `ReferGaussian` semantic assignment and query scoring on top of `worldtube`.
3. Focus next query task on contact-heavy actions such as `cut the lemon`.
4. Promote `entitybank -> semantic_priors -> native_queries` into the default evaluation path for `worldtube`.

## D-NeRF Comparison Deliverables

1. Build a consolidated benchmark table for `mutant` and `standup`.
2. Export frame comparisons for:
   - `baseline 4DGS`
   - `chrono density`
   - `stellar_core`
   - `stellar_spacetime` or `stellar_worldtube` depending on scene stability
3. Export GIF comparisons from existing `test/.../renders` outputs.
4. Save the artifacts under `reports/dnerf_comparisons/`.

## HyperNeRF Expansion Deliverables

1. Keep `cut-lemon1` as the main query/semantics showcase scene.
2. Add at least one more local HyperNeRF smoke scene from the already-mounted dataset.
3. Save for each tested scene:
   - metrics report
   - selected key frames
   - comparison GIF
   - render/video path index
4. Save the artifacts under `reports/hypernerf_showcase/`.

## Immediate Execution Order

1. Export D-NeRF comparison tables and visual assets from existing runs.
2. Generalize HyperNeRF local-scene preparation.
3. Run an additional HyperNeRF smoke benchmark on one local scene.
4. Sync the reports and update the benchmark summary files.
