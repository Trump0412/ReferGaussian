# Query-Guided Entity Pipeline

ReferGaussian now has a detector-guided proposal stage for query-conditioned entity discovery.

## Pipeline

1. `query -> GroundingDINO`
   - Run zero-shot detection on sampled source frames.
   - Save `bbox`, `center_xy`, and five-point prompt sets per detection.

2. `bbox / prompt points -> proposal filter`
   - Use the detector output as coarse spatial prompts.
   - Prefer proposals or worldtube entities whose projected centers repeatedly fall inside the detected regions.

3. `proposal/worldtube reassignment`
   - Reassign Gaussians with our worldtube trajectories and support windows.
   - Keep the final representation in ReferGaussian space, not in detector space.

4. `native/Qwen semantics`
   - Render every surviving entity.
   - Assign continuous semantics with Qwen over the entity library.

5. `query composition -> native render`
   - Match the textual query against the entity library.
   - Render source-background and model-background query videos.

## Current Outputs

The detector stage writes:

- `query_detection_proposals.json`
- `frames/*.png`

Each proposal stores:

- `bbox_xyxy`
- `center_xy`
- `prompt_points_xy`
- `time_value`
- `score`
- `label`
