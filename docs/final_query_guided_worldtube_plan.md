# Final Query-Guided Worldtube Plan

## Objective

Move ReferGaussian from:

- global full-scene clustering

to:

- query-conditioned entity discovery
- query-conditioned phrase-level tracking
- worldtube reassignment only on hit entities
- Qwen-guided continuous semantic assignment
- final dynamic query render

## Final Pipeline

1. `Qwen query planner`
   - Parse a user query into concrete detector phrases.
   - Add temporal hints such as before / during / after.

2. `Grounded SAM 2 phrase grounding`
   - Run phrase grounding per detector phrase.
   - Use Grounding DINO for the anchor detection.
   - Use SAM 2 to segment and track the phrase through the whole sequence.

3. `Prompt-conditioned worldtube reassignment`
   - Use the phrase tracks as spatial-temporal prompts.
   - Reassign only the relevant Gaussians / worldtubes.
   - Avoid clustering the entire scene.

4. `Query-specific entity library`
   - Save only the hit entities.
   - Render every hit entity individually.

5. `Qwen continuous semantics`
   - Assign identity and temporal semantics to the query-specific entities.
   - Keep phase changes explicit, such as one lemon before cutting and two halves after cutting.

6. `Dynamic query scoring and render`
   - Match the final query against the query-specific entity library.
   - Render source-background and model-background videos.

## Immediate Execution Steps

1. Install `Grounded SAM 2` in a dedicated environment.
2. Run `Qwen planner -> Grounded SAM 2` on `cut-lemon1`.
3. Save phrase-level tracks and previews.
4. Use those tracks to constrain worldtube reassignment.
5. Rebuild the query-specific entity library.
