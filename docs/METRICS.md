# Metric Definitions and Compatibility

ReferGaussian keeps the accepted-paper definitions and the executable legacy
evaluators distinguishable. They are not silently treated as the same metric.

## Paper-declared metrics

The paper defines:

- **Acc**: exact equality between the predicted referent set and the ground
  truth referent set, averaged over queries.
- **vIoU**: intersection over union of the full predicted and ground-truth
  spatiotemporal mask volumes, averaged over queries.

The currently released dense annotations do not expose a complete auditable
mapping from every predicted entity identity to every ground-truth instance
set, and the R4D dense tier does not provide exhaustive masks for every video
frame. The release therefore does not manufacture these two values from
different quantities. Evaluator JSON records their paper fields as `null`.

## Executable compatibility metrics

Historical outputs use the following compatibility aliases:

- R4D `Acc`: temporal binary accuracy at rendered sample timestamps.
- Public `Acc`: temporal binary accuracy over the public timeline.
- `vIoU`: arithmetic mean of 2D mask IoU over the evaluator's annotated mask
  frames. Missing required masks score zero.
- `tIoU`: intersection over union of predicted and ground-truth active time.
  R4D first applies nearest-sample hold from its sparse reconstruction test
  grid; Public uses its dense timeline directly.

New JSON output keeps `Acc`, `vIoU`, and `tIoU`/`temporal_tIoU` for backward
compatibility and adds explicit aliases:

- `temporal_frame_accuracy`;
- `mean_annotated_frame_iou`;
- `annotated_volume_iou`, computed as the sum of annotated-frame pixel
  intersections divided by the sum of annotated-frame pixel unions.

`annotated_volume_iou` is a useful area-weighted diagnostic. It is not labeled
as paper full-volume vIoU unless annotation coverage is proven exhaustive.

## Empty targets

- Empty ground truth and empty prediction receive `1.0` spatial and temporal
  IoU.
- Empty ground truth with any predicted activity receives `0.0` spatial and
  temporal IoU.
- A benchmark cannot obtain 100% from an empty list of queries: complete
  coverage checks require every expected query ID.

Every report must include overall, non-empty-only, and zero-target correctness
results together. Aggregation refuses to mix evaluator protocol IDs.

## Protocol registry

Scene/query scopes, source hashes, and protocol status are frozen in
[`configs/benchmarks/release_protocols.json`](../configs/benchmarks/release_protocols.json).
Paper-reported, dense-release, extension, and archival subset results must use
their own identifiers and output directories.
