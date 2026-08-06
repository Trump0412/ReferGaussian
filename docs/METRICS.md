# Time-Sensitive Evaluation Metrics

The released evaluators implement the time-sensitive protocol used in the
camera-ready paper for both released benchmarks:

- Public HyperNeRF: 4 scenes and 9 English queries.
- R4D-Bench-QA: 12 scenes and 89 English queries.

For query `q`, let `F_q` be its evaluated timeline, `T_q` the predicted active
frames, and `T*_q` the ground-truth active frames. `M_t` and `M*_t` are the
predicted and ground-truth binary masks at frame `t`.

## Acc

`Acc(q)` is frame-wise temporal activation accuracy over every frame in `F_q`:

```text
Acc(q) = (1 / |F_q|) sum_t 1[1[t in T_q] = 1[t in T*_q]]
```

## vIoU

`vIoU(q)` combines temporal localization and spatial overlap:

```text
vIoU(q) = (1 / |T_q union T*_q|)
           sum_(t in T_q intersection T*_q) IoU(M_t, M*_t)
```

Frames outside the temporal intersection contribute zero through the temporal
union denominator. A valid zero-target query obtains `Acc = vIoU = 1` only
when the prediction is empty throughout the evaluated timeline. An unresolved
selection, missing required output, or incomplete manifest fails strict
evaluation; it is not converted into an empty prediction.

Reported `Acc` and `vIoU` are arithmetic means of the query-level values.
`tIoU`, temporal precision/recall, overlap-frame mean IoU, and annotated-volume
IoU are emitted only as diagnostics.

Every report records the protocol id, query manifest, source hashes, and
coverage status. Formal runs require complete query and spatial coverage.
