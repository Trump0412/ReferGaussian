# Reproducibility Record (2026-04-16)

## Environment

- Host: `myautodl`
- Project root: `<AUTODL_ROOT>/ReferGaussian`
- Main env: `<AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310`
- Reproduction artifacts: `reports/reproducibility_20260416`

## A) Keyboard Reconstruction Target (PSNR 28.4051)

Evaluated run:

`<AUTODL_ROOT>/ReferGaussian/runs/stellar_tube_bench12_best04034_lrlow_20260402_keyboard/hypernerf/keyboard`

Command:

```bash
conda run -p <AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310 \
  python scripts/collect_metrics.py \
  --run-dir <AUTODL_ROOT>/ReferGaussian/runs/stellar_tube_bench12_best04034_lrlow_20260402_keyboard/hypernerf/keyboard \
  --write-summary
```

Result snapshot:

- PSNR: `28.4050827` (rounded `28.4051`)
- SSIM: `0.8866783`
- LPIPS-vgg: `0.2071507`

Artifacts:

- `reports/reproducibility_20260416/keyboard_metrics.json`
- `reports/reproducibility_20260416/keyboard_summary.md`

## B) R4D-Bench-QA Evaluation

### Historical public snapshot

- Valid queries: `18 / 36`
- Acc: `97.4990%`
- vIoU: `8.1029%`
- tIoU: `97.4908%`

### Re-run with current evaluation entrypoint

```bash
conda run -p <AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310 \
  python scripts/evaluate_ours_benchmark.py \
  --benchmark <AUTODL_ROOT>/data/Ours_benchmark.json \
  --query-root-map reports/reproducibility_20260416/query_root_map_from_results_latest.json \
  --dataset-dir-map reports/reproducibility_20260416/dataset_dir_map_from_results_latest.json \
  --output-json reports/reproducibility_20260416/ours_benchmark_eval_reproduced_from_latest.json \
  --output-md reports/reproducibility_20260416/ours_benchmark_eval_reproduced_from_latest.md \
  --skip-missing
```

Current run snapshot:

- Valid queries: `15 / 36`
- Acc: `98.0296%`
- vIoU: `5.0668%`
- tIoU: `48.7067%`

Note: differences are due to protocol/mapping coverage and evaluation-scope differences.

## C) 4DLangSplat (Americano)

```bash
conda run -p <AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310 \
  python scripts/evaluate_public_query_protocol.py \
  --protocol-json <AUTODL_ROOT>/ReferGaussian/reports/4dlangsplat_compare/protocol_splits/americano.json \
  --annotation-dir <AUTODL_ROOT>/ReferGaussian/data/benchmarks/4dlangsplat/HyperNeRF-Annotation/americano \
  --dataset-dir <AUTODL_ROOT>/ReferGaussian/data/hypernerf/misc/americano \
  --query-root <AUTODL_ROOT>/ReferGaussian/runs/stellar_tube_4dlangsplat_refresh_20260328_americano/hypernerf/americano/entitybank/query_guided \
  --output-json reports/reproducibility_20260416/4dlangsplat_americano_public_eval_reproduced.json \
  --output-md reports/reproducibility_20260416/4dlangsplat_americano_public_eval_reproduced.md
```

Result:

- Queries: `3`
- Acc: `97.72%`
- vIoU: `69.35%`
- temporal tIoU: `94.34%`

