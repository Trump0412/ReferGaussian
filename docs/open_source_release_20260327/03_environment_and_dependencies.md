# Environment And Dependencies

## 1. 推荐环境矩阵

仓库目前有四套常见环境。

| 环境 | 默认路径 | 用途 |
| --- | --- | --- |
| main CUDA 12.1 | `<AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310` | 日常训练、评测、导出 |
| official baseline | `<AUTODL_ROOT>/.conda-envs/gs4d-baseline-py37` | 对齐官方 `4DGaussians` 兼容配置 |
| Grounded-SAM-2 | `<AUTODL_ROOT>/.conda-envs/grounded-sam2-py310` | query planning、GSAM2 tracking、Qwen 相关脚本 |
| DA3 | `<AUTODL_ROOT>/.conda-envs/da3-gs-py310` | 可选几何 bootstrap |

## 2. 主环境安装

先安装主训练环境：

```bash
cd <AUTODL_ROOT>/ReferGaussian
bash scripts/setup_baseline_env.sh cuda121
```

如果需要官方兼容环境：

```bash
bash scripts/setup_baseline_env.sh official
```

安装后建议立刻做检查：

```bash
conda run -p <AUTODL_ROOT>/.conda-envs/gs4d-cuda121-py310 \
  python scripts/check_install.py
```

这个检查会验证：

- `torch`
- `diff_gaussian_rasterization`
- `simple_knn`
- `refergaussian.temporal`
- upstream config loader

## 3. CUDA 与缓存目录约定

默认约定如下：

- `CUDA_HOME=/usr/local/cuda-12.1`
- conda env root: `<AUTODL_ROOT>/.conda-envs`
- conda pkg cache: `<AUTODL_ROOT>/.conda-pkgs`
- pip cache: `<AUTODL_ROOT>/.cache/pip`

如果机器上 CUDA 不在默认路径，需要先设置：

```bash
export GS4D_CUDA_HOME=/path/to/cuda
```

## 4. 数据准备

主仓库自带数据准备脚本：

```bash
bash scripts/prepare_dnerf.sh
bash scripts/prepare_hypernerf.sh
python scripts/check_dataset_layout.py --skip-scene-load
```

默认数据目录：

- `data/dnerf/`
- `data/hypernerf/`
- `data/dynerf/`

## 5. Query / 语义环境安装

如果需要 query grounding、GSAM2、Qwen 语义阶段，再安装：

```bash
bash scripts/setup_grounded_sam2.sh
```

可选地预取 GroundingDINO 模型：

```bash
bash scripts/setup_query_detector.sh
```

这套环境主要服务：

- `plan_query_entities.py`
- `run_grounded_sam2_query.py`
- `export_qwen_semantics.py`
- `select_qwen_query_entities.py`

## 6. DA3 可选环境

如果需要 DA3 bootstrap：

```bash
bash scripts/setup_da3_env.sh
```

如果还要启用 `gsplat` 导出视频版本：

```bash
DA3_INSTALL_GSPLAT=1 bash scripts/setup_da3_env.sh
```

DA3 环境只用于初始化，不影响主训练环境设计。

## 7. 外部依赖边界

| 依赖 | 目录 | 作用 |
| --- | --- | --- |
| `4DGaussians` | `external/4DGaussians` | 主训练与渲染骨架 |
| `Grounded-SAM-2` | `external/Grounded-SAM-2` | 2D query tracking |
| `Depth-Anything-3` | `external/Depth-Anything-3` | 可选 bootstrap |
| `gsplat` | `external/gsplat` | DA3 可选导出依赖 |

## 8. 最小可运行依赖组合

### 只做重建

只需要：

- `setup_baseline_env.sh`
- 数据集

### 做语义导出

需要：

- 主训练环境
- 已完成训练的 `run_dir`

### 做 query grounding

需要：

- 主训练环境
- GSAM2/Qwen 环境
- 已完成训练的 `run_dir`
- 数据集原图目录

### 做 DA3 bootstrap

需要：

- DA3 环境
- 可选 `gsplat`

## 9. 开源时建议保留的环境文档重点

对外 README 最需要讲清楚的不是每个 pip 包，而是：

- 主环境和 query 环境是分开的。
- `DA3` 是可选环境。
- 只做重建时不需要 GSAM2/Qwen。
- 语义与 query 阶段建立在已有 `run_dir` 上，不会重新训练底层 scene representation。

