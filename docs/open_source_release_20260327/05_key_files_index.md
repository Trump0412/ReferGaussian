# Key Files Index

## 1. 先看哪些文件

如果只想快速掌握项目，优先看下面这些文件。

### 1.1 总入口

- `README.md`
  - 仓库总说明与 quick start。
- `docs/open_source_release_20260327/`
  - 本次整理出来的开源入口文档。

### 1.2 训练与运行入口

- `scripts/common.sh`
  - 环境选择、`gs_python`、数据集路径解析、GPU 监控。
- `scripts/train_baseline.sh`
  - baseline 4DGS 训练入口。
- `scripts/train_stellar_tube.sh`
  - weak tube 训练入口。
- `scripts/train_stellar_worldtube.sh`
  - worldtube 训练入口。
- `scripts/eval_stellar_tube.sh`
  - weak tube 评测入口。
- `scripts/eval_stellar_worldtube.sh`
  - worldtube 评测入口。

### 1.3 temporal 与 primitive

- `refergaussian/temporal/modules.py`
  - `identity / mlp / density / stellar` warp 定义。
- `refergaussian/temporal/warp_io.py`
  - warp 的构建、保存、加载。
- `external/4DGaussians/scene/gaussian_model.py`
  - per-Gaussian temporal state、tube/worldtube support、support-aware optimization。
- `external/4DGaussians/gaussian_renderer/__init__.py`
  - render-time worldtube expansion。
- `external/4DGaussians/train.py`
  - reconstruction 与 support regularization 的训练主循环。
- `external/4DGaussians/render.py`
  - 评测与渲染入口。
- `external/4DGaussians/arguments/__init__.py`
  - 新增 temporal/worldtube 参数定义。

### 1.4 entitybank

- `scripts/export_entitybank.py`
  - entitybank 导出 CLI。
- `refergaussian/entitybank/tube_bank.py`
  - 从 checkpoint 读取 temporal 参数并采样 trajectory/tube stats。
- `refergaussian/entitybank/export.py`
  - support-aware clustering、entity 导出。

### 1.5 语义主线

- `scripts/export_semantic_slots.py`
- `scripts/export_semantic_tracks.py`
- `scripts/export_semantic_priors.py`
- `scripts/export_native_semantics.py`
- `refergaussian/semantics/slots.py`
- `refergaussian/semantics/tracks.py`
- `refergaussian/semantics/priors.py`
- `refergaussian/semantics/native_assignment.py`
- `refergaussian/semantics/trase_bridge.py`

### 1.6 Query grounding 主线

- `scripts/run_query_specific_worldtube_pipeline.sh`
  - 最完整的单 query pipeline 入口。
- `scripts/plan_query_entities.py`
  - Qwen query planner。
- `scripts/run_grounded_sam2_query.py`
  - GSAM2 tracking 入口。
- `refergaussian/semantics/qwen_query_planner.py`
  - query 解析与 temporal hints。
- `refergaussian/semantics/grounded_sam2_backend.py`
  - multi-anchor detection/tracking。
- `refergaussian/semantics/query_proposal_bridge.py`
  - 2D tracks 到 learned worldtube 的重分配。
- `refergaussian/semantics/qwen_assignment.py`
  - Qwen entity assignment。
- `refergaussian/semantics/query_render.py`
  - final query render 与 validation 输出。

### 1.7 应用与编辑

- `scripts/run_scene_deepfill_removal_experiment.sh`
  - 应用侧实体剔除主脚本。
- `scripts/build_query_removal_bundle.py`
  - subset run、filtered ply、removal bundle 导出。
- `scripts/expand_seed_entity.py`
  - 从 query tracks/proposal 扩展实体。

## 2. 代码按功能分目录整理

下面这个分组最适合作为开源时的目录说明。

### A. 核心代码

- `refergaussian/temporal/`
- `refergaussian/entitybank/`
- `refergaussian/semantics/`
- `external/4DGaussians/scene/gaussian_model.py`
- `external/4DGaussians/gaussian_renderer/__init__.py`
- `external/4DGaussians/train.py`
- `external/4DGaussians/render.py`
- `external/4DGaussians/arguments/__init__.py`

### B. 训练与评测脚本

- `scripts/train_*.sh`
- `scripts/eval_*.sh`
- `scripts/collect_metrics.py`
- `scripts/check_install.py`
- `scripts/check_dataset_layout.py`

### C. 语义与 query 脚本

- `scripts/export_entitybank.py`
- `scripts/export_semantic_*.py`
- `scripts/export_native_semantics.py`
- `scripts/export_qwen_semantics.py`
- `scripts/plan_query_entities.py`
- `scripts/run_grounded_sam2_query.py`
- `scripts/build_query_proposal_dir.py`
- `scripts/render_query_video.py`
- `scripts/select_qwen_query_entities.py`
- `scripts/run_query_specific_worldtube_pipeline.sh`
- `scripts/run_public_query_protocol.sh`

### D. 应用脚本

- `scripts/run_scene_deepfill_removal_experiment.sh`
- `scripts/build_query_removal_bundle.py`
- `scripts/expand_seed_entity.py`

### E. 配置与说明

- `configs/`
- `patches/`
- `docs/`

## 3. 开源时建议保留的目录

建议首版开源至少保留：

- `README.md`
- `docs/`
- `configs/`
- `refergaussian/`
- `scripts/`
- `patches/`
- `external/4DGaussians/`

如果不想直接把外部依赖整仓放进去，也至少要保留你们实际改过的 upstream 文件列表与 patch 说明。

## 4. 首版开源时可不放的内容

这些目录通常不适合作为首版公开仓库直接上传：

- `data/`
- `runs/`
- `logs/`
- `output.png`
- 大体量临时 `reports/` 实验汇报稿

## 5. 一句话记忆版

如果后面有人问“关键代码在哪”，最短回答就是：

- 训练改动看 `external/4DGaussians`
- 时间模块看 `refergaussian/temporal`
- 时空实体导出看 `refergaussian/entitybank`
- 语义/query/app 看 `refergaussian/semantics` 和 `scripts/run_query_specific_worldtube_pipeline.sh`

