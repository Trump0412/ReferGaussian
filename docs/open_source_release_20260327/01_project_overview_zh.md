# ReferGaussian 项目总览

## 1. 项目定位

ReferGaussian 是一个围绕动态 4D Gaussian 表示搭建的统一研究仓库。它不是三个松散拼接的小项目，而是一条连续主线：

`动态重建 -> 时空 primitive -> entity bank -> training-free 语义 grounding -> query-conditioned 渲染/编辑`

仓库里的目标可以概括成三件事：

- 做出比 vanilla `4DGS` 更强的动态时空表示。
- 把语义建立在结构化 dynamic entities 上，而不是重新为每个场景训练一个 dense language field。
- 在同一个表示上继续支持 query grounding、实体剔除、query-conditioned rendering 等应用侧能力。

## 2. 现在已经实现了什么

### 2.1 重建主线

当前仓库已经包含以下重建分支：

- `baseline_4dgs`
  - 官方 `4DGaussians` 基线复现。
- `chronometric_4dgs`
  - 时间重参数化分支，支持 `identity / mlp / density` warp。
- `stellar_core`
  - 引入 per-Gaussian temporal state。
- `stellar_spacetime`
  - 时间开始直接进入 primitive 的局部支撑域。
- `stellar_tube`
  - 弱时空管近似，当前对部分重建任务非常实用。
- `stellar_worldtube`
  - 显式 worldtube 多样本渲染，是当前最完整的 spacetime primitive 分支。

### 2.2 语义与 grounding 主线

在重建结果之上，仓库已经实现：

- `entitybank` 导出
- `semantic_slots`
- `semantic_tracks`
- `semantic_priors`
- `native semantic assignment`
- `Qwen-based semantic assignment / query planning`
- `Grounded-SAM-2` query detection and tracking
- `TRASE bridge`
- `query-conditioned entity reassignment`
- `query-conditioned rendering`

### 2.3 应用与编辑主线

应用侧已经包含：

- query-specific entitybank 构建
- query video 渲染
- subset Gaussian 过滤与 removal bundle 导出
- `deepfill`/补全导向的实体剔除实验脚本

## 3. 重建、语义、应用是不是同一套基座

结论分三层理解最准确。

| 问题 | 结论 | 说明 |
| --- | --- | --- |
| 是不是同一个仓库基座 | 是 | 都在 `ReferGaussian` 内部完成 |
| 是不是同一个训练/渲染主入口 | 是 | 重建统一从 `external/4DGaussians/train.py` 和 `render.py` 进入 |
| 是不是共享同一套中间产物协议 | 是，但主要针对 temporal 分支 | 语义和应用默认依赖 `config.yaml`、`point_cloud/iteration_*`、`temporal_params.pth`、`entitybank/` |
| 是不是所有结果都来自完全相同的模型分支 | 不一定 | 当前不同任务可能分别基于 `stellar_tube` 或 `stellar_worldtube` 的 run |
| 是不是共享同一个下游语义/应用接口 | 是 | `entitybank -> slots -> tracks -> priors -> query pipeline` 是共用的 |

所以更准确的说法是：

- **代码基座是统一的。**
- **artifact contract 是统一的。**
- **实验分支不一定完全相同。**

还需要额外说明：

- `baseline_4dgs` 主要服务重建基线对比。
- `entitybank -> semantics -> query -> application` 这条链通常建立在 `stellar_*` 这类带 temporal state 的分支上。

如果后面写论文或准备开源，建议把论文总方法名写成 `ReferGaussian`，把 `stellar_tube` 和 `stellar_worldtube` 表述为它的两种实现形态或两种近似级别。

## 4. 统一的数据与产物流

端到端的数据流如下：

1. 数据集进入 `external/4DGaussians` 的 reader。
2. 使用 `baseline / chrono / stellar_*` 分支进行训练。
3. 在 `run_dir/point_cloud/iteration_*` 下保存 Gaussian 与 temporal 参数。
4. 用 `export_entitybank.py` 导出 `trajectory_samples.npz`、`tube_bank.json`、`cluster_stats.json`、`entities.json`。
5. 在 `entitybank` 上继续导出 `semantic_slots`、`semantic_tracks`、`semantic_priors`、`native/qwen assignments`。
6. Query 时使用 `Qwen planner + Grounded-SAM-2 + query proposal bridge` 做 query-conditioned worldtube reassignment。
7. 最终得到 query-specific render、mask、diagnostics，或进一步做 subset removal / editing。

## 5. 统一基座下的关键外部依赖

| 组件 | 角色 | 是否核心 |
| --- | --- | --- |
| `external/4DGaussians` | 训练、渲染、数据协议、基础 Gaussian 模型 | 核心 |
| `refergaussian/temporal` | 时间 warp 与时间状态辅助模块 | 核心 |
| `refergaussian/entitybank` | 时空实体导出与聚类 | 核心 |
| `refergaussian/semantics` | 语义导出、query planning、query grounding、rendering | 核心 |
| `external/Grounded-SAM-2` | 2D query detection/tracking backend | 语义阶段重要外部件 |
| `Qwen` | query planning 与 entity semantic assignment | 语义阶段重要外部件 |
| `external/Depth-Anything-3` | 可选 bootstrap | 非核心 |
| `4dgs4mm / TRASE assets` | 兼容桥接与对比实验 | 可选 |

## 6. 开源时最重要的统一口径

建议后续对外统一这么讲：

- ReferGaussian/ReferGaussian 的核心不是“给 4DGS 加一个语义头”，而是把 dynamic Gaussian 改造成具有显式 temporal support 的 generalized spacetime primitive。
- 语义阶段不重新训练底层 scene representation，而是基于 `entitybank` 做 training-free grounding。
- 应用侧能力也是消费同一个 `entitybank` 和 query-specific reassignment，而不是另起一个编辑系统。
