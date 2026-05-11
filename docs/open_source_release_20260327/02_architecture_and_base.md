# Architecture And Base

## 1. 总体分层

仓库最适合按五层来理解。

| 层 | 作用 | 关键目录/文件 |
| --- | --- | --- |
| L0 Upstream base | 官方 `4DGaussians` 训练、渲染、数据协议 | `external/4DGaussians/` |
| L1 Temporal warp | 时间重参数化与 warp 持久化 | `refergaussian/temporal/` |
| L2 Spacetime primitive | temporal extent、tube、worldtube、support-aware optimization | `external/4DGaussians/scene/gaussian_model.py`、`external/4DGaussians/gaussian_renderer/__init__.py` |
| L3 Entity bank | 从训练好的 Gaussian 导出时空实体与统计量 | `refergaussian/entitybank/` |
| L4 Semantics and query | slots/tracks/priors、native semantics、Qwen、GSAM2、query render | `refergaussian/semantics/` |

## 2. 真正的基座在哪里

### 2.1 训练与渲染基座

训练与渲染仍然建立在 `external/4DGaussians` 上。关键入口是：

- `external/4DGaussians/train.py`
- `external/4DGaussians/render.py`
- `external/4DGaussians/arguments/__init__.py`
- `external/4DGaussians/scene/gaussian_model.py`
- `external/4DGaussians/gaussian_renderer/__init__.py`

也就是说，ReferGaussian 不是另起炉灶写了一个完全新的 renderer，而是在现有 4DGS 主线中插入新的时间状态、worldtube 支撑和优化规则。

### 2.2 ReferGaussian 自己新增的基座

ReferGaussian 新增的“基础层”主要是三块：

- `refergaussian/temporal`
  - 时间 warp、warp 正则、warp 存取。
- `refergaussian/entitybank`
  - 读取训练结果、采样时空轨迹、聚类成 entities。
- `refergaussian/semantics`
  - 在 entity 级别做语义、query、render、应用。

## 3. 当前架构最重要的修改点

### 3.1 Temporal warp

`refergaussian/temporal/modules.py` 提供：

- `IdentityWarp`
- `MonotonicMLPWarp`
- `DensityIntegralWarp`
- `StellarMetricWarp`

这些 warp 在 deformation 之前对时间做重参数化。它们解决的是“每个 Gaussian 如何测量局部时间”。

### 3.2 Temporal primitive

真正让方法开始脱离 vanilla `4DGS` 的，是给每个 Gaussian 增加：

- `time_anchor`
- `time_scale`
- `time_velocity`
- `time_acceleration`

这部分主要在 `external/4DGaussians/scene/gaussian_model.py` 中实现。

### 3.3 Weak tube 与 worldtube

当前仓库实际上有两种 spacetime primitive 近似：

- `stellar_tube`
  - 把局部时间支撑压进 covariance 修正。
  - 每个 primitive 在 query time 仍然只 rasterize 一次。
- `stellar_worldtube`
  - 在渲染时把一个 parent Gaussian 展开成多个 child time samples。
  - 更接近显式 local spacetime integral approximation。

建议论文叙事把这两者统一成同一个 generalized representation 的两个实现版本：

- `weak tube approximation`
- `explicit worldtube approximation`

### 3.4 Support-aware optimization

这部分也落在 `gaussian_model.py` 和 `train.py` 里，核心是：

- support regularization
- tube ratio regularization
- activity-aware densify
- tube-aware split/clone
- support-aware prune protect

所以 ReferGaussian 的变化不仅是 render-time sample 数增加，而是训练规则也已经随 primitive 改变。

## 4. Entity bank 是几何和语义之间的桥

仓库的统一性很大程度上来自 `entitybank`。

训练完成后，`export_entitybank.py` 会从 `run_dir` 中导出：

- `trajectory_samples.npz`
- `tube_bank.json`
- `cluster_stats.json`
- `entities.json`

其中的统计量已经不只是几何位置，还包括：

- trajectories
- gate/support
- displacement
- occupancy
- visibility
- support factor
- effective support
- tube ratio

也就是说，下游语义不是直接“贴”在 splat 上，而是建立在时空实体上。

## 5. 语义与应用如何挂到同一基座上

语义和应用都直接消费 `entitybank`：

1. `slots.py`
   - 为 entity 生成 prompt candidates 和 role hints。
2. `tracks.py`
   - 为每帧导出时序轨迹。
3. `priors.py`
   - 把 entity 分成 `static / dynamic / interaction` 三个语义头。
4. `native_assignment.py` 与 `qwen_assignment.py`
   - 做 entity-centric semantic assignment。
5. `query_proposal_bridge.py`
   - 把 2D query tracks 重新对齐到 learned worldtubes。
6. `query_render.py`
   - 渲染 query-conditioned overlay / mask / validation。
7. `build_query_removal_bundle.py`
   - 在相同 entity selection 上构造 subset run 和移除结果。

## 6. DA3 在架构里的位置

`DA3` 只在 bootstrap 层起作用。

它负责：

- 当初始化点云弱时提供更强的几何先验。
- 产出 `gs_ply` 或转换后的 `fused.ply`。

它不负责：

- 替代 ReferGaussian 的主训练过程。
- 替代 entitybank。
- 替代下游 query grounding。

因此在对外描述时，应把 `DA3` 写成一个可选初始化路径，而不是方法主体。

## 7. 对论文最友好的统一说法

最简洁准确的说法是：

- 上游仍然沿用 4DGS 的工程主干。
- 我们把 Gaussian primitive 从 `time-conditioned 3D splat` 扩展成 `support-aware spacetime unit`。
- 再把语义 grounding 的最小单位从单个 splat 换成 structured dynamic entity。

