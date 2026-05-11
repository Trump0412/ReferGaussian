# ReferGaussian / ReferGaussian 方法说明

## 1. 方法目标

我们的目标是把动态 Gaussian 从“在时刻 `t` 被查询的一团 3D 渲染原语”改造成“具有显式 temporal support 的时空单元”，并在此基础上实现 **training-free 4D semantic grounding**。

更直白地说，我们想解决两个问题：

1. 让动态 Gaussian 自己知道“它在什么时间段活跃、沿着什么局部轨迹运动”。
2. 让语义 grounding 建立在这些稳定的时空实体上，而不是重新给每个场景训练一个 dense language field。

因此，整条方法主线可以写成：

`视频序列 -> ReferGaussian 重建 -> entity bank -> semantic slots/tracks/priors -> query-conditioned grounding/rendering/editing`

这里的 `training-free` 是指语义阶段不再对底层 scene representation 做场景特定的语义微调。底层几何表示依然通过 photometric reconstruction 训练得到，但语义 grounding、query selection 和应用阶段都在训练后的结构化实体之上完成。

## 2. 核心想法

 `4DGS` 里，时间更多像是 deformation 网络的输入条件；Gaussian 本体仍然更接近一个空间原语。

ReferGaussian 的关键变化是：给每个 Gaussian 额外加入一组 temporal state，使它不再只是空间里的一个点状/椭球状 blob，而是一个局部的时空支撑单元。

对第 `i` 个 Gaussian，我们写成：

`Theta_i = {mu_i, Sigma_i, alpha_i, f_i, a_i, s_i, v_i, u_i}`

其中：

- `mu_i, Sigma_i, alpha_i, f_i` 分别是中心、协方差、不透明度和外观特征。
- `a_i` 是时间锚点，表示这个 Gaussian 主要活跃在什么时间附近。
- `s_i` 是时间尺度，表示它的时间支撑范围有多宽。
- `v_i` 是速度。
- `u_i` 是加速度。

时间不再只是“喂给网络的一个标量”，而是 primitive 自身的一部分。

## 3. 时间重参数化

为了让不同 Gaussian 对同一个全局时间有不同的局部时间度量，我们还允许时间先经过一个可学习映射：

`tau_i = phi(t, c_i)`

其中 `c_i` 是 Gaussian 的局部上下文，例如：

- 归一化位置
- `time_anchor`
- `time_scale`
- `time_velocity`
- `time_acceleration`
- 运动强度

这部分在仓库里对应 `identity / mlp / density / stellar` 四类 warp。

这一步解决的是“时间怎么量化更合理”，真正的贡献：后面把 temporal support 写进 primitive 本体。

## 4. 显式 temporal support

在查询时刻 `t`，我们定义：

`Delta t_i = t - a_i`

并用一个时间门控函数表示该 Gaussian 在时刻 `t` 的活跃程度：

`g_i(t) = exp(-0.5 * beta * (Delta t_i / s_i)^2)`

这可以理解为：离 `a_i` 越近，Gaussian 越活跃；离得越远，它的贡献就越小。

同时，我们用局部 worldline drift 描述 Gaussian 的运动：

`d_i(t) = v_i * Delta t_i + 0.5 * u_i * Delta t_i^2`

于是它在时刻 `t` 的中心会变成：

`mu_i(t) = mu_i + lambda_d * d_i(t)`

这时一个 Gaussian 已经不是“固定位置 + 外部 deformation”的纯空间原语，而是一个知道自己何时活跃、如何随时间漂移的局部时空单元。

## 5. 两种实现形态：weak tube 与 worldtube

ReferGaussian 在当前实现里有两种具体落地方式。

### 5.1 Weak tube

`stellar_tube` 是一种弱时空管近似。

它的思想是：在当前查询时间附近，对 Gaussian 的局部时间支撑做一个近似，把这段时空支撑折算成额外的 covariance 修正，然后仍然只 rasterize 一次。

它的优点是：

- 不需要改底层 CUDA rasterizer。
- 工程上稳定。
- 在当前部分重建任务里表现很强。

它的缺点是：

- 它仍然是把时空支撑“压缩”成一次 3D rasterization，表达能力是近似的。

### 5.2 Worldtube

`stellar_worldtube` 是更显式的实现。

它把一个 parent Gaussian 看作一小段局部 worldtube。在渲染时，不是只在时间 `t` 上渲染一次，而是在其局部 temporal support 内采样多个 child samples，再把这些 child samples 一起 rasterize。

对第 `i` 个 Gaussian，我们在局部时间窗内采样 `K` 个 segment。每个样本有：

- 自己的局部时间位置
- 自己的 child center
- 自己的 temporal gate
- 自己的 sample weight

最终图像由这些 child samples 共同生成。

这里要特别澄清两点：

- 当前实现里的 `K` 是全局超参数，而不是“运动越大就自动分出更多 child”。
- 真正自适应的是每个 Gaussian 的 `effective support`、child 的时间持续长度、权重和 occupancy；运动更复杂时，child 会分布在更宽的局部时间支撑内，但 child 数量本身不自动变化。

但非常关键的一点是：

- child sample 只是在渲染时展开出来的；
- 梯度、densification 统计、优化状态仍然回传到原始 parent Gaussian。

所以它仍然兼容现有 3D Gaussian rasterizer，只是把它变成了一个 **local spacetime integral approximation**。

## 6. 自适应 temporal support

显式 temporal support 带来一个新的问题：支撑域不能太短，也不能太长。

- 太短时，模型会退化回接近 point-like 的行为。
- 太长时，会出现 temporal blur。

因此我们根据运动强度、空间尺度和可见性，自适应调节 support。

一个直观的指标是 `tube ratio`，它衡量的是：

- 这段时间里 Gaussian 走过的轨迹长度
- 相对于它自身空间尺度到底是偏短还是偏长

更准确地说，这个“时间窗”不是整段视频上的全局窗口，而是每个 Gaussian 在查询时刻附近的局部积分窗口。它由 `time_scale`、`tube_span` 和自适应得到的 `support_factor` 共同决定，并且会受到速度、加速度、空间尺度和可见性的共同影响。

因此不能简单理解成“速度越大，窗口越长”。真正被控制的是：

- 这段局部运动相对于 Gaussian 空间尺度是否足够显著；
- 当前 Gaussian 是否在该时段内足够可见；
- 它的 temporal support 是否需要被拉宽或收窄，才能避免退化成 point-like 行为或出现 temporal blur。

这一步的意义非常大，因为后面不管是 densify/prune，还是 entity clustering，都会继续消费这些 support statistics。

## 7. Support-aware optimization

当 primitive 已经变成时空单元后，训练规则也必须跟着变。

我们的优化不再只关心重建误差，还会显式加入：

- velocity regularization
- acceleration regularization
- support regularization
- tube ratio regularization

同时，densify/prune/split/clone 也不再只看图像梯度，而会结合 temporal activity。

通俗地说：

- 短时但关键的动态结构，不应该因为像素占比小就被过早剪掉。
- split 出来的新 Gaussian，也不应该只是复制一个完全一样的时间状态，而应该沿当前 worldtube 继续细分。

因此，ReferGaussian 的变化不仅在“怎么渲染”，也在“怎么训练”和“怎么维护表示结构”。

## 8. 从 Gaussian 到 structured dynamic entities

训练结束后，我们不会停在一堆独立的动态 Gaussian 上，而是继续把它们组织成 entity。

为此，我们会对每个 Gaussian 采样：

- trajectory
- displacement
- velocity / acceleration
- support window
- occupancy
- visibility
- effective support
- tube ratio

然后基于这些统计量做 support-aware clustering，导出：

- `trajectory_samples.npz`
- `tube_bank.json`
- `cluster_stats.json`
- `entities.json`

这里的 `support-aware clustering` 在当前主线实现里不是对比学习，也不是端到端 learned clustering。它是一个基于 worldtube 统计量的非参数聚类流程：

- 先根据 `occupancy / support / visibility / opacity / path length / centrality` 计算 salience；
- 再选出高 salience 的 core Gaussian；
- 在这些 core 上用 `kmeans2` 初始化聚类；
- 然后结合 support window overlap、velocity gap、RGB gap、occupancy 和 fragment 规则做 merge / split / absorb。

因此，当前 release 主线的 entitybank 构建应被表述为 `support-aware worldtube clustering`，而不是 `contrastive clustering`。

这一步非常关键，因为它把“底层原语”变成了“可解释的动态实体”。

换句话说，后面的语义 grounding 不再是对 splat 逐点打标签，而是对结构化 entity 做语义映射。

## 9. Training-free semantic interface

在 entity bank 之上，我们构建三个层次的语义接口：

1. `semantic slots`
   - 为每个 entity 生成 prompt candidates、role hints、基础描述。
2. `semantic tracks`
   - 为每一帧导出 entity 的中心、范围、可见性和时序状态。
3. `semantic priors`
   - 把 entity 分解到三个语义头：
     - `static semantics`
     - `dynamic semantics`
     - `interaction semantics`

这种设计的核心思想是：

- 我们不直接对整场景学习一个 dense semantic field。
- 我们先建立稳定的动态实体，再对这些实体施加语义。

还需要区分“主线语义接口”和“实验性分支”：

- 主线 `semantic slots / tracks / priors` 都建立在已经导出的 entity bank 之上，不重新训练底层场景表示，也不默认训练一个对比式语义 embedding。
- 仓库里存在一个单独的 `joint embedding` proposal 分支，用于 query proposal 的实验性筛选；但它不是当前 release 里 entitybank 聚类的主线实现。

因此语义不是“贴在渲染层上”的，而是建立在时空支撑结构上。

## 10. Query-conditioned grounding

对于开放词汇 query，我们采用 query-conditioned 但 training-free 的 grounding pipeline。

流程是：

1. 用 Qwen planner 从 query 中提取主体名词、状态变化线索、时间提示。
2. 用 Grounded-SAM-2 在 2D 图像域产生 detection 和 tracking 证据。
3. 但 2D mask 不直接作为最终输出，而是作为线索去重新选择和重组 learned worldtubes。
4. 在 query-specific entitybank 上做 semantic assignment 和 query selection。
5. 输出最终的 overlay、mask、validation。

当前默认的 query-specific proposal 构建也不是“始终使用 `kmeans2`”。更准确地说：

- 默认流程先根据正负图像证据、worldtube support 和几何一致性得到候选 Gaussian pool；
- refine 阶段优先尝试 `HDBSCAN`，不可用时才回退到 `kmeans2`；
- 其中出现的 `contrastive margin` 只是正负证据分数差的启发式项，不应写成“采用了对比学习”。

这里最重要的不是“有一个 detector”，而是：

- detector 只提供图像域证据；
- 最终结果仍然回到 ReferGaussian 的时空实体空间里完成。

因此，query grounding 仍然保持跨时间、跨视角、可渲染的一致性。

## 11. 应用侧实体剔除

应用侧的实体剔除，也是建立在同一条 query pipeline 上。

也就是说，系统先通过 query-specific grounding 找到需要编辑/移除的实体，然后再：

- 过滤出对应的 Gaussian 子集
- 构造 subset run
- 导出 removal bundle
- 在需要时结合 deep-fill 或补全策略处理结果

所以应用侧并不是另一个完全独立的编辑模型，而是同一个时空实体表示在下游的一种使用方式。

## 12. 方法和现有工作的差异

如果用最简洁的方式概括，ReferGaussian 和现有几类方法的差异如下：

- 相对 vanilla `4DGS`
  - 不是只把时间作为 deformation 条件，而是把时间写进 primitive 的支撑域。
- 相对只做语言头或语义头的方法
  - 不是在原表示上外挂语义，而是先重构出结构化时空实体，再做 grounding。
- 相对 scene-specific semantic optimization
  - 语义阶段不重新训练底层 scene representation，而是在 entity bank 上做 training-free grounding。

## 13. 论文 method 建议写法

如果直接写论文 `Method` 章节，建议组织成下面几节：

1. `Method Overview`
2. `ReferGaussian Representation with Explicit Temporal Support`
3. `Weak Tube and Worldtube Rendering`
4. `Support-Aware Optimization`
5. `Entity-Centric Training-Free Semantic Interface`
6. `Query-Conditioned Grounding and Rendering`

如果只保留一句话作为方法总述，建议写成：

> 我们将动态 Gaussian 从 time-conditioned rendering primitive 推广为具有显式 temporal support 的 generalized spacetime unit，并在其上构建了一个面向结构化 dynamic entities 的 training-free 4D semantic grounding framework。
