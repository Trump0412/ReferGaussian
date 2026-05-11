# Release Scope And Backup

## 1. 建议公开的首版仓库范围

首版开源建议聚焦“方法实现 + 运行入口 + 说明文档”，优先保留：

- `README.md`
- `docs/`
- `configs/`
- `refergaussian/`
- `scripts/`
- `patches/`
- `external/4DGaussians/`

如果对外部仓库不想直接 vendoring，可以保留：

- setup 脚本
- patch 说明
- 修改过的 upstream 文件清单

但至少要保证别人能重建你们的改动位置。

## 2. 建议暂不直接公开的大体量内容

这些内容更适合后续按数据、模型、benchmark 分开处理：

- `data/`
- `runs/`
- `logs/`
- 大量阶段性 `reports/`
- 本地输出图片、视频和中间实验目录

## 3. 当前建议的公开仓库骨架

```text
ReferGaussian/
├── README.md
├── docs/
│   └── open_source_release_20260327/
├── configs/
├── refergaussian/
├── scripts/
├── patches/
└── external/
    └── 4DGaussians/
```

如果需要发布 query 功能，再补充：

- `external/Grounded-SAM-2` 的安装说明
- Qwen 模型下载说明

如果需要发布 DA3 bootstrap，再补充：

- `external/Depth-Anything-3` 的安装说明

## 4. 本次整理的备份路径

本次建议生成的备份目录路径：

- `<AUTODL_ROOT>/ReferGaussian_backups/open_source_release_20260327/`

本次实际归档文件路径：

- `<AUTODL_ROOT>/ReferGaussian_backups/open_source_release_20260327/refergaussian_open_source_core_20260327.tar.gz`

本次额外保存的文档快照：

- `<AUTODL_ROOT>/ReferGaussian_backups/open_source_release_20260327/open_source_release_20260327/`
- `<AUTODL_ROOT>/ReferGaussian_backups/open_source_release_20260327/README.md`

建议其中至少包含：

- 新整理的 `docs/open_source_release_20260327/`
- 更新后的 `README.md`
- 一份关键源码归档

## 5. 公开前最终检查清单

1. README 是否直接说明仓库目标与三条任务主线。
2. 环境安装是否能区分主环境、query 环境、DA3 环境。
3. 是否明确说明 `DA3` 只是 bootstrap。
4. 是否明确说明 reconstruction、semantics、application 共享统一 artifact contract。
5. 是否给出 `stellar_tube` 与 `stellar_worldtube` 的关系。
6. 是否列出关键文件与真正修改过的 upstream 文件。
7. 是否去掉本地绝对路径、私人实验备注、无关临时结果。
8. 是否确认大体量 `runs/data/logs` 不会误提交。
9. 是否保留至少一个最小可运行命令链。
10. 是否保留中文 method 草稿，方便后续转成英文论文写法。
