# ReferGaussian Open-Source Release Pack

This directory consolidates the project into a release-oriented reading path so the repository can be explained, packaged, and opened more easily.

The main conclusions are:

- Reconstruction, semantics, and application editing live in one codebase and share one artifact contract.
- The common base is `ReferGaussian + external/4DGaussians`; for temporal branches, downstream stages keep consuming the same `run_dir`, `point_cloud`, `temporal_params`, and `entitybank`.
- The reported tasks do not always use the exact same branch variant (`stellar_tube` vs. `stellar_worldtube`), but they do use the same repository-level base and the same export interface.
- `DA3` is an optional bootstrap path, not the core representation.

Recommended reading order:

1. [Project Overview](01_project_overview_zh.md)
2. [Architecture And Base](02_architecture_and_base.md)
3. [Environment And Dependencies](03_environment_and_dependencies.md)
4. [Runbook And Pipeline](04_runbook_and_pipeline.md)
5. [Key Files Index](05_key_files_index.md)
6. [Method CN](06_method_cn.md)
7. [Release Scope And Backup](07_release_scope_and_backup.md)

If you only need three documents before writing the paper or preparing open source, read:

1. [Project Overview](01_project_overview_zh.md)
2. [Key Files Index](05_key_files_index.md)
3. [Method CN](06_method_cn.md)
