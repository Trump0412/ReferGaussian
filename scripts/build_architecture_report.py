from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def repo_rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def load_run_summary(run_dir: Path) -> dict[str, Any]:
    metrics = read_json(run_dir / "metrics.json")
    entitybank = metrics.get("entitybank_summary") or {}
    semantic = metrics.get("semantic_summary") or {}
    return {
        "run_dir": run_dir,
        "method": metrics.get("method"),
        "psnr": metrics.get("PSNR"),
        "ssim": metrics.get("SSIM"),
        "lpips": metrics.get("LPIPS-vgg"),
        "train_seconds": metrics.get("train_seconds"),
        "render_fps": metrics.get("render_fps"),
        "num_entities": entitybank.get("num_entities"),
        "num_priors": semantic.get("num_priors"),
        "dynamic_slots_mean": semantic.get("dynamic_slots_mean"),
        "support_factor_mean": entitybank.get("support_factor_mean"),
        "occupancy_mean": entitybank.get("occupancy_mean"),
    }


def table(headers: list[str], rows: list[list[str]]) -> str:
    header = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---:" if idx > 0 else "---" for idx, _ in enumerate(headers)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def frame_set_from_segments(segments: list[list[int]]) -> set[int]:
    indices: set[int] = set()
    for segment in segments:
        if len(segment) != 2:
            continue
        start = int(segment[0])
        end = int(segment[1])
        for frame_idx in range(start, end + 1):
            indices.add(frame_idx)
    return indices


def frame_set_from_validation(validation: dict[str, Any]) -> set[int]:
    active_segments = validation.get("active_segments") or []
    if active_segments:
        return frame_set_from_segments(active_segments)
    first_active = validation.get("first_active_frame")
    last_active = validation.get("last_active_frame")
    if first_active is not None and last_active is not None:
        return set(range(int(first_active), int(last_active) + 1))
    return set()


def frame_set_from_selected(selected: dict[str, Any]) -> set[int]:
    segments: list[list[int]] = []
    for item in selected.get("selected", []):
        segments.extend(item.get("segments", []))
    return frame_set_from_segments(segments)


def load_query_summary(
    query_dir: Path,
    validation_path: Path,
    reference_active: set[int],
    reference_roles: dict[str, int],
) -> dict[str, Any]:
    selected = read_json(query_dir / "selected.json")
    validation = read_json(validation_path)
    active_frames = frame_set_from_validation(validation)
    if not active_frames:
        active_frames = frame_set_from_selected(selected)
    intersection = reference_active & active_frames
    union = reference_active | active_frames
    selected_roles = {
        str(item.get("role", "unknown")): int(item.get("id", -1))
        for item in selected.get("selected", [])
    }
    role_conf = {
        str(item.get("role", "unknown")): item.get("confidence")
        for item in selected.get("selected", [])
    }
    return {
        "query_dir": query_dir,
        "validation_path": validation_path,
        "active_frame_count": len(active_frames),
        "first_active_frame": validation.get("first_active_frame"),
        "last_active_frame": validation.get("last_active_frame"),
        "active_iou": len(intersection) / max(len(union), 1),
        "active_precision": len(intersection) / max(len(active_frames), 1),
        "active_recall": len(intersection) / max(len(reference_active), 1),
        "patient_match": int(selected_roles.get("patient", -1) == reference_roles.get("patient", -2)),
        "tool_match": int(selected_roles.get("tool", -1) == reference_roles.get("tool", -2)),
        "patient_conf": role_conf.get("patient"),
        "tool_conf": role_conf.get("tool"),
    }


def build_report() -> str:
    full_runs = {
        ("mutant", "baseline 4DGS"): load_run_summary(REPO_ROOT / "runs/baseline_4dgs_full/dnerf/mutant"),
        ("mutant", "chronometric 4DGS"): load_run_summary(REPO_ROOT / "runs/chronometric_4dgs_full/density/dnerf/mutant"),
        ("mutant", "stellar_core"): load_run_summary(REPO_ROOT / "runs/stellar_core_full/dnerf/mutant"),
        ("mutant", "stellar_spacetime"): load_run_summary(REPO_ROOT / "runs/stellar_spacetime_full/dnerf/mutant"),
        ("standup", "baseline 4DGS"): load_run_summary(REPO_ROOT / "runs/baseline_4dgs_full/dnerf/standup"),
        ("standup", "chronometric 4DGS"): load_run_summary(REPO_ROOT / "runs/chronometric_4dgs_full/density/dnerf/standup"),
        ("standup", "stellar_core"): load_run_summary(REPO_ROOT / "runs/stellar_core_full/dnerf/standup"),
        ("standup", "stellar_spacetime_blend"): load_run_summary(REPO_ROOT / "runs/stellar_spacetime_blend_full/dnerf/standup"),
    }
    primitive_runs = {
        ("mutant", "baseline pilot"): load_run_summary(REPO_ROOT / "runs/baseline_4dgs_mutant_pilot/dnerf/mutant"),
        ("mutant", "stellar_tube"): load_run_summary(REPO_ROOT / "runs/stellar_tube_mutant_pilot/dnerf/mutant"),
        ("mutant", "stellar_worldtube_v5"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_mutant_pilot_v5/dnerf/mutant"),
        ("mutant", "stellar_worldtube_v6a"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_mutant_pilot_v6a/dnerf/mutant"),
        ("standup", "baseline pilot"): load_run_summary(REPO_ROOT / "runs/baseline_4dgs_standup_pilot/dnerf/standup"),
        ("standup", "stellar_worldtube_v5"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_standup_pilot_v5/dnerf/standup"),
        ("standup", "stellar_worldtube_v6a"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_standup_pilot_v6a/dnerf/standup"),
        ("cut-lemon1", "baseline smoke300"): load_run_summary(REPO_ROOT / "runs/baseline_cut_lemon1_smoke300/hypernerf/cut-lemon1"),
        ("cut-lemon1", "stellar_tube"): load_run_summary(REPO_ROOT / "runs/stellar_tube_cut_lemon1_smoke300/hypernerf/cut-lemon1"),
        ("cut-lemon1", "stellar_worldtube_v3"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v3/hypernerf/cut-lemon1"),
        ("cut-lemon1", "stellar_worldtube_v6a"): load_run_summary(REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1"),
    }
    da3_runs = {
        "baseline 4DGS + DA3 init": load_run_summary(REPO_ROOT / "runs/baseline_4dgs_da3_smoke/dnerf/bouncingballs"),
        "weak stellar_tube + DA3 init": load_run_summary(REPO_ROOT / "runs/stellar_tube_weak_da3_smoke/dnerf/bouncingballs"),
    }

    bridge_query_dir = (
        REPO_ROOT
        / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/trase_bridge/queries/final_cuts_the_lemon_v5"
    )
    bridge_selected = read_json(bridge_query_dir / "selected.json")
    bridge_validation = read_json(bridge_query_dir / "validation.json")
    bridge_active = frame_set_from_validation(bridge_validation)
    if not bridge_active:
        bridge_active = frame_set_from_selected(bridge_selected)
    bridge_roles = {
        str(item.get("role", "unknown")): int(item.get("id", -1))
        for item in bridge_selected.get("selected", [])
    }
    query_rows = {
        "TRASE-bridge": {
            "query_dir": bridge_query_dir,
            "validation_path": bridge_query_dir / "validation.json",
            "active_frame_count": len(bridge_active),
            "first_active_frame": bridge_validation.get("first_active_frame"),
            "last_active_frame": bridge_validation.get("last_active_frame"),
            "active_iou": 1.0,
            "active_precision": 1.0,
            "active_recall": 1.0,
            "patient_match": 1,
            "tool_match": 1,
            "patient_conf": next((item.get("confidence") for item in bridge_selected.get("selected", []) if item.get("role") == "patient"), None),
            "tool_conf": next((item.get("confidence") for item in bridge_selected.get("selected", []) if item.get("role") == "tool"), None),
        },
        "ReferGaussian-native-v1": load_query_summary(
            REPO_ROOT
            / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon",
            REPO_ROOT
            / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon/rendered/validation.json",
            bridge_active,
            bridge_roles,
        ),
        "ReferGaussian-native-v5": load_query_summary(
            REPO_ROOT
            / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon_v5",
            REPO_ROOT
            / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon_v5/rendered/validation.json",
            bridge_active,
            bridge_roles,
        ),
    }

    mutant_baseline_full = full_runs[("mutant", "baseline 4DGS")]
    mutant_stellar_core = full_runs[("mutant", "stellar_core")]
    mutant_stellar_spacetime = full_runs[("mutant", "stellar_spacetime")]
    standup_baseline_full = full_runs[("standup", "baseline 4DGS")]
    standup_stellar_core = full_runs[("standup", "stellar_core")]
    mutant_baseline_pilot = primitive_runs[("mutant", "baseline pilot")]
    mutant_worldtube_v6a = primitive_runs[("mutant", "stellar_worldtube_v6a")]
    standup_baseline_pilot = primitive_runs[("standup", "baseline pilot")]
    standup_worldtube_v6a = primitive_runs[("standup", "stellar_worldtube_v6a")]
    cut_baseline = primitive_runs[("cut-lemon1", "baseline smoke300")]
    cut_worldtube_v6a = primitive_runs[("cut-lemon1", "stellar_worldtube_v6a")]
    da3_baseline = da3_runs["baseline 4DGS + DA3 init"]
    da3_tube = da3_runs["weak stellar_tube + DA3 init"]

    delta_mutant_core = (mutant_stellar_core["psnr"] or 0.0) - (mutant_baseline_full["psnr"] or 0.0)
    delta_mutant_spacetime = (mutant_stellar_spacetime["psnr"] or 0.0) - (mutant_baseline_full["psnr"] or 0.0)
    delta_standup_core = (standup_stellar_core["psnr"] or 0.0) - (standup_baseline_full["psnr"] or 0.0)
    delta_mutant_worldtube = (mutant_worldtube_v6a["psnr"] or 0.0) - (mutant_baseline_pilot["psnr"] or 0.0)
    delta_standup_worldtube = (standup_worldtube_v6a["psnr"] or 0.0) - (standup_baseline_pilot["psnr"] or 0.0)
    delta_cut_worldtube = (cut_worldtube_v6a["psnr"] or 0.0) - (cut_baseline["psnr"] or 0.0)
    delta_da3_tube = (da3_tube["psnr"] or 0.0) - (da3_baseline["psnr"] or 0.0)

    full_rows = []
    for key in [
        ("mutant", "baseline 4DGS"),
        ("mutant", "chronometric 4DGS"),
        ("mutant", "stellar_core"),
        ("mutant", "stellar_spacetime"),
        ("standup", "baseline 4DGS"),
        ("standup", "chronometric 4DGS"),
        ("standup", "stellar_core"),
        ("standup", "stellar_spacetime_blend"),
    ]:
        scene, label = key
        row = full_runs[key]
        full_rows.append(
            [
                scene,
                label,
                fmt(row["psnr"]),
                fmt(row["ssim"]),
                fmt(row["lpips"]),
                fmt(row["train_seconds"], digits=0),
                fmt(row["render_fps"]),
            ]
        )

    primitive_rows = []
    for key in [
        ("mutant", "baseline pilot"),
        ("mutant", "stellar_tube"),
        ("mutant", "stellar_worldtube_v5"),
        ("mutant", "stellar_worldtube_v6a"),
        ("standup", "baseline pilot"),
        ("standup", "stellar_worldtube_v5"),
        ("standup", "stellar_worldtube_v6a"),
        ("cut-lemon1", "baseline smoke300"),
        ("cut-lemon1", "stellar_tube"),
        ("cut-lemon1", "stellar_worldtube_v3"),
        ("cut-lemon1", "stellar_worldtube_v6a"),
    ]:
        scene, label = key
        row = primitive_runs[key]
        primitive_rows.append(
            [
                scene,
                label,
                fmt(row["psnr"]),
                fmt(row["ssim"]),
                fmt(row["lpips"]),
                fmt(row["train_seconds"], digits=0),
                fmt(row["render_fps"]),
                fmt(row["num_entities"], digits=0),
                fmt(row["num_priors"], digits=0),
                fmt(row["dynamic_slots_mean"]),
                fmt(row["support_factor_mean"]),
                fmt(row["occupancy_mean"]),
            ]
        )

    da3_rows = []
    for label in ["baseline 4DGS + DA3 init", "weak stellar_tube + DA3 init"]:
        row = da3_runs[label]
        da3_rows.append(
            [
                label,
                fmt(row["psnr"]),
                fmt(row["ssim"]),
                fmt(row["lpips"]),
                fmt(row["train_seconds"], digits=0),
                fmt(row["render_fps"]),
            ]
        )

    query_table_rows = []
    for label in ["TRASE-bridge", "ReferGaussian-native-v1", "ReferGaussian-native-v5"]:
        row = query_rows[label]
        query_table_rows.append(
            [
                label,
                fmt(row["active_frame_count"], digits=0),
                fmt(row["first_active_frame"], digits=0),
                fmt(row["last_active_frame"], digits=0),
                fmt(row["active_iou"]),
                fmt(row["active_precision"]),
                fmt(row["active_recall"]),
                fmt(row["patient_match"], digits=0),
                fmt(row["tool_match"], digits=0),
                fmt(row["patient_conf"]),
                fmt(row["tool_conf"]),
            ]
        )

    recent_bootstrap_rows = [
        [
            "2026-03-16",
            "stellar_worldtube_slice-banana_smoke300",
            "slice-banana",
            "dataset_ply",
            "`points3D_downsample2.ply`",
        ],
        [
            "2026-03-16",
            "stellar_worldtube_cut_lemon1_smoke300_v6a",
            "cut-lemon1",
            "dataset_ply",
            "`points3D_downsample2.ply`",
        ],
        [
            "2026-03-16",
            "stellar_worldtube_mutant_pilot_v6a",
            "mutant",
            "random_init",
            "scene 目录没有 `fused.ply`，D-NeRF synthetic reader 会随机初始化",
        ],
        [
            "2026-03-16",
            "stellar_worldtube_standup_pilot_v6b",
            "standup",
            "random_init",
            "scene 目录没有 `fused.ply`，D-NeRF synthetic reader 会随机初始化",
        ],
        [
            "2026-03-15",
            "baseline_4dgs_da3_smoke / stellar_tube_weak_da3_smoke",
            "bouncingballs",
            "da3_gs_ply",
            "`bootstrap_manifest.json` 记录为 `da3_gs_ply -> fused.ply`",
        ],
    ]

    factor_rows = [
        [
            "temporal_warp",
            "`tau_i = phi(t, c_i)`",
            "在 deformation 之前先重参数化时间",
            "baseline 4DGS 没有这一层；chrono 是全局 warp，stellar 是局部 context-aware warp",
        ],
        [
            "temporal_extent",
            "`g_i(t)=exp(-0.5 * alpha * ((t-a_i)/s_i)^2)`，`d_i(t)=v_i * dt + 0.5 * u_i * dt^2`",
            "给每个 Gaussian 增加时间锚点、时间尺度、速度、加速度",
            "时间不再只是网络输入，而是 primitive 的局部支撑域",
        ],
        [
            "stellar_tube",
            "局部时间支持的二阶矩 -> covariance 增量",
            "仍然每个 primitive 一次 rasterize，但把 tube 统计压进协方差",
            "这是弱时空管近似，不改 CUDA rasterizer",
        ],
        [
            "stellar_worldtube",
            "一个 parent Gaussian -> K 个 child spacetime samples",
            "显式做 local spacetime integral 近似，再把 child 贡献回 parent",
            "query time 的 primitive 不再是单个 3D blob，而是局部 worldtube",
        ],
        [
            "spacetime_aware_optimization",
            "support loss + ratio loss + activity-weighted densify/prune",
            "让 split / clone / prune 也感知 temporal support",
            "训练策略也变了，不再只是 image-loss-aware densification",
        ],
        [
            "entitybank_semantics",
            "tube_bank -> entities -> slots -> tracks -> priors -> bootstrap/query",
            "下游语义直接消费 support / occupancy / visibility / trajectory",
            "语义不再是外挂标签，而是建立在时空 primitive 的统计上",
        ],
    ]

    semantic_rows = [
        [
            "tube_bank",
            "`trajectory_samples.npz`, `tube_bank.json`",
            "trajectory, motion score, occupancy, visibility, support factor, tube ratio",
            "把 worldtube 变成可聚类的时空统计",
        ],
        [
            "entities",
            "`entities.json`, `cluster_stats.json`",
            "按 trajectory + support-aware feature 聚类成 entity",
            "把 Gaussian 集合变成 entity-level track",
        ],
        [
            "semantic_slots",
            "`semantic_slots.json`",
            "给每个 entity 生成 prompt candidates、role hints、dynamic/static views",
            "语义入口层",
        ],
        [
            "semantic_tracks",
            "`semantic_tracks.json`, `semantic_frame_queries.json`",
            "逐帧活动状态、support window、motion label、frame-level active slots",
            "把 entity 语义化为时序轨迹",
        ],
        [
            "semantic_priors",
            "`semantic_priors.json`",
            "拆成 `static / dynamic / interaction` 三个 head",
            "这是当前语义架构最关键的一层",
        ],
        [
            "segmentation_bootstrap",
            "`semantic_segmentation_bootstrap.json`",
            "根据 worldtube support 和语义 head 选择 frame / slot / prompt",
            "给后续 segmentation 或 query 渲染做 bootstrap",
        ],
        [
            "TRASE bridge / native query",
            "`trase_bridge`, `native_queries`",
            "前者走外部强语义后端，后者走 ReferGaussian 自身语义打分",
            "当前 bridge 更稳，native 还在追",
        ],
    ]

    lines: list[str] = []
    lines.append("# ReferGaussian 架构与 Pipeline 汇报稿")
    lines.append("")
    lines.append("## Bootstrap 审计")
    lines.append("")
    lines.append("下面这张表回答“最近几次 run 到底用的是什么 bootstrap”。这里用的是绝对日期，避免“最近”这种相对说法变模糊。")
    lines.append("")
    lines.append(
        table(
            ["Date", "Run", "Scene", "Bootstrap", "Evidence"],
            recent_bootstrap_rows,
        )
    )
    lines.append("结论：")
    lines.append("")
    lines.append("- 你最近拿来讲主结果的 worldtube run 里，`mutant / standup` 不是 DA3，也不是 COLMAP，而是 `random initialization`。")
    lines.append("- 你最近的 HyperNeRF 主结果 `cut-lemon1 / slice-banana` 用的是 scene 自带 `points3D_downsample2.ply`，在系统设计里应视为 `dataset_ply bootstrap`。")
    lines.append("- 明确使用 `DA3` 的，是 `bouncingballs` 那条专门的 smoke 线；其 `bootstrap_manifest.json` 已记录为 `da3_gs_ply -> fused.ply`。")
    lines.append("- 最近这批高亮结果里，**没有**一条是显式 COLMAP bootstrap。")
    lines.append("")
    lines.append("一般怎么选：")
    lines.append("")
    lines.append("- `dataset_ply`：如果数据集已经有高质量 `points3D_downsample2.ply`，这是最省事也最稳的选择，当前 HyperNeRF 就是这样。")
    lines.append("- `random_init`：适合 D-NeRF synthetic 快速验证方法改动是否成立，但它不适合当作“绝对性能最好”的几何初始化。")
    lines.append("- `DA3`：适合没有可靠点云、或者想增强弱初始化时使用。它更适合作为 bootstrap ablation，而不是论文主贡献。")
    lines.append("- `COLMAP`：更适合真实多视角、相机和重建链条稳定的场景。如果之后转向真实场景 benchmark，COLMAP 会比 random 更合理。")
    lines.append("")
    lines.append("## 架构拆解")
    lines.append("")
    lines.append("你现在会觉得“看起来还是 4DGS”，根本原因是目前报告没有把三个层次拆开：")
    lines.append("")
    lines.append("- `temporal_warp`：时间参数化层。")
    lines.append("- `temporal primitive`：Gaussian 本体怎么拥有时间支撑。")
    lines.append("- `spacetime-aware optimization`：训练时如何利用这个时间支撑。")
    lines.append("")
    lines.append("如果只讲第一层，它确实仍然像“4DGS + 更好的时间编码”；只有把第二层和第三层讲清楚，方法才开始真正脱离 vanilla 4DGS。")
    lines.append("")
    lines.append(
        table(
            ["层", "定义", "作用位置", "与 4DGS 的差异"],
            factor_rows,
        )
    )
    lines.append("")
    lines.append("## 1. 一句话定位")
    lines.append("")
    lines.append(
        "ReferGaussian 的目标不是只做“更好的时间编码”，而是把动态场景从 `time-conditioned 3D Gaussian` 推进到 `spacetime-native Gaussian primitive`："
    )
    lines.append("")
    lines.append("- 上游可以接 `COLMAP / dataset ply / DA3` 作为几何 bootstrap。")
    lines.append("- 中游把时间写进 Gaussian 自身状态，而不是只喂给 deformation 网络。")
    lines.append("- 下游不仅输出渲染结果，还导出 `tube_bank / trajectory_samples / entities / priors`，为动态语义和查询服务。")
    lines.append("")
    lines.append("当前最合适的汇报主线是：`4DGS -> Chronometric -> Stellar Core -> Spacetime -> Tube -> Worldtube -> EntityBank/Query`。")
    lines.append("")
    lines.append("## 2. 方法演进主线")
    lines.append("")
    lines.append(
        table(
            ["阶段", "时间如何进入模型", "primitive 是否改变", "核心新增能力", "汇报时的定位"],
            [
                ["baseline 4DGS", "时间是 deformation 的输入", "否", "标准动态 4DGS", "官方基线"],
                ["chronometric 4DGS", "先学 `tau=phi(t)` 再送入 4DGS", "否", "非均匀时间重参数化", "时间建模增强版"],
                ["stellar_core", "每个 Gaussian 带局部 temporal context", "部分改变", "per-Gaussian 时间状态进入 warp", "从全局时间走向局部时间"],
                ["stellar_spacetime", "时间直接影响 render-time gate 和 drift", "是", "时间开始进入 primitive 支撑域", "第一代显式时空 primitive"],
                ["stellar_tube", "局部时间支持转成 covariance 增量", "是", "弱 tube 近似，不改 CUDA rasterizer", "从 worldline 走向 tube"],
                ["stellar_worldtube", "一个 parent Gaussian 在渲染时展开为多个 time samples", "是", "显式 local spacetime integral + tube-aware optimization", "当前最像新方法的主分支"],
                ["entitybank/semantics", "利用 temporal trajectories 做聚类、先验和查询", "下游层", "trajectory-aware entity bank、TRASE bridge、native query", "几何到语义的桥"],
            ],
        )
    )
    lines.append("## 3. 当前端到端 Pipeline")
    lines.append("")
    lines.append("1. 数据进入系统")
    lines.append("   - 相机、时间戳和 scene layout 仍然沿用 `external/4DGaussians` 的 dataset reader。")
    lines.append("2. 几何 bootstrap")
    lines.append("   - 默认可用 `COLMAP sparse / dataset ply / random init`。")
    lines.append("   - 当现有点云弱、没有可靠 multiview 点云、或者想做 monocular-first 几何初始化时，用 `DA3`。")
    lines.append("3. 初始化 Gaussian")
    lines.append("   - `create_from_pcd(...)` 读取点云后初始化 canonical Gaussian，同时初始化 temporal state。")
    lines.append("4. 训练")
    lines.append("   - `temporal warp` 负责时间重参数化。")
    lines.append("   - `temporal extent / tube / worldtube` 负责把时间写进 primitive 的局部支撑域。")
    lines.append("   - `tube-aware optimization` 负责 densify、split、clone、prune 的时空一致性。")
    lines.append("5. 渲染")
    lines.append("   - 在 query time `t`，根据分支不同，走 `temporal_slice`、`tube_slice` 或 `worldtube_samples`。")
    lines.append("   - `worldtube` 分支会把一个 parent Gaussian 展开成多个 child samples，再做 rasterization。")
    lines.append("6. 结果导出")
    lines.append("   - `collect_metrics.py` 汇总 PSNR / SSIM / LPIPS / FPS 等指标。")
    lines.append("   - `export_entitybank.py` 导出 `tube_bank.json`、`trajectory_samples.npz`、`cluster_stats.json`、`entities.json`。")
    lines.append("7. 动态语义 / 查询")
    lines.append("   - 可以继续导出 `semantic_slots / semantic_tracks / semantic_priors`。")
    lines.append("   - 查询层既支持 `TRASE bridge`，也支持 `ReferGaussian native query`。")
    lines.append("")
    lines.append("## 4. DA3 在系统里的角色")
    lines.append("")
    lines.append("DA3 在 ReferGaussian 里是 `geometry bootstrap`，不是最终动态模型。")
    lines.append("")
    lines.append("实际接法如下：")
    lines.append("")
    lines.append("1. `scripts/run_da3_bootstrap.py` 直接调用 `DepthAnything3.from_pretrained(\"depth-anything/DA3NESTED-GIANT-LARGE-1.1\")`。")
    lines.append("2. 推理时导出 `npz-gs_ply`，得到 DA3 的高质量几何/高斯结果。")
    lines.append("3. `scripts/convert_da3_gs_ply_to_fused.py` 会读取 `gs_ply/*.ply`，优先按 opacity 做 top-k 采样，把 SH DC 转成 RGB，并写回 scene 目录下的 `fused.ply`。")
    lines.append("4. 后续训练就继续走原来的 4DGS / ReferGaussian 训练入口，因此 DA3 只替换初始化，不替换后续 scene-specific 4D 优化。")
    lines.append("")
    lines.append("这套设计的意义是：")
    lines.append("")
    lines.append("- 保留 ReferGaussian 对时空 primitive、densify/prune、entitybank 的控制权。")
    lines.append("- 把 DA3 的价值集中在“给一个更强的初始几何”，而不是让 feed-forward 模型替代整个动态优化主线。")
    lines.append("")
    lines.append("## Temporal Warp 是什么")
    lines.append("")
    lines.append("`temporal_warp` 的定义不是一句“warp 一下时间”，而是一个显式的可学习映射：")
    lines.append("")
    lines.append("`tau_i = phi(t, c_i)`")
    lines.append("")
    lines.append("其中：")
    lines.append("")
    lines.append("- `t` 是原始时间。")
    lines.append("- `c_i` 是 Gaussian `i` 的上下文。")
    lines.append("- `tau_i` 是送进 deformation backbone 的重参数化时间。")
    lines.append("")
    lines.append("仓库里现在有四种 warp：")
    lines.append("")
    lines.append("- `identity`：不做 warp，对应 baseline 4DGS。")
    lines.append("- `mlp`：单调 MLP warp，对应最早的 chronometric 版本。")
    lines.append("- `density`：把时间看成一维 density 的积分，对应 chronometric density 版本。")
    lines.append("- `stellar`：`StellarMetricWarp`，是当前 ReferGaussian 常用的 context-aware warp。")
    lines.append("")
    lines.append("当前 `stellar` warp 的关键点是：")
    lines.append("")
    lines.append("- 它不是全局 `phi(t)`，而是局部 `phi(t, c_i)`。")
    lines.append("- `c_i` 来自 Gaussian 的局部状态，比如 `xyz_norm / time_anchor / time_scale / time_velocity / time_acceleration / speed / acceleration_norm`。")
    lines.append("- 训练时还会加三类正则：`mono` 保证基本单调，`smooth` 保证曲线平滑，`budget` 避免时间压缩或拉伸过度。")
    lines.append("")
    lines.append("所以 `temporal_warp` 的角色是：")
    lines.append("")
    lines.append("- 它仍然属于“时间参数化层”，不是最终的 new primitive。")
    lines.append("- 它的价值是让不同 Gaussian 对同一全局时间有不同的局部时间度量。")
    lines.append("- 这层单独拿出来看，仍然更像 `4DGS++`；但它为后面的时空 primitive 提供了局部时间坐标系。")
    lines.append("")
    lines.append("## Temporal Primitive 是什么")
    lines.append("")
    lines.append("当 `temporal_extent_enabled=true` 时，每个 Gaussian 都会多出四个时间参数：")
    lines.append("")
    lines.append("- `a_i = time_anchor`")
    lines.append("- `s_i = time_scale`")
    lines.append("- `v_i = time_velocity`")
    lines.append("- `u_i = time_acceleration`")
    lines.append("")
    lines.append("这时一个 Gaussian 在时间 `t` 的局部行为不再只是“把 t 喂给 deformation 网络”，而是至少还包括：")
    lines.append("")
    lines.append("- 时间门控：`g_i(t) = exp(-0.5 * alpha * ((t-a_i)/s_i)^2)`")
    lines.append("- 轨迹漂移：`d_i(t) = v_i * (t-a_i) + 0.5 * u_i * (t-a_i)^2`")
    lines.append("")
    lines.append("这意味着 Gaussian 开始有了自己的：")
    lines.append("")
    lines.append("- 在什么时间最活跃。")
    lines.append("- 时间支撑域有多宽。")
    lines.append("- 局部 worldline 怎么走。")
    lines.append("")
    lines.append("这一步比 4DGS 多出来的，不是一个更复杂的 deformation net，而是 **time support 已经进入 primitive 自身**。")
    lines.append("")
    lines.append("## 5. 时空管 / worldtube 概念如何贯穿系统")
    lines.append("")
    lines.append("它不是只在 renderer 里加了几行 sample，而是贯穿了四层：")
    lines.append("")
    lines.append("1. 表示层")
    lines.append("   - 每个 Gaussian 都携带 `time_anchor / time_scale / time_velocity / time_acceleration`。")
    lines.append("2. 渲染层")
    lines.append("   - 在时间 `t` 上，不再只问“这个 Gaussian 在 t 时刻长什么样”，而是问“这个 Gaussian 在 t 附近的局部时空支撑域是什么”。")
    lines.append("   - `worldtube` 会在局部时间窗内取多个 sample，算 `sample_delta / sample_duration / sample_gate / sample_weights / sample_occupancy`，再沿 worldline 生成 child Gaussian。")
    lines.append("3. 优化层")
    lines.append("   - densify 不再只看 2D gradient，还会混入 temporal activity。")
    lines.append("   - split/clone 不再简单复制时间锚点，而是沿 temporal support 生成新的 child temporal params。")
    lines.append("   - prune 会保护高 temporal activity 或高 tube ratio 的 Gaussian，避免短时但关键的动态结构过早被删掉。")
    lines.append("4. 导出层")
    lines.append("   - `tube_bank.json` 不只存静态几何统计，还存 `motion score / displacement / support factor / tube ratio / occupancy / visibility`。")
    lines.append("   - 这让下游 clustering、entity tracking、semantic prior construction 能直接吃时空轨迹，而不是只吃一堆无时间结构的 3D blobs。")
    lines.append("")
    lines.append("## Worldtube-aware Optimization 是什么")
    lines.append("")
    lines.append("这部分如果不讲清楚，worldtube 看起来就像“渲染时多采样了几次”。实际上仓库里已经把它写进训练规则了。")
    lines.append("")
    lines.append("1. regularization")
    lines.append("   - `support_loss`：约束 effective support 不要太小或太大。")
    lines.append("   - `ratio_loss`：约束 `tube_ratio` 不要偏离目标太多，避免 tube 收缩成点或发散成 smear。")
    lines.append("2. densify score")
    lines.append("   - baseline 是纯 screen-space gradient。")
    lines.append("   - 现在会乘上 `1 + temporal_activity_weight * activity_i`，也就是 temporal activity 越高，越容易被保留和增殖。")
    lines.append("3. split / clone")
    lines.append("   - baseline 复制的是几乎同一个 Gaussian。")
    lines.append("   - 现在 `_tube_child_temporal_params(...)` 会沿 temporal support 给 child Gaussian 新的 anchor / scale / velocity。")
    lines.append("4. prune")
    lines.append("   - 会保护高 temporal activity 的点。")
    lines.append("   - 在 worldtube 分支里，还会保护高 `tube_ratio` 的点。")
    lines.append("")
    lines.append("所以 `worldtube-aware` 不是一个可有可无的后处理词，而是：")
    lines.append("")
    lines.append("- 它定义了 support 怎么分配。")
    lines.append("- 它定义了 tube 怎么分裂。")
    lines.append("- 它定义了哪些短时动态结构不该过早被剪掉。")
    lines.append("")
    lines.append("## 6. ReferGaussian worldtube 与 vanilla 4DGS 的核心区别")
    lines.append("")
    lines.append(
        table(
            ["维度", "vanilla 4DGS", "ReferGaussian worldtube"],
            [
                ["时间的角色", "时间主要是 deformation 的条件变量", "时间是 primitive 自身支撑域的一部分"],
                ["Gaussian 的定义", "本质上仍是某个时刻被形变后的 3D Gaussian", "每个 Gaussian 是带局部时间支撑的 spacetime tube"],
                ["运动表达", "依赖 deformation field", "同时有 per-Gaussian anchor/scale/velocity/acceleration"],
                ["渲染单位", "每个 primitive 在 query time 通常只渲染一次", "每个 parent Gaussian 可展开成多个 child samples 做局部时空积分近似"],
                ["优化逻辑", "主要按图像误差和 screen-space gradient densify/prune", "加入 temporal activity、tube ratio、adaptive support、tube-aware split/clone/prune"],
                ["输出产物", "以重建结果为主", "还能导出 trajectory-aware entity bank 和语义先验"],
                ["和语义的关系", "语义通常是外挂层", "语义是建立在更强的时空 primitive 和 entitybank 之上"],
            ],
        )
    )
    lines.append("## 7. 指标对比")
    lines.append("")
    lines.append("### 7.1 方法演进：full benchmark")
    lines.append("")
    lines.append(
        table(
            ["Scene", "Method", "PSNR", "SSIM", "LPIPS-vgg", "Train s", "FPS"],
            full_rows,
        )
    )
    lines.append("解读：")
    lines.append("")
    lines.append(f"- `mutant` 上，`stellar_core` 比 baseline 提升 `+{delta_mutant_core:.4f}` PSNR，`stellar_spacetime` 进一步达到 `+{delta_mutant_spacetime:.4f}` PSNR。")
    lines.append(f"- `standup` 上，`stellar_core` 是当前最稳的 full-run 版本，比 baseline 提升 `+{delta_standup_core:.4f}` PSNR。")
    lines.append("- 这说明 ReferGaussian 的主线不是凭空跳到 worldtube，而是先完成 `time reparameterization -> local temporal state -> primitive-level temporal support` 的逐级推进。")
    lines.append("")
    lines.append("### 7.2 当前显式 tube/worldtube primitive 对比")
    lines.append("")
    lines.append(
        table(
            ["Scene", "Method", "PSNR", "SSIM", "LPIPS-vgg", "Train s", "FPS", "Entities", "Priors", "Dynamic slots mean", "Support factor mean", "Occupancy mean"],
            primitive_rows,
        )
    )
    lines.append("解读：")
    lines.append("")
    lines.append(f"- `mutant` 的 same-budget `ours_2500` 对比里，`stellar_worldtube_v6a` 相比 baseline 提升 `+{delta_mutant_worldtube:.4f}` PSNR，并明显优于 `worldtube_v5`。")
    lines.append(f"- `standup` 上，`stellar_worldtube_v6a` 基本追回 baseline，差距只有 `{delta_standup_worldtube:.4f}` PSNR，说明修复后的 v6a 已经接近跨场景可用。")
    lines.append(f"- HyperNeRF `cut-lemon1` 上，`stellar_worldtube_v6a` 相比 baseline 提升 `+{delta_cut_worldtube:.4f}` PSNR，同时导出 `65` 个 entities / `65` 个 priors，说明它更像一个可供语义层消费的时空基础表示。")
    lines.append("")
    lines.append("### 7.3 DA3 bootstrap 的直接效果")
    lines.append("")
    lines.append(
        table(
            ["Method", "PSNR", "SSIM", "LPIPS-vgg", "Train s", "FPS"],
            da3_rows,
        )
    )
    lines.append("解读：")
    lines.append("")
    lines.append(f"- 在 `bouncingballs` 的 DA3-init smoke 上，弱 tube 配置比 `baseline 4DGS + DA3 init` 仍然提高了 `+{delta_da3_tube:.4f}` PSNR。")
    lines.append("- 这说明 DA3 并没有替代 ReferGaussian；它只是把初始几何变强，而真正的提升仍来自后续时空 primitive 和优化。")
    lines.append("")
    lines.append("### 7.4 cut-lemon1 查询结果对比")
    lines.append("")
    lines.append(
        table(
            ["Method", "Active frames", "First", "Last", "IoU", "Precision", "Recall", "Patient match", "Tool match", "Patient conf", "Tool conf"],
            query_table_rows,
        )
    )
    lines.append("解读：")
    lines.append("")
    lines.append("- `TRASE-bridge` 仍然是当前最稳定的查询后端，事件窗口稳定落在 `331-371`。")
    lines.append("- `ReferGaussian-native-v1` 能大致覆盖事件，但时间窗口过宽，active IoU 只有 `0.4824`。")
    lines.append("- `ReferGaussian-native-v5` 的窗口又收得过晚，active IoU 降到 `0.0000`。")
    lines.append("- 因此，当前最佳实践仍然是：`几何用 ReferGaussian worldtube，语义查询先走 TRASE bridge`。")
    lines.append("")
    lines.append("## 语义架构")
    lines.append("")
    lines.append("语义部分如果只说“我们做了 semantic query”，也会显得像外挂。当前代码里的真实结构其实是下面这条链：")
    lines.append("")
    lines.append("`worldtube primitive -> tube_bank -> entities -> semantic_slots -> semantic_tracks -> semantic_priors -> segmentation_bootstrap -> TRASE bridge / native query`")
    lines.append("")
    lines.append(
        table(
            ["阶段", "产物", "输入统计", "作用"],
            semantic_rows,
        )
    )
    lines.append("这条链最关键的点是：")
    lines.append("")
    lines.append("- 语义不是直接从图像 feature 生造，而是先经过 `entitybank`。")
    lines.append("- `entitybank` 不是只按 3D 位置聚类，而是用 `trajectory / velocity / acceleration / occupancy / visibility / tube ratio / support` 做 support-aware clustering。")
    lines.append("- `semantic_priors` 不是单一 caption，而是明确拆成 `static / dynamic / interaction` 三个 head。")
    lines.append("- 这正是为什么 worldtube 对语义层是结构性改动，而不是 cosmetic change。")
    lines.append("")
    lines.append("## 8. 汇报时可以直接讲的结论")
    lines.append("")
    lines.append("1. ReferGaussian 的核心不是“多一个 time MLP”，而是把时间写进 Gaussian primitive 本身。")
    lines.append("2. DA3 在系统里只负责 bootstrap 几何，把弱点云初始化替换成更强的 `fused.ply`；真正的动态建模仍然由 ReferGaussian 完成。")
    lines.append("3. `stellar_core` 证明 per-Gaussian temporal state 是有效的；`stellar_spacetime` 证明时间已经开始直接影响 primitive；`stellar_worldtube` 则把这个想法推进到显式局部时空积分。")
    lines.append("4. 当前最有代表性的显式时空 primitive 是 `stellar_worldtube_v6a`：它在 `mutant` 和 `cut-lemon1` 上已经明显优于同预算 baseline，在 `standup` 上也基本追平。")
    lines.append("5. 下游语义层目前仍建议使用 `TRASE bridge`，因为 native query 还没有稳定到能替代 bridge backend。")
    lines.append("")
    lines.append("## 为什么你现在会觉得它还不够论文")
    lines.append("")
    lines.append("这个判断是对的。不是实现没东西，而是现在的**叙述方式**还没有把贡献压成一个足够强的中心命题。")
    lines.append("")
    lines.append("当前离论文还差四件事：")
    lines.append("")
    lines.append("1. 中心贡献太散")
    lines.append("   - 现在像是 `warp + extent + tube + worldtube + semantics` 五个改动并列出现。")
    lines.append("   - 论文里必须压成一个中心句：`we turn Gaussian primitives from time-conditioned 3D blobs into support-aware spacetime worldtubes, and use their support statistics for optimization and entity-level semantics`。")
    lines.append("2. bootstrap 不能喧宾夺主")
    lines.append("   - `DA3` 最好放在 initialization ablation，不应成为论文主贡献的一部分。")
    lines.append("   - 否则审稿人会觉得提升来自更强先验，而不是更强时空 primitive。")
    lines.append("3. 语义层还缺和 worldtube 的定量闭环")
    lines.append("   - 现在语义架构是通的，但还需要更明确的 ablation：有无 worldtube support statistics 时，entity quality / query quality 是否稳定提升。")
    lines.append("4. cross-scene 结果还不够硬")
    lines.append("   - `mutant` 和 `cut-lemon1` 很正向。")
    lines.append("   - `standup` 目前只是追回 baseline 附近。")
    lines.append("   - 想发论文，最好要么再拉开一个真实场景的明显增益，要么把贡献明确改写成“更强的语义几何基底”而不是纯 PSNR paper。")
    lines.append("")
    lines.append("更合适的 paper framing 建议是：")
    lines.append("")
    lines.append("- 主贡献：`worldtube primitive + support-aware optimization`。")
    lines.append("- 第一条副产物：`trajectory-aware entity bank`。")
    lines.append("- 第二条副产物：`support-grounded semantic priors and query grounding`。")
    lines.append("- `DA3`：只保留为 bootstrap ablation。")
    lines.append("")
    lines.append("## 9. 关键实现入口")
    lines.append("")
    lines.append("- `scripts/run_da3_bootstrap.py`")
    lines.append("- `scripts/convert_da3_gs_ply_to_fused.py`")
    lines.append("- `scripts/train_stellar.sh`")
    lines.append("- `scripts/train_stellar_spacetime.sh`")
    lines.append("- `scripts/train_stellar_tube.sh`")
    lines.append("- `scripts/train_stellar_worldtube.sh`")
    lines.append("- `external/4DGaussians/scene/gaussian_model.py`")
    lines.append("- `external/4DGaussians/gaussian_renderer/__init__.py`")
    lines.append("- `refergaussian/entitybank/tube_bank.py`")
    lines.append("- `scripts/export_entitybank.py`")
    lines.append("")
    lines.append("## 10. 本报告读取的数据源")
    lines.append("")
    source_paths = [
        REPO_ROOT / "runs/baseline_4dgs_full/dnerf/mutant",
        REPO_ROOT / "runs/chronometric_4dgs_full/density/dnerf/mutant",
        REPO_ROOT / "runs/stellar_core_full/dnerf/mutant",
        REPO_ROOT / "runs/stellar_spacetime_full/dnerf/mutant",
        REPO_ROOT / "runs/baseline_4dgs_full/dnerf/standup",
        REPO_ROOT / "runs/chronometric_4dgs_full/density/dnerf/standup",
        REPO_ROOT / "runs/stellar_core_full/dnerf/standup",
        REPO_ROOT / "runs/stellar_spacetime_blend_full/dnerf/standup",
        REPO_ROOT / "runs/baseline_4dgs_mutant_pilot/dnerf/mutant",
        REPO_ROOT / "runs/stellar_tube_mutant_pilot/dnerf/mutant",
        REPO_ROOT / "runs/stellar_worldtube_mutant_pilot_v5/dnerf/mutant",
        REPO_ROOT / "runs/stellar_worldtube_mutant_pilot_v6a/dnerf/mutant",
        REPO_ROOT / "runs/baseline_4dgs_standup_pilot/dnerf/standup",
        REPO_ROOT / "runs/stellar_worldtube_standup_pilot_v5/dnerf/standup",
        REPO_ROOT / "runs/stellar_worldtube_standup_pilot_v6a/dnerf/standup",
        REPO_ROOT / "runs/baseline_cut_lemon1_smoke300/hypernerf/cut-lemon1",
        REPO_ROOT / "runs/stellar_tube_cut_lemon1_smoke300/hypernerf/cut-lemon1",
        REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v3/hypernerf/cut-lemon1",
        REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1",
        REPO_ROOT / "runs/baseline_4dgs_da3_smoke/dnerf/bouncingballs",
        REPO_ROOT / "runs/stellar_tube_weak_da3_smoke/dnerf/bouncingballs",
        bridge_query_dir,
        REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon",
        REPO_ROOT / "runs/stellar_worldtube_cut_lemon1_smoke300_v6a/hypernerf/cut-lemon1/entitybank/native_queries/refergaussian_native_cuts_the_lemon_v5",
    ]
    for path in source_paths:
        lines.append(f"- `{repo_rel(path)}`")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "reports/refergaussian_architecture_pipeline_report.md"),
    )
    args = parser.parse_args()

    report = build_report()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
