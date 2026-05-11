#!/usr/bin/env python3
"""
run_ours_benchmark_master.py

主编排脚本：在 Ours_benchmark.json 上运行完整评测流程。
- 对已有weaktube重建的场景，直接跑query pipeline
- 对需要训练的场景，先训练再跑query pipeline
- 对需要下载的场景，尝试下载后训练
- 使用两张GPU并行（GPU0 + GPU1）
- 最终运行evaluate_ours_benchmark.py汇总指标

用法:
  python scripts/run_ours_benchmark_master.py [--skip-training] [--skip-download] [--gpu0 0] [--gpu1 1]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ============================================================
# 路径配置
# ============================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_JSON = Path(os.environ.get("OURS_BENCHMARK_JSON", str(REPO_ROOT / "data" / "benchmarks" / "r4d_bench_qa" / "benchmark.json")))
REPORT_DIR = REPO_ROOT / "reports" / "ours_benchmark_eval"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

QUERY_ROOT_MAP_JSON = REPORT_DIR / "query_root_map.json"
DATASET_DIR_MAP_JSON = REPORT_DIR / "dataset_dir_map.json"
EVAL_JSON = REPORT_DIR / "ours_benchmark_eval.json"
EVAL_MD = REPORT_DIR / "ours_benchmark_eval.md"
LOG_FILE = REPORT_DIR / "master.log"

COMMON_SH = REPO_ROOT / "scripts" / "common.sh"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_stellar_tube.sh"
EVAL_SCRIPT = REPO_ROOT / "scripts" / "eval_stellar_tube.sh"
QUERY_PIPELINE = REPO_ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh"
EXPORT_ENTITYBANK = REPO_ROOT / "scripts" / "export_entitybank.py"
EVALUATE_SCRIPT = REPO_ROOT / "scripts" / "evaluate_ours_benchmark.py"

# 最优weaktube超参（与4DLangSplat最优结果对齐: span040_sigma032）
BEST_WEAKTUBE_ENV = {
    "TEMPORAL_TUBE_SAMPLES": "3",
    "TEMPORAL_TUBE_SPAN": "0.40",
    "TEMPORAL_TUBE_SIGMA": "0.32",
    "TEMPORAL_TUBE_WEIGHT_POWER": "1.0",
    "TEMPORAL_TUBE_COVARIANCE_MIX": "0.05",
    "TEMPORAL_DRIFT_SCALE": "1.0",
    "TEMPORAL_GATE_MIX": "1.0",
    "TEMPORAL_DRIFT_MIX": "1.0",
    "TEMPORAL_ACCELERATION_ENABLED": "0",
    "TEMPORAL_VELOCITY_REG_WEIGHT": "0.0",
    "TEMPORAL_ACCELERATION_REG_WEIGHT": "0.0",
}

WEAKTUBE_TRAIN_ARGS = [
    "--iterations", "14000",
    "--coarse_iterations", "3000",
    "--test_iterations", "3000", "7000", "14000",
    "--save_iterations", "7000", "14000",
    "--checkpoint_iterations", "7000", "14000",
]

# ============================================================
# 场景配置
# ============================================================
# query_id前缀 → 场景信息
SCENE_CONFIG = {
    # ----------- 已有最优weaktube重建 (可立即跑query pipeline) -----------
    "espresso": {
        "dataset": "hypernerf",
        "scene": "misc/espresso",
        "run_namespace": "stellar_tube_4dlangsplat_refresh_20260328_espresso",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/misc/espresso"),
        "status": "ready",  # 已有完整重建+entitybank
    },
    "americano": {
        "dataset": "hypernerf",
        "scene": "misc/americano",
        "run_namespace": "stellar_tube_4dlangsplat_refresh_20260328_americano",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/misc/americano"),
        "status": "ready",
    },
    "cut_lemon": {
        "dataset": "hypernerf",
        "scene": "interp/cut-lemon1",
        "run_namespace": "stellar_tube_cutlemon_refresh_20260329",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/interp/cut-lemon1"),
        "status": "ready",
    },
    "split_cookie": {
        "dataset": "hypernerf",
        "scene": "misc/split-cookie",
        "run_namespace": "stellar_tube_full6_20260328_histplus_span040_sigma032",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/misc/split-cookie"),
        "status": "ready",
    },
    # ----------- 有数据，需要训练 (HyperNeRF) -----------
    "keyboard": {
        "dataset": "hypernerf",
        "scene": "misc/keyboard",
        "run_namespace": "stellar_tube_ours_benchmark_keyboard",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/misc/keyboard"),
        "status": "needs_training",
        "gpu": 0,
        "port": 6401,
    },
    "torchchocolate": {
        "dataset": "hypernerf",
        "scene": "interp/torchocolate",
        "run_namespace": "stellar_tube_ours_benchmark_torchocolate",
        "dataset_dir": str(REPO_ROOT / "data/hypernerf/interp/torchocolate"),
        "status": "needs_training",
        "gpu": 1,
        "port": 6402,
    },
    # ----------- 有数据，需要训练 (dynerf) -----------
    "coffee_martini": {
        "dataset": "dynerf",
        "scene": "coffee_martini",
        "run_namespace": "stellar_tube_ours_benchmark_coffee_martini",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/coffee_martini"),
        "status": "needs_training",
        "gpu": 0,
        "port": 6403,
    },
    "flame_steak": {
        "dataset": "dynerf",
        "scene": "flame_steak",
        "run_namespace": "stellar_tube_ours_benchmark_flame_steak",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/flame_steak"),
        "status": "needs_training",
        "gpu": 1,
        "port": 6404,
    },
    # ----------- 需要下载+训练 (Neu3D/dynerf) -----------
    "cook-spinach": {
        "dataset": "dynerf",
        "scene": "cook_spinach",
        "run_namespace": "stellar_tube_ours_benchmark_cook_spinach",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/cook_spinach"),
        "status": "needs_download",
        "gpu": 0,
        "port": 6405,
    },
    "cut_roasted_beef": {
        "dataset": "dynerf",
        "scene": "cut_roasted_beef",
        "run_namespace": "stellar_tube_ours_benchmark_cut_roasted_beef",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/cut_roasted_beef"),
        "status": "needs_download",
        "gpu": 1,
        "port": 6406,
    },
    "flame_salmon": {
        "dataset": "dynerf",
        "scene": "flame_salmon_1",
        "run_namespace": "stellar_tube_ours_benchmark_flame_salmon",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/flame_salmon_1"),
        "status": "needs_download",
        "gpu": 0,
        "port": 6407,
    },
    "sear_steak": {
        "dataset": "dynerf",
        "scene": "sear_steak",
        "run_namespace": "stellar_tube_ours_benchmark_sear_steak",
        "dataset_dir": str(REPO_ROOT / "data/dynerf/sear_steak"),
        "status": "needs_download",
        "gpu": 1,
        "port": 6408,
    },
}

# query_id → 场景key的映射（根据query_id前缀推断）
def get_scene_key(query_id: str) -> str | None:
    """从query_id推断场景key。"""
    # 特殊前缀映射
    prefix_map = {
        "cut_lemon": "cut_lemon",
        "espresso": "espresso",
        "keyboard": "keyboard",
        "torchchocolate": "torchchocolate",
        "cook-spinach": "cook-spinach",
        "cook_spinach": "cook-spinach",
        "cut_roasted_beef": "cut_roasted_beef",
        "flame_salmon": "flame_salmon",
        "sear_steak": "sear_steak",
        "split_cookie": "split_cookie",
        "americano": "americano",
        "coffee_martini": "coffee_martini",
        "flame_steak": "flame_steak",
    }
    for prefix, key in prefix_map.items():
        if query_id.startswith(prefix):
            return key
    return None


# ============================================================
# 日志工具
# ============================================================
def log(msg: str) -> None:
    ts = time.strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ============================================================
# 运行辅助
# ============================================================
def run_cmd(cmd: list[str], env: dict | None = None, cwd: Path | None = None,
            check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    """运行命令，输出到终端和日志。"""
    env_merged = {**os.environ}
    if env:
        env_merged.update(env)
    # 去掉代理
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY", "ftp_proxy", "FTP_PROXY"]:
        env_merged.pop(k, None)
    env_merged["HF_ENDPOINT"] = env_merged.get("HF_ENDPOINT", "https://hf-mirror.com")
    return subprocess.run(
        cmd,
        env=env_merged,
        cwd=str(cwd or REPO_ROOT),
        check=check,
        capture_output=capture,
    )


def gs_python(args: list[str], env: dict | None = None) -> None:
    """使用主conda环境运行python脚本。"""
    gs_env = os.environ.get("GS4D_ENV_PATH", str(REPO_ROOT / ".cache" / "refergaussian" / "conda-envs" / "gs4d-cuda121-py310"))
    cmd = [
        "conda", "run", "--no-capture-output", "-p", gs_env,
        "python", *args,
    ]
    run_cmd(cmd, env=env)


# ============================================================
# 重建检查
# ============================================================
def get_run_dir(scene_key: str) -> Path:
    cfg = SCENE_CONFIG[scene_key]
    ns = cfg["run_namespace"]
    ds = cfg["dataset"]
    scene = cfg["scene"].split("/")[-1]
    return REPO_ROOT / "runs" / ns / ds / scene


def is_trained(scene_key: str) -> bool:
    run_dir = get_run_dir(scene_key)
    pc_dir = run_dir / "point_cloud"
    if not pc_dir.exists():
        return False
    # 检查是否有14000 iteration checkpoint
    iter_dirs = list(pc_dir.glob("iteration_*"))
    return len(iter_dirs) > 0


def has_entitybank(scene_key: str) -> bool:
    run_dir = get_run_dir(scene_key)
    eb = run_dir / "entitybank"
    return (eb / "entities.json").exists() and (eb / "tube_bank.json").exists()


# ============================================================
# 训练函数
# ============================================================
def train_scene(scene_key: str, gpu: int) -> bool:
    cfg = SCENE_CONFIG[scene_key]
    log(f"[train] 开始训练 {scene_key} on GPU{gpu}")

    ns = cfg["run_namespace"]
    env = {
        **os.environ,
        **BEST_WEAKTUBE_ENV,
        "GS_RUN_NAMESPACE": ns,
        "GS_PORT": str(cfg["port"]),
        "CUDA_VISIBLE_DEVICES": str(gpu),
    }
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)

    train_cmd = [
        "bash", str(TRAIN_SCRIPT),
        cfg["dataset"], cfg["scene"],
        *WEAKTUBE_TRAIN_ARGS,
    ]
    log(f"[train] cmd: {' '.join(train_cmd[:4])} ...")
    result = subprocess.run(
        train_cmd,
        cwd=str(REPO_ROOT),
        env=env,
    )
    if result.returncode != 0:
        log(f"[train] 训练失败 {scene_key}: returncode={result.returncode}")
        return False

    # 确保entitybank已导出
    run_dir = get_run_dir(scene_key)
    if not has_entitybank(scene_key):
        log(f"[train] 导出entitybank for {scene_key}")
        gs_python([
            str(EXPORT_ENTITYBANK),
            "--run-dir", str(run_dir),
        ], env={"CUDA_VISIBLE_DEVICES": str(gpu)})

    log(f"[train] 训练完成 {scene_key}")
    return True


# ============================================================
# Query Pipeline
# ============================================================
def run_query_pipeline(
    query_id: str,
    query_text: str,
    run_dir: Path,
    dataset_dir: Path,
    gpu: int | None = None,
) -> Path | None:
    """运行单个query的完整pipeline，返回query输出目录。"""
    query_name = query_id  # 使用query_id作为query_name，方便后续映射
    final_validation = run_dir / "entitybank" / "query_guided" / query_name / "final_query_render_sourcebg" / "validation.json"

    if final_validation.exists():
        log(f"[query] 跳过已完成 {query_id}: {final_validation}")
        return final_validation.parent.parent

    log(f"[query] 运行 {query_id}: \"{query_text[:40]}...\"")
    env = {**os.environ}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY"]:
        env.pop(k, None)
    env["HF_ENDPOINT"] = "https://hf-mirror.com"

    cmd = [
        "bash", str(QUERY_PIPELINE),
        str(run_dir),
        str(dataset_dir),
        query_text,
        query_name,
    ]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    if result.returncode != 0:
        log(f"[query] Pipeline失败 {query_id}: returncode={result.returncode}")
        return None

    output_dir = run_dir / "entitybank" / "query_guided" / query_name
    log(f"[query] 完成 {query_id} -> {output_dir}")
    return output_dir


# ============================================================
# 下载缺失Neu3D数据
# ============================================================
def download_neu3d_scene(scene_key: str) -> bool:
    """尝试从Neural 3D Video数据集下载场景。"""
    cfg = SCENE_CONFIG[scene_key]
    scene_name = cfg["scene"]  # e.g., "cook_spinach"
    data_dir = Path(cfg["dataset_dir"])

    if data_dir.exists() and any(data_dir.iterdir()):
        log(f"[download] {scene_key} 数据已存在: {data_dir}")
        return True

    log(f"[download] 尝试下载 {scene_key} Neu3D数据...")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Neural 3D Video数据集下载（通过4DGaussians建议的方式）
    # 尝试从 hf-mirror 或其他可用来源
    # 实际数据需要从 Facebook Research 的 Neural 3D Video Synthesis 项目获取
    download_script = REPORT_DIR / f"download_{scene_name}.sh"
    download_script.write_text(f"""#!/usr/bin/env bash
# 自动生成的Neu3D场景下载脚本
# 场景: {scene_name}
# 
# 从 Neural 3D Video Synthesis 数据集下载
# 官方来源: https://github.com/facebookresearch/Neural_3D_Video
# 
# 注意：需要从官方数据集获取原始多视角视频，然后:
# 1. python scripts/preprocess_dynerf.py --datadir data/dynerf/{scene_name}
# 2. bash colmap.sh data/dynerf/{scene_name} llff
# 3. python scripts/downsample_point.py data/dynerf/{scene_name}/colmap/dense/workspace/fused.ply \\
#       data/dynerf/{scene_name}/points3D_downsample2.ply
#
# 检查是否有预处理好的数据在 GR4D-Bench 或其他位置

DATA_DIR="{data_dir}"
GR4DBench_DIR="{REPO_ROOT / 'data' / 'GR4D-Bench' / 'data' / 'scenes' / scene_name}"

# 检查GR4D-Bench
if [[ -d "$GR4DBench_DIR/images" ]]; then
    echo "Found sparse frames in GR4D-Bench (not full multi-view data)"
    echo "Full Neu3D data required for reconstruction"
fi

echo "Scene {scene_name} data not available for full reconstruction."
echo "Queries for this scene will be skipped in evaluation."
exit 1
""")
    download_script.chmod(0o755)
    log(f"[download] {scene_key} Neu3D原始多视角数据需手动下载。跳过该场景。")
    log(f"[download] 参考脚本: {download_script}")
    return False


# ============================================================
# 主流程
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="Ours_benchmark 主评测流程")
    parser.add_argument("--skip-training", action="store_true", help="跳过训练步骤")
    parser.add_argument("--skip-download", action="store_true", help="跳过下载步骤")
    parser.add_argument("--skip-query", action="store_true", help="跳过query pipeline")
    parser.add_argument("--gpu0", type=int, default=0, help="GPU 0 编号")
    parser.add_argument("--gpu1", type=int, default=1, help="GPU 1 编号")
    parser.add_argument("--only-eval", action="store_true", help="只运行最终评测")
    parser.add_argument("--only-scenes", nargs="+", default=None, help="只处理指定场景")
    args = parser.parse_args()

    LOG_FILE.write_text(f"=== Ours Benchmark Master [{time.strftime('%Y-%m-%d %H:%M:%S')}] ===\n")
    log("开始 Ours_benchmark 评测流程")

    # 加载benchmark
    benchmark = json.loads(BENCHMARK_JSON.read_text(encoding="utf-8"))
    log(f"加载benchmark: {len(benchmark)} 条查询")

    # 按场景分组
    scene_queries: dict[str, list[dict]] = {}
    for item in benchmark:
        qid = item["query_id"]
        skey = get_scene_key(qid)
        if skey is None:
            log(f"[warn] 未能识别场景: {qid}")
            continue
        if skey not in scene_queries:
            scene_queries[skey] = []
        scene_queries[skey].append(item)

    log(f"识别到 {len(scene_queries)} 个场景: {list(scene_queries.keys())}")

    if args.only_scenes:
        scene_queries = {k: v for k, v in scene_queries.items() if k in args.only_scenes}
        log(f"过滤后场景: {list(scene_queries.keys())}")

    if args.only_eval:
        run_final_evaluation(benchmark)
        return

    # ========================================================
    # Phase 1: 处理"needs_download"场景（尝试下载）
    # ========================================================
    if not args.skip_download:
        log("=== Phase 1: 下载缺失数据集 ===")
        for skey, queries in scene_queries.items():
            cfg = SCENE_CONFIG.get(skey)
            if cfg and cfg["status"] == "needs_download":
                download_neu3d_scene(skey)

    # ========================================================
    # Phase 2: 并行训练（GPU 0 + GPU 1）
    # ========================================================
    if not args.skip_training:
        log("=== Phase 2: 并行训练 ===")
        # 分配训练队列
        gpu0_queue = []  # GPU 0 的训练场景列表
        gpu1_queue = []  # GPU 1 的训练场景列表

        for skey, queries in scene_queries.items():
            cfg = SCENE_CONFIG.get(skey)
            if cfg is None:
                continue
            if cfg["status"] in ("needs_training", "needs_download"):
                if not is_trained(skey):
                    # 检查数据是否存在
                    data_dir = Path(cfg["dataset_dir"])
                    if not data_dir.exists():
                        log(f"[train] 跳过 {skey}：数据不存在 {data_dir}")
                        continue
                    assigned_gpu = cfg.get("gpu", args.gpu0)
                    if assigned_gpu == 0:
                        gpu0_queue.append(skey)
                    else:
                        gpu1_queue.append(skey)
                else:
                    log(f"[train] {skey} 已有训练结果，跳过训练")
                    if not has_entitybank(skey):
                        log(f"[train] {skey} 需要导出entitybank")
                        run_dir = get_run_dir(skey)
                        gs_python([str(EXPORT_ENTITYBANK), "--run-dir", str(run_dir)])

        # 并行训练两队列（每队列内串行，两队列并行）
        def train_queue(queue: list[str], gpu: int) -> None:
            for skey in queue:
                log(f"[train-queue] GPU{gpu} 开始训练 {skey}")
                train_scene(skey, gpu)

        with ThreadPoolExecutor(max_workers=2) as ex:
            f0 = ex.submit(train_queue, gpu0_queue, args.gpu0) if gpu0_queue else None
            f1 = ex.submit(train_queue, gpu1_queue, args.gpu1) if gpu1_queue else None
            for f in [f0, f1]:
                if f:
                    try:
                        f.result()
                    except Exception as e:
                        log(f"[train] 训练队列异常: {e}")

    # ========================================================
    # Phase 3: 运行Query Pipeline（所有可用场景）
    # ========================================================
    if not args.skip_query:
        log("=== Phase 3: 运行Query Pipeline ===")
        query_root_map: dict[str, str] = {}
        dataset_dir_map: dict[str, str] = {}

        # 按GPU分组，并行跑两路query pipeline
        # 这里简单地串行跑每个场景的所有查询
        for skey, queries in scene_queries.items():
            cfg = SCENE_CONFIG.get(skey)
            if cfg is None:
                continue

            run_dir = get_run_dir(skey)
            dataset_dir = Path(cfg["dataset_dir"])

            # 检查是否可以运行query pipeline
            if not run_dir.exists():
                log(f"[query] 跳过 {skey}: run_dir不存在 {run_dir}")
                continue
            if not dataset_dir.exists():
                log(f"[query] 跳过 {skey}: dataset_dir不存在 {dataset_dir}")
                continue
            pc_dir = run_dir / "point_cloud"
            if not pc_dir.exists() or not list(pc_dir.glob("iteration_*")):
                log(f"[query] 跳过 {skey}: 没有完成的训练checkpoint")
                continue
            if not has_entitybank(skey):
                log(f"[query] {skey} 缺少entitybank，尝试导出...")
                gs_python([str(EXPORT_ENTITYBANK), "--run-dir", str(run_dir)])
                if not has_entitybank(skey):
                    log(f"[query] 跳过 {skey}: entitybank导出失败")
                    continue

            log(f"[query] 处理场景 {skey} ({len(queries)} 个查询)")
            for item in queries:
                qid = item["query_id"]
                query_text = item.get("question", "")
                if not query_text:
                    log(f"[query] 跳过 {qid}: 空查询文本")
                    continue

                output_dir = run_query_pipeline(
                    qid, query_text, run_dir, dataset_dir,
                    gpu=cfg.get("gpu", args.gpu0),
                )
                if output_dir:
                    query_root_map[qid] = str(output_dir)
                    dataset_dir_map[qid] = str(dataset_dir)

        # 保存映射
        QUERY_ROOT_MAP_JSON.write_text(
            json.dumps(query_root_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        DATASET_DIR_MAP_JSON.write_text(
            json.dumps(dataset_dir_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        log(f"query_root_map: {len(query_root_map)} 条")

    # ========================================================
    # Phase 4: 最终评测
    # ========================================================
    run_final_evaluation(benchmark)


def run_final_evaluation(benchmark: list) -> None:
    """运行evaluate_ours_benchmark.py汇总指标。"""
    log("=== Phase 4: 最终评测 ===")

    if not QUERY_ROOT_MAP_JSON.exists():
        log("[eval] 没有query_root_map，跳过评测")
        return

    gs_python([
        str(EVALUATE_SCRIPT),
        "--benchmark", str(BENCHMARK_JSON),
        "--query-root-map", str(QUERY_ROOT_MAP_JSON),
        "--dataset-dir-map", str(DATASET_DIR_MAP_JSON) if DATASET_DIR_MAP_JSON.exists() else "",
        "--output-json", str(EVAL_JSON),
        "--output-md", str(EVAL_MD),
        "--skip-missing",
    ])
    log(f"评测完成: {EVAL_JSON}")


if __name__ == "__main__":
    main()
