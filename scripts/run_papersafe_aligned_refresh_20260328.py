#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "papersafe_aligned_refresh_20260328"
GS_PYTHON = Path(os.environ.get("GS4D_PYTHON", str(Path(os.environ.get("GS4D_ENV_ROOT", str(Path.home() / ".cache" / "refergaussian" / "conda-envs"))) / "gs4d-cuda121-py310" / "bin" / "python")))
GSAM2_PYTHON = Path(os.environ.get("GS4D_GSAM2_PYTHON", str(Path(os.environ.get("GS4D_ENV_ROOT", str(Path.home() / ".cache" / "refergaussian" / "conda-envs"))) / "grounded-sam2-py310" / "bin" / "python")))

PAPERPUSH_ROOT = Path(os.environ.get("PAPERPUSH_ROOT", str(PROJECT_ROOT.parent)))

SCENES = {
    "americano": {
        "run_dir": PROJECT_ROOT / "runs" / "stellar_tube_4dlangsplat_refresh_20260328_americano" / "hypernerf" / "americano",
        "dataset_dir": PROJECT_ROOT / "data" / "hypernerf" / "misc" / "americano",
        "annotation_dir": PROJECT_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "americano",
        "protocol_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/americano_protocol.json",
        "source_eval_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260327_papersafe_trackprimary_support_americano/americano_eval.json",
    },
    "chickchicken": {
        "run_dir": PROJECT_ROOT / "runs" / "stellar_tube_4dlangsplat_refresh_20260328_chickchicken" / "hypernerf" / "chickchicken",
        "dataset_dir": PROJECT_ROOT / "data" / "hypernerf" / "interp" / "chickchicken",
        "annotation_dir": PROJECT_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "chickchicken",
        "protocol_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/chickchicken_protocol.json",
        "source_eval_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/chickchicken_public_query_eval_fullanchors.json",
    },
    "espresso": {
        "run_dir": PROJECT_ROOT / "runs" / "stellar_tube_4dlangsplat_refresh_20260328_espresso" / "hypernerf" / "espresso",
        "dataset_dir": PROJECT_ROOT / "data" / "hypernerf" / "misc" / "espresso",
        "annotation_dir": PROJECT_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "espresso",
        "protocol_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/espresso_protocol.json",
        "source_eval_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260327_papersafe_trackprimary_support_espresso/espresso_eval.json",
    },
    "split-cookie": {
        "run_dir": PROJECT_ROOT / "runs" / "stellar_tube_full6_20260328_histplus_span040_sigma032" / "hypernerf" / "split-cookie",
        "dataset_dir": PROJECT_ROOT / "data" / "hypernerf" / "misc" / "split-cookie",
        "annotation_dir": PROJECT_ROOT / "data" / "benchmarks" / "4dlangsplat" / "HyperNeRF-Annotation" / "split-cookie",
        "protocol_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/split_cookie_protocol.json",
        "source_eval_json": PAPERPUSH_ROOT / "gs_4dlangsplat_paperpush_20260325/split_cookie_public_query_eval_accpush.json",
    },
}


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _run_wrapped(wrapper: str, script_path: Path, args: list[str | Path], env: dict[str, str] | None = None) -> None:
    python_exe = GS_PYTHON if wrapper == "gs_python" else GSAM2_PYTHON
    env_items = os.environ.copy()
    if env:
        env_items.update(env)
    existing_pythonpath = env_items.get("PYTHONPATH", "")
    prefix = f"{PROJECT_ROOT}:{PROJECT_ROOT / 'external/4DGaussians'}"
    env_items["PYTHONPATH"] = f"{prefix}:{existing_pythonpath}" if existing_pythonpath else prefix
    env_items.setdefault("OMP_NUM_THREADS", "1")
    cmd = [str(python_exe), str(script_path), *[str(item) for item in args]]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env_items)


def _ensure_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    os.symlink(str(target_path), str(link_path), target_is_directory=target_path.is_dir())


def _prepare_query_root(target_query_dir: Path, source_query_dir: Path) -> None:
    if target_query_dir.exists():
        shutil.rmtree(target_query_dir)
    target_query_dir.mkdir(parents=True, exist_ok=True)
    source_query_plan = source_query_dir / "query_plan.json"
    if source_query_plan.exists():
        shutil.copy2(source_query_plan, target_query_dir / "query_plan.json")
    _ensure_symlink(target_query_dir / "grounded_sam2", source_query_dir / "grounded_sam2")


def _build_query_specific_assets(scene_key: str, query_item: dict, scene_cfg: dict, target_query_dir: Path) -> dict:
    source_validation = _read_json(Path(query_item["validation_path"]))
    source_selection_path = Path(source_validation["selection_path"])
    source_query_dir = source_selection_path.parent.parent.parent
    _prepare_query_root(target_query_dir, source_query_dir)

    proposal_dir = target_query_dir / "proposal_dir"
    query_entitybank_dir = target_query_dir / "query_entitybank"
    query_run_dir = target_query_dir / "query_worldtube_run"
    query_run_entitybank = query_run_dir / "entitybank"
    selection_path = query_run_entitybank / "selected_query_qwen.json"
    tracks_path = target_query_dir / "grounded_sam2" / "grounded_sam2_query_tracks.json"
    query_text = str(query_item["query"])

    _run_wrapped(
        "gs_python",
        PROJECT_ROOT / "scripts" / "build_query_proposal_dir.py",
        [
            "--run-dir", scene_cfg["run_dir"],
            "--dataset-dir", scene_cfg["dataset_dir"],
            "--tracks-path", tracks_path,
            "--output-dir", proposal_dir,
            "--max-track-frames", "16",
            "--proposal-keep-ratio", "0.03",
            "--min-gaussians", "256",
            "--max-gaussians", "4096",
            "--opacity-power", "0.0",
            "--cluster-mode", "support_only",
            "--seed-ratio", "0.05",
            "--expansion-factor", "4.0",
        ],
    )

    _run_wrapped(
        "gs_python",
        PROJECT_ROOT / "scripts" / "export_entitybank.py",
        [
            "--run-dir", scene_cfg["run_dir"],
            "--proposal-dir", proposal_dir,
            "--proposal-strict",
            "--output-dir", query_entitybank_dir,
            "--max-entities", "12",
            "--min-gaussians-per-entity", "32",
        ],
    )

    query_run_dir.mkdir(parents=True, exist_ok=True)
    _ensure_symlink(query_run_dir / "config.yaml", scene_cfg["run_dir"] / "config.yaml")
    _ensure_symlink(query_run_dir / "point_cloud", scene_cfg["run_dir"] / "point_cloud")
    _ensure_symlink(query_run_dir / "test", scene_cfg["run_dir"] / "test")
    _ensure_symlink(query_run_entitybank, query_entitybank_dir)

    for script_name in [
        "export_semantic_slots.py",
        "export_semantic_tracks.py",
        "export_semantic_priors.py",
        "export_native_semantics.py",
    ]:
        _run_wrapped(
            "gs_python",
            PROJECT_ROOT / "scripts" / script_name,
            ["--run-dir", query_run_dir],
        )

    _run_wrapped(
        "gsam2_python",
        PROJECT_ROOT / "scripts" / "export_qwen_semantics.py",
        ["--run-dir", query_run_dir, "--query", query_text, "--max-entities", "12"],
    )

    _run_wrapped(
        "gs_python",
        PROJECT_ROOT / "scripts" / "select_qwen_query_entities.py",
        [
            "--assignments-path", query_run_entitybank / "semantic_assignments_qwen.json",
            "--query", query_text,
            "--query-plan-path", target_query_dir / "query_plan.json",
            "--output-path", selection_path,
        ],
        env={"QUERY_SKIP_QWEN_SELECTION": "1"},
    )

    _run_wrapped(
        "gs_python",
        PROJECT_ROOT / "scripts" / "render_query_video.py",
        [
            "--run-dir", query_run_dir,
            "--dataset-dir", scene_cfg["dataset_dir"],
            "--selection-path", selection_path,
            "--output-dir", target_query_dir / "final_query_render_sourcebg",
            "--background-mode", "source",
        ],
    )

    return {
        "query_slug": str(query_item["query_slug"]),
        "query": query_text,
        "source_validation_path": str(query_item["validation_path"]),
        "source_query_dir": str(source_query_dir),
        "selection_path": str(selection_path),
        "query_root": str(target_query_dir),
    }


def _render_scene(scene_key: str) -> dict:
    scene_cfg = SCENES[scene_key]
    source_eval = _read_json(scene_cfg["source_eval_json"])
    scene_root = REPORT_ROOT / scene_key
    query_root = scene_root / "query_root"
    if scene_root.exists():
        shutil.rmtree(scene_root)
    query_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "scene": scene_key,
        "run_dir": str(scene_cfg["run_dir"]),
        "dataset_dir": str(scene_cfg["dataset_dir"]),
        "annotation_dir": str(scene_cfg["annotation_dir"]),
        "protocol_json": str(scene_cfg["protocol_json"]),
        "source_eval_json": str(scene_cfg["source_eval_json"]),
        "query_root": str(query_root),
        "queries": [],
    }

    for query_item in source_eval["queries"]:
        query_slug = str(query_item["query_slug"])
        target_query_dir = query_root / query_slug
        manifest["queries"].append(
            _build_query_specific_assets(scene_key=scene_key, query_item=query_item, scene_cfg=scene_cfg, target_query_dir=target_query_dir)
        )

    output_json = scene_root / f"{scene_key}_aligned_refresh_eval.json"
    output_md = scene_root / f"{scene_key}_aligned_refresh_eval.md"
    _run_wrapped(
        "gs_python",
        PROJECT_ROOT / "scripts" / "evaluate_public_query_protocol.py",
        [
            "--protocol-json", scene_cfg["protocol_json"],
            "--annotation-dir", scene_cfg["annotation_dir"],
            "--dataset-dir", scene_cfg["dataset_dir"],
            "--query-root", query_root,
            "--output-json", output_json,
            "--output-md", output_md,
        ],
    )

    manifest["output_json"] = str(output_json)
    manifest["output_md"] = str(output_md)
    manifest["summary"] = _read_json(output_json).get("summary", {})
    _write_json(scene_root / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", choices=sorted(SCENES.keys()), action="append", default=None)
    args = parser.parse_args()

    scenes = args.scene or ["americano", "chickchicken", "espresso", "split-cookie"]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    results = []
    for scene_key in scenes:
        results.append(_render_scene(scene_key))

    summary_rows = []
    for item in results:
        scene_summary = item.get("summary", {})
        summary_rows.append(
            {
                "scene": item["scene"],
                "Acc": scene_summary.get("Acc"),
                "vIoU": scene_summary.get("vIoU"),
                "temporal_tIoU": scene_summary.get("temporal_tIoU"),
                "output_json": item.get("output_json"),
            }
        )
    _write_json(REPORT_ROOT / "summary.json", {"results": summary_rows})


if __name__ == "__main__":
    main()
