#!/usr/bin/env python3
"""Validate the public repository before creating a release."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DENSE_RELEASE_SCENES = {
    "americano": 10,
    "coffee_martini": 7,
    "cook-spinach": 7,
    "sear_steak": 7,
    "cut_roasted_beef": 7,
    "cut_lemon": 8,
    "espresso": 8,
    "flame_salmon": 7,
    "flame_steak": 7,
    "keyboard": 6,
    "split_cookie": 8,
    "torchchocolate": 7,
}
TEXT_SUFFIXES = {".cff", ".html", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
FORBIDDEN_PATTERNS = {
    "legacy project name": re.compile(r"HyperGaussian", re.IGNORECASE),
    "release placeholder": re.compile(r"XXXX\.XXXXX|Author[0-9]"),
    "internal server path": re.compile(r"/root/autodl-tmp|/home/chenbp|seetacloud\.com"),
    "temporary fix label": re.compile(r"\bhotfix\b", re.IGNORECASE),
    "deprecated transfer artifact": re.compile(r"\btrase\b", re.IGNORECASE),
}
FORBIDDEN_RELEASE_PATHS = (
    ".gitattributes",
    "configs/benchmarks/reconstruction_release_v1.json",
    "patches",
    "refergaussian/temporal",
    "refergaussian/semantics/joint_embedding_cluster.py",
    "scripts/build_joint_query_proposal_dir.py",
    "scripts/build_query_proposal_dir.py",
    "scripts/collect_metrics.py",
    "scripts/eval.sh",
    "scripts/eval_baseline.sh",
    "scripts/fullframe_metrics.py",
    "scripts/plot_time_warp.py",
    "scripts/quick_subset_metrics.py",
    "scripts/run_gsam_ablation.py",
    "scripts/run_matched_reconstruction.py",
    "scripts/run_query_batch_two_gpu.py",
    "scripts/train.sh",
    "scripts/train_baseline.sh",
    "tests/test_matched_reconstruction_runner.py",
    "tests/test_metrics_cache_contract.py",
    "tests/test_temporal_warp_schedule_contract.py",
    "docs/assets/Fig2.png",
    "docs/assets/Fig3.png",
)
FORBIDDEN_ARTIFACT_SUFFIXES = {
    ".ckpt", ".pth", ".pt", ".safetensors", ".ply", ".npz", ".npy",
    ".mp4", ".tar", ".gz", ".zip",
}
FORBIDDEN_ARTIFACT_DIRS = {
    "checkpoints", "results", "reports", "runs", "outputs", "weights", "models",
    "reproducibility", "experiments", "server_sync", "backups", "tmp",
}
REQUIRED_RUNTIME_FILES = (
    "configs/benchmarks/release_protocols.json",
    "configs/benchmarks/r4d_query_text_en.json",
    "docs/METRICS.md",
    "docs/assets/framework.png",
    "refergaussian/run_identity.py",
    "refergaussian/semantics/semantic_renderer.py",
    "refergaussian/semantics/surface_mask_field.py",
    "refergaussian/semantics/mask_supported_lifting.py",
    "refergaussian/semantics/select_qwen_query_entities.py",
    "refergaussian/semantics/grounded_sam2_backend.py",
    "scripts/bootstrap_external.sh",
    "scripts/setup.sh",
    "scripts/setup_4dgs_env.sh",
    "scripts/download_models.sh",
    "scripts/download_4dlangsplat_annotations.sh",
    "scripts/download_r4d_bench_qa.sh",
    "scripts/prepare_hypernerf.sh",
    "scripts/train_4dgs.py",
    "scripts/run_benchmark.py",
    "scripts/build_mask_supported_proposal_dir.py",
    "scripts/export_entitybank.py",
    "scripts/render_query_video.py",
    "scripts/rerender_query_outputs.py",
    "scripts/run_query_specific_worldtube_pipeline.sh",
    "scripts/check_query_runtime.py",
    "scripts/check_grounded_sam2_import.py",
    "scripts/select_qwen_query_entities.py",
    "scripts/build_4dlangsplat_query_protocol.py",
    "scripts/download_hf_snapshot.py",
    "scripts/aggregate_public_query_evaluations.py",
    "scripts/evaluate_public_query_protocol.py",
    "scripts/evaluate_ours_benchmark.py",
    "scripts/validate_4dgs_run.py",
)
REQUIRED_RUNTIME_TOKENS = {
    "scripts/bootstrap_external.sh": "Pinned, unmodified external dependencies",
    "scripts/setup.sh": "scripts/setup_grounded_sam2.sh",
    "scripts/setup_4dgs_env.sh": "kornia==0.7.3",
    "scripts/download_models.sh": "Qwen/Qwen3-VL-8B-Instruct",
    "scripts/download_4dlangsplat_annotations.sh": "d127a280446206fc97887a304de790a1fe6af5ff",
    "scripts/download_r4d_bench_qa.sh": "gsam2_python",
    "scripts/prepare_hypernerf.sh": "interp_chickchicken.zip",
    "scripts/train_4dgs.py": "validate_query_ready_4dgs_run",
    "scripts/run_benchmark.py": "release_r4d_dense89_renderer_consistent",
    "scripts/build_mask_supported_proposal_dir.py": "build_mask_supported_lifting_proposal_dir",
    "scripts/export_entitybank.py": "--proposal-supervision-mode",
    "scripts/render_query_video.py": "--eval-profile",
    "scripts/rerender_query_outputs.py": "--require-complete",
    "scripts/run_query_specific_worldtube_pipeline.sh": "require_4dgs_run",
    "scripts/select_qwen_query_entities.py": "refergaussian.semantics.select_qwen_query_entities import main",
    "scripts/build_4dlangsplat_query_protocol.py": "video_annotations.json",
    "scripts/download_hf_snapshot.py": "resolved_revision",
    "scripts/aggregate_public_query_evaluations.py": "--require-complete",
    "scripts/evaluate_public_query_protocol.py": "spatial_coverage_complete",
    "scripts/evaluate_ours_benchmark.py": "spatial_coverage_complete",
    "scripts/write_empty_query_selection.py": '"selection_status": "semantic_empty"',
    "scripts/run_query_batch.py": "batch_provenance.json",
    "scripts/validate_4dgs_run.py": "validate_query_ready_4dgs_run",
    "scripts/run_query_guided_grounded_sam2.sh": "QUERY_OUTPUT_ROOT_OVERRIDE",
    "scripts/check_query_runtime.py": "--require-qwen",
    "scripts/check_grounded_sam2_import.py": "sam2._C",
    "refergaussian/semantics/grounded_sam2_backend.py": "local_files_only=local_files_only",
    "refergaussian/semantics/query_render.py": "selection_resolution_complete",
    "refergaussian/semantics/select_qwen_query_entities.py": "_finalize_selection_status",
}
ENGLISH_RUNTIME_FILES = (
    "refergaussian/semantics/qwen_query_planner.py",
    "refergaussian/semantics/qwen_assignment.py",
    "refergaussian/semantics/select_qwen_query_entities.py",
    "refergaussian/semantics/mask_supported_lifting.py",
)
OBJECT_SPECIFIC_ALIAS_TERMS = (
    "martini glass", "cocktail glass", "wine glass", "drinking glass",
    "round mouse", "computer mouse", "wireless mouse",
)


def tracked_files() -> list[Path]:
    """Return release files from a Git checkout or source archive."""
    try:
        output = subprocess.check_output(
            ["git", "ls-files"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        excluded = {".git", "__pycache__", "data", "external", "models", "runs"}
        return [
            path for path in ROOT.rglob("*")
            if path.is_file() and not any(part in excluded for part in path.relative_to(ROOT).parts)
        ]
    return [ROOT / line for line in output.splitlines() if line.strip()]


def check_forbidden_text(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path == Path(__file__).resolve() or path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            match = pattern.search(text)
            if match:
                errors.append(f"{path.relative_to(ROOT)}: {label}: {match.group(0)!r}")
    return errors


def check_forbidden_paths(paths: list[Path]) -> list[str]:
    tracked = [path.relative_to(ROOT).as_posix() for path in paths]
    errors: list[str] = []
    for forbidden in FORBIDDEN_RELEASE_PATHS:
        if any(path == forbidden or path.startswith(f"{forbidden}/") for path in tracked):
            errors.append(f"Non-mainline release path must not be shipped: {forbidden}")
    return errors


def check_no_experiment_artifacts(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_ARTIFACT_SUFFIXES:
            errors.append(f"Experiment/checkpoint artifact must not be tracked: {relative}")
        if any(part.lower() in FORBIDDEN_ARTIFACT_DIRS for part in relative.parts[:-1]):
            errors.append(f"Experiment output directory must not be tracked: {relative}")
    return errors


def check_public_metadata() -> list[str]:
    errors: list[str] = []
    expected_title = "R4DGS: Referring Segmentation in 4D Gaussian Splatting"
    expected_doi = "10.1145/3767308.3836021"
    expected_repository = "https://github.com/Trump0412/R4DGS"
    expected_page = "https://trump0412.github.io/R4DGS/"
    for relative in ("README.md", "CITATION.cff", "docs/index.html"):
        text = (ROOT / relative).read_text(encoding="utf-8")
        if expected_title not in text:
            errors.append(f"Camera-ready title is missing from {relative}")
        if expected_doi not in text:
            errors.append(f"Camera-ready DOI is missing from {relative}")
        if expected_repository not in text:
            errors.append(f"Renamed repository URL is missing from {relative}")
        if "Chu Liuxin" in text:
            errors.append(f"Camera-ready author order must be 'Liuxin Chu' in {relative}")
    for relative in ("README.md", "CITATION.cff"):
        if expected_page not in (ROOT / relative).read_text(encoding="utf-8"):
            errors.append(f"Renamed GitHub Pages URL is missing from {relative}")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    page = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    if "Accepted at" in page:
        errors.append("Project page must not display an 'Accepted at' label")
    for stale in ("266 sentence", "3-scene / 7-query", "91.62", "66.48", "method_overview.svg"):
        if stale in readme or stale in page:
            errors.append(f"Public README/page contains stale camera-ready content: {stale}")
    for required in ("76.5", "34.4", "93.01", "62.95", "99.43", "68.13"):
        if required not in readme + page:
            errors.append(f"Paper-reported metric is missing from README/page: {required}")
    for label, text in (("README", readme), ("project page", page), ("CITATION", citation)):
        for forbidden in ("R4D-Bench reconstruction", "PSNR", "SSIM", "LPIPS", "Dynamic Scene Reconstruction"):
            if forbidden in text:
                errors.append(f"{label} still presents reconstruction content: {forbidden}")
    return errors


def check_paper_page_assets() -> list[str]:
    expected = {
        "teaser2.png": "c3320760387377f993aa201e5411195518c39e358213025ba21bea2982eb1f8c",
        "Fig4.png": "7ad1bc37b53cf6f0ca3a408f7f9597fcec7f0121e9cb2c2bdcccb9b90e511c96",
        "framework.png": "5233022f304436211899646b1ca59f36becad59308bfb1244fcc34ae28bce7ed",
        "dataset.png": "26b0cf192c03d884ee899c775261f4baeacfc1516d1230b78114d7fd814f81f7",
        "vl_model_radar_6axis.png": "131528ae7b45d929190cbb5b88c0ae2f9f888a88336694e5a2fda99d35a0e3d6",
        "runtime.png": "3b58f94b4302dadaaf06b2611b5259b46f4014fc20c1a43bb38e729c5604128c",
        "Fig5.png": "b519203708824fd8e0822977968ba2a82136e1829a3f36a04b59148260c3c62c",
    }
    errors: list[str] = []
    for name, expected_hash in expected.items():
        path = ROOT / "docs" / "assets" / name
        if not path.is_file():
            errors.append(f"Paper figure is missing from project page assets: {name}")
            continue
        actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual_hash != expected_hash:
            errors.append(f"Project page asset differs from the camera-ready figure: {name}")

    page = (ROOT / "docs/index.html").read_text(encoding="utf-8")
    local_images = set(re.findall(r'<img[^>]+src="assets/([^"?#]+)', page))
    unexpected = sorted(name for name in local_images if name.lower().endswith(".png") and name not in expected)
    if unexpected:
        errors.append("Project page includes figures absent from the camera-ready paper: " + ", ".join(unexpected))
    return errors


def check_readme_scripts() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    references = sorted(set(re.findall(r"(?<![\w/])(scripts/[A-Za-z0-9_.-]+)", text)))
    return [f"README references missing file: {path}" for path in references if not (ROOT / path).is_file()]


def check_release_queries() -> list[str]:
    path = ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(str(query_id).rsplit("_q", 1)[0] for query_id in payload)
    actual = dict(sorted(counts.items()))
    expected = dict(sorted(DENSE_RELEASE_SCENES.items()))
    errors: list[str] = []
    if actual != expected:
        errors.append(f"R4D dense release scene counts differ: expected={expected}, actual={actual}")
    if sum(actual.values()) != 89 or len(actual) != 12:
        errors.append(f"R4D dense release must contain 89 queries over 12 scenes; got {sum(actual.values())}/{len(actual)}")
    non_english = [
        str(query_id) for query_id, query_text in payload.items()
        if re.search(r"[\u4e00-\u9fff]", str(query_text))
    ]
    if non_english:
        errors.append("R4D release queries must be English: " + ", ".join(sorted(non_english)[:10]))
    return errors


def check_protocol_registry() -> list[str]:
    path = ROOT / "configs" / "benchmarks" / "release_protocols.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    protocols = payload.get("protocols", {})
    expected = {
        "paper_r4d_reported": (12, 266),
        "release_r4d_dense89": (12, 89),
        "release_r4d_dense89_renderer_consistent": (12, 89),
        "legacy_r4d_filtered58": (8, 58),
        "paper_public3": (3, 7),
        "release_public4_extension": (4, 9),
    }
    errors: list[str] = []
    for protocol_id, (scene_count, query_count) in expected.items():
        row = protocols.get(protocol_id)
        if not isinstance(row, dict):
            errors.append(f"Protocol registry entry is missing: {protocol_id}")
            continue
        if int(row.get("scene_count", -1)) != scene_count or int(row.get("query_count", -1)) != query_count:
            errors.append(f"Protocol {protocol_id} must be {scene_count} scenes/{query_count} queries")
    query_map = ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
    actual_hash = hashlib.sha256(query_map.read_bytes()).hexdigest()
    for protocol_id in ("release_r4d_dense89", "release_r4d_dense89_renderer_consistent"):
        if protocols.get(protocol_id, {}).get("english_query_map_sha256") != actual_hash:
            errors.append(f"Protocol {protocol_id} English query-map hash does not match")
    return errors


def check_runtime_contracts() -> list[str]:
    errors: list[str] = []
    for relative in REQUIRED_RUNTIME_FILES:
        if not (ROOT / relative).is_file():
            errors.append(f"Required query runtime file is missing: {relative}")
    for relative, token in REQUIRED_RUNTIME_TOKENS.items():
        path = ROOT / relative
        if path.is_file() and token not in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(f"Required query runtime contract is missing: {relative} -> {token}")
    return errors


def check_runtime_release_guards() -> list[str]:
    errors: list[str] = []
    for relative in ENGLISH_RUNTIME_FILES:
        path = ROOT / relative
        if re.search(r"[\u4e00-\u9fff]", path.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"Runtime source must remain English-only: {relative}")

    planner = (ROOT / "refergaussian/semantics/qwen_query_planner.py").read_text(encoding="utf-8")
    selector = (ROOT / "refergaussian/semantics/select_qwen_query_entities.py").read_text(encoding="utf-8")
    for term in OBJECT_SPECIFIC_ALIAS_TERMS:
        if term in planner or term in selector:
            errors.append(f"Runtime must not contain object-specific alias rule: {term!r}")

    lifting = (ROOT / "refergaussian/semantics/mask_supported_lifting.py").read_text(encoding="utf-8")
    if "_is_thin_object_phrase" in lifting:
        errors.append("Lifting must use geometry, not object-name thinness rules")

    profiles = (ROOT / "scripts/query_eval_profiles.sh").read_text(encoding="utf-8")
    for forbidden in (
        "QUERY_LIFT_ALLOW_SPARSE_FALLBACK=1",
        "QUERY_SKIP_QWEN_EXPORT=1",
        "QUERY_SKIP_QWEN_SELECTION=1",
        "QUERY_LIFT_BOOTSTRAP_PROXY_EVIDENCE_ONLY=1",
        "QUERY_LIFT_BOOTSTRAP_FINAL_RENDER_METRICS=0",
        "QUERY_STATIC_SELECT_WITHOUT_QWEN=1",
        "QUERY_ALLOW_BASELINE_4DGS_RUN",
    ):
        if forbidden in profiles:
            errors.append(f"Published profiles contain a forbidden bypass: {forbidden}")

    pipeline = (ROOT / "scripts/run_query_specific_worldtube_pipeline.sh").read_text(encoding="utf-8")
    if "QUERY_ALLOW_FULLSCENE_FALLBACK" in pipeline:
        errors.append("Published query pipeline must not contain full-scene fallback")
    if "QUERY_RETRY_RELAXED_GSAM2:-1" in pipeline:
        errors.append("Published query pipeline must not enable relaxed GSAM2 retry by default")
    for token in ("require_4dgs_run", "check_query_runtime.py", "check_grounded_sam2_import.py"):
        if token not in pipeline:
            errors.append(f"Published query pipeline is missing guard: {token}")

    query_render = (ROOT / "refergaussian/semantics/query_render.py").read_text(encoding="utf-8")
    for forbidden in ("ours_fallback_source", '"glass", "cup", "bottle"', "cloud_only_fallback"):
        if forbidden in query_render:
            errors.append(f"Query renderer contains a forbidden fallback/rule: {forbidden}")

    setup_gsam = (ROOT / "scripts/setup_grounded_sam2.sh").read_text(encoding="utf-8")
    for revision in (
        "e6a8e8809b8f1bfa2238b6d080f3d05cc76bd251",
        "12bdfa3120f3e7ec7b434d90674b3396eccf88eb",
        "transformers==5.3.0",
        "huggingface_hub==1.7.2",
    ):
        if revision not in setup_gsam:
            errors.append(f"Grounded-SAM2 setup must pin {revision}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for token in (
        "bash scripts/setup.sh",
        "bash scripts/download_models.sh",
        "scripts/train_4dgs.py",
        "scripts/run_benchmark.py 4dlangsplat",
        "scripts/run_benchmark.py r4d-bench",
    ):
        if token not in readme:
            errors.append(f"README is missing release contract: {token}")
    if re.search(r"time[-_ ]agnostic", readme, re.IGNORECASE):
        errors.append("README must expose only the two released dynamic-query benchmarks")
    return errors


def main() -> int:
    paths = tracked_files()
    errors = check_forbidden_text(paths)
    errors.extend(check_forbidden_paths(paths))
    errors.extend(check_no_experiment_artifacts(paths))
    errors.extend(check_public_metadata())
    errors.extend(check_paper_page_assets())
    errors.extend(check_readme_scripts())
    errors.extend(check_release_queries())
    errors.extend(check_protocol_registry())
    errors.extend(check_runtime_contracts())
    errors.extend(check_runtime_release_guards())
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1
    print("[ok] public release preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
