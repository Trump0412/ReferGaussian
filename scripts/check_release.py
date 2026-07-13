#!/usr/bin/env python3
"""Validate the public repository before creating a release."""

from __future__ import annotations

import json
import re
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RELEASE_SCENES = {
    "coffee_martini": 7,
    "sear_steak": 7,
    "cut_roasted_beef": 7,
    "cut_lemon": 8,
    "espresso": 8,
    "keyboard": 6,
    "split_cookie": 8,
    "torchchocolate": 7,
}
FORBIDDEN_PATTERNS = {
    "legacy project name": re.compile(r"HyperGaussian", re.IGNORECASE),
    "release placeholder": re.compile(r"XXXX\.XXXXX|Author[0-9]"),
    "internal server path": re.compile(r"/root/autodl-tmp|/home/chenbp|seetacloud\.com"),
    "temporary fix label": re.compile(r"\bhotfix\b", re.IGNORECASE),
    "deprecated transfer artifact": re.compile(r"\btrase\b", re.IGNORECASE),
}
TEXT_SUFFIXES = {".cff", ".html", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
REQUIRED_RUNTIME_FILES = (
    "refergaussian/semantics/semantic_renderer.py",
    "refergaussian/semantics/surface_mask_field.py",
    "refergaussian/semantics/mask_supported_lifting.py",
    "refergaussian/semantics/select_qwen_query_entities.py",
    "scripts/build_joint_query_proposal_dir.py",
    "scripts/export_entitybank.py",
    "scripts/render_query_video.py",
    "scripts/run_query_specific_worldtube_pipeline.sh",
    "scripts/select_qwen_query_entities.py",
    "scripts/build_4dlangsplat_query_protocol.py",
    "scripts/evaluate_ours_benchmark.py",
    "scripts/validate_refergaussian_run.py",
    "scripts/train_baseline.sh",
    "scripts/eval_baseline.sh",
)
REQUIRED_RUNTIME_TOKENS = {
    "scripts/build_joint_query_proposal_dir.py": "mask_supported_lifting",
    "scripts/export_entitybank.py": "--proposal-supervision-mode",
    "scripts/render_query_video.py": "--eval-profile",
    "scripts/run_query_specific_worldtube_pipeline.sh": "mask_supported_lifting",
    "scripts/select_qwen_query_entities.py": "refergaussian.semantics.select_qwen_query_entities import main",
    "scripts/build_4dlangsplat_query_protocol.py": "video_annotations.json",
    "scripts/evaluate_ours_benchmark.py": "--query-manifest",
    "scripts/validate_refergaussian_run.py": "validate_refergaussian_run",
    "scripts/eval_baseline.sh": "external/4DGaussians/render.py",
}
ENGLISH_RUNTIME_FILES = (
    "refergaussian/semantics/qwen_query_planner.py",
    "refergaussian/semantics/select_qwen_query_entities.py",
    "refergaussian/semantics/mask_supported_lifting.py",
)
OBJECT_SPECIFIC_ALIAS_TERMS = (
    "martini glass",
    "cocktail glass",
    "wine glass",
    "drinking glass",
    "round mouse",
    "computer mouse",
    "wireless mouse",
)


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files"], cwd=ROOT, text=True
    )
    return [ROOT / line for line in output.splitlines() if line.strip()]


def check_forbidden_text(paths: list[Path]) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if path == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in FORBIDDEN_PATTERNS.items():
            match = pattern.search(text)
            if match:
                errors.append(f"{path.relative_to(ROOT)}: {label}: {match.group(0)!r}")
    return errors


def check_readme_scripts() -> list[str]:
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    references = sorted(set(re.findall(r"(?<![\w/])(scripts/[A-Za-z0-9_.-]+)", text)))
    return [f"README references missing file: {path}" for path in references if not (ROOT / path).is_file()]


def check_release_queries() -> list[str]:
    path = ROOT / "configs" / "benchmarks" / "r4d_query_text_en.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    counts = Counter(str(query_id).rsplit("_q", 1)[0] for query_id in payload)
    actual = {scene: counts.get(scene, 0) for scene in RELEASE_SCENES}
    errors: list[str] = []
    if actual != RELEASE_SCENES:
        errors.append(f"R4D release scene counts differ: expected={RELEASE_SCENES}, actual={actual}")
    if sum(actual.values()) != 58:
        errors.append(f"R4D release query count is {sum(actual.values())}, expected 58")
    non_english = [
        str(query_id)
        for query_id, query_text in payload.items()
        if re.search(r"[\u4e00-\u9fff]", str(query_text))
    ]
    if non_english:
        errors.append(
            "R4D release query text must be English; found CJK text for: "
            + ", ".join(sorted(non_english)[:10])
        )
    return errors


def check_runtime_contracts() -> list[str]:
    """Catch deleted dependencies and CLI drift in the documented query path."""
    errors: list[str] = []
    for relative_path in REQUIRED_RUNTIME_FILES:
        if not (ROOT / relative_path).is_file():
            errors.append(f"Required query runtime file is missing: {relative_path}")
    for relative_path, token in REQUIRED_RUNTIME_TOKENS.items():
        path = ROOT / relative_path
        if path.is_file() and token not in path.read_text(encoding="utf-8", errors="replace"):
            errors.append(f"Required query runtime contract is missing: {relative_path} -> {token}")
    return errors


def check_runtime_release_guards() -> list[str]:
    """Keep the documented path English-only, generic, and free of silent recovery."""
    errors: list[str] = []
    for relative_path in ENGLISH_RUNTIME_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            continue
        if re.search(r"[\u4e00-\u9fff]", path.read_text(encoding="utf-8", errors="replace")):
            errors.append(f"Runtime source must remain English-only: {relative_path}")

    planner_text = (ROOT / "refergaussian/semantics/qwen_query_planner.py").read_text(
        encoding="utf-8", errors="replace"
    )
    selector_text = (ROOT / "refergaussian/semantics/select_qwen_query_entities.py").read_text(
        encoding="utf-8", errors="replace"
    )
    for term in OBJECT_SPECIFIC_ALIAS_TERMS:
        if term in planner_text or term in selector_text:
            errors.append(f"Runtime must not contain object-specific alias rule: {term!r}")

    lifting_text = (ROOT / "refergaussian/semantics/mask_supported_lifting.py").read_text(
        encoding="utf-8", errors="replace"
    )
    if "_is_thin_object_phrase" in lifting_text:
        errors.append("Lifting must use geometry, not object-name thinness rules")

    profile_text = (ROOT / "scripts/query_eval_profiles.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    if "QUERY_LIFT_ALLOW_SPARSE_FALLBACK=1" in profile_text:
        errors.append("Published profiles must not enable sparse-candidate fallback")

    pipeline_text = (ROOT / "scripts/run_query_specific_worldtube_pipeline.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    if "QUERY_ALLOW_FULLSCENE_FALLBACK" in pipeline_text:
        errors.append("Published query pipeline must not contain full-scene fallback")
    if "QUERY_RETRY_RELAXED_GSAM2:-1" in pipeline_text:
        errors.append("Published query pipeline must not enable relaxed GSAM2 retry by default")
    if "require_refergaussian_run" not in pipeline_text:
        errors.append("Published query pipeline must validate the ReferGaussian training identity")

    grounded_sam_text = (ROOT / "scripts/run_query_guided_grounded_sam2.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    if "GSAM2_QUERY_PLAN_STRICT_FALLBACK:-1" in grounded_sam_text:
        errors.append("Published Stage-1 path must not enable non-strict planner fallback by default")

    baseline_eval_text = (ROOT / "scripts/eval_baseline.sh").read_text(
        encoding="utf-8", errors="replace"
    )
    if "--warp_enabled" in baseline_eval_text or "temporal_warp_type" in baseline_eval_text:
        errors.append("Baseline evaluation must not enable ReferGaussian temporal warp")

    query_render_text = (ROOT / "refergaussian/semantics/query_render.py").read_text(
        encoding="utf-8", errors="replace"
    )
    if "ours_fallback_source" in query_render_text:
        errors.append("Query rendering must not substitute source RGB for model renders")
    if '"glass", "cup", "bottle"' in query_render_text:
        errors.append("Query rendering must not contain object-specific intent rules")
    if "cloud_only_fallback" in query_render_text:
        errors.append("Query rendering must not silently replace an empty Gaussian projection")
    return errors


def main() -> int:
    errors = check_forbidden_text(tracked_files())
    errors.extend(check_readme_scripts())
    errors.extend(check_release_queries())
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
