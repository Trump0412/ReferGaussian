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
}
TEXT_SUFFIXES = {".cff", ".html", ".json", ".md", ".py", ".sh", ".toml", ".yaml", ".yml"}
REQUIRED_RUNTIME_FILES = (
    "refergaussian/semantics/semantic_renderer.py",
    "refergaussian/semantics/surface_mask_field.py",
    "refergaussian/semantics/mask_supported_lifting.py",
    "scripts/build_joint_query_proposal_dir.py",
    "scripts/export_entitybank.py",
    "scripts/render_query_video.py",
    "scripts/run_query_specific_worldtube_pipeline.sh",
    "scripts/evaluate_ours_benchmark.py",
)
REQUIRED_RUNTIME_TOKENS = {
    "scripts/build_joint_query_proposal_dir.py": "mask_supported_lifting",
    "scripts/export_entitybank.py": "--proposal-supervision-mode",
    "scripts/render_query_video.py": "--eval-profile",
    "scripts/run_query_specific_worldtube_pipeline.sh": "QUERY_ALLOW_FULLSCENE_FALLBACK",
    "scripts/evaluate_ours_benchmark.py": "--query-manifest",
}


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


def main() -> int:
    errors = check_forbidden_text(tracked_files())
    errors.extend(check_readme_scripts())
    errors.extend(check_release_queries())
    errors.extend(check_runtime_contracts())
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1
    print("[ok] public release preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
