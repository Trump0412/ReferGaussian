"""Validate that a query run was trained with the released ReferGaussian path."""

from __future__ import annotations

from pathlib import Path


REQUIRED_CONFIG = {
    "phase": "refergaussian",
    "temporal_warp_type": "refergaussian",
}
REQUIRED_QUERY_ARTIFACTS = ("point_cloud", "test")
_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off"}


def _read_flat_yaml(path: Path) -> dict[str, str]:
    """Read the small flat config written by ``scripts/train.sh``.

    Query validation should not depend on an optional YAML package.  The public
    training wrapper writes one scalar ``key: value`` per line, which is all
    that is needed to identify a released ReferGaussian run.
    """

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip().split("#", 1)[0].strip().strip("\"'")
    return values


def validate_refergaussian_run(run_dir: str | Path) -> list[str]:
    """Return human-readable reasons why ``run_dir`` is not a released run."""

    run_path = Path(run_dir)
    if not run_path.is_dir():
        return [f"run directory is missing: {run_path}"]

    config_path = run_path / "config.yaml"
    if not config_path.is_file():
        return [f"missing ReferGaussian training identity: {config_path}"]

    config = _read_flat_yaml(config_path)
    errors: list[str] = []
    for key, expected in REQUIRED_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            errors.append(f"{key}={actual!r}, expected {expected!r}")

    if config.get("warp_enabled", "").lower() not in _TRUTHY:
        errors.append("warp_enabled is not true")
    return errors


def validate_query_ready_refergaussian_run(run_dir: str | Path) -> list[str]:
    """Validate both ReferGaussian identity and artifacts required by query inference."""

    run_path = Path(run_dir)
    errors = validate_refergaussian_run(run_path)
    if errors:
        return errors

    for artifact in REQUIRED_QUERY_ARTIFACTS:
        path = run_path / artifact
        if not path.is_dir():
            errors.append(f"missing query-render artifact: {path}")
    return errors


def validate_query_ready_baseline_4dgs_run(run_dir: str | Path) -> list[str]:
    """Validate an explicitly requested vanilla 4DGS query backbone."""
    run_path = Path(run_dir)
    if not run_path.is_dir():
        return [f"run directory is missing: {run_path}"]
    config_path = run_path / "config.yaml"
    if not config_path.is_file():
        return [f"missing baseline training identity: {config_path}"]

    config = _read_flat_yaml(config_path)
    errors: list[str] = []
    if config.get("phase") != "baseline":
        errors.append(f"phase={config.get('phase')!r}, expected 'baseline'")
    if config.get("warp_enabled", "").lower() not in _FALSY:
        errors.append("warp_enabled is not false")
    for artifact in REQUIRED_QUERY_ARTIFACTS:
        path = run_path / artifact
        if not path.is_dir():
            errors.append(f"missing query-render artifact: {path}")
    return errors
