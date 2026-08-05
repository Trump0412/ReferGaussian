"""Validation helpers for frozen upstream 4DGS inputs."""

from __future__ import annotations

from pathlib import Path


def validate_query_ready_4dgs_run(run_dir: str | Path) -> list[str]:
    """Return reasons why an upstream 4DGS run is not query-ready.

    ReferGaussian does not train the scene representation.  Query inference
    consumes the standard files produced by the pinned 4DGaussians project:
    its serialized arguments, Gaussian checkpoint, and test-camera renders.
    """

    run_path = Path(run_dir)
    if not run_path.is_dir():
        return [f"run directory is missing: {run_path}"]

    errors: list[str] = []
    cfg_args = run_path / "cfg_args"
    if not cfg_args.is_file():
        errors.append(f"missing upstream 4DGS arguments: {cfg_args}")

    point_cloud_root = run_path / "point_cloud"
    point_clouds = sorted(point_cloud_root.glob("iteration_*/point_cloud.ply"))
    if not point_clouds:
        errors.append(
            "missing upstream 4DGS checkpoint: "
            f"{point_cloud_root / 'iteration_*/point_cloud.ply'}"
        )

    test_root = run_path / "test"
    renders = sorted(test_root.glob("ours_*/renders/*"))
    if not any(path.is_file() for path in renders):
        errors.append(
            "missing upstream 4DGS test renders: "
            f"{test_root / 'ours_*/renders/*'}"
        )
    return errors
