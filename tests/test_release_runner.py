"""Regression tests for the strict release batch-runner contract."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_query_batch_two_gpu.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_release_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _manifest_row(**overrides: object) -> dict:
    row = {
        "query_id": "scene_q1",
        "query": "the target object",
        "run_dir": "runs/refergaussian/hypernerf/scene",
        "dataset_dir": "data/hypernerf/misc/scene",
        "output_root": "reports/release/query_root",
        "gpu": 0,
    }
    row.update(overrides)
    return row


class ReleaseRunnerTest(unittest.TestCase):
    def test_strict_manifest_accepts_plain_query_row(self) -> None:
        errors = RUNNER.validate_release_manifest(
            [_manifest_row()], profile="public_time_shape_v4_recall"
        )
        self.assertEqual(errors, [])

    def test_strict_manifest_rejects_per_query_environment_override(self) -> None:
        errors = RUNNER.validate_release_manifest(
            [_manifest_row(env={"QUERY_SKIP_QWEN_SELECTION": "1"})],
            profile="public_time_shape_v4_recall",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("env override", errors[0])

    def test_strict_manifest_rejects_mixed_profiles(self) -> None:
        errors = RUNNER.validate_release_manifest(
            [_manifest_row(profile="r4d_shape_v4_recall")],
            profile="public_time_shape_v4_recall",
        )
        self.assertEqual(len(errors), 1)
        self.assertIn("differs from --profile", errors[0])

    def test_strict_runner_clears_inherited_query_tuning_only(self) -> None:
        env = {
            "QUERY_SKIP_QWEN_SELECTION": "1",
            "GS_QUERY_ALLOW_DIRECT_2D_MASKS": "1",
            "GSAM2_REUSE_QUERY_PLAN": "1",
            "QWEN_MODEL_PATH": "/models/qwen",
            "PATH": "/usr/bin",
        }

        RUNNER._clear_inherited_release_tuning(env)

        self.assertEqual(env, {"QWEN_MODEL_PATH": "/models/qwen", "PATH": "/usr/bin"})


if __name__ == "__main__":
    unittest.main()
