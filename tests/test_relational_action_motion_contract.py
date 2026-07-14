"""Regression coverage for generic relative-motion action verification."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "refergaussian" / "semantics" / "select_qwen_query_entities.py"


def _load_relative_motion_score():
    module = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_relative_motion_score"
    )
    namespace = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SELECTOR_PATH), "exec"), namespace)
    return namespace["_relative_motion_score"]


class RelationalActionMotionContractTest(unittest.TestCase):
    def test_static_relative_geometry_has_negligible_motion(self) -> None:
        motion_score = _load_relative_motion_score()
        time = np.arange(12, dtype=np.float32)
        first = np.stack([time * 0.0, time * 0.0, time * 0.0], axis=1)
        second = np.stack([np.full_like(time, 2.0), time * 0.0, time * 0.0], axis=1)

        metrics = motion_score(first, 1.0, second, 1.0)

        self.assertIsNotNone(metrics)
        self.assertLess(float(metrics["normalized_relative_extent"]), 1.0e-6)

    def test_relative_translation_is_detected_independently_of_shared_motion(self) -> None:
        motion_score = _load_relative_motion_score()
        time = np.arange(12, dtype=np.float32)
        shared_camera_like_motion = np.stack([time * 2.0, time * -1.0, time * 0.0], axis=1)
        first = shared_camera_like_motion
        second = shared_camera_like_motion + np.stack([time * 0.05, time * 0.0, time * 0.0], axis=1)

        metrics = motion_score(first, 1.0, second, 1.0)

        self.assertIsNotNone(metrics)
        self.assertGreater(float(metrics["normalized_relative_extent"]), 0.1)

    def test_profile_enables_motion_gate_without_scene_branches(self) -> None:
        profile = (ROOT / "scripts" / "query_eval_profiles.sh").read_text(encoding="utf-8")
        r4d_block = profile.split("r4d_renderer_geometry_v7)", 1)[1].split(";;", 1)[0]
        public_block = profile.split("public_time_boundary_gated_v5|boundary_gated_gaussian_v5)", 1)[1].split(";;", 1)[0]
        self.assertIn("QUERY_RELATIONAL_ACTION_MOTION_GATE=1", r4d_block)
        self.assertNotIn("QUERY_RELATIONAL_ACTION_MOTION_GATE=1", public_block)
        self.assertNotIn("torchchocolate|blowtorch|chocolate", profile)


if __name__ == "__main__":
    unittest.main()
