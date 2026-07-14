"""Regression coverage for benchmark-level entity-lifecycle output semantics."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SELECTOR_PATH = ROOT / "refergaussian" / "semantics" / "select_qwen_query_entities.py"


def _load_plan_window_helper():
    module = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
    names = {
        "_env_flag",
        "_uses_entity_lifecycle_temporal_output",
        "_query_plan_window_test_range",
    }
    nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"Any": Any, "os": os, "np": np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SELECTOR_PATH), "exec"), namespace)
    return namespace["_query_plan_window_test_range"]


class EntityLifecycleTemporalContractTest(unittest.TestCase):
    def test_lifecycle_mode_ignores_planner_state_window(self) -> None:
        plan_window = _load_plan_window_helper()
        plan = {
            "refined_temporal_window": {
                "start_frame_index": 10,
                "end_frame_index": 20,
            }
        }
        with patch.dict(
            "os.environ",
            {"QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT": "1"},
            clear=False,
        ):
            self.assertEqual(plan_window(plan, np.asarray([0.0, 0.5, 1.0], dtype=np.float32)), (None, None))

    def test_profile_enables_lifecycle_mode_only_for_r4d_geometry_profile(self) -> None:
        profile = (ROOT / "scripts" / "query_eval_profiles.sh").read_text(encoding="utf-8")
        r4d_block = profile.split("r4d_renderer_geometry_v7)", 1)[1].split(";;", 1)[0]
        public_block = profile.split("public_time_boundary_gated_v5|boundary_gated_gaussian_v5)", 1)[1].split(";;", 1)[0]
        self.assertIn("QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT=1", r4d_block)
        self.assertNotIn("QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT=1", public_block)

    def test_selection_records_auditable_lifecycle_source(self) -> None:
        source = SELECTOR_PATH.read_text(encoding="utf-8")
        self.assertIn("qwen_plan_entity_lifecycle", source)
        self.assertIn("synchronized_entity_lifecycle", source)


if __name__ == "__main__":
    unittest.main()
