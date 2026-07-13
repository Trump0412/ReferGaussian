"""Regression tests for generic multi-frame alpha threshold calibration."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from refergaussian.semantics.query_render import EntityCloud
from refergaussian.semantics.semantic_renderer import PreparedSemanticFrame, render_selection_mask


ROOT = Path(__file__).resolve().parents[1]
LIFTING_PATH = ROOT / "refergaussian" / "semantics" / "mask_supported_lifting.py"


def _load_function(name: str, namespace: dict[str, object]):
    module = ast.parse(LIFTING_PATH.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(LIFTING_PATH), "exec"),
        namespace,
    )
    return namespace[name]


class AlphaThresholdCalibrationContractTest(unittest.TestCase):
    def test_calibration_grid_parses_a_stable_numeric_profile(self) -> None:
        alpha_levels = _load_function("_alpha_calibration_levels", {"os": os, "np": np})
        with patch.dict(
            os.environ,
            {"QUERY_LIFT_ALPHA_CALIBRATION_LEVELS": "0.18:0.015,bad,0.03:0.002,0.18:0.015"},
            clear=False,
        ):
            self.assertEqual(alpha_levels(), [(0.18, 0.015), (0.03, 0.002)])

    def test_geometry_grid_is_bounded_and_deduplicated(self) -> None:
        geometry_levels = _load_function("_alpha_geometry_levels", {"os": os, "np": np})
        with patch.dict(
            os.environ,
            {"QUERY_LIFT_ALPHA_GEOMETRY_LEVELS": "1:18,bad,6:32,1:18,100:999"},
            clear=False,
        ):
            self.assertEqual(geometry_levels(), [(1.0, 18), (6.0, 32), (8.0, 64)])

    def test_calibration_reserves_a_bounded_largest_candidate_probe(self) -> None:
        calibrated_ids: list[tuple[int, ...]] = []

        def env_flag(_name: str, _default: bool) -> bool:
            return True

        def env_int(_name: str, _default: int, minimum: int = 1) -> int:
            return max(3, minimum)

        def utility(row: dict[str, object], target_area_ratio: float) -> float:
            del target_area_ratio
            return float(row["rank"])

        def calibrate(ids, **_kwargs):
            calibrated_ids.append(tuple(np.asarray(ids, dtype=np.int64).tolist()))
            return {
                "alpha_relative_threshold": 0.03,
                "alpha_absolute_threshold": 0.002,
                "alpha_sigma_scale": 3.0,
                "alpha_max_splat_radius": 24,
                "alpha_calibration_method": "multiframe_rendered_overlap",
                "alpha_calibration_frame_count": 8,
                "alpha_calibration_utility": 1.0,
                "alpha_calibration_trials": [],
            }

        def selection_metrics(ids, **kwargs):
            return {
                "gaussian_count": int(np.asarray(ids).size),
                "score": 1.0,
                "rendered_iou_stage1": 0.5,
                "precision": 0.7,
                "recall": 0.6,
                "area_ratio": 0.7,
                "outer_leakage": 0.1,
                "active_frame_coverage": 1.0,
                "area_cv": 0.0,
                "mean_pred_area": 1.0,
                "alpha_relative_threshold": kwargs["relative_threshold"],
                "alpha_absolute_threshold": kwargs["absolute_threshold"],
            }

        calibrate_rows = _load_function(
            "_calibrate_v4_candidate_rows",
            {
                "Any": object,
                "np": np,
                "_env_flag": env_flag,
                "_env_int": env_int,
                "_coverage_v4_selection_utility": utility,
                "_calibrate_alpha_threshold": calibrate,
                "_selection_metrics": selection_metrics,
            },
        )
        rows = [
            {"name": "top", "ids": np.asarray([1]), "rank": 4.0, "gaussian_count": 10},
            {"name": "middle", "ids": np.asarray([2]), "rank": 3.0, "gaussian_count": 20},
            {"name": "small", "ids": np.asarray([3]), "rank": 2.0, "gaussian_count": 30},
            {"name": "largest", "ids": np.asarray([4]), "rank": 1.0, "gaussian_count": 100},
        ]

        calibrate_rows(rows, support=object(), num_gaussians=128, device="cpu", target_area_ratio=0.7)

        self.assertEqual(calibrated_ids, [(1,), (2,), (4,)])
        self.assertEqual(rows[-1]["alpha_relative_threshold"], 0.03)
        self.assertEqual(rows[-1]["alpha_absolute_threshold"], 0.002)
        self.assertEqual(rows[-1]["alpha_sigma_scale"], 3.0)
        self.assertEqual(rows[-1]["alpha_max_splat_radius"], 24)

    def test_renderer_entity_cloud_carries_calibrated_thresholds(self) -> None:
        fields = EntityCloud.__dataclass_fields__
        self.assertIn("alpha_relative_threshold", fields)
        self.assertIn("alpha_absolute_threshold", fields)
        self.assertIn("alpha_sigma_scale", fields)
        self.assertIn("alpha_max_splat_radius", fields)

    def test_splat_radius_is_an_explicit_rendering_control(self) -> None:
        prepared = PreparedSemanticFrame(
            frame_index=0,
            time_value=0.0,
            image_id="frame",
            width=17,
            height=17,
            image_scale=1.0,
            gaussian_ids=torch.tensor([0], dtype=torch.long),
            centers_xy=torch.tensor([[8.0, 8.0]], dtype=torch.float32),
            sigma_px=torch.tensor([4.0], dtype=torch.float32),
            alpha_weight=torch.tensor([1.0], dtype=torch.float32),
            depth=torch.tensor([1.0], dtype=torch.float32),
        )
        weights = torch.tensor([1.0], dtype=torch.float32)
        compact, _ = render_selection_mask(
            prepared,
            weights,
            relative_threshold=0.0,
            absolute_threshold=1.0e-4,
            max_splat_radius=1,
        )
        expanded, _ = render_selection_mask(
            prepared,
            weights,
            relative_threshold=0.0,
            absolute_threshold=1.0e-4,
            max_splat_radius=12,
        )
        self.assertGreater(int(expanded.sum()), int(compact.sum()))

    def test_bootstrap_caps_an_oversized_seed_before_refinement(self) -> None:
        lifting = LIFTING_PATH.read_text(encoding="utf-8")
        self.assertIn("if seed_ids.size > selection_cap", lifting)
        self.assertIn("membership[seed_ids] = 1.0", lifting)

    def test_profile_enables_calibration_without_object_specific_branching(self) -> None:
        profile = (ROOT / "scripts" / "query_eval_profiles.sh").read_text(encoding="utf-8")
        self.assertIn("QUERY_LIFT_ALPHA_THRESHOLD_CALIBRATION", profile)
        self.assertIn("QUERY_LIFT_ALPHA_CALIBRATION_LEVELS", profile)
        self.assertIn("QUERY_LIFT_ALPHA_GEOMETRY_LEVELS", profile)
        self.assertNotIn("americano|espresso|cookie", profile)


if __name__ == "__main__":
    unittest.main()
