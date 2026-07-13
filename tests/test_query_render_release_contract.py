"""Release contracts for strict, scene-agnostic query rendering."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from refergaussian.semantics.query_render import (
    _apply_render_profile_env_defaults,
    _find_render_dir,
    _opacity_logits_from_probabilities,
    _query_intent_mode,
)


class QueryRenderReleaseContractTest(unittest.TestCase):
    def test_missing_render_is_not_replaced_by_source_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            with self.assertRaisesRegex(FileNotFoundError, "never substituted"):
                _find_render_dir(run_dir)

    def test_intent_mode_does_not_special_case_object_categories(self) -> None:
        self.assertEqual(
            _query_intent_mode("the glass cup with liquid above the midpoint"),
            "generic",
        )

    def test_generic_structure_still_handles_component_queries(self) -> None:
        self.assertEqual(_query_intent_mode("the package broken into pieces"), "multi_component")
        self.assertEqual(_query_intent_mode("the whole package"), "single_component")

    def test_v4_profile_uses_alpha_projection_matching_lifting_geometry(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _apply_render_profile_env_defaults("public_time_shape_v4_recall")

            self.assertEqual(os.environ["GS_QUERY_CLOUD_RENDER_MODE"], "gaussian_alpha")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_GATE_THRESHOLD"], "0.01")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_REL_THRESHOLD"], "0.18")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_SIGMA_SCALE"], "1.0")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_MAX_SPLAT_RADIUS"], "18")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_REQUIRE_SUCCESS"], "1")
            self.assertEqual(os.environ["GS_QUERY_ALPHA_REQUIRE_OPACITY"], "1")

    def test_probability_opacity_round_trips_to_logits(self) -> None:
        probabilities = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
        logits = _opacity_logits_from_probabilities(probabilities)

        self.assertIsNotNone(logits)
        restored = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(restored, probabilities, rtol=1.0e-5, atol=1.0e-5)


if __name__ == "__main__":
    unittest.main()
