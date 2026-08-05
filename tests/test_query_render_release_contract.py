"""Release contracts for strict, scene-agnostic query rendering."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from refergaussian.semantics.query_render import (
    QueryTrack,
    _apply_render_profile_env_defaults,
    _explicit_camera_image_ids,
    _frame_mask_at_render_times,
    _find_render_dir,
    _fuse_query_and_cloud_masks,
    _fusion_options_for_profile,
    _opacity_logits_from_probabilities,
    _query_intent_mode,
    _query_track_match_for_time,
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

    def test_v5_profile_enforces_synchronized_boundary_gated_gaussians(self) -> None:
        with patch.dict(os.environ, {"GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY": "1"}, clear=True):
            _apply_render_profile_env_defaults("public_time_boundary_gated_v5")

            self.assertEqual(os.environ["GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY"], "0")
            self.assertEqual(os.environ["GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY"], "1")
            self.assertEqual(os.environ["GS_QUERY_ALLOW_DIRECT_2D_MASKS"], "0")

            options = _fusion_options_for_profile("public_time_boundary_gated_v5")
            query_mask = np.zeros((48, 48), dtype=bool)
            query_mask[14:34, 14:34] = True
            cloud_mask = np.zeros((48, 48), dtype=bool)
            cloud_mask[8:40, 8:40] = True
            fused, _, _, source = _fuse_query_and_cloud_masks(
                query_mask,
                cloud_mask,
                fusion_options=options,
            )

            self.assertIsNotNone(fused)
            self.assertIn("gaussian", str(source))
            self.assertFalse(str(source).startswith("query_track"))
            self.assertFalse(np.any(np.asarray(fused, dtype=bool) & ~cloud_mask))

            ring_cloud = cloud_mask.copy()
            ring_cloud[22:26, 22:26] = False
            fused_ring, _, _, _ = _fuse_query_and_cloud_masks(
                query_mask,
                ring_cloud,
                fusion_options=options,
            )
            self.assertIsNotNone(fused_ring)
            self.assertFalse(np.any(np.asarray(fused_ring, dtype=bool) & ~ring_cloud))

    def test_v5_rejects_stale_stage1_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            mask_path = Path(temp_dir) / "mask.png"
            Image.fromarray(np.full((16, 16), 255, dtype=np.uint8)).save(mask_path)
            track = QueryTrack(
                phrase="target",
                frames=[{"frame_index": 0, "time_value": 0.0, "mask_path": str(mask_path)}],
            )
            with patch.dict(os.environ, {"GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY": "0"}, clear=True):
                mask, meta = _query_track_match_for_time(track, time_value=0.8, tolerance=0.01, strict=True)

            self.assertIsNone(mask)
            self.assertEqual(meta["stage1_match_mode"], "strict_rejected_stale")

    def test_v5_numeric_profile_preserves_the_formal_gaussian_contract(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            _apply_render_profile_env_defaults("public_time_boundary_gated_v5_numeric")

            self.assertEqual(os.environ["GS_QUERY_ALLOW_STALE_STAGE1_BOUNDARY"], "0")
            self.assertEqual(os.environ["GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY"], "1")
            self.assertEqual(os.environ["GS_QUERY_ALLOW_DIRECT_2D_MASKS"], "0")
            self.assertEqual(os.environ["GS_QUERY_CLOUD_RENDER_MODE"], "gaussian_alpha")

    def test_v5_numeric_profile_only_disables_qualitative_exports(self) -> None:
        profiles = (Path(__file__).resolve().parents[1] / "scripts" / "query_eval_profiles.sh").read_text(
            encoding="utf-8"
        )
        block = profiles.split("public_time_boundary_gated_v5_numeric|boundary_gated_gaussian_v5_numeric)", 1)[1].split(
            ";;", 1
        )[0]
        self.assertIn("apply_query_eval_profile public_time_boundary_gated_v5", block)
        self.assertIn("QUERY_SKIP_ENTITY_LIBRARY=1", block)
        self.assertIn("QUERY_SKIP_VIDEO_EXPORT=1", block)
        self.assertIn("QUERY_SKIP_OVERLAY_FRAME_EXPORT=1", block)
        self.assertIn("QUERY_SKIP_DIAGNOSTIC_EXPORT=1", block)
        self.assertIn("GS_QUERY_EXPORT_ENTITY_LIFECYCLE=0", block)

        pipeline = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_query_specific_worldtube_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('${QUERY_SKIP_DIAGNOSTIC_EXPORT:-0}" != "1"', pipeline)

    def test_time_agnostic_profile_uses_exact_numeric_render_contract(self) -> None:
        profiles = (Path(__file__).resolve().parents[1] / "scripts" / "query_eval_profiles.sh").read_text(
            encoding="utf-8"
        )
        block = profiles.split("public_time_agnostic_v1)", 1)[1].split(";;", 1)[0]
        self.assertIn("QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT=1", block)
        self.assertIn("QUERY_FAST_VALIDATION_ONLY=1", block)
        self.assertIn("QUERY_RENDER_ACTIVE_MASKS_ONLY=0", block)
        self.assertIn("QUERY_RENDER_REQUESTED_CAMERAS_ONLY=1", block)
        self.assertIn("GSAM2_ENABLE_INSTANCE_CANDIDATES=1", block)
        self.assertIn("GSAM2_INSTANCE_MAX_CANDIDATES=3", block)
        self.assertNotIn("QUERY_AUTO_SKIP_QWEN_FOR_DECLARED_MULTIHYPOTHESIS=1", block)
        self.assertIn("QUERY_USE_RENDERER_GEOMETRY=1", block)
        self.assertIn("QUERY_REQUIRE_RENDERER_GEOMETRY=1", block)
        self.assertIn("QUERY_LIFT_MASK_AWARE_PREFILTER=1", block)

        pipeline = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "run_query_specific_worldtube_pipeline.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("QUERY_RENDER_IMAGE_IDS_JSON", pipeline)
        self.assertIn('--image-ids-json "${QUERY_RENDER_IMAGE_IDS_JSON}"', pipeline)

        with patch.dict(os.environ, {}, clear=True):
            _apply_render_profile_env_defaults("public_time_agnostic_v1")
            self.assertEqual(os.environ["QUERY_RENDER_REQUESTED_CAMERAS_ONLY"], "1")
            self.assertEqual(os.environ["GS_QUERY_REQUIRE_SYNCHRONIZED_STAGE1_BOUNDARY"], "1")
            self.assertEqual(os.environ["QUERY_REQUIRE_RENDERER_GEOMETRY"], "1")

    def test_explicit_camera_grid_can_be_restricted_to_requested_ids(self) -> None:
        reference_ids = ["000001", "000003", "000005"]
        requested_ids = ["000002", "000003", "000002"]

        self.assertEqual(
            _explicit_camera_image_ids(reference_ids, requested_ids, requested_only=True),
            ["000002", "000003"],
        )
        self.assertEqual(
            _explicit_camera_image_ids(reference_ids, requested_ids, requested_only=False),
            ["000001", "000003", "000005", "000002"],
        )

    def test_probability_opacity_round_trips_to_logits(self) -> None:
        probabilities = np.asarray([0.1, 0.5, 0.9], dtype=np.float32)
        logits = _opacity_logits_from_probabilities(probabilities)

        self.assertIsNotNone(logits)
        restored = 1.0 / (1.0 + np.exp(-logits))
        np.testing.assert_allclose(restored, probabilities, rtol=1.0e-5, atol=1.0e-5)

    def test_selection_segments_keep_their_temporal_meaning_on_new_cameras(self) -> None:
        active = _frame_mask_at_render_times(
            [[1, 1]],
            np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            np.asarray([0.0, 0.49, 0.75, 1.0], dtype=np.float32),
        )

        self.assertEqual(active.astype(int).tolist(), [0, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
