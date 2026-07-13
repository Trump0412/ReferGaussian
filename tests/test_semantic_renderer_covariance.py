"""Regression tests for covariance-aware semantic Gaussian splatting."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from refergaussian.semantics.semantic_renderer import prepare_semantic_frame_inputs, render_selection_mask


class _PinholeCamera:
    image_size = np.asarray([64, 64], dtype=np.uint32)

    def points_to_local_points(self, points: np.ndarray) -> np.ndarray:
        return np.asarray(points, dtype=np.float32)

    def project(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=np.float32)
        return np.stack([32.0 + 20.0 * points[:, 0] / points[:, 2], 32.0 + 20.0 * points[:, 1] / points[:, 2]], axis=1)


class SemanticRendererCovarianceTest(unittest.TestCase):
    def test_selection_priority_overrides_generic_frame_cap(self) -> None:
        prepared = prepare_semantic_frame_inputs(
            camera=_PinholeCamera(),
            frame_index=0,
            image_id="000001",
            time_value=0.0,
            points=np.asarray([[0.0, 0.0, 2.0], [0.1, 0.0, 2.0]], dtype=np.float32),
            spatial_scale=np.full((2, 3), 0.05, dtype=np.float32),
            opacity=np.asarray([0.0, 12.0], dtype=np.float32),
            visibility_gate=np.ones((2,), dtype=np.float32),
            selection_priority=np.asarray([10.0, 1.0], dtype=np.float32),
            max_gaussians=1,
            device="cpu",
        )
        self.assertTrue(np.array_equal(prepared.gaussian_ids.cpu().numpy(), np.asarray([0], dtype=np.int64)))

    def test_covariance_projection_creates_anisotropic_splat(self) -> None:
        prepared = prepare_semantic_frame_inputs(
            camera=_PinholeCamera(),
            frame_index=0,
            image_id="000001",
            time_value=0.0,
            points=np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32),
            spatial_scale=np.asarray([[0.05, 0.05, 0.05]], dtype=np.float32),
            opacity=np.asarray([8.0], dtype=np.float32),
            visibility_gate=np.asarray([1.0], dtype=np.float32),
            covariance_world=np.asarray([[0.36, 0.0, 0.0, 0.0025, 0.0, 0.0025]], dtype=np.float32),
            device="cpu",
        )
        self.assertIsNotNone(prepared.covariance_xy)
        mask, _ = render_selection_mask(
            prepared,
            torch.ones((1,), dtype=torch.float32),
            relative_threshold=0.0,
            absolute_threshold=0.01,
            max_splat_radius=32,
        )
        ys, xs = np.where(mask)
        self.assertGreater(xs.max() - xs.min(), ys.max() - ys.min())


if __name__ == "__main__":
    unittest.main()
