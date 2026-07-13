"""Contract tests for query-local renderer geometry caches."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from refergaussian.semantics.renderer_geometry import (
    GEOMETRY_MANIFEST_NAME,
    GEOMETRY_SCHEMA_VERSION,
    RendererGeometryCache,
)


class RendererGeometryCacheTest(unittest.TestCase):
    def _write_cache(self, root: Path) -> None:
        np.save(root / "gaussian_ids.npy", np.asarray([2, 5, 9], dtype=np.int64))
        np.save(
            root / "centers.npy",
            np.asarray(
                [
                    [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
                    [[4.0, 0.0, 1.0], [5.0, 0.0, 1.0], [6.0, 0.0, 1.0]],
                ],
                dtype=np.float32,
            ),
        )
        np.save(root / "opacity_logit.npy", np.zeros((2, 3), dtype=np.float32))
        np.save(root / "covariance_packed.npy", np.ones((2, 3, 6), dtype=np.float32))
        np.save(root / "projection_world_view.npy", np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0))
        np.save(root / "projection_full.npy", np.repeat(np.eye(4, dtype=np.float32)[None], 2, axis=0))
        np.save(root / "projection_image_sizes.npy", np.asarray([[100, 80], [100, 80]], dtype=np.int32))
        (root / GEOMETRY_MANIFEST_NAME).write_text(
            json.dumps(
                {
                    "schema_version": GEOMETRY_SCHEMA_VERSION,
                    "image_ids": ["000001", "000002"],
                    "time_values": [0.0, 1.0],
                }
            ),
            encoding="utf-8",
        )

    def test_resolves_exact_frame_and_gaussian_subset(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)
            cache = RendererGeometryCache(root)
            frame = cache.resolve("000002", 0.0, np.asarray([9, 2], dtype=np.int64))

        self.assertTrue(frame.exact_image_id)
        self.assertEqual(frame.image_id, "000002")
        self.assertTrue(np.array_equal(frame.centers[:, 0], np.asarray([6.0, 4.0], dtype=np.float32)))
        self.assertEqual(frame.covariance_packed.shape, (2, 6))

    def test_resolves_renderer_projection_camera(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)
            cache = RendererGeometryCache(root)
            camera = cache.resolve_projection_camera("000001", 0.0)
            pixels = camera.project(np.asarray([[-1.0, -1.0, 1.0], [1.0, 1.0, 1.0]], dtype=np.float32))
            local = camera.points_to_local_points(np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32))

        self.assertTrue(np.allclose(pixels, np.asarray([[0.0, 0.0], [100.0, 80.0]], dtype=np.float32)))
        self.assertTrue(np.allclose(local, np.asarray([[0.0, 0.0, 2.0]], dtype=np.float32)))

    def test_missing_id_and_frame_are_hard_errors(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_cache(root)
            cache = RendererGeometryCache(root)
            with self.assertRaises(KeyError):
                cache.columns_for_gaussian_ids(np.asarray([4], dtype=np.int64))
            with self.assertRaises(KeyError):
                cache.resolve("missing", 0.5, require_exact_image_id=True)


if __name__ == "__main__":
    unittest.main()
