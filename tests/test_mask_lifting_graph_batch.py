"""Regression coverage for batched radius-neighborhood construction."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]


def _load_radius_neighbor_helper():
    source_path = ROOT / "refergaussian" / "semantics" / "mask_supported_lifting.py"
    module = ast.parse(source_path.read_text(encoding="utf-8"))
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_radius_neighbor_lists"
    )
    namespace = {"np": np, "cKDTree": cKDTree}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source_path), "exec"), namespace)
    return namespace["_radius_neighbor_lists"]


class MaskLiftingGraphBatchTest(unittest.TestCase):
    def test_batched_radius_query_matches_per_point_reference(self) -> None:
        radius_neighbor_lists = _load_radius_neighbor_helper()
        points = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.0, 0.1, 0.0],
                [0.9, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        tree = cKDTree(points)
        radius = 0.16

        expected = [
            np.asarray(tree.query_ball_point(point, r=radius), dtype=np.int64)
            for point in points
        ]
        actual = radius_neighbor_lists(tree, points, radius)

        self.assertEqual(len(actual), len(expected))
        for actual_neighbors, expected_neighbors in zip(actual, expected):
            self.assertListEqual(actual_neighbors.tolist(), sorted(expected_neighbors.tolist()))


if __name__ == "__main__":
    unittest.main()
