"""Unit tests for Stage-1 anchor-window coverage."""

from __future__ import annotations

import unittest

from refergaussian.semantics.track_windows import required_anchor_window_radius


class TrackWindowTest(unittest.TestCase):
    def test_radius_bridges_anchor_gaps(self) -> None:
        self.assertEqual(
            required_anchor_window_radius(
                [0, 360, 660],
                first_frame_index=0,
                last_frame_index=690,
            ),
            180,
        )

    def test_radius_covers_edges_and_single_anchor(self) -> None:
        self.assertEqual(
            required_anchor_window_radius(
                [50],
                first_frame_index=0,
                last_frame_index=100,
            ),
            50,
        )
        self.assertEqual(
            required_anchor_window_radius(
                [20, 80],
                first_frame_index=0,
                last_frame_index=100,
            ),
            30,
        )

    def test_invalid_ranges_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            required_anchor_window_radius([], first_frame_index=0, last_frame_index=1)
        with self.assertRaises(ValueError):
            required_anchor_window_radius([0], first_frame_index=2, last_frame_index=1)
        with self.assertRaises(ValueError):
            required_anchor_window_radius([10], first_frame_index=0, last_frame_index=5)


if __name__ == "__main__":
    unittest.main()
