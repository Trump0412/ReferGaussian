"""Tests for scene-agnostic query-planner phrase expansion."""

from __future__ import annotations

import unittest

from refergaussian.semantics.qwen_query_planner import _state_detector_phrase_additions


class QueryPlannerPhraseTest(unittest.TestCase):
    def test_state_phrase_is_composed_for_an_unseen_object(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the package broken into several pieces",
            ["package"],
        )
        self.assertIn("broken package", phrases)
        self.assertIn("broken package pieces", phrases)
        self.assertIn("package pieces", phrases)

    def test_state_phrase_preserves_query_specific_noun_context(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the closed plastic container",
            ["container"],
        )
        self.assertIn("closed container", phrases)
        self.assertIn("closed plastic container", phrases)

    def test_content_state_is_derived_from_the_query(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the vessel with liquid above the midpoint",
            ["vessel"],
        )
        self.assertIn("liquid above midpoint", phrases)


if __name__ == "__main__":
    unittest.main()
