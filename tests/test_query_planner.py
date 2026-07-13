"""Tests for scene-agnostic query-planner phrase expansion."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from refergaussian.semantics.qwen_query_planner import (
    _canonicalize_phrase,
    _count_neutral_detector_phrases,
    _normalize_plan,
    _qwen_gpu_memory_budget_gib,
    _qwen_max_new_tokens,
    _state_detector_phrase_additions,
)
from refergaussian.semantics.select_qwen_query_entities import (
    _candidate_phrase_score,
    _expand_counted_subject_phrases,
    _select_phrase_ids,
)


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

    def test_action_context_is_not_promoted_to_the_singular_query_subject(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["knife", "beef steak"],
                "query_subject_phrases": ["knife", "beef steak"],
                "must_track_phrases": ["knife", "beef steak"],
            },
            "The knife while it is cutting the beef steak.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["knife"])
        self.assertEqual(plan["query_subject_phrases"], ["knife"])
        self.assertEqual(plan["detector_phrases"], ["knife"])
        self.assertEqual(plan["must_track_phrases"], ["knife"])

    def test_explicit_multi_target_query_preserves_all_requested_subjects(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["red block", "blue block"],
                "primary_subject_phrases": ["red block", "blue block"],
                "query_subject_phrases": ["red block", "blue block"],
            },
            "Both the red block and the blue block while they are moving.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["red block", "blue block"])
        self.assertEqual(plan["query_subject_phrases"], ["red block", "blue block"])

    def test_normalization_keeps_object_names_and_matches_spelling_generically(self) -> None:
        self.assertEqual(_canonicalize_phrase("cocktail-glass"), "cocktail glass")
        self.assertEqual(_canonicalize_phrase("mouse-pad"), "mouse pad")
        score = _candidate_phrase_score("mouse pad", {"proposal_alias": "mousepad"})
        self.assertGreaterEqual(score, 0.9)

    def test_counted_subjects_use_generic_distinct_entity_selection(self) -> None:
        phrases = _expand_counted_subject_phrases(["both cups"])
        self.assertEqual(phrases, ["cup", "cup"])
        candidates = [
            {"id": 1, "proposal_alias": "red cup", "quality": 0.9},
            {"id": 2, "proposal_alias": "blue cup", "quality": 0.8},
        ]
        selected_ids, selected_by_phrase = _select_phrase_ids(candidates, phrases)
        self.assertEqual(selected_ids, [1, 2])
        self.assertEqual(selected_by_phrase["cup"], [1, 2])

    def test_count_neutral_detector_variant_is_not_category_specific(self) -> None:
        phrases = _count_neutral_detector_phrases(["two packages"])
        self.assertIn("two packages", phrases)
        self.assertIn("packages", phrases)
        self.assertIn("package", phrases)

    def test_qwen_budget_uses_available_large_gpu_memory_without_a_hidden_16gib_cap(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REFERGAUSSIAN_QWEN_GPU_RESERVE_GIB", None)
            os.environ.pop("REFERGAUSSIAN_QWEN_GPU_MAX_GIB", None)
            self.assertEqual(_qwen_gpu_memory_budget_gib(32), 28)
            self.assertEqual(_qwen_gpu_memory_budget_gib(12), 8)
        with patch.dict(
            os.environ,
            {
                "REFERGAUSSIAN_QWEN_GPU_RESERVE_GIB": "3",
                "REFERGAUSSIAN_QWEN_GPU_MAX_GIB": "20",
            },
            clear=False,
        ):
            self.assertEqual(_qwen_gpu_memory_budget_gib(32), 20)

    def test_qwen_generation_budget_defaults_to_existing_limit_and_is_bounded(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REFERGAUSSIAN_QWEN_MAX_NEW_TOKENS", None)
            self.assertEqual(_qwen_max_new_tokens(), 1024)
        with patch.dict(
            os.environ,
            {"REFERGAUSSIAN_QWEN_MAX_NEW_TOKENS": "640"},
            clear=False,
        ):
            self.assertEqual(_qwen_max_new_tokens(), 640)


if __name__ == "__main__":
    unittest.main()
