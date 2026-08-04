"""Release contract for resolved, semantic-empty, and unresolved selections."""

from __future__ import annotations

import unittest

from refergaussian.semantics.select_qwen_query_entities import (
    PROMPT_TEMPLATE,
    _finalize_selection_status,
)
from refergaussian.semantics.qwen_query_planner import QUERY_PLAN_TEMPLATE


class SelectionResolutionContractTest(unittest.TestCase):
    def test_release_prompts_do_not_name_benchmark_examples(self) -> None:
        prompt_text = f"{QUERY_PLAN_TEMPLATE}\n{PROMPT_TEMPLATE}".lower()

        for term in ("glass", "cup", "cookie", "keyboard", "chocolate", "midpoint"):
            self.assertNotIn(term, prompt_text)

    def test_selected_entity_is_resolved(self) -> None:
        payload = _finalize_selection_status({"selected": [{"id": 7}], "empty": False})

        self.assertEqual(payload["selection_status"], "resolved")
        self.assertFalse(payload["semantic_empty"])
        self.assertFalse(payload["unresolved"])

    def test_verified_empty_answer_stays_semantic_empty(self) -> None:
        payload = _finalize_selection_status(
            {
                "selected": [],
                "empty": True,
                "selection_status": "semantic_empty",
                "selection_status_reason": "visual_absence_verified",
            }
        )

        self.assertEqual(payload["selection_status"], "semantic_empty")
        self.assertTrue(payload["semantic_empty"])
        self.assertFalse(payload["unresolved"])

    def test_unclassified_empty_answer_fails_closed(self) -> None:
        payload = _finalize_selection_status(
            {"selected": [], "empty": True, "selection_mode": "phrase_match_empty"}
        )

        self.assertEqual(payload["selection_status"], "unresolved")
        self.assertFalse(payload["semantic_empty"])
        self.assertTrue(payload["unresolved"])

    def test_nonresolved_status_cannot_hide_selected_entities(self) -> None:
        payload = _finalize_selection_status(
            {
                "selected": [{"id": 3}],
                "empty": False,
                "selection_status": "semantic_empty",
            }
        )

        self.assertEqual(payload["selection_status"], "unresolved")


if __name__ == "__main__":
    unittest.main()
