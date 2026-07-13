"""Release contracts for strict, scene-agnostic query rendering."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from refergaussian.semantics.query_render import _find_render_dir, _query_intent_mode


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


if __name__ == "__main__":
    unittest.main()
