"""Tests for converting public 4DLangSplat annotations into query protocols."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "scripts" / "build_4dlangsplat_query_protocol.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_public_protocol_builder", BUILDER_PATH)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


class PublicProtocolBuilderTest(unittest.TestCase):
    def test_state_annotations_become_query_ids_and_ranges(self) -> None:
        rows = BUILDER.build_scene_queries(
            "espresso",
            {
                "glass cup": {
                    "empty glass cup": [[0, 340]],
                    "full glass cup": [[600, 775]],
                }
            },
        )
        by_id = {row["query_slug"]: row for row in rows}

        self.assertEqual(by_id["espresso__the_empty_glass_cup"]["query"], "the empty glass cup")
        self.assertEqual(
            by_id["espresso__the_empty_glass_cup"]["targets"][0]["target_ranges"],
            [[0, 340]],
        )
        self.assertEqual(
            by_id["espresso__the_glass_cup"]["targets"][0]["target_ranges"],
            [[0, 340], [600, 775]],
        )

    def test_protocol_scene_uses_the_expected_hypernerf_group(self) -> None:
        rows = BUILDER.build_scene_queries(
            "chickchicken",
            {"chicken container": {"closed chicken container": [[0, 25]]}},
        )

        self.assertTrue(rows)
        self.assertEqual(rows[0]["scene"], "HyperNeRF/interp/chickchicken")


if __name__ == "__main__":
    unittest.main()
