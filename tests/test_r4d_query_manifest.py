"""Tests for R4D manifest layout overrides used in isolated reproductions."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_r4d_query_manifest.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
from query_text_utils import choose_english_query_text

SPEC = importlib.util.spec_from_file_location("refergaussian_r4d_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


class R4DQueryManifestTest(unittest.TestCase):
    def test_run_namespace_replaces_only_the_layout_prefix(self) -> None:
        self.assertEqual(
            MANIFEST._run_relative_path("refergaussian/hypernerf/keyboard", "clean_reproduction"),
            "clean_reproduction/hypernerf/keyboard",
        )

    def test_vetted_english_map_preserves_hand_side_and_object_state(self) -> None:
        query, source = choose_english_query_text(
            {"query_id": "keyboard_q1", "question": "正在键盘上打字的左手。"}
        )
        self.assertEqual(query, "The left hand while it is typing on the keyboard.")
        self.assertEqual(source, "r4d_query_text_en")

        steak_query, _ = choose_english_query_text(
            {"query_id": "sear_steak_q2", "question": "被夹子夹起的牛排。"}
        )
        self.assertEqual(steak_query, "The beef steak being held by the tongs.")


if __name__ == "__main__":
    unittest.main()
