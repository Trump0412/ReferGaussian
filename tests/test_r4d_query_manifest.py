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


if __name__ == "__main__":
    unittest.main()
