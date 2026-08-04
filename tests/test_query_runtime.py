"""Regression tests for pinned Qwen release provenance."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_query_runtime.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_query_runtime", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNTIME = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNTIME)


class QueryRuntimeTest(unittest.TestCase):
    def test_release_manifest_requires_the_pinned_revision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "refergaussian_snapshot.json").write_text(
                json.dumps(
                    {
                        "repo_id": RUNTIME.DEFAULT_QWEN_REPO_ID,
                        "repo_type": "model",
                        "resolved_revision": RUNTIME.DEFAULT_QWEN_REVISION,
                    }
                ),
                encoding="utf-8",
            )

            self.assertEqual(RUNTIME._pinned_manifest_errors(model_dir), [])

    def test_release_manifest_rejects_missing_or_different_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            self.assertIn("missing", RUNTIME._pinned_manifest_errors(model_dir)[0])
            (model_dir / "refergaussian_snapshot.json").write_text(
                json.dumps(
                    {
                        "repo_id": "unrelated/model",
                        "repo_type": "model",
                        "resolved_revision": "deadbeef",
                    }
                ),
                encoding="utf-8",
            )

            errors = RUNTIME._pinned_manifest_errors(model_dir)
            self.assertEqual(len(errors), 2)
            self.assertTrue(any("repo_id" in error for error in errors))
            self.assertTrue(any("resolved revision" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
