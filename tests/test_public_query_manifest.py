"""Tests that Public batch manifests follow generated annotation protocols."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_public_query_manifest.py"
if str(SCRIPT_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("refergaussian_public_manifest", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MANIFEST = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MANIFEST)


class PublicQueryManifestTest(unittest.TestCase):
    def test_time_sensitive_rows_preserve_protocol_id_and_text(self) -> None:
        payload = {
            "queries": [
                {
                    "scene": "HyperNeRF/misc/espresso",
                    "query_slug": "espresso__the_empty_glass_cup",
                    "query": "the empty glass cup",
                    "category": "temporal_state_reference",
                },
                {
                    "scene": "HyperNeRF/misc/espresso",
                    "query_slug": "espresso__the_glass_cup",
                    "query": "the glass cup",
                    "category": "static_reference",
                },
            ]
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "protocol.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            rows = MANIFEST._time_sensitive_protocol_rows(path)

        self.assertEqual(rows, [("espresso", "espresso__the_empty_glass_cup", "the empty glass cup")])
