"""Tests that Public batch manifests follow generated annotation protocols."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from unittest import mock
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
    def test_release_exposes_only_dynamic_public_protocols(self) -> None:
        self.assertEqual(
            MANIFEST.FORMAL_PUBLIC_PROTOCOLS,
            frozenset({"paper_public3", "release_public4_extension"}),
        )

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

    def test_run_namespace_replaces_only_the_layout_prefix(self) -> None:
        self.assertEqual(
            MANIFEST._run_relative_path("refergaussian/hypernerf/espresso", "clean_reproduction"),
            "clean_reproduction/hypernerf/espresso",
        )

    def test_scene_filter_separates_paper_three_from_four_scene_extension(self) -> None:
        rows = [
            ("americano", "a_q1", "query a"),
            ("espresso", "e_q1", "query e"),
            ("split-cookie", "s_q1", "query s"),
            ("chickchicken", "c_q1", "query c"),
        ]

        selected = MANIFEST._filter_protocol_scenes(
            rows,
            ["americano", "split-cookie", "espresso"],
        )

        self.assertEqual([row[0] for row in selected], ["americano", "espresso", "split-cookie"])
        self.assertNotIn("chickchicken", {row[0] for row in selected})

    def test_formal_identity_rejects_a_canary_subset(self) -> None:
        registry = MANIFEST._load_protocol_registry()
        rows = [
            (query_id.split("__", 1)[0], query_id, "query")
            for query_id in MANIFEST.PUBLIC_PROTOCOL_QUERY_IDS["paper_public3"]
        ]
        MANIFEST._validate_formal_identity(
            rows,
            protocol_id="paper_public3",
            registry=registry,
        )
        with self.assertRaisesRegex(ValueError, "requires 7 queries"):
            MANIFEST._validate_formal_identity(
                rows[:1],
                protocol_id="paper_public3",
                registry=registry,
            )

    def test_gs_root_precedes_deprecated_alias(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {"GS_DATA_ROOT": "/new", "REFERGAUSSIAN_DATA_ROOT": "/old"},
            clear=True,
        ):
            self.assertEqual(
                MANIFEST._root_env_default("GS_DATA_ROOT", "REFERGAUSSIAN_DATA_ROOT", "data"),
                "/new",
            )
