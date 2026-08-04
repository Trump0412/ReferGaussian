"""Tests for R4D manifest layout overrides used in isolated reproductions."""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


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

    def test_release_query_map_has_exact_dense_protocol_size(self) -> None:
        payload = json.loads(MANIFEST.R4D_ENGLISH_QUERY_MAP_PATH.read_text(encoding="utf-8"))
        registry = MANIFEST._load_protocol_registry()
        self.assertEqual(
            len(payload),
            registry["protocols"][MANIFEST.FORMAL_R4D_PROTOCOL]["query_count"],
        )
        self.assertEqual(len(MANIFEST.SCENE_CONFIG), 12)

    def test_formal_source_validator_checks_hashes_and_categories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark = root / "benchmark.json"
            metadata = root / "metadata.json"
            english = root / "english.json"
            benchmark.write_text(json.dumps([{"query_id": "scene_q1"}]), encoding="utf-8")
            metadata.write_text(
                json.dumps(
                    [
                        {
                            "query_id": "scene_q1",
                            "query_type": "A",
                            "scene": "scene",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            english.write_text(json.dumps({"scene_q1": "The target."}), encoding="utf-8")
            registry = {
                "protocols": {
                    MANIFEST.FORMAL_R4D_PROTOCOL: {
                        "query_count": 1,
                        "category_counts": {"temporal_single_target": 1},
                        "dense_gt_sha256": MANIFEST._sha256(benchmark),
                        "query_metadata_sha256": MANIFEST._sha256(metadata),
                        "english_query_map_sha256": MANIFEST._sha256(english),
                    }
                }
            }
            with mock.patch.object(MANIFEST, "R4D_ENGLISH_QUERY_MAP_PATH", english):
                rows, _paths, hashes = MANIFEST._validate_formal_sources(
                    benchmark_path=benchmark,
                    query_metadata_path=metadata,
                    registry=registry,
                )
        self.assertEqual(set(rows), {"scene_q1"})
        self.assertEqual(hashes["benchmark_sha256"], registry["protocols"][MANIFEST.FORMAL_R4D_PROTOCOL]["dense_gt_sha256"])


if __name__ == "__main__":
    unittest.main()
