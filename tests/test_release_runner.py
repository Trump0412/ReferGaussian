"""Regression tests for the strict release batch-runner contract."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_query_batch.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_release_runner", RUNNER_PATH)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _manifest_row(**overrides: object) -> dict:
    row = {
        "query_id": "scene_q1",
        "query": "the target object",
        "run_dir": "runs/refergaussian/hypernerf/scene",
        "dataset_dir": "data/hypernerf/misc/scene",
        "output_root": "reports/release/query_root",
        "gpu": 0,
    }
    row.update(overrides)
    return row


def _formal_public_rows(
    *, profile: str = "public_time_boundary_gated_v5"
) -> list[dict]:
    registry = RUNNER._load_protocol_registry()
    rows: list[dict] = []
    for query_id in sorted(RUNNER.PUBLIC_PROTOCOL_QUERY_IDS["paper_public3"]):
        scene = query_id.split("__", 1)[0]
        hashes = registry["public_annotation_sha256"][scene]
        rows.append(
            _manifest_row(
                query_id=query_id,
                scene=scene,
                profile=profile,
                protocol_id="paper_public3",
                protocol_complete=True,
                protocol_registry_version=registry["registry_version"],
                source_hashes={
                    "video_annotations_sha256": hashes["video_annotations"],
                    "coco_annotations_sha256": hashes["coco_masks"],
                    "protocol_json_sha256": "f" * 64,
                },
            )
        )
    return rows


class ReleaseRunnerTest(unittest.TestCase):
    def test_time_agnostic_runner_materializes_exact_test_camera_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            protocol_path = root / "protocol.json"
            protocol_path.write_text(
                json.dumps(
                    {
                        "queries": [
                            {
                                "query_slug": "scene__time_agnostic__target",
                                "category": "time_agnostic_reference",
                                "evaluation_image_ids": ["000010", "000020"],
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            item = _manifest_row(
                query_id="scene__time_agnostic__target",
                profile="public_time_agnostic_v1",
                source_paths={"protocol_json": str(protocol_path)},
            )

            output_path = RUNNER._prepare_public_time_agnostic_render_ids(
                item,
                query_output_root=str(root),
                strict_release=True,
            )

            self.assertIsNotNone(output_path)
            payload = json.loads(Path(str(output_path)).read_text(encoding="utf-8"))
            self.assertEqual(payload["image_ids"], ["000010", "000020"])
            self.assertEqual(payload["query_id"], item["query_id"])

    def test_time_agnostic_strict_runner_fails_without_protocol_ids(self) -> None:
        item = _manifest_row(
            query_id="scene__time_agnostic__target",
            profile="public_time_agnostic_v1",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(ValueError, "no readable protocol_json"):
                RUNNER._prepare_public_time_agnostic_render_ids(
                    item,
                    query_output_root=temp_dir,
                    strict_release=True,
                )

    def test_non_strict_runner_keeps_its_exploratory_default(self) -> None:
        self.assertEqual(
            RUNNER.resolve_profile(None, strict_release=False),
            RUNNER.EXPLORATORY_DEFAULT_PROFILE,
        )

    def test_strict_runner_requires_an_explicit_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit --profile"):
            RUNNER.resolve_profile(None, strict_release=True)

    def test_strict_runner_accepts_an_explicit_profile(self) -> None:
        self.assertEqual(
            RUNNER.resolve_profile("public_time_boundary_gated_v5", strict_release=True),
            "public_time_boundary_gated_v5",
        )

    def test_strict_runner_accepts_renderer_consistent_r4d_profile(self) -> None:
        self.assertEqual(
            RUNNER.resolve_profile("r4d_renderer_consistent", strict_release=True),
            "r4d_renderer_consistent",
        )
        self.assertEqual(
            RUNNER.PROTOCOL_PROFILES["release_r4d_dense89_renderer_consistent"],
            frozenset({"r4d_renderer_consistent"}),
        )

    def test_strict_runner_rejects_non_release_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "is not allowed"):
            RUNNER.resolve_profile("public_time_shape_v4_recall", strict_release=True)

    def test_strict_manifest_accepts_complete_registered_protocol(self) -> None:
        errors = RUNNER.validate_release_manifest(
            _formal_public_rows(),
            profile="public_time_boundary_gated_v5",
            protocol_id="paper_public3",
        )
        self.assertEqual(errors, [])

    def test_strict_manifest_rejects_per_query_environment_override(self) -> None:
        rows = _formal_public_rows()
        rows[0]["env"] = {"QUERY_SKIP_QWEN_SELECTION": "1"}
        errors = RUNNER.validate_release_manifest(
            rows,
            profile="public_time_boundary_gated_v5",
            protocol_id="paper_public3",
        )
        self.assertTrue(any("env override" in error for error in errors))

    def test_strict_manifest_rejects_mixed_profiles(self) -> None:
        rows = _formal_public_rows()
        rows[0]["profile"] = "r4d_boundary_gated_v5"
        errors = RUNNER.validate_release_manifest(
            rows,
            profile="public_time_boundary_gated_v5",
            protocol_id="paper_public3",
        )
        self.assertTrue(any("differs from --profile" in error for error in errors))

    def test_strict_manifest_rejects_incomplete_subset(self) -> None:
        rows = _formal_public_rows()[:1]
        rows[0]["protocol_complete"] = False
        errors = RUNNER.validate_release_manifest(
            rows,
            profile="public_time_boundary_gated_v5",
            protocol_id="paper_public3",
        )
        self.assertTrue(any("not marked complete" in error for error in errors))
        self.assertTrue(any("requires 7 rows" in error for error in errors))

    def test_strict_loader_rejects_malformed_missing_and_duplicate_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.jsonl"
            row = _manifest_row()
            manifest.write_text(
                "{bad json}\n"
                + json.dumps({"query_id": "missing"})
                + "\n"
                + json.dumps(row)
                + "\n"
                + json.dumps(row)
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "invalid JSON"):
                RUNNER.load_manifest(str(manifest), strict=True)

    def test_non_strict_loader_records_skipped_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(_manifest_row()) + "\n" + "not-json\n",
                encoding="utf-8",
            )
            rows, audit = RUNNER.load_manifest_with_audit(str(manifest), strict=False)
        self.assertEqual(len(rows), 1)
        self.assertEqual(audit["skipped_rows"], 1)

    def test_strict_gpu_grouping_rejects_unscheduled_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside active GPU"):
            RUNNER.group_by_gpu([_manifest_row(gpu=2)], [0, 1], strict=True)

    def test_execution_contract_detects_missing_or_duplicate_results(self) -> None:
        scheduled = [_manifest_row(query_id="q1"), _manifest_row(query_id="q2")]
        errors = RUNNER._execution_contract_errors(
            scheduled,
            [{"query_id": "q1"}, {"query_id": "q1"}],
        )
        self.assertTrue(any("duplicate" in error for error in errors))
        self.assertTrue(any("differ" in error for error in errors))

    def test_strict_runner_clears_inherited_query_tuning_only(self) -> None:
        env = {
            "QUERY_SKIP_QWEN_SELECTION": "1",
            "GS_QUERY_ALLOW_DIRECT_2D_MASKS": "1",
            "GSAM2_REUSE_QUERY_PLAN": "1",
            "QWEN_MODEL_PATH": "/models/qwen",
            "PATH": "/usr/bin",
        }

        RUNNER._clear_inherited_release_tuning(env)

        self.assertEqual(env, {"QWEN_MODEL_PATH": "/models/qwen", "PATH": "/usr/bin"})

    def test_strict_runner_rejects_a_dirty_git_checkout(self) -> None:
        errors = RUNNER._source_tree_release_errors(
            strict_release=True,
            commit="a" * 40,
            status=" M scripts/query.py",
        )

        self.assertEqual(len(errors), 1)
        self.assertIn("clean Git worktree", errors[0])

    def test_source_archive_without_git_metadata_is_supported(self) -> None:
        errors = RUNNER._source_tree_release_errors(
            strict_release=True,
            commit=None,
            status=None,
        )

        self.assertEqual(errors, [])

    def test_provenance_environment_redacts_secrets(self) -> None:
        env = {
            "QWEN_MODEL_PATH": "/models/qwen",
            "HF_HOME": "/cache/hf",
            "HF_TOKEN": "must-not-leak",
            "QWEN_API_KEY": "must-not-leak",
            "PATH": "/usr/bin",
        }

        safe = RUNNER._safe_release_environment(env)

        self.assertEqual(
            safe,
            {"HF_HOME": "/cache/hf", "QWEN_MODEL_PATH": "/models/qwen"},
        )

    def test_sha256_file_records_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "artifact.txt"
            path.write_text("refergaussian\n", encoding="utf-8")
            record = RUNNER._sha256_file(path)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["bytes"], len("refergaussian\n"))
        self.assertEqual(len(str(record["sha256"])), 64)

    def test_qwen_provenance_records_manifest_and_weight_sizes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir)
            (model_dir / "refergaussian_snapshot.json").write_text(
                '{"resolved_revision":"abc"}\n', encoding="utf-8"
            )
            (model_dir / "config.json").write_text("{}\n", encoding="utf-8")
            (model_dir / "model-00001.safetensors").write_bytes(b"weights")

            record = RUNNER._qwen_model_provenance(
                {"REFERGAUSSIAN_QWEN_MODEL": str(model_dir)}
            )

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(len(record["metadata"]), 2)
        self.assertEqual(
            record["weight_files"],
            [{"name": "model-00001.safetensors", "bytes": 7}],
        )


if __name__ == "__main__":
    unittest.main()
