"""Tests for the bounded matched-reconstruction release harness."""

from __future__ import annotations

import copy
import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "scripts" / "run_matched_reconstruction.py"
REGISTRY_PATH = REPO_ROOT / "configs" / "benchmarks" / "reconstruction_release_v1.json"
SPEC = importlib.util.spec_from_file_location(
    "refergaussian_matched_reconstruction", RUNNER_PATH
)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _registry() -> dict:
    return RUNNER.read_json(REGISTRY_PATH)


def _metrics(psnr: float, ssim: float, vgg: float, alex: float) -> dict:
    return {
        "PSNR": psnr,
        "SSIM": ssim,
        "LPIPS-vgg": vgg,
        "LPIPS-alex": alex,
    }


class ProtocolValidationTest(unittest.TestCase):
    def test_git_output_preserves_porcelain_status_columns(self) -> None:
        completed = SimpleNamespace(
            returncode=0,
            stdout=" M arguments/__init__.py\n?? local.py\n",
            stderr="",
        )
        with mock.patch.object(RUNNER.subprocess, "run", return_value=completed):
            output = RUNNER._git_output(Path("/tmp/example"), "status", "--porcelain=v1")

        self.assertEqual(output, " M arguments/__init__.py\n?? local.py")

    def test_executable_release_protocol_is_valid(self) -> None:
        protocol = _registry()["identities"]["release_reconstruction_v1"]
        RUNNER.validate_protocol("release_reconstruction_v1", protocol)
        self.assertEqual(protocol["scene_count"], 12)
        self.assertFalse(protocol["is_paper_reproduction"])

    def test_reported_paper_identity_is_not_executable(self) -> None:
        protocol = _registry()["identities"]["paper_reported_12_scene_table"]
        self.assertEqual(
            protocol["reported_metrics"]["baseline_4dgs"],
            {"PSNR": 20.3208, "SSIM": 0.7027, "LPIPS": 0.3971},
        )
        self.assertEqual(
            protocol["reported_metrics"]["refergaussian"],
            {"PSNR": 20.4159, "SSIM": 0.7069, "LPIPS": 0.3942},
        )
        with self.assertRaisesRegex(RUNNER.HarnessError, "unresolved or non-executable"):
            RUNNER.validate_protocol("paper_reported_12_scene_table", protocol)

    def test_release_rejects_subset_metric_mode(self) -> None:
        protocol = copy.deepcopy(
            _registry()["identities"]["release_reconstruction_v1"]
        )
        protocol["shared"]["metric_mode"] = "quick_subset"
        with self.assertRaisesRegex(RUNNER.HarnessError, "full mode"):
            RUNNER.validate_protocol("release_reconstruction_v1", protocol)

    def test_environment_tube_values_cannot_diverge_from_protocol(self) -> None:
        protocol = copy.deepcopy(
            _registry()["identities"]["release_reconstruction_v1"]
        )
        protocol["refergaussian"]["frozen_environment"][
            "TEMPORAL_TUBE_SIGMA"
        ] = "0.99"
        with self.assertRaisesRegex(RUNNER.HarnessError, "TEMPORAL_TUBE_SIGMA"):
            RUNNER.validate_protocol("release_reconstruction_v1", protocol)

    def test_external_patch_snapshot_is_content_addressed(self) -> None:
        protocol = copy.deepcopy(
            _registry()["identities"]["release_reconstruction_v1"]
        )
        external = protocol["external_4dgaussians"]
        self.assertEqual(len(external["patched_diff_sha256"]), 64)
        self.assertEqual(
            {item["path"] for item in external["untracked_files"]},
            {"arguments/hypernerf/cut-lemon1.py", "utils/config_utils.py"},
        )

        external.pop("patched_diff_sha256")
        with self.assertRaisesRegex(RUNNER.HarnessError, "patched diff"):
            RUNNER.validate_protocol("release_reconstruction_v1", protocol)

    def test_external_generated_file_hash_is_validated(self) -> None:
        protocol = copy.deepcopy(
            _registry()["identities"]["release_reconstruction_v1"]
        )
        protocol["external_4dgaussians"]["untracked_files"][0]["sha256"] = "bad"
        with self.assertRaisesRegex(RUNNER.HarnessError, "file SHA-256"):
            RUNNER.validate_protocol("release_reconstruction_v1", protocol)

    def test_mutable_parameter_override_is_rejected(self) -> None:
        protocol = _registry()["identities"]["release_reconstruction_v1"]
        expected = RUNNER.expected_parameter_environment(protocol)
        with self.assertRaisesRegex(RUNNER.HarnessError, "TEMPORAL_TUBE_SIGMA"):
            RUNNER.validate_no_mutable_parameter_overrides(
                {"TEMPORAL_TUBE_SIGMA": "0.99"}, expected
            )


class SceneSubsetTest(unittest.TestCase):
    def setUp(self) -> None:
        self.protocol = _registry()["identities"]["release_reconstruction_v1"]

    def test_full_scene_selection_is_complete(self) -> None:
        selected, complete = RUNNER.select_scene_ids(self.protocol, None)
        self.assertTrue(complete)
        self.assertEqual(selected, self.protocol["scene_ids"])

    def test_canary_subset_is_ordered_and_incomplete(self) -> None:
        selected, complete = RUNNER.select_scene_ids(
            self.protocol, ["torchchocolate", "americano"]
        )
        self.assertEqual(selected, ["americano", "torchchocolate"])
        self.assertFalse(complete)

    def test_unknown_or_duplicate_subset_is_rejected(self) -> None:
        with self.assertRaisesRegex(RUNNER.HarnessError, "Unknown scene"):
            RUNNER.select_scene_ids(self.protocol, ["not_a_scene"])
        with self.assertRaisesRegex(RUNNER.HarnessError, "duplicates"):
            RUNNER.select_scene_ids(self.protocol, ["americano", "americano"])

    def test_dense_scene_layout_is_exactly_frozen(self) -> None:
        protocol = copy.deepcopy(self.protocol)
        protocol["scenes"][0]["scene"] = "misc/a-different-scene"
        protocol["scenes"][0]["source_relpath"] = (
            "hypernerf/misc/a-different-scene"
        )
        with self.assertRaisesRegex(RUNNER.HarnessError, "frozen dense R4D layout"):
            RUNNER.validate_protocol("release_reconstruction_v1", protocol)


class DefaultConfigContractTest(unittest.TestCase):
    def test_explicit_default_config_is_accepted(self) -> None:
        scene = {
            "id": "americano",
            "config_relpath": "external/4DGaussians/arguments/hypernerf/default.py",
            "default_config_declared": True,
        }
        RUNNER.validate_default_config_declaration(
            scene, "external/4DGaussians/arguments/hypernerf/default.py"
        )

    def test_undeclared_default_config_is_rejected(self) -> None:
        scene = {
            "id": "americano",
            "config_relpath": "external/4DGaussians/arguments/hypernerf/default.py",
            "default_config_declared": False,
        }
        with self.assertRaisesRegex(RUNNER.HarnessError, "without an explicit"):
            RUNNER.validate_default_config_declaration(
                scene, "external/4DGaussians/arguments/hypernerf/default.py"
            )

    def test_resolved_config_must_match_registry(self) -> None:
        scene = {
            "id": "cut_lemon",
            "config_relpath": "external/4DGaussians/arguments/hypernerf/cut-lemon1.py",
            "default_config_declared": False,
        }
        with self.assertRaisesRegex(RUNNER.HarnessError, "registry requires"):
            RUNNER.validate_default_config_declaration(
                scene, "external/4DGaussians/arguments/hypernerf/default.py"
            )


class SceneEqualAggregationTest(unittest.TestCase):
    def test_scene_equal_mean_keeps_low_psnr_scene(self) -> None:
        per_scene = {
            "scene_a": {
                "status": "complete",
                "metrics": {
                    "baseline_4dgs": _metrics(5.0, 0.4, 0.6, 0.5),
                    "refergaussian": _metrics(7.0, 0.5, 0.5, 0.4),
                },
            },
            "scene_b": {
                "status": "complete",
                "metrics": {
                    "baseline_4dgs": _metrics(25.0, 0.8, 0.2, 0.1),
                    "refergaussian": _metrics(27.0, 0.9, 0.1, 0.05),
                },
            },
        }
        aggregate = RUNNER.aggregate_scene_equal(
            per_scene, ["scene_a", "scene_b"]
        )
        self.assertEqual(aggregate["scene_count"], 2)
        self.assertFalse(aggregate["post_hoc_psnr_filtering"])
        self.assertAlmostEqual(
            aggregate["methods"]["baseline_4dgs"]["PSNR"], 15.0
        )
        self.assertAlmostEqual(
            aggregate["methods"]["refergaussian"]["LPIPS-vgg"], 0.3
        )

    def test_incomplete_scene_set_cannot_be_aggregated(self) -> None:
        per_scene = {
            "scene_a": {
                "status": "complete",
                "metrics": {
                    "baseline_4dgs": _metrics(20.0, 0.7, 0.3, 0.2),
                    "refergaussian": _metrics(21.0, 0.8, 0.2, 0.1),
                },
            }
        }
        with self.assertRaisesRegex(RUNNER.HarnessError, "requires all declared"):
            RUNNER.aggregate_scene_equal(per_scene, ["scene_a", "scene_b"])


if __name__ == "__main__":
    unittest.main()
