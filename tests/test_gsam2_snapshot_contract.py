"""Regression tests for pinned offline Grounded-SAM2 runtime loading."""

from __future__ import annotations

import unittest
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


class GroundedSamSnapshotContractTest(unittest.TestCase):
    def test_runtime_resolves_pinned_sam2_checkpoint_locally(self) -> None:
        text = (ROOT / "refergaussian/semantics/grounded_sam2_backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("HF_MODEL_ID_TO_FILENAMES", text)
        self.assertIn("revision=sam2_model_revision", text)
        self.assertIn("local_files_only=local_files_only", text)
        self.assertIn("use_fast=True", text)
        self.assertNotIn("SAM2VideoPredictor.from_pretrained(sam2_model_id", text)

    def test_launcher_defaults_to_local_only_weights(self) -> None:
        text = (ROOT / "scripts" / "run_query_guided_grounded_sam2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("GSAM2_LOCAL_FILES_ONLY:-1", text)
        self.assertIn("--local-files-only", text)
        self.assertIn("HF_HUB_OFFLINE=1", text)
        self.assertIn("TRANSFORMERS_OFFLINE=1", text)

    def test_setup_verifies_the_same_pinned_local_cache(self) -> None:
        text = (ROOT / "scripts" / "setup_grounded_sam2.sh").read_text(encoding="utf-8")
        self.assertIn("hf_hub_download(", text)
        self.assertIn("revision=sam2_model_revision", text)
        self.assertIn("local_files_only=local_files_only", text)
        self.assertIn("GSAM2_INSTALL_EDITABLE:-1", text)
        self.assertIn("GSAM2_DOWNLOAD_WEIGHTS", text)
        self.assertIn("HF_HUB_OFFLINE=\"${local_only}\"", text)
        self.assertIn("validate_pinned_assets 1", text)
        self.assertIn("sam2 imported from another checkout", text)
        self.assertIn("pip uninstall -y SAM-2", text)
        self.assertIn("--force-reinstall -e .", text)

    def test_gsam2_runner_prioritizes_the_current_pinned_checkout(self) -> None:
        text = (ROOT / "scripts" / "common.sh").read_text(encoding="utf-8")
        self.assertIn('local gsam2_pythonpath="${GS_ROOT}/external/Grounded-SAM-2:${PYTHONPATH:-}"', text)
        self.assertIn('PYTHONPATH="${gsam2_pythonpath}"', text)

    def test_query_pipeline_checks_external_model_and_sam2_provenance_first(self) -> None:
        text = (ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("check_grounded_sam2_import.py", text)
        self.assertIn("check_query_runtime.py", text)

    def test_provenance_checker_rejects_foreign_sam2_extension(self) -> None:
        text = (ROOT / "scripts" / "check_grounded_sam2_import.py").read_text(encoding="utf-8")
        self.assertIn("sam2._C resolved outside", text)

    def test_multi_instance_profile_keeps_full_sequence_tracking_opt_in(self) -> None:
        profiles = ROOT / "scripts" / "query_eval_profiles.sh"
        command = (
            f"source {profiles}; "
            "apply_query_eval_profile r4d_multi_instance_boundary_v6; "
            "printf '%s' \"${GSAM2_INSTANCE_FULL_SEQUENCE_TRACKS:-}\""
        )
        enabled = subprocess.check_output(["bash", "-lc", command], text=True)
        self.assertEqual(enabled, "1")

        command = (
            f"source {profiles}; "
            "apply_query_eval_profile public_time_boundary_gated_v5; "
            "printf '%s' \"${GSAM2_INSTANCE_FULL_SEQUENCE_TRACKS:-}\""
        )
        disabled = subprocess.check_output(["bash", "-lc", command], text=True)
        self.assertEqual(disabled, "")

    def test_multi_instance_profile_enables_only_the_declared_group_fast_path(self) -> None:
        profiles = ROOT / "scripts" / "query_eval_profiles.sh"
        command = (
            f"source {profiles}; "
            "apply_query_eval_profile r4d_multi_instance_boundary_v6; "
            "printf '%s' \"${QUERY_AUTO_SKIP_QWEN_FOR_DECLARED_MULTIHYPOTHESIS:-}\""
        )
        enabled = subprocess.check_output(["bash", "-lc", command], text=True)
        self.assertEqual(enabled, "1")

        command = (
            f"source {profiles}; "
            "apply_query_eval_profile public_time_boundary_gated_v5; "
            "printf '%s' \"${QUERY_AUTO_SKIP_QWEN_FOR_DECLARED_MULTIHYPOTHESIS:-}\""
        )
        disabled = subprocess.check_output(["bash", "-lc", command], text=True)
        self.assertEqual(disabled, "")

    def test_pipeline_gates_fast_path_on_lifted_instance_contract(self) -> None:
        text = (ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("check_declared_multi_instance.py", text)
        self.assertIn("--entitybank-path", text)
        self.assertIn("QUERY_SKIP_QWEN_SELECTION=1", text)

    def test_counted_instance_variants_are_suppressed_only_after_a_group_exists(self) -> None:
        text = (ROOT / "refergaussian/semantics/grounded_sam2_backend.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("resolved_counted_candidate_heads", text)
        self.assertIn("suppressed_instance_variant_phrases", text)
        self.assertIn("and phrase_head in resolved_counted_candidate_heads", text)


if __name__ == "__main__":
    unittest.main()
