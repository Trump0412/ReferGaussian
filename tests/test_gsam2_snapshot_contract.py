"""Regression tests for pinned offline Grounded-SAM2 runtime loading."""

from __future__ import annotations

import unittest
from pathlib import Path


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
        self.assertIn("local_files_only=True", text)
        self.assertIn("GSAM2_INSTALL_EDITABLE:-1", text)
        self.assertIn("sam2 imported from another checkout", text)

    def test_query_pipeline_checks_external_model_and_sam2_provenance_first(self) -> None:
        text = (ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("check_grounded_sam2_import.py", text)
        self.assertIn("check_query_runtime.py", text)

    def test_provenance_checker_rejects_foreign_sam2_extension(self) -> None:
        text = (ROOT / "scripts" / "check_grounded_sam2_import.py").read_text(encoding="utf-8")
        self.assertIn("sam2._C resolved outside", text)


if __name__ == "__main__":
    unittest.main()
