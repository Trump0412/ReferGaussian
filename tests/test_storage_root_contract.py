"""Regression tests for relocatable data and run roots."""

from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class StorageRootContractTest(unittest.TestCase):
    def test_common_uses_configurable_data_root_for_each_dataset_layout(self) -> None:
        common = ROOT / "scripts" / "common.sh"
        command = (
            "export GS_DATA_ROOT=/tmp/refergaussian-data; "
            f"source {common}; "
            "dataset_source_path hypernerf interp/torchocolate; printf '\\n'; "
            "dataset_source_path dynerf coffee_martini"
        )
        output = subprocess.check_output(["bash", "-lc", command], text=True)
        self.assertEqual(
            output,
            "/tmp/refergaussian-data/hypernerf/interp/torchocolate\n"
            "/tmp/refergaussian-data/dynerf/coffee_martini",
        )

    def test_train_and_eval_wrappers_use_configurable_run_root(self) -> None:
        expected = "${GS_RUN_ROOT}/${RUN_NAMESPACE}/${DATASET}/${SCENE##*/}"
        for name in ("train.sh", "train_baseline.sh", "eval.sh", "eval_baseline.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn(expected, text, name)

    def test_hypernerf_preparation_uses_data_root_and_fails_without_point_cloud(self) -> None:
        text = (ROOT / "scripts" / "prepare_hypernerf.sh").read_text(encoding="utf-8")
        self.assertIn('DATA_ROOT="${GS_DATA_ROOT}/hypernerf"', text)
        self.assertIn('DOWNLOAD_ROOT="${GS_DATA_ROOT}/downloads"', text)
        self.assertIn("COLMAP is required to finish", text)
        self.assertIn('if [[ ! -s "${target_dir}/points3D_downsample2.ply" ]]', text)

    def test_matched_training_wrappers_fix_the_release_iteration_budget(self) -> None:
        for name in ("train.sh", "train_baseline.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("REFERGAUSSIAN_ITERATIONS:-14000", text, name)
            self.assertIn("--iterations '${TRAIN_ITERATIONS}'", text, name)
            self.assertIn("iterations: ${TRAIN_ITERATIONS}", text, name)

    def test_full_metric_eval_streams_only_the_test_camera_split(self) -> None:
        for name in ("eval.sh", "eval_baseline.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            for option in (
                "--skip_train",
                "--skip_video",
                "--only_split test",
                "--stream_write",
                "--no_video_file",
            ):
                self.assertIn(option, text, f"{name}: {option}")


if __name__ == "__main__":
    unittest.main()
