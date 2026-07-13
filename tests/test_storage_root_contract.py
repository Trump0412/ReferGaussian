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
        for name in ("train.sh", "train_baseline.sh", "eval.sh", "eval_baseline.sh", "eval_baseline_subset.sh"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn(expected, text, name)


if __name__ == "__main__":
    unittest.main()
