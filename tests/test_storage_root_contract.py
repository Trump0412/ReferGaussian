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

    def test_hypernerf_preparation_uses_data_root_and_fails_without_point_cloud(self) -> None:
        text = (ROOT / "scripts" / "prepare_hypernerf.sh").read_text(encoding="utf-8")
        self.assertIn('DATA_ROOT="${GS_DATA_ROOT}/hypernerf"', text)
        self.assertIn('DOWNLOAD_ROOT="${GS_DATA_ROOT}/downloads"', text)
        self.assertIn("COLMAP is required to finish", text)
        self.assertIn('if [[ ! -s "${target_dir}/points3D_downsample2.ply" ]]', text)

    def test_manifest_builders_default_to_frozen_4dgs_inputs(self) -> None:
        for name in ("build_public_query_manifest.py", "build_r4d_query_manifest.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn('"baseline_4dgs"', text, name)
            self.assertNotIn('"refergaussian/hypernerf/', text, name)


if __name__ == "__main__":
    unittest.main()
