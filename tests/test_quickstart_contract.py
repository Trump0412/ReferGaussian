"""Contracts for the concise public Quick Start entrypoints."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "scripts" / "train_4dgs.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_train_4dgs", TRAIN_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)


class QuickStartContractTest(unittest.TestCase):
    def _dry_run(self, benchmark: str) -> str:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            return subprocess.check_output(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_benchmark.py"),
                    benchmark,
                    "--gpus",
                    "0",
                    "2",
                    "--data-root",
                    str(root / "data"),
                    "--run-root",
                    str(root / "runs"),
                    "--output",
                    str(root / "output"),
                    "--dry-run",
                ],
                cwd=ROOT,
                text=True,
            )

    def test_public_entrypoint_expands_strict_dynamic_protocol(self) -> None:
        output = self._dry_run("4dlangsplat")

        self.assertIn("release_public4_extension", output)
        self.assertIn("public_time_boundary_gated_v5_numeric", output)
        self.assertIn("aggregate_public_query_evaluations.py", output)
        self.assertNotIn("time_agnostic", output)

    def test_r4d_entrypoint_expands_strict_dense_protocol(self) -> None:
        output = self._dry_run("r4d-bench")

        self.assertIn("release_r4d_dense89_renderer_consistent", output)
        self.assertIn("rerender_query_outputs.py", output)
        self.assertIn("evaluate_ours_benchmark.py", output)

    def test_standard_4dgs_wrapper_uses_the_documented_layout(self) -> None:
        args = argparse.Namespace(
            dataset="hypernerf",
            scene="misc/americano",
            source=None,
            output=None,
            config=None,
            data_root=Path("/tmp/data"),
            run_root=Path("/tmp/runs"),
            gpu=0,
            port=6009,
            render_only=False,
            dry_run=True,
        )

        source, output, commands = TRAIN.build_commands(args)

        self.assertEqual(source, Path("/tmp/data/hypernerf/misc/americano"))
        self.assertEqual(output, Path("/tmp/runs/baseline_4dgs/hypernerf/americano"))
        self.assertTrue(commands[0][1].endswith("external/4DGaussians/train.py"))
        self.assertTrue(commands[1][1].endswith("external/4DGaussians/render.py"))

    def test_readme_and_page_embed_the_method_figure(self) -> None:
        method_figure = ROOT / "docs" / "assets" / "framework.png"
        self.assertTrue(method_figure.is_file())
        self.assertEqual(method_figure.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        page = (ROOT / "docs" / "index.html").read_text(encoding="utf-8")

        self.assertIn("docs/assets/framework.png", readme)
        self.assertIn("assets/framework.png", page)
        self.assertNotIn("method_overview.svg", readme + page)
        self.assertNotIn("Accepted at", page)
        self.assertNotRegex(readme.lower(), r"time[-_ ]agnostic")


if __name__ == "__main__":
    unittest.main()
