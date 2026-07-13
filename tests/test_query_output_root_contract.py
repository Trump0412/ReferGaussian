"""Regression tests for one coherent output root per batch query."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class QueryOutputRootContractTest(unittest.TestCase):
    def test_stage_one_honors_the_batch_output_root_override(self) -> None:
        text = (ROOT / "scripts" / "run_query_guided_grounded_sam2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn('if [[ -n "${QUERY_OUTPUT_ROOT_OVERRIDE:-}" ]]', text)
        self.assertIn('OUTPUT_ROOT="${QUERY_OUTPUT_ROOT_OVERRIDE}"', text)
        self.assertIn('OUTPUT_ROOT="${GS_ROOT}/${QUERY_OUTPUT_ROOT_OVERRIDE}"', text)

    def test_pipeline_and_stage_one_share_the_override_contract(self) -> None:
        pipeline = (ROOT / "scripts" / "run_query_specific_worldtube_pipeline.sh").read_text(
            encoding="utf-8"
        )
        stage_one = (ROOT / "scripts" / "run_query_guided_grounded_sam2.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("QUERY_OUTPUT_ROOT_OVERRIDE", pipeline)
        self.assertIn("QUERY_OUTPUT_ROOT_OVERRIDE", stage_one)


if __name__ == "__main__":
    unittest.main()
