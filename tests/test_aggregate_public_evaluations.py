"""Tests for complete, query-weighted public-result aggregation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "aggregate_public_query_evaluations.py"
SPEC = importlib.util.spec_from_file_location("refergaussian_public_aggregate", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AGGREGATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AGGREGATOR)


class AggregatePublicEvaluationsTest(unittest.TestCase):
    def test_incomplete_input_cannot_be_aggregated_as_complete(self) -> None:
        output = AGGREGATOR.aggregate_payloads(
            [
                {
                    "coverage": {"expected_queries": 2, "missing_query_ids": ["scene_q2"]},
                    "queries": [
                        {
                            "query_slug": "scene_q1",
                            "Acc": 1.0,
                            "vIoU": 1.0,
                            "temporal_tIoU": 1.0,
                            "temporal_precision": 1.0,
                            "temporal_recall": 1.0,
                            "temporal_gt_active_count": 1,
                        },
                        {"query_slug": "scene_q2", "Acc": None},
                    ],
                }
            ],
            expected_queries=2,
        )

        self.assertFalse(output["summary"]["complete"])
        self.assertEqual(output["missing_query_ids"], ["scene_q2"])

    def test_different_metric_protocols_cannot_be_mixed(self) -> None:
        payload = {
            "coverage": {"expected_queries": 1, "missing_query_ids": []},
            "queries": [{"query_slug": "scene_q1", "Acc": 1.0}],
        }

        with self.assertRaisesRegex(ValueError, "different metric protocols"):
            AGGREGATOR.aggregate_payloads(
                [
                    {**payload, "metric_protocol": {"id": "protocol_a"}},
                    {
                        "coverage": {"expected_queries": 1, "missing_query_ids": []},
                        "queries": [{"query_slug": "scene_q2", "Acc": 1.0}],
                        "metric_protocol": {"id": "protocol_b"},
                    },
                ],
                expected_queries=2,
            )


if __name__ == "__main__":
    unittest.main()
