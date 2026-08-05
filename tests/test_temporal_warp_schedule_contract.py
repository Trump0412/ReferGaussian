"""Regression tests for the shared temporal optimization schedule."""

from __future__ import annotations

import unittest
from pathlib import Path

from refergaussian.temporal.warp_io import set_temporal_warp_learning_rate


ROOT = Path(__file__).resolve().parents[1]


class _Optimizer:
    def __init__(self) -> None:
        self.param_groups = [{"lr": 0.1}, {"lr": 0.2}]


class TemporalWarpScheduleContractTest(unittest.TestCase):
    def test_schedule_updates_every_warp_parameter_group(self) -> None:
        optimizer = _Optimizer()

        set_temporal_warp_learning_rate(optimizer, 1.2e-4)

        self.assertEqual([group["lr"] for group in optimizer.param_groups], [1.2e-4, 1.2e-4])

    def test_external_patch_keeps_warp_schedule_explicit_and_independent(self) -> None:
        text = (ROOT / "patches" / "4dgaussians_temporal_warp_schedule.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('hyper.temporal_warp_lr_schedule == "shared_exponential"', text)
        self.assertIn("gaussians.temporal_scheduler_args(iteration)", text)
        self.assertIn(
            "+        temporal_warp, hyper.temporal_warp_lr_init",
            text,
        )
        argument_patch = (ROOT / "patches" / "4dgaussians_refergaussian.patch").read_text(
            encoding="utf-8"
        )
        self.assertIn('self.temporal_warp_lr_schedule = "constant"', argument_patch)
        self.assertIn("self.temporal_warp_lr_init = 0.00016", argument_patch)

        train_wrapper = (ROOT / "scripts" / "train.sh").read_text(encoding="utf-8")
        self.assertIn("TEMPORAL_WARP_LR_INIT", train_wrapper)
        self.assertIn("TEMPORAL_WARP_LR_SCHEDULE", train_wrapper)
        self.assertIn("temporal_warp_lr_schedule:", train_wrapper)


if __name__ == "__main__":
    unittest.main()
