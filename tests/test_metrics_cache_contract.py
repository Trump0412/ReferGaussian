"""Release contracts for metric caching without changing metric definitions."""

from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class MetricsCacheContractTest(unittest.TestCase):
    def test_external_metric_patch_caches_both_lpips_networks(self) -> None:
        text = (ROOT / "patches" / "4dgaussians_metrics_cache.patch").read_text(encoding="utf-8")
        self.assertIn("LPIPS(net_type='vgg').cuda()", text)
        self.assertIn("LPIPS(net_type='alex').cuda()", text)
        self.assertIn("with torch.no_grad():", text)

    def test_public_metric_helpers_reuse_one_vgg_network(self) -> None:
        for name in ("fullframe_metrics.py", "quick_subset_metrics.py"):
            text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
            self.assertIn("LPIPS(net_type=\"vgg\")", text)
            self.assertNotIn("lpips(render, gt", text)


if __name__ == "__main__":
    unittest.main()
