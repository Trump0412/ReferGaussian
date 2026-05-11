import argparse
import os
import sys

import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPSTREAM_ROOT = os.path.join(REPO_ROOT, "external", "4DGaussians")
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, UPSTREAM_ROOT)

from refergaussian.temporal import build_temporal_warp, load_temporal_warp
from refergaussian.temporal.warp_viz import save_warp_artifacts


def load_cfg_args(run_dir):
    cfg_path = os.path.join(run_dir, "cfg_args")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing cfg_args in {run_dir}")
    with open(cfg_path, "r", encoding="utf-8") as handle:
        return eval(handle.read(), {"Namespace": argparse.Namespace, "__builtins__": {}})  # noqa: S307


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()

    cfg_args = load_cfg_args(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warp = build_temporal_warp(cfg_args, device=device)
    if not load_temporal_warp(warp, args.run_dir, iteration=args.iteration):
        raise FileNotFoundError(f"No temporal warp weights found under {args.run_dir}")
    output_dir = args.output_dir or os.path.join(args.run_dir, "temporal_warp", "latest")
    save_warp_artifacts(warp, output_dir)


if __name__ == "__main__":
    main()
