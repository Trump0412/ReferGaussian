#!/usr/bin/env python3
"""Lift synchronized Stage-1 masks into a training-free Gaussian entity set."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = PROJECT_ROOT / "external" / "4DGaussians"
for candidate in (PROJECT_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.mask_supported_lifting import (
    build_mask_supported_lifting_proposal_dir,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--tracks-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-track-frames", type=int, default=24)
    parser.add_argument("--min-gaussians", type=int, default=192)
    parser.add_argument("--max-gaussians", type=int, default=1280)
    parser.add_argument("--max-gaussians-per-frame", type=int, default=18000)
    parser.add_argument("--gate-threshold", type=float, default=0.01)
    parser.add_argument("--graph-knn", type=int, default=24)
    parser.add_argument("--graph-radius-scale", type=float, default=1.35)
    args = parser.parse_args()

    output_dir = build_mask_supported_lifting_proposal_dir(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        tracks_path=Path(args.tracks_path),
        output_dir=Path(args.output_dir),
        max_track_frames=int(args.max_track_frames),
        min_gaussians=int(args.min_gaussians),
        max_gaussians=int(args.max_gaussians),
        max_gaussians_per_frame=int(args.max_gaussians_per_frame),
        gate_threshold=float(args.gate_threshold),
        graph_knn=int(args.graph_knn),
        graph_radius_scale=float(args.graph_radius_scale),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
