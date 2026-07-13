#!/usr/bin/env python3
"""Export renderer-consistent Gaussian states for query lifting or rendering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refergaussian.semantics.renderer_geometry import (
    _source_entries,
    export_renderer_geometry,
    frame_requests_from_tracks,
    gaussian_ids_from_entitybank,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the real fine-stage ReferGaussian state at source-camera times. "
            "This is a geometry cache, not a 2D mask export."
        )
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    frames = parser.add_mutually_exclusive_group(required=True)
    frames.add_argument("--tracks-path", help="Export active Stage-1 track frames.")
    frames.add_argument("--all-dataset-frames", action="store_true", help="Export every source-camera frame.")
    parser.add_argument(
        "--entitybank-dir",
        default=None,
        help="Restrict export to the union of Gaussian ids selected in entities.json.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    if args.tracks_path:
        frame_requests = frame_requests_from_tracks(args.tracks_path)
    else:
        frame_requests = _source_entries(dataset_dir)
    gaussian_ids = None if args.entitybank_dir is None else gaussian_ids_from_entitybank(args.entitybank_dir)
    output_dir = export_renderer_geometry(
        run_dir=Path(args.run_dir),
        dataset_dir=dataset_dir,
        output_dir=Path(args.output_dir),
        frame_requests=frame_requests,
        gaussian_ids=gaussian_ids,
        overwrite=bool(args.overwrite),
    )
    print(output_dir)


if __name__ == "__main__":
    main()
