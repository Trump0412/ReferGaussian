#!/usr/bin/env python3
"""Export renderer-consistent Gaussian states for query lifting or rendering."""

from __future__ import annotations

import argparse
import json
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


def _frame_requests_from_image_ids(path: str | Path, dataset_dir: Path) -> list[dict]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    values = payload.get("image_ids", []) if isinstance(payload, dict) else payload
    if not isinstance(values, list) or not values:
        raise ValueError(f"No image_ids found in {path}")
    requested = {str(value).strip() for value in values if str(value).strip()}
    entries = _source_entries(dataset_dir)
    available = {str(entry["image_id"]) for entry in entries}
    missing = sorted(requested - available)
    if missing:
        raise FileNotFoundError(
            "Requested renderer-geometry image ids are absent from the dataset: "
            + ", ".join(missing[:8])
        )
    return [entry for entry in entries if str(entry["image_id"]) in requested]


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
    frames.add_argument(
        "--image-ids-json",
        help="Export only the exact source-camera image ids declared by an evaluation protocol.",
    )
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
    elif args.image_ids_json:
        frame_requests = _frame_requests_from_image_ids(args.image_ids_json, dataset_dir)
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
