#!/usr/bin/env python3
"""Re-render existing query entities without rerunning Stage-1 or Qwen.

This utility is intended for renderer ablations and release verification.  It
keeps the previously produced query-worldtube, Qwen selection, and Stage-1
tracks fixed, then writes evaluator-compatible masks under a separate output
root.  It never synthesizes a selection or falls back to a full-scene entity.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from refergaussian.semantics import render_hypernerf_query_video


def _read_manifest(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
            required = {"query_id", "dataset_dir", "output_root"}
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"Manifest row {line_number} is missing: {', '.join(missing)}")
            rows.append(row)
    if not rows:
        raise ValueError(f"No query rows found in {path}")
    return rows


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _paths_for_row(row: dict, source_output_root: Path | None, target_output_root: Path) -> tuple[Path, Path, Path, Path]:
    """Return source query root, source run, selection, and target render path."""
    query_id = str(row["query_id"])
    source_root = (source_output_root if source_output_root is not None else Path(str(row["output_root"]))) / query_id
    source_run_dir = source_root / "query_worldtube_run"
    selection_path = source_run_dir / "entitybank" / "selected_query_qwen.json"
    target_render_dir = target_output_root / query_id / "final_query_render_sourcebg"
    return source_root, source_run_dir, selection_path, target_render_dir


@contextmanager
def _render_environment(*, validation_only: bool, gpu: int | None) -> Iterator[None]:
    keys = (
        "CUDA_VISIBLE_DEVICES",
        "QUERY_FAST_VALIDATION_ONLY",
        "QUERY_SKIP_VIDEO_EXPORT",
        "QUERY_SKIP_OVERLAY_FRAME_EXPORT",
        "QUERY_SAVE_KEY_FRAMES",
        "QUERY_RENDER_ACTIVE_MASKS_ONLY",
    )
    previous = {key: os.environ.get(key) for key in keys}
    try:
        if gpu is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        if validation_only:
            os.environ["QUERY_FAST_VALIDATION_ONLY"] = "1"
            os.environ["QUERY_SKIP_VIDEO_EXPORT"] = "1"
            os.environ["QUERY_SKIP_OVERLAY_FRAME_EXPORT"] = "1"
            os.environ["QUERY_SAVE_KEY_FRAMES"] = "0"
            # The evaluators require an explicit binary mask for every rendered
            # frame, including an all-black mask when the query is inactive.
            os.environ["QUERY_RENDER_ACTIVE_MASKS_ONLY"] = "0"
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-render fixed query selections into a fresh evaluator-compatible output root."
    )
    parser.add_argument("--manifest", required=True, help="Original JSONL manifest used to create the selections.")
    parser.add_argument(
        "--output-root",
        required=True,
        help="New output root. Existing Stage-1/Qwen outputs are never modified.",
    )
    parser.add_argument(
        "--source-output-root",
        default=None,
        help="Override the manifest output_root that contains the existing query outputs.",
    )
    parser.add_argument("--profile", required=True, help="Render profile to apply to every query.")
    parser.add_argument("--query-id", action="append", dest="query_ids", help="Only re-render this official query id; repeat as needed.")
    parser.add_argument("--gpu", type=int, default=None, help="Optional visible GPU index for alpha-splat rendering.")
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--background-mode", choices=("source", "render"), default="source")
    parser.add_argument(
        "--export-visuals",
        action="store_true",
        help="Write overlay/lifecycle frames and videos. The default writes only numerical masks and validation.",
    )
    parser.add_argument("--force", action="store_true", help="Replace an existing target render directory.")
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit non-zero unless every selected query rendered successfully.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    source_output_root = None if args.source_output_root is None else Path(args.source_output_root).resolve()
    target_output_root = Path(args.output_root).resolve()
    requested_ids = None if not args.query_ids else {str(value) for value in args.query_ids}
    rows = _read_manifest(manifest_path)
    if requested_ids is not None:
        rows = [row for row in rows if str(row["query_id"]) in requested_ids]
        missing_ids = sorted(requested_ids - {str(row["query_id"]) for row in rows})
        if missing_ids:
            raise ValueError("Requested query ids absent from manifest: " + ", ".join(missing_ids))

    results: list[dict] = []
    query_root_map: dict[str, str] = {}
    dataset_dir_map: dict[str, str] = {}
    with _render_environment(validation_only=not args.export_visuals, gpu=args.gpu):
        for row in rows:
            query_id = str(row["query_id"])
            source_root, source_run_dir, selection_path, target_render_dir = _paths_for_row(
                row,
                source_output_root,
                target_output_root,
            )
            target_query_root = target_render_dir.parent
            record = {
                "query_id": query_id,
                "source_query_root": str(source_root),
                "source_run_dir": str(source_run_dir),
                "selection_path": str(selection_path),
                "target_query_root": str(target_query_root),
                "target_render_dir": str(target_render_dir),
                "dataset_dir": str(row["dataset_dir"]),
                "profile": args.profile,
                "started_at_utc": _utc_now(),
            }
            start = time.monotonic()
            try:
                if not source_run_dir.is_dir():
                    raise FileNotFoundError(f"Missing query-worldtube run: {source_run_dir}")
                if not selection_path.is_file():
                    raise FileNotFoundError(f"Missing Qwen selection: {selection_path}")
                if not Path(str(row["dataset_dir"])).is_dir():
                    raise FileNotFoundError(f"Missing dataset directory: {row['dataset_dir']}")
                if target_render_dir.exists():
                    if not args.force:
                        validation_path = target_render_dir / "validation.json"
                        if validation_path.is_file():
                            record["status"] = "skipped_existing"
                            record["validation_path"] = str(validation_path)
                            query_root_map[query_id] = str(target_query_root)
                            dataset_dir_map[query_id] = str(row["dataset_dir"])
                            continue
                        raise FileExistsError(
                            f"Target render directory exists without validation; pass --force: {target_render_dir}"
                        )
                    shutil.rmtree(target_render_dir)
                rendered_dir = render_hypernerf_query_video(
                    run_dir=source_run_dir,
                    dataset_dir=Path(str(row["dataset_dir"])),
                    selection_path=selection_path,
                    output_dir=target_render_dir,
                    fps=args.fps,
                    stride=args.stride,
                    background_mode=args.background_mode,
                    eval_profile=args.profile,
                )
                validation_path = rendered_dir / "validation.json"
                if not validation_path.is_file():
                    raise RuntimeError(f"Renderer completed without validation.json: {validation_path}")
                record["status"] = "ok"
                record["validation_path"] = str(validation_path)
                query_root_map[query_id] = str(target_query_root)
                dataset_dir_map[query_id] = str(row["dataset_dir"])
            except Exception as exc:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                print(f"[rerender] {query_id}: {record['error']}", file=sys.stderr)
            finally:
                record["elapsed_seconds"] = round(time.monotonic() - start, 3)
                record["finished_at_utc"] = _utc_now()
                results.append(record)

    target_output_root.mkdir(parents=True, exist_ok=True)
    _write_json(target_output_root / "rerender_summary.json", {
        "manifest": str(manifest_path),
        "source_output_root_override": None if source_output_root is None else str(source_output_root),
        "profile": args.profile,
        "validation_only": not bool(args.export_visuals),
        "results": results,
    })
    _write_json(target_output_root / "query_root_map.json", query_root_map)
    _write_json(target_output_root / "dataset_dir_map.json", dataset_dir_map)

    failures = [record for record in results if record.get("status") == "failed"]
    complete = len(results) == len(rows) and not failures
    print(
        f"Re-rendered {len(results) - len(failures)}/{len(rows)} queries; "
        f"output={target_output_root}; complete={complete}"
    )
    return 0 if (not args.require_complete or complete) else 2


if __name__ == "__main__":
    raise SystemExit(main())
