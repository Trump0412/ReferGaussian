import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
EXTERNAL_ROOT = REPO_ROOT / "external" / "4DGaussians"
for candidate in (REPO_ROOT, EXTERNAL_ROOT):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from refergaussian.semantics.query_render import render_hypernerf_query_video


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def _read_simple_yaml(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        payload[key.strip()] = value.strip()
    return payload


def _resolve_test_times(run_dir: Path) -> np.ndarray:
    config = _read_simple_yaml(run_dir / "config.yaml")
    dataset_dir = Path(config.get("source_path", ""))

    # HyperNeRF path: dataset.json + metadata.json
    if dataset_dir and (dataset_dir / "dataset.json").exists() and (dataset_dir / "metadata.json").exists():
        dataset_payload = _read_json(dataset_dir / "dataset.json")
        metadata_payload = _read_json(dataset_dir / "metadata.json")
        all_ids = list(dataset_payload["ids"])
        if dataset_payload.get("val_ids"):
            val_ids = set(dataset_payload["val_ids"])
            test_ids = [image_id for image_id in all_ids if image_id in val_ids]
        else:
            i_train = [index for index in range(len(all_ids)) if index % 4 == 0]
            i_test = (np.asarray(i_train, dtype=np.int64) + 2)[:-1]
            test_ids = [all_ids[int(index)] for index in i_test]
        max_time = max(float(metadata_payload[image_id]["warp_id"]) for image_id in all_ids)
        return np.asarray(
            [float(metadata_payload[image_id]["warp_id"]) / max(max_time, 1.0) for image_id in test_ids],
            dtype=np.float32,
        )

    # DyNeRF / generic path: derive test times from rendered test frames in the run dir
    gt_candidates = sorted((run_dir / "test").glob("ours_*/renders"))
    if gt_candidates:
        render_files = sorted(gt_candidates[-1].glob("*.png"))
        n = len(render_files)
        if n <= 1:
            return np.zeros((n,), dtype=np.float32)
        return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    # Fallback: use all frames from the dataset directory
    if dataset_dir and dataset_dir.exists():
        # DyNeRF: look for cam*/images
        cam_dirs = sorted(dataset_dir.glob("cam*/images"))
        if cam_dirs:
            image_files = sorted(cam_dirs[0].glob("*.png"))
            n = len(image_files)
            if n <= 1:
                return np.zeros((n,), dtype=np.float32)
            return np.linspace(0.0, 1.0, num=n, dtype=np.float32)

    raise FileNotFoundError(
        f"Cannot resolve test times for run_dir={run_dir}. "
        "Expected HyperNeRF dataset.json/metadata.json or DyNeRF test renders."
    )


def _sample_time_values(run_dir: Path) -> np.ndarray:
    payload = np.load(run_dir / "entitybank" / "trajectory_samples.npz")
    return payload["time_values"].astype(np.float32)


def _resample_mask(sample_mask: np.ndarray, sample_times: np.ndarray, test_times: np.ndarray) -> np.ndarray:
    sample_mask = np.asarray(sample_mask, dtype=np.float32)
    indices = np.abs(test_times[:, None] - sample_times[None, :]).argmin(axis=1)
    return sample_mask[indices] > 0.5


def _ranges_from_mask(mask: np.ndarray) -> list[list[int]]:
    indices = np.where(np.asarray(mask, dtype=bool))[0]
    if indices.size == 0:
        return []
    ranges: list[list[int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        ranges.append([start, prev])
        start = value
        prev = value
    ranges.append([start, prev])
    return ranges


def _entity_segments(assignment: dict[str, Any], sample_times: np.ndarray, test_times: np.ndarray) -> list[list[int]]:
    sample_mask = np.zeros((sample_times.shape[0],), dtype=bool)
    support_window = assignment.get("support_window", {})
    frame_start = int(support_window.get("frame_start", 0))
    frame_end = int(support_window.get("frame_end", 0))
    if frame_end > frame_start:
        sample_mask[max(frame_start, 0) : min(frame_end, sample_times.shape[0])] = True
    for phase_name in ("moving", "stationary"):
        for segment in assignment.get("phase_segments", {}).get(phase_name, []):
            if len(segment) != 2:
                continue
            start = max(int(segment[0]), 0)
            end = min(int(segment[1]) + 1, sample_times.shape[0])
            if end > start:
                sample_mask[start:end] = True
    if not sample_mask.any():
        sample_mask[:] = True
    return _ranges_from_mask(_resample_mask(sample_mask.astype(np.float32), sample_times, test_times))


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "entity"


def build_entity_library(
    run_dir: Path,
    dataset_dir: Path,
    assignments_path: Path,
    output_root: Path,
    fps: int,
    stride: int,
    background_mode: str,
) -> Path:
    assignments_payload = _read_json(assignments_path)
    assignments = assignments_payload.get("assignments", [])
    sample_times = _sample_time_values(run_dir)
    test_times = _resolve_test_times(run_dir)

    output_root.mkdir(parents=True, exist_ok=True)
    index_rows = []

    for assignment in assignments:
        entity_id = int(assignment["entity_id"])
        label_text = assignment.get("qwen_text", {}).get("category") or assignment.get("native_text", {}).get("static_text") or assignment.get("entity_type", "entity")
        entity_dir = output_root / f"entity_{entity_id:04d}_{_slugify(str(label_text))[:48]}"
        entity_dir.mkdir(parents=True, exist_ok=True)

        segments = _entity_segments(assignment, sample_times, test_times)
        selection_payload = {
            "selected": [
                {
                    "id": entity_id,
                    "role": "entity",
                    "entity_type": assignment.get("entity_type", "entity"),
                    "confidence": float(assignment.get("quality", 0.0)),
                    "segments": segments,
                }
            ],
            "empty": False,
            "reason": "",
            "query_slots": {
                "query": f"entity {entity_id}",
            },
            "semantic_source": assignments_payload.get("semantic_source", "entity_library"),
        }
        selection_path = entity_dir / "selected_entity.json"
        _write_json(selection_path, selection_payload)

        semantic_summary = {
            "entity_id": entity_id,
            "semantic_source": assignment.get("semantic_source", assignments_payload.get("semantic_source")),
            "qwen_enabled": bool(assignment.get("qwen_enabled", False)),
            "entity_type": assignment.get("entity_type"),
            "semantic_head": assignment.get("semantic_head"),
            "temporal_mode": assignment.get("temporal_mode"),
            "quality": float(assignment.get("quality", 0.0)),
            "support_window": assignment.get("support_window", {}),
            "role_scores": assignment.get("role_scores", {}),
            "concept_tags": assignment.get("concept_tags", []),
            "semantic_terms": assignment.get("semantic_terms", []),
            "native_text": assignment.get("native_text", {}),
            "qwen_text": assignment.get("qwen_text", {}),
            "qwen_temporal_segments": assignment.get("qwen_temporal_segments", []),
            "prompt_groups": assignment.get("prompt_groups", {}),
            "interaction_partners": assignment.get("interaction_partners", []),
            "segments_test_frames": segments,
            "source_assignment_path": str(assignments_path),
        }
        semantic_path = entity_dir / "semantic_summary.json"
        _write_json(semantic_path, semantic_summary)

        render_dir = entity_dir / f"rendered_{background_mode}"
        if not (render_dir / "overlay.mp4").exists() or not (render_dir / "validation.json").exists():
            render_dir = render_hypernerf_query_video(
                run_dir=run_dir,
                dataset_dir=dataset_dir,
                selection_path=selection_path,
                output_dir=render_dir,
                fps=fps,
                stride=stride,
                background_mode=background_mode,
            )

        index_rows.append(
            {
                "entity_id": entity_id,
                "entity_dir": str(entity_dir),
                "selection_path": str(selection_path),
                "semantic_summary_path": str(semantic_path),
                "render_dir": str(render_dir),
                "qwen_enabled": bool(assignment.get("qwen_enabled", False)),
                "label": str(label_text),
                "entity_type": assignment.get("entity_type", "entity"),
                "semantic_head": assignment.get("semantic_head", "unknown"),
                "quality": float(assignment.get("quality", 0.0)),
                "concept_tags": assignment.get("concept_tags", [])[:12],
                "global_desc": assignment.get("qwen_text", {}).get("global_desc")
                or assignment.get("native_text", {}).get("global_desc", ""),
            }
        )

    index_payload = {
        "schema_version": 1,
        "run_dir": str(run_dir),
        "dataset_dir": str(dataset_dir),
        "assignments_path": str(assignments_path),
        "num_entities": len(index_rows),
        "background_mode": background_mode,
        "fps": int(fps),
        "stride": int(stride),
        "entities": index_rows,
    }
    _write_json(output_root / "entity_index.json", index_payload)

    lines = [
        "# Entity Library",
        "",
        f"- run_dir: `{run_dir}`",
        f"- assignments: `{assignments_path}`",
        f"- num_entities: `{len(index_rows)}`",
        "",
        "| entity_id | qwen | label | semantic_head | quality | render_dir |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in index_rows:
        lines.append(
            f"| {row['entity_id']} | {row['qwen_enabled']} | {row['label']} | {row['semantic_head']} | {row['quality']:.4f} | `{row['render_dir']}` |"
        )
    (output_root / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-dir", required=True)
    parser.add_argument("--assignments-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--fps", type=int, default=6)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--background-mode", choices=["render", "source"], default="source")
    args = parser.parse_args()

    output_root = build_entity_library(
        run_dir=Path(args.run_dir),
        dataset_dir=Path(args.dataset_dir),
        assignments_path=Path(args.assignments_path),
        output_root=Path(args.output_root),
        fps=args.fps,
        stride=args.stride,
        background_mode=args.background_mode,
    )
    print(output_root)


if __name__ == "__main__":
    main()
