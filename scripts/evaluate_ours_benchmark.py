#!/usr/bin/env python3
"""
evaluate_ours_benchmark.py

评测脚本：读取 Ours_benchmark.json 作为 ground-truth，
评测 ReferGaussian query pipeline 的输出结果。
指标：Acc（时序准确率）、vIoU（视频 mask IoU）、tIoU（时序 IoU）。

用法:
  python scripts/evaluate_ours_benchmark.py \
    --benchmark data/benchmarks/r4d_bench_qa/benchmark.json \
    --query-root-map /path/to/query_root_map.json \
    --output-json reports/ours_benchmark_eval.json \
    [--output-md reports/ours_benchmark_eval.md]

query_root_map.json 格式:
{
  "espresso_q1": "/path/to/run_dir/entitybank/query_guided/espresso_q1",
  ...
}
"""
from __future__ import annotations

import argparse
import bisect
import json
import struct
import zlib
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# RLE (COCO format) decoder
# ---------------------------------------------------------------------------

def _decode_rle(rle: dict) -> np.ndarray:
    """Decode COCO RLE mask to binary numpy array (H, W)."""
    counts_raw = rle["counts"]
    size = rle["size"]  # [H, W]
    h, w = int(size[0]), int(size[1])

    if isinstance(counts_raw, str):
        # Encoded RLE string (COCO binary compressed RLE)
        counts = _decode_rle_string(counts_raw, h * w)
    else:
        counts = [int(c) for c in counts_raw]

    mask = np.zeros(h * w, dtype=np.uint8)
    pos = 0
    val = 0
    for count in counts:
        mask[pos: pos + count] = val
        pos += count
        val = 1 - val

    # COCO RLE is column-major (Fortran order)
    return mask.reshape((h, w), order="F").astype(bool)


def _decode_rle_string(encoded: str, n: int) -> list[int]:
    """Decode COCO compressed RLE string into run-length list."""
    counts = []
    m = 0
    p = 0
    while p < len(encoded):
        x = 0
        k = 0
        more = True
        while more:
            c = ord(encoded[p]) - 48
            p += 1
            x |= (c & 0x1f) << (5 * k)
            more = (c & 0x20) != 0
            k += 1
        if x & 1:
            x = ~(x >> 1)
        else:
            x = x >> 1
        if m > 0:
            x += counts[m - 1]
        counts.append(x)
        m += 1
    return counts


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else 0.0


def _mask_iou(pred: np.ndarray, gt: np.ndarray) -> float:
    p = np.asarray(pred, dtype=bool)
    g = np.asarray(gt, dtype=bool)
    inter = float(np.logical_and(p, g).sum())
    union = float(np.logical_or(p, g).sum())
    return _safe_div(inter, union)


def _decode_segmentation(seg: dict | list, image_size: tuple[int, int] | None = None) -> np.ndarray | None:
    """Decode COCO segmentation (RLE dict or polygon list) to binary numpy array (H, W).
    
    Args:
        seg: Either an RLE dict {"counts": ..., "size": [H, W]} or a polygon list [[x1, y1, ...]].
        image_size: (H, W) tuple, required for polygon format when size is not in seg.
    
    Returns:
        Binary (bool) numpy array of shape (H, W), or None if decoding fails.
    """
    if isinstance(seg, dict):
        # COCO RLE format
        try:
            return _decode_rle(seg)
        except Exception:
            return None
    elif isinstance(seg, list):
        # COCO polygon format: [[x1, y1, x2, y2, ...], ...]
        # Determine image size
        h, w = None, None
        if image_size is not None:
            h, w = image_size
        if h is None or w is None:
            return None
        from PIL import Image as PILImage, ImageDraw
        mask_img = PILImage.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask_img)
        for polygon in seg:
            if len(polygon) < 6:
                continue  # Need at least 3 points
            pts = [(polygon[i], polygon[i + 1]) for i in range(0, len(polygon), 2)]
            draw.polygon(pts, fill=255)
        return np.asarray(mask_img) > 0
    return None


def _load_metadata(dataset_dir: Path) -> dict[str, int]:
    """Load HyperNeRF metadata.json -> image_id: time_id mapping."""
    meta_path = dataset_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    raw = _read_json(meta_path)
    return {k: int(v["time_id"]) for k, v in raw.items()}


def _build_image_id_to_frame_index(validation_frames: list[dict]) -> dict[str, int]:
    """Build mapping from image_id → frame_index in validation output."""
    return {str(f["image_id"]): int(f["frame_index"]) for f in validation_frames}


def _frame_id_to_image_id_hypernerf(frame_id: int) -> str:
    """HyperNeRF: frame_id → 6-digit image_id string."""
    return f"{int(frame_id):06d}"


def _frame_id_to_image_id_dynerf(frame_id: int) -> str:
    """dynerf: frame_id → 4-digit image_id string (cam00/images/XXXX.png)."""
    return f"{int(frame_id):04d}"


def _detect_dataset_type(dataset_dir: Path) -> str:
    """Detect if this is a hypernerf or dynerf dataset."""
    if (dataset_dir / "metadata.json").exists():
        return "hypernerf"
    if (dataset_dir / "poses_bounds.npy").exists():
        return "dynerf"
    # Try to detect from config.yaml in parent dirs
    return "hypernerf"


def _infer_dataset_info_from_query_output(query_output_dir: Path) -> tuple[str | None, Path | None]:
    """Infer dataset type and dataset_dir from query output path.

    Expected path pattern:
      .../runs/<namespace>/<dataset_type>/<scene>/entitybank/query_guided/<qid>
    """
    parts = query_output_dir.parts
    ds_type = None
    scene_name = None
    if "runs" in parts:
        idx = parts.index("runs")
        if idx + 3 < len(parts):
            cand = parts[idx + 2]
            if cand in ("hypernerf", "dynerf", "dnerf"):
                ds_type = cand
                scene_name = parts[idx + 3]

    if ds_type is None:
        return None, None

    # Infer dataset root for known benchmark scenes.
    repo_root = Path(__file__).resolve().parents[1]
    if ds_type == "dynerf":
        if scene_name:
            return ds_type, repo_root / "data" / "dynerf" / scene_name
        return ds_type, None

    if ds_type == "hypernerf":
        # Benchmark scenes are split across misc/interp. Infer from scene name.
        scene_to_subdir = {
            "espresso": ("misc", "espresso"),
            "americano": ("misc", "americano"),
            "split-cookie": ("misc", "split-cookie"),
            "keyboard": ("misc", "keyboard"),
            "cut-lemon1": ("interp", "cut-lemon1"),
            "torchocolate": ("interp", "torchocolate"),
        }
        if scene_name in scene_to_subdir:
            sub, name = scene_to_subdir[scene_name]
            return ds_type, repo_root / "data" / "hypernerf" / sub / name
        if scene_name:
            # Best-effort fallback for other HyperNeRF layouts.
            return ds_type, repo_root / "data" / "hypernerf" / "misc" / scene_name
        return ds_type, None

    return ds_type, None


# ---------------------------------------------------------------------------
# Per-query evaluation
# ---------------------------------------------------------------------------

def evaluate_query(
    query_item: dict,
    query_output_dir: Path,
    dataset_dir: Path | None = None,
) -> dict:
    """Evaluate a single query against Ours_benchmark.json ground truth."""
    query_id = str(query_item["query_id"])
    gt = query_item.get("ground_truth", {})
    existence_frames: list[int] = [int(f) for f in gt.get("existence_frames", [])]
    gt_frames: list[dict] = gt.get("frames", [])

    # -----------------------------------------------------------------------
    # Load validation.json
    # -----------------------------------------------------------------------
    validation_path = query_output_dir / "final_query_render_sourcebg" / "validation.json"
    if not validation_path.exists():
        return {
            "query_id": query_id,
            "status": "missing_validation",
            "Acc": None,
            "vIoU": None,
            "tIoU": None,
        }

    validation = _read_json(validation_path)
    val_frames = validation.get("frames", [])
    binary_mask_dir = Path(validation.get("frame_exports", {}).get("binary_masks", ""))

    if not val_frames:
        return {
            "query_id": query_id,
            "status": "empty_validation",
            "Acc": None,
            "vIoU": None,
            "tIoU": None,
        }

    # -----------------------------------------------------------------------
    # Determine dataset type and build image_id lookup
    # -----------------------------------------------------------------------
    ds_type = "hypernerf"
    metadata: dict[str, int] = {}
    if dataset_dir is not None and dataset_dir.exists():
        ds_type = _detect_dataset_type(dataset_dir)
        if ds_type == "hypernerf":
            metadata = _load_metadata(dataset_dir)
    else:
        inferred_ds_type, inferred_dataset_dir = _infer_dataset_info_from_query_output(query_output_dir)
        if inferred_ds_type is not None:
            ds_type = inferred_ds_type
        if inferred_dataset_dir is not None and inferred_dataset_dir.exists():
            if ds_type == "hypernerf":
                metadata = _load_metadata(inferred_dataset_dir)

    # Build lookup: image_id (str) → {frame_index, query_active}
    val_by_image_id: dict[str, dict] = {str(f["image_id"]): f for f in val_frames}

    # Build predicted time_id → query_active mapping
    # For HyperNeRF: use metadata.json to get time_id from image_id
    # For dynerf: use frame_index directly as time_id
    pred_by_time_id: dict[int, bool] = {}
    for f in val_frames:
        image_id_str = str(f["image_id"])
        if ds_type == "hypernerf" and metadata:
            tid = metadata.get(image_id_str)
            if tid is None:
                continue
        else:
            # dynerf: time_id = frame_index (0-indexed)
            tid = int(f["frame_index"])
        pred_by_time_id[tid] = bool(f["query_active"])

    # -----------------------------------------------------------------------
    # Determine total_frames
    # -----------------------------------------------------------------------
    if metadata:
        total_frames = max(metadata.values()) + 1
    elif pred_by_time_id:
        total_frames = max(pred_by_time_id.keys()) + 1
    else:
        total_frames = len(val_frames)

    # -----------------------------------------------------------------------
    # Build GT timeline
    # -----------------------------------------------------------------------
    # Convert existence_frames (list of frame_id) to time_ids (sparse annotation samples)
    gt_sampled_tids: list[int] = []  # the actual annotated time_ids
    if ds_type == "hypernerf" and metadata:
        for fid in existence_frames:
            image_id_str = _frame_id_to_image_id_hypernerf(fid)
            tid = metadata.get(image_id_str)
            if tid is not None:
                gt_sampled_tids.append(tid)
    else:
        for fid in existence_frames:
            gt_sampled_tids.append(int(fid))

    # KEY FIX 1: existence_frames are SPARSE SAMPLES of an active temporal range.
    # Build gt_time_ids as the DENSE RANGE [min, max] of sampled tids.
    # This correctly labels all rendered frames within the activity window as active.
    if gt_sampled_tids:
        gt_min_tid = min(gt_sampled_tids)
        gt_max_tid = max(gt_sampled_tids)
        gt_time_ids: set[int] = set(range(gt_min_tid, gt_max_tid + 1))
    else:
        # Negative query: entity does not exist
        gt_time_ids = set()
        gt_min_tid = -1
        gt_max_tid = -1

    # -----------------------------------------------------------------------
    # Temporal accuracy (Acc) - evaluated over all rendered frames
    # GT active = frame's time_id falls within [gt_min_tid, gt_max_tid]
    # -----------------------------------------------------------------------
    acc_correct = 0
    acc_total = 0
    for tid, pred_active in pred_by_time_id.items():
        gt_active = tid in gt_time_ids
        if pred_active == gt_active:
            acc_correct += 1
        acc_total += 1
    acc = _safe_div(acc_correct, acc_total)

    # -----------------------------------------------------------------------
    # Temporal IoU (tIoU) - full timeline comparison
    # -----------------------------------------------------------------------
    # Build full binary arrays over total_frames
    gt_full = np.zeros(total_frames, dtype=bool)
    pred_full = np.zeros(total_frames, dtype=bool)

    for tid in gt_time_ids:
        if 0 <= tid < total_frames:
            gt_full[tid] = True
    for tid, active in pred_by_time_id.items():
        if 0 <= tid < total_frames and active:
            pred_full[tid] = True

    temporal_inter = int(np.logical_and(pred_full, gt_full).sum())
    temporal_union = int(np.logical_or(pred_full, gt_full).sum())

    # Special case: negative query (empty gt and empty pred) → tIoU = 1.0
    if temporal_union == 0:
        t_iou = 1.0
    else:
        t_iou = _safe_div(temporal_inter, temporal_union)

    # -----------------------------------------------------------------------
    # Visual IoU (vIoU) - mask comparison at annotated GT frames
    # -----------------------------------------------------------------------
    # KEY FIX 2: Build sorted list of (time_id, frame_index) for rendered frames
    # so we can do nearest-neighbor lookup (GT annotated frames may not be in
    # the rendered test set, so we use the closest rendered frame instead).
    sorted_tids = sorted(pred_by_time_id.keys())
    # Build time_id → frame_index lookup for rendered frames
    val_by_time_id: dict[int, dict] = {}
    for f in val_frames:
        image_id_str = str(f["image_id"])
        if ds_type == "hypernerf" and metadata:
            tid = metadata.get(image_id_str)
            if tid is None:
                continue
        else:
            tid = int(f["frame_index"])
        val_by_time_id[tid] = f

    iou_sum = 0.0
    iou_count = 0
    mask_found = 0
    mask_missing = 0

    # Determine image size for polygon mask decoding.
    # Use the first available rendered binary mask to get (H, W).
    pred_image_size: tuple[int, int] | None = None
    if binary_mask_dir and binary_mask_dir.exists():
        sample_masks = sorted(binary_mask_dir.glob("*.png"))
        if sample_masks:
            with Image.open(sample_masks[0]) as _img:
                _arr = np.asarray(_img)
                pred_image_size = (_arr.shape[0], _arr.shape[1])  # (H, W)

    # Build frame_id → GT masks
    gt_frame_masks: dict[int, np.ndarray] = {}
    for gt_frame in gt_frames:
        fid = int(gt_frame["frame_id"])
        masks_in_frame = []
        for mask_item in gt_frame.get("masks", []):
            seg = mask_item.get("segmentation")
            if seg:
                mask_arr = _decode_segmentation(seg, image_size=pred_image_size)
                if mask_arr is not None:
                    masks_in_frame.append(mask_arr)
        if masks_in_frame:
            # Merge all masks in this frame (union)
            merged = masks_in_frame[0].copy()
            for m in masks_in_frame[1:]:
                if m.shape == merged.shape:
                    merged = merged | m
            gt_frame_masks[fid] = merged

    for fid, gt_mask in gt_frame_masks.items():
        # Map frame_id → time_id
        if ds_type == "hypernerf" and metadata:
            image_id_str = _frame_id_to_image_id_hypernerf(fid)
            tid = metadata.get(image_id_str)
        else:
            tid = int(fid)

        if tid is None:
            mask_missing += 1
            continue

        # KEY FIX 2: Nearest-neighbor lookup for rendered frame.
        # GT annotated frames may not align with rendered test frames,
        # so find the closest rendered frame by time_id.
        if not sorted_tids:
            mask_missing += 1
            continue
        pos = bisect.bisect_left(sorted_tids, tid)
        if pos == 0:
            nearest_tid = sorted_tids[0]
        elif pos >= len(sorted_tids):
            nearest_tid = sorted_tids[-1]
        else:
            # Pick whichever of sorted_tids[pos-1] or sorted_tids[pos] is closer
            before = sorted_tids[pos - 1]
            after = sorted_tids[pos]
            nearest_tid = before if (tid - before) <= (after - tid) else after

        val_frame = val_by_time_id.get(nearest_tid)
        if val_frame is None:
            mask_missing += 1
            continue

        pred_active = bool(pred_by_time_id.get(nearest_tid, False))
        gt_active = tid in gt_time_ids  # whether this GT frame is in the active range

        # vIoU: only compute at frames where GT is active (GT annotated frame)
        # If pred is also active → compute IoU of masks
        # If pred not active but GT is → IoU = 0 (entity present but not detected)
        if not gt_active:
            # GT says entity not active at this frame → skip for vIoU
            continue

        frame_idx = int(val_frame["frame_index"])
        mask_found += 1

        if not binary_mask_dir or not binary_mask_dir.exists():
            # No mask output dir
            iou_sum += 0.0
            iou_count += 1
            continue

        pred_mask_path = binary_mask_dir / f"{frame_idx:05d}.png"
        if pred_active:
            if pred_mask_path.exists():
                with Image.open(pred_mask_path) as img:
                    pred_mask = np.asarray(img.convert("L"), dtype=np.uint8) > 0
                # Resize pred_mask to match GT mask if sizes differ
                if pred_mask.shape != gt_mask.shape:
                    pred_pil = Image.fromarray(pred_mask.astype(np.uint8) * 255)
                    pred_pil = pred_pil.resize((gt_mask.shape[1], gt_mask.shape[0]), Image.NEAREST)
                    pred_mask = np.asarray(pred_pil) > 0
                iou = _mask_iou(pred_mask, gt_mask)
            else:
                iou = 0.0
        else:
            # Pred not active but GT is active → IoU = 0
            iou = 0.0
        iou_sum += iou
        iou_count += 1

    v_iou = _safe_div(iou_sum, iou_count)

    return {
        "query_id": query_id,
        "question": str(query_item.get("question", "")),
        "status": "ok",
        "Acc": acc,
        "vIoU": v_iou,
        "tIoU": t_iou,
        "dataset_type": ds_type,
        "total_frames": total_frames,
        "gt_active_count": int(len(gt_time_ids)),
        "pred_active_count": int(sum(1 for v in pred_by_time_id.values() if v)),
        "temporal_inter": temporal_inter,
        "temporal_union": temporal_union,
        "mask_found": mask_found,
        "mask_missing": mask_missing,
        "vIoU_count": iou_count,
        "validation_path": str(validation_path),
        "dataset_dir_used": str(dataset_dir) if dataset_dir is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReferGaussian outputs against Ours_benchmark.json"
    )
    parser.add_argument("--benchmark", required=True, help="Path to Ours_benchmark.json")
    parser.add_argument(
        "--query-root-map", required=True,
        help="JSON file mapping query_id → query_output_dir path"
    )
    parser.add_argument("--dataset-dir-map", default=None,
                        help="JSON file mapping query_id or scene_name → dataset_dir (optional)")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--skip-missing", action="store_true",
                        help="Skip queries with missing validation (don't fail)")
    args = parser.parse_args()

    benchmark = json.loads(Path(args.benchmark).read_text(encoding="utf-8"))
    query_root_map: dict[str, str] = json.loads(
        Path(args.query_root_map).read_text(encoding="utf-8")
    )
    dataset_dir_map: dict[str, str] = {}
    if args.dataset_dir_map:
        dataset_dir_map = json.loads(
            Path(args.dataset_dir_map).read_text(encoding="utf-8")
        )

    per_query: list[dict] = []
    for item in benchmark:
        query_id = str(item["query_id"])
        query_output_dir_str = query_root_map.get(query_id)
        if query_output_dir_str is None:
            if not args.skip_missing:
                raise ValueError(f"No query_root_map entry for {query_id}")
            per_query.append({
                "query_id": query_id,
                "question": str(item.get("question", "")),
                "status": "not_in_map",
                "Acc": None, "vIoU": None, "tIoU": None,
            })
            continue

        query_output_dir = Path(query_output_dir_str)
        dataset_dir_str = dataset_dir_map.get(query_id) or dataset_dir_map.get(
            "_".join(query_id.split("_")[:-1])
        )
        dataset_dir = Path(dataset_dir_str) if dataset_dir_str else None

        result = evaluate_query(item, query_output_dir, dataset_dir)
        per_query.append(result)
        status_str = result.get("status", "?")
        acc = result.get("Acc")
        viou = result.get("vIoU")
        tiou = result.get("tIoU")
        acc_s = f"{acc*100:.2f}%" if acc is not None else "n/a"
        viou_s = f"{viou*100:.2f}%" if viou is not None else "n/a"
        tiou_s = f"{tiou*100:.2f}%" if tiou is not None else "n/a"
        print(f"[eval] {query_id}: {status_str}  Acc={acc_s}  vIoU={viou_s}  tIoU={tiou_s}")

    # Aggregate metrics (only over queries with valid results)
    valid = [r for r in per_query if r.get("Acc") is not None]
    n = len(valid)
    summary = {
        "total_queries": len(per_query),
        "valid_queries": n,
        "Acc": float(np.mean([r["Acc"] for r in valid])) if valid else None,
        "vIoU": float(np.mean([r["vIoU"] for r in valid])) if valid else None,
        "tIoU": float(np.mean([r["tIoU"] for r in valid])) if valid else None,
    }

    print("\n=== Summary ===")
    print(f"Valid queries: {n} / {len(per_query)}")
    if summary["Acc"] is not None:
        print(f"Acc:  {summary['Acc']*100:.4f}%")
        print(f"vIoU: {summary['vIoU']*100:.4f}%")
        print(f"tIoU: {summary['tIoU']*100:.4f}%")

    payload = {
        "benchmark": str(args.benchmark),
        "summary": summary,
        "per_query": per_query,
    }
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved: {out_path}")

    if args.output_md:
        lines = [
            "# Ours_benchmark Evaluation",
            "",
            f"- Total queries: {summary['total_queries']}",
            f"- Valid queries: {summary['valid_queries']}",
        ]
        if summary["Acc"] is not None:
            lines += [
                f"- Acc:  `{summary['Acc']*100:.4f}%`",
                f"- vIoU: `{summary['vIoU']*100:.4f}%`",
                f"- tIoU: `{summary['tIoU']*100:.4f}%`",
                "",
                "| Query | Status | Acc(%) | vIoU(%) | tIoU(%) |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
            for r in per_query:
                def fmt(v):
                    return f"{v*100:.2f}" if v is not None else "n/a"
                lines.append(
                    f"| {r['query_id']} | {r.get('status','')} | {fmt(r.get('Acc'))} | {fmt(r.get('vIoU'))} | {fmt(r.get('tIoU'))} |"
                )
        Path(args.output_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Saved: {args.output_md}")


if __name__ == "__main__":
    main()
