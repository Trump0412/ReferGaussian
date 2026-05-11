#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build contact sheets and an index for a rendered entity library."
    )
    parser.add_argument("--library-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--thumb-width", type=int, default=320)
    parser.add_argument("--thumb-height", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip().replace("\n", " ")
    return " ".join(text.split())


def extract_frame(video_path: Path, frame_index: int) -> Image.Image:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 0:
        frame_index = max(0, min(frame_index, total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to decode frame from: {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame)


def choose_frame_index(validation: dict[str, Any]) -> int:
    for key in ("first_contact_frame", "first_active_frame", "last_active_frame"):
        value = validation.get(key)
        if isinstance(value, int):
            return value
    count = validation.get("frame_count")
    if isinstance(count, int) and count > 0:
        return count // 2
    return 0


def build_thumb(
    image: Image.Image,
    text_lines: list[str],
    width: int,
    height: int,
    font: ImageFont.ImageFont,
) -> Image.Image:
    canvas = Image.new("RGB", (width, height), (20, 20, 20))
    preview_h = height - 56
    preview = ImageOps.fit(image, (width, preview_h), method=Image.Resampling.LANCZOS)
    canvas.paste(preview, (0, 0))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, preview_h, width, height), fill=(0, 0, 0))
    y = preview_h + 6
    for line in text_lines[:3]:
        draw.text((8, y), line[:52], fill=(255, 255, 255), font=font)
        y += 16
    return canvas


def page_name(page_idx: int) -> str:
    return f"entity_overview_page_{page_idx:02d}.png"


def build_contact_sheet(
    thumbs: list[Image.Image],
    page_idx: int,
    columns: int,
    rows: int,
    width: int,
    height: int,
    output_dir: Path,
) -> Path:
    gap = 12
    header_h = 52
    page_w = columns * width + (columns + 1) * gap
    page_h = header_h + rows * height + (rows + 1) * gap
    page = Image.new("RGB", (page_w, page_h), (245, 245, 245))
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    draw.text((gap, 16), f"ReferGaussian Entity Library Page {page_idx}", fill=(0, 0, 0), font=font)
    for idx, thumb in enumerate(thumbs):
        row = idx // columns
        col = idx % columns
        x = gap + col * (width + gap)
        y = header_h + gap + row * (height + gap)
        page.paste(thumb, (x, y))
    page_path = output_dir / page_name(page_idx)
    page.save(page_path)
    return page_path


def main() -> None:
    args = parse_args()
    library_root = args.library_root.resolve()
    output_dir = (args.output_dir or (library_root / "overview")).resolve()
    ensure_dir(output_dir)
    thumbs_dir = output_dir / "thumbs"
    ensure_dir(thumbs_dir)

    index = load_json(library_root / "entity_index.json")
    entities = index.get("entities", [])
    font = ImageFont.load_default()
    page_capacity = args.columns * args.rows

    entity_rows: list[dict[str, Any]] = []
    page_images: list[Image.Image] = []
    page_paths: list[Path] = []
    page_idx = 1

    for entity in entities:
        entity_id = int(entity["entity_id"])
        entity_dir = Path(entity["entity_dir"])
        semantic_path = Path(entity["semantic_summary_path"])
        render_dir = Path(entity["render_dir"])
        overlay_path = render_dir / "overlay.mp4"
        validation_path = render_dir / "validation.json"
        summary = load_json(semantic_path)
        validation = load_json(validation_path)

        qwen_text = summary.get("qwen_text", {})
        category = safe_text(qwen_text.get("category"))
        label = safe_text(entity.get("label")) or safe_text(category) or f"entity {entity_id}"
        description = safe_text(qwen_text.get("canonical_description")) or safe_text(qwen_text.get("global_desc"))
        frame_index = choose_frame_index(validation)
        frame = extract_frame(overlay_path, frame_index)
        thumb = build_thumb(
            frame,
            [
                f"id {entity_id:04d} | {label}",
                f"type {entity.get('entity_type', '')} | head {entity.get('semantic_head', '')}",
                description or safe_text(entity.get("global_desc")),
            ],
            args.thumb_width,
            args.thumb_height,
            font,
        )
        thumb_path = thumbs_dir / f"entity_{entity_id:04d}.png"
        thumb.save(thumb_path)
        page_images.append(thumb)

        entity_rows.append(
            {
                "entity_id": entity_id,
                "label": label,
                "category": category,
                "description": description,
                "entity_dir": str(entity_dir),
                "overlay_path": str(overlay_path),
                "mask_path": str(render_dir / "mask.mp4"),
                "validation_path": str(validation_path),
                "thumb_path": str(thumb_path),
                "frame_index": frame_index,
                "active_frame_count": validation.get("active_frame_count"),
                "contact_frame_count": validation.get("contact_frame_count"),
                "qwen_enabled": bool(entity.get("qwen_enabled")),
            }
        )

        if len(page_images) == page_capacity:
            page_paths.append(
                build_contact_sheet(
                    page_images,
                    page_idx,
                    args.columns,
                    args.rows,
                    args.thumb_width,
                    args.thumb_height,
                    output_dir,
                )
            )
            page_images = []
            page_idx += 1

    if page_images:
        page_paths.append(
            build_contact_sheet(
                page_images,
                page_idx,
                args.columns,
                args.rows,
                args.thumb_width,
                args.thumb_height,
                output_dir,
            )
        )

    metadata = {
        "library_root": str(library_root),
        "num_entities": len(entity_rows),
        "num_pages": len(page_paths),
        "page_paths": [str(path) for path in page_paths],
        "entities": entity_rows,
    }
    with (output_dir / "overview_index.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    lines = [
        "# Entity Library Overview",
        "",
        f"- library_root: `{library_root}`",
        f"- num_entities: `{len(entity_rows)}`",
        f"- num_pages: `{len(page_paths)}`",
        "",
        "## Pages",
        "",
    ]
    for path in page_paths:
        lines.append(f"- `{path.name}`")
    lines.extend(["", "## Entities", ""])
    for row in entity_rows:
        lines.append(
            f"- `entity {row['entity_id']:04d}` | `{row['label']}` | "
            f"`active={row['active_frame_count']}` | `contact={row['contact_frame_count']}` | "
            f"`{Path(row['thumb_path']).name}`"
        )
    with (output_dir / "README.md").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")

    print(output_dir)


if __name__ == "__main__":
    main()
