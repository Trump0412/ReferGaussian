import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_font(size: int = 20) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int]) -> None:
    font = _load_font(20)
    left, top = xy
    bbox = draw.textbbox((left, top), text, font=font)
    draw.rounded_rectangle(
        (bbox[0] - 4, bbox[1] - 2, bbox[2] + 4, bbox[3] + 2),
        radius=4,
        fill=(18, 18, 18),
    )
    draw.text((left, top), text, fill=fill, font=font)


def _entity_color(entity_id: int) -> tuple[int, int, int]:
    hue = float((int(entity_id) * 0.61803398875) % 1.0)
    sat = 0.70
    val = 1.0
    rgb = np.asarray(__import__("colorsys").hsv_to_rgb(hue, sat, val), dtype=np.float32)
    return tuple(int(round(float(channel) * 255.0)) for channel in rgb)


def _overlay_mask(image: Image.Image, mask: np.ndarray, color: tuple[int, int, int], alpha: int) -> Image.Image:
    base = image.convert("RGBA")
    overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
    overlay[mask, 0] = color[0]
    overlay[mask, 1] = color[1]
    overlay[mask, 2] = color[2]
    overlay[mask, 3] = int(np.clip(alpha, 0, 255))
    return Image.alpha_composite(base, Image.fromarray(overlay, mode="RGBA"))


def _mask_bbox(mask: np.ndarray) -> list[int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0 or ys.size == 0:
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def regenerate_overlays(validation_path: Path, overwrite: bool = False) -> Path:
    validation = _read_json(validation_path)
    output_dir = validation_path.parent
    overlay_dir = output_dir / "overlay_frames"
    mask_dir = Path(validation["frame_exports"]["binary_masks"])
    background_dir = Path(validation.get("background_frame_dir") or "")

    if not background_dir.exists():
        raise FileNotFoundError(f"Missing background_frame_dir for {validation_path}: {background_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing binary_masks for {validation_path}: {mask_dir}")

    if overlay_dir.exists():
        if overlay_dir.is_symlink() or overwrite:
            if overlay_dir.is_symlink():
                overlay_dir.unlink()
            else:
                for child in overlay_dir.iterdir():
                    if child.is_file() or child.is_symlink():
                        child.unlink()
        else:
            raise FileExistsError(f"Overlay directory already exists: {overlay_dir}")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    query_text = str(validation.get("query", "query"))
    frames = list(validation.get("frames", []))
    for frame_row in frames:
        frame_index = int(frame_row["frame_index"])
        image_id = str(frame_row["image_id"])
        source_path = background_dir / f"{image_id}.png"
        mask_path = mask_dir / f"{frame_index:05d}.png"
        if not source_path.exists():
            continue
        if not mask_path.exists():
            continue

        with Image.open(source_path) as base_image:
            overlay = base_image.convert("RGB")
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8) > 0

        role_rows = list(frame_row.get("roles", []))
        entity_rows = [row for row in role_rows if bool(row.get("active", False))]
        color = _entity_color(int(entity_rows[0]["entity_id"])) if entity_rows else (255, 96, 96)
        overlay = _overlay_mask(overlay, mask, color, alpha=120).convert("RGB")

        bbox = _mask_bbox(mask)
        draw = ImageDraw.Draw(overlay, "RGBA")
        if bbox is not None:
            left, top, right, bottom = bbox
            draw.rectangle((left, top, right, bottom), outline=color + (255,), width=4)
            label = f"entity #{int(entity_rows[0]['entity_id'])}" if entity_rows else "mask"
            _draw_label(draw, (max(0, left), max(0, top - 26)), label, color)

        headline = f"ReferGaussian query render: {query_text}"
        visible_count = int(sum(1 for row in role_rows if row.get("active") and row.get("displayable")))
        status = (
            f"frame {frame_index:04d}  time={float(frame_row.get('time_value', 0.0)):.3f}"
            f"  active={'yes' if bool(frame_row.get('query_active')) else 'no'}  entities={visible_count}"
        )
        _draw_label(draw, (16, 16), headline, (240, 240, 240))
        _draw_label(draw, (16, 44), status, (220, 220, 220))
        overlay.save(overlay_dir / f"{frame_index:05d}.png")

    validation["frame_exports"]["overlay_frames"] = str(overlay_dir)
    with open(validation_path, "w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False)
    return overlay_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-path", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = regenerate_overlays(Path(args.validation_path), overwrite=bool(args.overwrite))
    print(output)


if __name__ == "__main__":
    main()
