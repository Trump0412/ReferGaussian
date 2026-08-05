#!/usr/bin/env python3
"""Build the COCO-category time-agnostic protocol used by 4D LangSplat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_4dlangsplat_query_protocol import GROUP_BY_SCENE, slugify


def build_scene_queries(scene: str, coco_payload: dict) -> list[dict]:
    group = GROUP_BY_SCENE.get(scene)
    if group is None:
        raise ValueError(f"Unsupported 4D LangSplat scene-group mapping for {scene}")

    used_category_ids = {
        int(annotation["category_id"])
        for annotation in coco_payload.get("annotations", [])
    }
    frame_ids_by_category: dict[int, set[int]] = {}
    for annotation in coco_payload.get("annotations", []):
        category_id = int(annotation["category_id"])
        frame_ids_by_category.setdefault(category_id, set()).add(int(annotation["image_id"]))
    evaluation_image_ids = [
        str(image["file_name"]).split("_")[0]
        for image in coco_payload.get("images", [])
    ]

    rows = []
    for category in coco_payload.get("categories", []):
        category_id = int(category["id"])
        if category_id not in used_category_ids:
            continue
        category_name = " ".join(str(category["name"]).strip().split())
        query_slug = f"{scene}__time_agnostic__{slugify(category_name)}"
        rows.append(
            {
                "scene": f"HyperNeRF/{group}/{scene}",
                "query_slug": query_slug,
                "query": category_name,
                "category": "time_agnostic_reference",
                "target_category_id": category_id,
                "target_category_name": category_name,
                "annotated_frame_count": len(frame_ids_by_category.get(category_id, set())),
                "test_frame_count": len(evaluation_image_ids),
                "evaluation_image_ids": evaluation_image_ids,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-root", required=True)
    parser.add_argument("--scene", default=None)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    root = Path(args.annotation_root)
    scenes = [args.scene] if args.scene else sorted(
        path.parent.parent.name for path in root.glob("*/train/_annotations.coco.json")
    )
    queries: list[dict] = []
    for scene in scenes:
        coco_path = root / scene / "train" / "_annotations.coco.json"
        if not coco_path.is_file():
            raise FileNotFoundError(coco_path)
        payload = json.loads(coco_path.read_text(encoding="utf-8"))
        queries.extend(build_scene_queries(scene, payload))

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "category": "time_agnostic_reference",
                "aggregation": "macro_category_mean_of_annotated_frame_iou",
                "queries": queries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(output_path)


if __name__ == "__main__":
    main()
