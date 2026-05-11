import argparse
import os
import json
import shutil
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_ROOT = Path(os.environ.get("GR4D_BENCH_ROOT", str(REPO_ROOT / "data" / "GR4D-Bench")))
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "data" / "benchmarks" / "gr4d_curated_v1"
CUT_LEMON_SUPPORT_DIR = (
    REPO_ROOT
    / "runs/stellar_worldtube_cut-lemon1_quality5k/hypernerf/cut-lemon1/entitybank/query_guided"
)


def _gr4d_scene(scene: str) -> Path:
    return DEFAULT_SOURCE_ROOT / "data" / "scenes" / scene


SCENE_CONFIGS = [
    {
        "scene": "cut-lemon1",
        "family": "hypernerf",
        "raw_scene_rel": "data/hypernerf/interp/cut-lemon1",
        "annotation_scene_abs": str((REPO_ROOT / "data/hypernerf/interp/cut-lemon1").resolve()),
        "source_reference_abs": str((CUT_LEMON_SUPPORT_DIR / "cut_the_lemon_final/query_plan.json").resolve()),
        "scene_support_abs": str(CUT_LEMON_SUPPORT_DIR.resolve()),
        "selection_rationale": (
            "This is the strongest in-house HyperNeRF scene for query-guided reconstruction. "
            "It directly exposes state transition, interaction grouping, and generic reference over the lemon halves."
        ),
        "track_notes": {
            "cut_lemon1_whole_lemon": "whole lemon before separation",
            "cut_lemon1_left_lemon_half": "left lemon half after separation",
            "cut_lemon1_right_lemon_half": "right lemon half after separation",
            "cut_lemon1_knife": "knife",
            "cut_lemon1_hand": "hand",
        },
        "queries": [
            {
                "query_id": "cut_lemon1_curated_q1",
                "text_en": "The whole lemon before cutting begins.",
                "text_zh": "切割开始前的完整柠檬。",
                "target_track_ids": ["cut_lemon1_whole_lemon"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "A",
                "temporal_relation": "before",
                "expected_temporal_anchor": "cutting",
                "focus_tags": ["dynamic_semantic_query", "state_transition"],
                "derived_from": ["cut_the_lemon_final/query_plan.json"],
            },
            {
                "query_id": "cut_lemon1_curated_q2",
                "text_en": "All lemon halves after the cut is complete.",
                "text_zh": "切开完成后的所有柠檬半块。",
                "target_track_ids": ["cut_lemon1_left_lemon_half", "cut_lemon1_right_lemon_half"],
                "target_count": 2,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "after",
                "expected_temporal_anchor": "after_separation",
                "focus_tags": ["dynamic_semantic_query", "generic_reference", "multi_target"],
                "derived_from": ["cut_the_lemon/query_plan.json", "cut_the_lemon_final/query_plan.json"],
                "note": "This is the clearest custom query for your all-matching reference claim.",
            },
            {
                "query_id": "cut_lemon1_curated_q3",
                "text_en": "Everything that directly participates in cutting the lemon.",
                "text_zh": "所有直接参与切柠檬动作的对象。",
                "target_track_ids": ["cut_lemon1_whole_lemon", "cut_lemon1_knife", "cut_lemon1_hand"],
                "target_count": 3,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "while",
                "expected_temporal_anchor": "cutting",
                "focus_tags": ["dynamic_semantic_query", "interaction_group", "multi_target"],
                "derived_from": ["cut_the_lemon_maskguided/query_plan.json"],
            },
        ],
    },
    {
        "scene": "split-cookie",
        "family": "hypernerf",
        "raw_scene_rel": "data/hypernerf/misc/split-cookie",
        "annotation_scene_abs": str(_gr4d_scene("split-cookie").resolve()),
        "source_reference_abs": str((_gr4d_scene("split-cookie") / "video_annotations4.json").resolve()),
        "selection_rationale": (
            "This HyperNeRF scene is ideal for generic reference after object splitting. "
            "It highlights the difference between a whole object and all resulting fragments."
        ),
        "track_notes": {
            "split_cookie_whole_cookie": "complete cookie before breakage",
            "split_cookie_cookie_pieces": "semantic union of all cookie fragments after breakage",
            "split_cookie_hands": "bare hands involved in breaking the cookie",
        },
        "queries": [
            {
                "query_id": "split_cookie_curated_q1",
                "text_en": "The whole cookie before it breaks apart.",
                "text_zh": "碎裂前的完整饼干。",
                "target_track_ids": ["split_cookie_whole_cookie"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "A",
                "temporal_relation": "before",
                "expected_temporal_anchor": "breaking",
                "focus_tags": ["dynamic_semantic_query", "state_transition"],
                "derived_from": ["video_annotations4.json: complete cookie"],
            },
            {
                "query_id": "split_cookie_curated_q2",
                "text_en": "All cookie pieces after the cookie has broken into smaller pieces.",
                "text_zh": "饼干碎成小块之后的所有饼干碎片。",
                "target_track_ids": ["split_cookie_cookie_pieces"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "after",
                "expected_temporal_anchor": "broken_into_pieces",
                "focus_tags": ["dynamic_semantic_query", "generic_reference", "fragment_group"],
                "derived_from": ["video_annotations4.json: cookie broken into smaller pieces"],
                "note": "This query is intentionally phrased to return all fragments as one semantic group.",
            },
            {
                "query_id": "split_cookie_curated_q3",
                "text_en": "Everything that directly participates in splitting the cookie.",
                "text_zh": "所有直接参与掰开饼干动作的对象。",
                "target_track_ids": ["split_cookie_whole_cookie", "split_cookie_hands"],
                "target_count": 2,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "while",
                "expected_temporal_anchor": "breaking",
                "focus_tags": ["dynamic_semantic_query", "interaction_group", "multi_target"],
                "derived_from": ["video_annotations4.json", "train/_annotations.coco4.json"],
            },
        ],
    },
    {
        "scene": "americano",
        "family": "hypernerf",
        "raw_scene_rel": "data/hypernerf/misc/americano",
        "annotation_scene_abs": str(_gr4d_scene("americano").resolve()),
        "source_reference_abs": str((_gr4d_scene("americano") / "americano_queries.json").resolve()),
        "selection_rationale": (
            "Two cup instances plus a liquid-state transition make americano the strongest public HyperNeRF scene "
            "for generic reference over all cups and for motion-conditioned state queries."
        ),
        "track_notes": {
            "americano_1": "coaster",
            "americano_2": "glass cup",
            "americano_3": "hands",
            "americano_4": "metal cup",
            "americano_5": "tray",
        },
        "queries": [
            {
                "query_id": "americano_curated_q1",
                "text_en": "All cups in the scene, including the glass cup and the pouring metal cup.",
                "text_zh": "场景中的所有杯子，包括玻璃杯和正在倾倒的金属杯。",
                "target_track_ids": ["americano_2", "americano_4"],
                "target_count": 2,
                "requires_motion_understanding": False,
                "query_type": "B",
                "focus_tags": ["generic_reference", "multi_target"],
                "derived_from": ["americano_q1", "americano_q4"],
            },
            {
                "query_id": "americano_curated_q2",
                "text_en": "The glass cup while the liquid inside is darkening.",
                "text_zh": "杯中液体正在变深时的玻璃杯。",
                "target_track_ids": ["americano_2"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "A",
                "temporal_relation": "while",
                "expected_temporal_anchor": "darkening",
                "focus_tags": ["dynamic_semantic_query", "state_conditioned"],
                "derived_from": ["americano_q1"],
            },
            {
                "query_id": "americano_curated_q3",
                "text_en": "Everything that directly participates in pouring the coffee.",
                "text_zh": "所有直接参与倒咖啡动作的对象。",
                "target_track_ids": ["americano_3", "americano_4"],
                "target_count": 2,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "while",
                "expected_temporal_anchor": "pouring",
                "focus_tags": ["dynamic_semantic_query", "multi_target", "interaction_group"],
                "derived_from": ["americano_q3", "americano_q4"],
            },
        ],
    },
    {
        "scene": "coffee_martini",
        "family": "neu3d",
        "raw_scene_rel": "data/neu3d/coffee_martini",
        "annotation_scene_abs": str(_gr4d_scene("coffee_martini").resolve()),
        "source_reference_abs": str((_gr4d_scene("coffee_martini") / "coffee_martini_queries.json").resolve()),
        "selection_rationale": (
            "A Neu3D pouring scene with two vessel instances and a clear human-action structure. "
            "It is useful for both multi-instance cup reference and event-conditioned semantic queries."
        ),
        "track_notes": {
            "coffee_martini_3": "man",
            "coffee_martini_4": "martini glass",
            "coffee_martini_5": "metal cup",
        },
        "queries": [
            {
                "query_id": "coffee_martini_curated_q1",
                "text_en": "All drink containers involved in the pour, including the martini glass and the metal cup.",
                "text_zh": "参与倾倒过程的所有容器，包括马提尼酒杯和金属杯。",
                "target_track_ids": ["coffee_martini_4", "coffee_martini_5"],
                "target_count": 2,
                "requires_motion_understanding": False,
                "query_type": "B",
                "focus_tags": ["generic_reference", "multi_target"],
                "derived_from": ["coffee_martini_q1", "coffee_martini_q2"],
            },
            {
                "query_id": "coffee_martini_curated_q2",
                "text_en": "The martini glass while it is being filled with coffee.",
                "text_zh": "正在被注入咖啡的马提尼酒杯。",
                "target_track_ids": ["coffee_martini_4"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "A",
                "temporal_relation": "while",
                "expected_temporal_anchor": "filling",
                "focus_tags": ["dynamic_semantic_query", "state_conditioned"],
                "derived_from": ["coffee_martini_q1"],
            },
            {
                "query_id": "coffee_martini_curated_q3",
                "text_en": "Everything that directly participates in pouring coffee into the martini glass.",
                "text_zh": "所有直接参与向马提尼酒杯倒咖啡的对象。",
                "target_track_ids": ["coffee_martini_3", "coffee_martini_4", "coffee_martini_5"],
                "target_count": 3,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "while",
                "expected_temporal_anchor": "leaning_and_pouring",
                "focus_tags": ["dynamic_semantic_query", "multi_target", "interaction_group"],
                "derived_from": ["coffee_martini_q1", "coffee_martini_q2", "coffee_martini_q3"],
            },
        ],
    },
    {
        "scene": "flame_steak",
        "family": "neu3d",
        "raw_scene_rel": "data/neu3d/flame_steak",
        "annotation_scene_abs": str(_gr4d_scene("flame_steak").resolve()),
        "source_reference_abs": str((_gr4d_scene("flame_steak") / "flame_steak_queries.json").resolve()),
        "selection_rationale": (
            "A Neu3D torching scene with explicit actor-tool interaction and a strong motion-conditioned action label."
        ),
        "track_notes": {
            "flame_steak_2": "chef",
            "flame_steak_4": "left hand",
            "flame_steak_8": "torch",
        },
        "queries": [
            {
                "query_id": "flame_steak_curated_q1",
                "text_en": "Everything that directly participates in torching the steak.",
                "text_zh": "所有直接参与喷枪炙烤牛排的对象。",
                "target_track_ids": ["flame_steak_2", "flame_steak_4", "flame_steak_8"],
                "target_count": 3,
                "requires_motion_understanding": True,
                "query_type": "B",
                "temporal_relation": "while",
                "expected_temporal_anchor": "operating_torch",
                "focus_tags": ["dynamic_semantic_query", "multi_target", "interaction_group"],
                "derived_from": ["flame_steak_q1", "flame_steak_q2", "flame_steak_q3"],
            },
            {
                "query_id": "flame_steak_curated_q2",
                "text_en": "The torch while it is moving freely to evenly sear the steak.",
                "text_zh": "正在自由移动以均匀炙烤牛排的喷枪。",
                "target_track_ids": ["flame_steak_8"],
                "target_count": 1,
                "requires_motion_understanding": True,
                "query_type": "A",
                "temporal_relation": "while",
                "expected_temporal_anchor": "emitting_flame_and_roaming",
                "focus_tags": ["dynamic_semantic_query", "tool_motion"],
                "derived_from": ["flame_steak_q1"],
            },
            {
                "query_id": "flame_steak_curated_q3",
                "text_en": "All objects that remain completely still throughout the video.",
                "text_zh": "在整个视频中始终完全静止的所有物体。",
                "target_track_ids": ["flame_steak_1", "flame_steak_3", "flame_steak_5", "flame_steak_6", "flame_steak_7"],
                "target_count": 5,
                "requires_motion_understanding": False,
                "query_type": "B",
                "focus_tags": ["generic_reference", "set_query"],
                "derived_from": ["flame_steak_q4"],
            },
        ],
    },
]


DROPPED_SCENES = [
    {"scene": "chickchicken", "reason": "Dropped to keep the final dataset to the five user-selected scenes."},
    {"scene": "keyboard", "reason": "Dropped to keep the final dataset to the five user-selected scenes."},
    {"scene": "cut_roasted_beef", "reason": "Dropped to keep the final dataset to the five user-selected scenes."},
    {"scene": "espresso", "reason": "Still useful, but not part of the final five-scene selection."},
    {"scene": "torchocolate", "reason": "Still useful, but not part of the final five-scene selection."},
    {"scene": "cook_spinach", "reason": "Valid Neu3D candidate, but not part of the final five-scene selection."},
    {"scene": "flame_salmon", "reason": "Valid Neu3D candidate, but not part of the final five-scene selection."},
    {"scene": "sear_steak", "reason": "Valid Neu3D candidate, but not part of the final five-scene selection."},
]


def _write_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _ensure_symlink(target: Path, link_path: Path) -> None:
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink():
        if link_path.resolve() == target.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"Refusing to replace non-symlink path: {link_path}")
    link_path.symlink_to(target)


def _build_readme(output_root: Path, scene_payloads: list[dict]) -> None:
    family_counts: dict[str, int] = {}
    for payload in scene_payloads:
        family_counts[payload["family"]] = family_counts.get(payload["family"], 0) + 1
    lines = [
        "# GR4D Curated v1",
        "",
        "This package now contains the final five-scene dataset requested for ReferGaussian.",
        f"It keeps `{family_counts.get('hypernerf', 0)}` HyperNeRF scenes and `{family_counts.get('neu3d', 0)}` Neu3D scenes.",
        "The scene list is fixed to:",
        "",
        "- `cut-lemon1`",
        "- `split-cookie`",
        "- `americano`",
        "- `coffee_martini`",
        "- `flame_steak`",
        "",
        "The query design focuses on:",
        "",
        "- motion-conditioned reconstruction and state transition,",
        "- dynamic semantic queries tied to interaction phases,",
        "- generic reference queries that should return all matching entities or fragments.",
        "",
        "Each scene keeps two or three high-value queries.",
        "",
        "## Selected scenes",
        "",
    ]
    for payload in scene_payloads:
        lines.append(f"- `{payload['scene']}` ({payload['family']}): {payload['selection_rationale']}")
    lines.extend(
        [
            "",
            "## Package layout",
            "",
            "- `gr4d_curated_v1_queries.json`: combined query file for all five scenes.",
            "- `scenes_manifest.json`: scene-level manifest with family, raw-scene, and support paths.",
            "- `scenes/<scene>/curated_queries.json`: per-scene reduced queries.",
            "- `scenes/<scene>/annotation_scene`: symlink to the main scene/annotation root for that scene.",
            "- `scenes/<scene>/raw_scene`: symlink to the local scene root used by this curated package.",
            "- `scenes/<scene>/scene_support`: optional extra support link for custom scenes such as `cut-lemon1`.",
            "",
            "## Dropped",
            "",
        ]
    )
    for item in DROPPED_SCENES:
        lines.append(f"- `{item['scene']}`: {item['reason']}")
    lines.append("")
    (output_root / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    output_root = Path(args.output_root)
    scenes_root = output_root / "scenes"
    selected_scene_names = {config["scene"] for config in SCENE_CONFIGS}
    if scenes_root.exists():
        for scene_dir in scenes_root.iterdir():
            if scene_dir.is_dir() and scene_dir.name not in selected_scene_names:
                shutil.rmtree(scene_dir)

    scene_payloads: list[dict] = []
    combined_queries: list[dict] = []
    family_counts: dict[str, int] = {}
    for config in SCENE_CONFIGS:
        scene = config["scene"]
        family = config["family"]
        raw_scene_path = REPO_ROOT / config["raw_scene_rel"]
        annotation_scene_path = Path(config["annotation_scene_abs"])
        source_reference_path = Path(config["source_reference_abs"])

        if not raw_scene_path.exists():
            raise FileNotFoundError(f"Raw scene missing: {raw_scene_path}")
        if not annotation_scene_path.exists():
            raise FileNotFoundError(f"Annotation scene missing: {annotation_scene_path}")
        if not source_reference_path.exists():
            raise FileNotFoundError(f"Source reference missing: {source_reference_path}")

        scene_output_dir = output_root / "scenes" / scene
        scene_output_dir.mkdir(parents=True, exist_ok=True)
        _ensure_symlink(annotation_scene_path, scene_output_dir / "annotation_scene")
        _ensure_symlink(raw_scene_path, scene_output_dir / "raw_scene")

        support_path = config.get("scene_support_abs")
        if support_path:
            _ensure_symlink(Path(support_path), scene_output_dir / "scene_support")
        elif (scene_output_dir / "scene_support").exists() or (scene_output_dir / "scene_support").is_symlink():
            (scene_output_dir / "scene_support").unlink()

        scene_query_payload = {
            "benchmark_name": "GR4D-Curated-v1",
            "scene": scene,
            "family": family,
            "raw_scene_rel": config["raw_scene_rel"],
            "annotation_scene_abs": str(annotation_scene_path.resolve()),
            "source_reference_abs": str(source_reference_path.resolve()),
            "scene_support_abs": None if not support_path else str(Path(support_path).resolve()),
            "selection_rationale": config["selection_rationale"],
            "track_notes": config["track_notes"],
            "queries": config["queries"],
        }
        _write_json(scene_output_dir / "curated_queries.json", scene_query_payload)

        scene_manifest = {
            "scene": scene,
            "family": family,
            "raw_scene_rel": config["raw_scene_rel"],
            "raw_scene_abs": str(raw_scene_path.resolve()),
            "annotation_scene_abs": str(annotation_scene_path.resolve()),
            "source_reference_abs": str(source_reference_path.resolve()),
            "scene_support_abs": None if not support_path else str(Path(support_path).resolve()),
            "selection_rationale": config["selection_rationale"],
            "query_count": len(config["queries"]),
            "track_notes": config["track_notes"],
        }
        _write_json(scene_output_dir / "scene_manifest.json", scene_manifest)

        scene_payloads.append(scene_manifest)
        family_counts[family] = family_counts.get(family, 0) + 1

        for query in config["queries"]:
            combined_query = dict(query)
            combined_query["scene"] = scene
            combined_query["family"] = family
            combined_query["raw_scene_rel"] = config["raw_scene_rel"]
            combined_queries.append(combined_query)

    combined_payload = {
        "benchmark_name": "GR4D-Curated-v1",
        "generated_on": date.today().isoformat(),
        "source_root": str(DEFAULT_SOURCE_ROOT.resolve()),
        "output_root": str(output_root.resolve()),
        "family_counts": family_counts,
        "focus": [
            "motion_conditioned_reconstruction",
            "dynamic_semantic_query",
            "generic_reference_multi_target_grounding",
        ],
        "total_scenes": len(scene_payloads),
        "total_queries": len(combined_queries),
        "queries": combined_queries,
    }
    _write_json(output_root / "gr4d_curated_v1_queries.json", combined_payload)

    manifest_payload = {
        "benchmark_name": "GR4D-Curated-v1",
        "generated_on": date.today().isoformat(),
        "source_root": str(DEFAULT_SOURCE_ROOT.resolve()),
        "output_root": str(output_root.resolve()),
        "family_counts": family_counts,
        "selected_scenes": scene_payloads,
        "dropped_or_deferred_scenes": DROPPED_SCENES,
    }
    _write_json(output_root / "scenes_manifest.json", manifest_payload)
    _build_readme(output_root, scene_payloads)
    print(output_root)


if __name__ == "__main__":
    main()
