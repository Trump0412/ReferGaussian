"""Tests for scene-agnostic query-planner phrase expansion."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from refergaussian.semantics.qwen_query_planner import (
    _canonicalize_phrase,
    _count_neutral_detector_phrases,
    _normalize_plan,
    _qwen_gpu_memory_budget_gib,
    _qwen_max_new_tokens,
    _query_semantic_profile,
    _state_detector_phrase_additions,
)
from refergaussian.semantics.select_qwen_query_entities import (
    _candidate_phrase_score,
    _build_candidate_visual_evidence,
    _compose_phrase_grounded_selection,
    _decorate_relation_disambiguation_candidates,
    _expand_counted_subject_phrases,
    _identity_attribute_verification,
    _select_phrase_ids,
)


class QueryPlannerPhraseTest(unittest.TestCase):
    def test_candidate_visual_evidence_uses_mask_overlay_crops(self) -> None:
        with TemporaryDirectory() as directory:
            overlay_path = Path(directory) / "overlay.png"
            Image.new("RGB", (640, 360), color=(10, 20, 30)).save(overlay_path)
            candidates = [{"id": 4, "stage1_object_id": 12, "proposal_alias": "target object"}]
            tracks_payload = {
                "tracks": [
                    {
                        "object_id": 12,
                        "frames": [
                            {
                                "active": True,
                                "frame_index": 20,
                                "bbox_xyxy": [100, 80, 300, 260],
                                "overlay_path": str(overlay_path),
                            }
                        ],
                    }
                ]
            }

            images, rows = _build_candidate_visual_evidence(candidates, tracks_payload)

        self.assertEqual(len(images), 1)
        self.assertEqual(rows[0]["candidate_id"], 4)
        self.assertEqual(rows[0]["stage1_object_id"], 12)
        self.assertEqual(rows[0]["kind"], "stage1_mask_overlay_crop")
        self.assertLess(images[0].width, 640)
        self.assertLess(images[0].height, 360)

    def test_relation_instance_visual_evidence_uses_two_temporal_moments(self) -> None:
        with TemporaryDirectory() as directory:
            overlay_paths = []
            for index in range(3):
                overlay_path = Path(directory) / f"overlay_{index}.png"
                Image.new("RGB", (640, 360), color=(10 + index, 20, 30)).save(overlay_path)
                overlay_paths.append(overlay_path)
            candidates = [
                {
                    "id": 4,
                    "stage1_object_id": 12,
                    "proposal_alias": "target instance 1",
                    "instance_selection_policy": "relation_disambiguation",
                }
            ]
            tracks_payload = {
                "tracks": [
                    {
                        "object_id": 12,
                        "anchor_frame_index": 20,
                        "frames": [
                            {
                                "active": True,
                                "frame_index": index * 20,
                                "bbox_xyxy": [100, 80, 300, 260],
                                "overlay_path": str(overlay_path),
                            }
                            for index, overlay_path in enumerate(overlay_paths)
                        ],
                    }
                ]
            }

            images, rows = _build_candidate_visual_evidence(candidates, tracks_payload)

        self.assertEqual(len(images), 2)
        self.assertEqual([row["frame_index"] for row in rows], [0, 20])

    def test_static_set_visual_evidence_uses_a_temporal_contact_sheet(self) -> None:
        with TemporaryDirectory() as directory:
            overlay_paths = []
            for index in range(3):
                overlay_path = Path(directory) / f"overlay_{index}.png"
                Image.new("RGB", (640, 360), color=(10 + index, 20, 30)).save(overlay_path)
                overlay_paths.append(overlay_path)
            candidates = [{"id": 4, "stage1_object_id": 12, "proposal_alias": "target object"}]
            tracks_payload = {
                "tracks": [
                    {
                        "object_id": 12,
                        "frames": [
                            {
                                "active": True,
                                "frame_index": index * 20,
                                "bbox_xyxy": [100, 80, 300, 260],
                                "overlay_path": str(overlay_path),
                            }
                            for index, overlay_path in enumerate(overlay_paths)
                        ],
                    }
                ]
            }

            images, rows = _build_candidate_visual_evidence(
                candidates,
                tracks_payload,
                temporal_contact_sheets=True,
            )

        self.assertEqual(len(images), 1)
        self.assertEqual(rows[0]["kind"], "stage1_temporal_contact_sheet")
        self.assertEqual(rows[0]["frame_indices"], [0, 20, 40])
        self.assertEqual(images[0].width, 768)
        self.assertEqual(images[0].height, 256)

    def test_state_phrase_is_composed_for_an_unseen_object(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the package broken into several pieces",
            ["package"],
        )
        self.assertIn("broken package", phrases)
        self.assertIn("broken package pieces", phrases)
        self.assertIn("package pieces", phrases)

    def test_state_phrase_preserves_query_specific_noun_context(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the closed plastic container",
            ["container"],
        )
        self.assertIn("closed container", phrases)
        self.assertIn("closed plastic container", phrases)

    def test_plan_preserves_non_temporal_identity_attributes(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["dark chocolate", "tray"],
                "primary_subject_phrases": ["chocolate"],
                "query_subject_phrases": ["chocolate"],
                "required_identity_attributes": ["white"],
            },
            "The white chocolate on the tray.",
        )

        self.assertEqual(plan["required_identity_attributes"], ["white"])

    def test_temporal_state_and_subject_head_do_not_enter_identity_gate(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["chocolate bar", "tray"],
                "primary_subject_phrases": ["chocolate bar"],
                "query_subject_phrases": ["chocolate bar"],
                # Exercise the temporal-hint recovery path when a VLM omits
                # the new temporal_state_attributes field.
                "required_identity_attributes": ["solid", "chocolate"],
                "temporal_hints": ["initial solid state", "melting process"],
            },
            "The solid chocolate before it starts melting.",
        )

        self.assertEqual(plan["required_identity_attributes"], [])
        self.assertEqual(
            plan["identity_attribute_filter"],
            [
                {"attribute": "solid", "reason": "temporal_hint"},
                {"attribute": "chocolate", "reason": "subject_head"},
            ],
        )

    def test_content_state_is_derived_from_the_query(self) -> None:
        phrases = _state_detector_phrase_additions(
            "the vessel with liquid above the midpoint",
            ["vessel"],
        )
        self.assertIn("liquid above midpoint", phrases)

    def test_action_context_is_not_promoted_to_the_singular_query_subject(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["knife", "beef steak"],
                "query_subject_phrases": ["knife", "beef steak"],
                "must_track_phrases": ["knife", "beef steak"],
            },
            "The knife while it is cutting the beef steak.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["knife"])
        self.assertEqual(plan["query_subject_phrases"], ["knife"])
        self.assertEqual(plan["detector_phrases"], ["knife"])
        self.assertEqual(plan["must_track_phrases"], ["knife"])

    def test_leading_subject_constrains_primary_planner_context_for_a_gerund_query(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["operator", "device", "material"],
                "primary_subject_phrases": ["operator", "device", "material"],
                "query_subject_phrases": ["operator", "device", "material"],
                "must_track_phrases": ["operator", "device", "material"],
            },
            "The operator moving the device to heat the material.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["operator"])
        self.assertEqual(plan["query_subject_phrases"], ["operator"])
        self.assertEqual(plan["relation_context_phrases"], ["device", "material"])
        self.assertEqual(plan["detector_phrases"], ["operator", "device", "material"])
        self.assertEqual(plan["must_track_phrases"], ["operator"])

    def test_while_clause_is_an_action_window_without_action_specific_vocabulary(self) -> None:
        profile = _query_semantic_profile("The object while it is moving.")
        self.assertTrue(profile["asks_action_window"])

    def test_action_track_segments_are_written_to_the_entity_selection(self) -> None:
        candidates = [
            {
                "id": 7,
                "proposal_phrase": "object",
                "proposal_variant": "main",
                "quality": 1.0,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [],
                "moving_segments_test": [],
            }
        ]
        with patch(
            "refergaussian.semantics.select_qwen_query_entities._track_state_segments_test",
            return_value=([[2, 5]], {"state_mode": "action"}),
        ):
            payload = _compose_phrase_grounded_selection(
                query="The object while it is moving.",
                query_plan_payload={"query_subject_phrases": ["object"]},
                candidates=candidates,
                pair_candidates=[],
                test_times=np.linspace(0.0, 1.0, num=10),
                tracks_payload=None,
                raw_phrase_payload={"subject_phrases": ["object"], "successor_phrases": []},
            )

        self.assertEqual(payload["selected"], [
            {
                "id": 7,
                "role": "entity",
                "confidence": 1.0,
                "reason": "Selected from track-derived action support for subject phrase 'object'.",
                "segments": [[2, 5]],
            }
        ])

    def test_progressive_relation_uses_planner_confirmed_action_window(self) -> None:
        candidates = [
            {
                "id": 7,
                "proposal_phrase": "operator",
                "proposal_variant": "main",
                "quality": 1.0,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [],
                "moving_segments_test": [],
            }
        ]
        with patch(
            "refergaussian.semantics.select_qwen_query_entities._track_state_segments_test",
            return_value=([[2, 5]], {"state_mode": "action"}),
        ):
            payload = _compose_phrase_grounded_selection(
                query="The operator moving the device.",
                query_plan_payload={
                    "query_subject_phrases": ["operator"],
                    "relation_context_phrases": ["device"],
                    "action_window_hint": "during the device movement",
                },
                candidates=candidates,
                pair_candidates=[],
                test_times=np.linspace(0.0, 1.0, num=10),
                tracks_payload=None,
                raw_phrase_payload={"subject_phrases": ["operator"], "successor_phrases": []},
            )

        self.assertEqual(payload["selected"][0]["segments"], [[2, 5]])
        self.assertIn("action support", payload["selected"][0]["reason"])

    def test_explicit_multi_target_query_preserves_all_requested_subjects(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["red block", "blue block"],
                "primary_subject_phrases": ["red block", "blue block"],
                "query_subject_phrases": ["red block", "blue block"],
            },
            "Both the red block and the blue block while they are moving.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["red block", "blue block"])
        self.assertEqual(plan["query_subject_phrases"], ["red block", "blue block"])

    def test_set_query_preserves_every_planner_subject_for_later_semantic_filtering(self) -> None:
        objects = ["object one", "object two", "object three", "object four", "object five", "object six"]
        plan = _normalize_plan(
            {
                "video_inventory_phrases": objects,
                "primary_subject_phrases": objects,
                "query_subject_phrases": objects,
                "must_track_phrases": objects,
            },
            "All objects that remain physically stationary throughout the video.",
        )

        self.assertEqual(plan["primary_subject_phrases"], objects)
        self.assertEqual(plan["query_subject_phrases"], objects)
        self.assertEqual(plan["must_track_phrases"], objects)

    def test_counted_leading_subject_survives_a_compact_vlm_plan(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["hands", "keyboard"],
                "primary_subject_phrases": ["hands"],
                "query_subject_phrases": ["hands"],
                "must_track_phrases": ["hands"],
            },
            "Both hands that are actively typing on the keyboard.",
        )

        self.assertEqual(plan["primary_subject_phrases"], ["both hands"])
        self.assertEqual(plan["query_subject_phrases"], ["both hands"])
        self.assertEqual(plan["requested_instance_count"], 2)
        self.assertIn("hand", plan["detector_phrases"])

    def test_counted_subject_preserves_modifiers_without_an_object_lexicon(self) -> None:
        plan = _normalize_plan(
            {
                "video_inventory_phrases": ["red cups", "table"],
                "primary_subject_phrases": ["red cups"],
                "query_subject_phrases": ["red cups"],
            },
            "The two red cups while they are being moved.",
        )

        self.assertEqual(plan["query_subject_phrases"], ["two red cups"])
        self.assertEqual(plan["requested_instance_count"], 2)

    def test_normalization_keeps_object_names_and_matches_spelling_generically(self) -> None:
        self.assertEqual(_canonicalize_phrase("cocktail-glass"), "cocktail glass")
        self.assertEqual(_canonicalize_phrase("mouse-pad"), "mouse pad")
        score = _candidate_phrase_score("mouse pad", {"proposal_alias": "mousepad"})
        self.assertGreaterEqual(score, 0.9)

    def test_counted_subjects_use_generic_distinct_entity_selection(self) -> None:
        phrases = _expand_counted_subject_phrases(["both cups"])
        self.assertEqual(phrases, ["cup", "cup"])
        candidates = [
            {"id": 1, "proposal_alias": "red cup", "quality": 0.9},
            {"id": 2, "proposal_alias": "blue cup", "quality": 0.8},
        ]
        selected_ids, selected_by_phrase = _select_phrase_ids(candidates, phrases)
        self.assertEqual(selected_ids, [1, 2])
        self.assertEqual(selected_by_phrase["cup"], [1, 2])

    def test_count_neutral_detector_variant_is_not_category_specific(self) -> None:
        phrases = _count_neutral_detector_phrases(["two packages"])
        self.assertIn("two packages", phrases)
        self.assertIn("packages", phrases)
        self.assertIn("package", phrases)

    def test_declared_multi_hypothesis_keeps_distinct_stage1_instances(self) -> None:
        candidates = [
            {
                "id": 3,
                "stage1_object_id": 11,
                "proposal_alias": "left hand",
                "proposal_phrase": "left hand",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
            {
                "id": 4,
                "stage1_object_id": 12,
                "proposal_alias": "left hand",
                "proposal_phrase": "left hand",
                "quality": 0.8,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
        ]
        payload = _compose_phrase_grounded_selection(
            query="The left hand while it is typing.",
            query_plan_payload={"query_subject_phrases": ["left hand"]},
            candidates=candidates,
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload={
                "instance_candidate_groups": [
                    {
                        "semantic_phrase": "left hand",
                        "object_ids": [11, 12],
                        "selection_policy": "multi_hypothesis",
                    }
                ]
            },
            raw_phrase_payload={"subject_phrases": ["left hand"], "successor_phrases": []},
        )

        self.assertEqual(payload["selection_mode"], "stage1_multi_hypothesis")
        self.assertEqual([item["id"] for item in payload["selected"]], [3, 4])
        self.assertTrue(all(item["segments"] == [[0, 9]] for item in payload["selected"]))

    def test_relation_disambiguation_keeps_only_the_visually_selected_instance_family(self) -> None:
        candidates = [
            {
                "id": 3,
                "stage1_object_id": 11,
                "stage1_instance_group_id": "instance_group_000",
                "stage1_instance_index": 0,
                "instance_selection_policy": "relation_disambiguation",
                "proposal_alias": "part instance 1",
                "proposal_phrase": "part",
                "static_text": "part",
                "quality": 0.99,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
            {
                "id": 4,
                "stage1_object_id": 12,
                "stage1_instance_group_id": "instance_group_000",
                "stage1_instance_index": 1,
                "instance_selection_policy": "relation_disambiguation",
                "proposal_alias": "part instance 2",
                "proposal_phrase": "part",
                "static_text": "part",
                "quality": 0.10,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
        ]
        payload = _compose_phrase_grounded_selection(
            query="The part moving the device.",
            query_plan_payload={"query_subject_phrases": ["part"]},
            candidates=candidates,
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload={
                "instance_candidate_groups": [
                    {
                        "semantic_phrase": "part",
                        "object_ids": [11, 12],
                        "selection_policy": "relation_disambiguation",
                    }
                ]
            },
            raw_phrase_payload={"subject_phrases": ["part instance 2"], "successor_phrases": []},
        )

        self.assertEqual([item["id"] for item in payload["selected"]], [4])

    def test_singular_plan_rejects_selector_added_interaction_partner(self) -> None:
        candidates = [
            {
                "id": 3,
                "proposal_alias": "part instance 1",
                "proposal_phrase": "part",
                "quality": 0.8,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
            {
                "id": 4,
                "proposal_alias": "device",
                "proposal_phrase": "device",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 9]],
            },
        ]
        payload = _compose_phrase_grounded_selection(
            query="The part moving the device.",
            query_plan_payload={
                "query_subject_phrases": ["part"],
                "relation_context_phrases": ["device"],
                "action_window_hint": "during the movement",
            },
            candidates=candidates,
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload=None,
            raw_phrase_payload={
                "subject_phrases": ["part instance 1", "device"],
                "successor_phrases": [],
            },
        )

        self.assertEqual([item["id"] for item in payload["selected"]], [3])
        self.assertEqual(payload["subject_phrases"], ["part instance 1"])

    def test_relation_geometry_overrides_arbitrary_instance_numbering(self) -> None:
        def track(object_id: int, phrase: str, bbox: list[int], *, anchor: int = 10) -> dict:
            return {
                "object_id": object_id,
                "phrase": phrase,
                "anchor_frame_index": anchor,
                "frames": [
                    {
                        "frame_index": frame_index,
                        "active": True,
                        "bbox_xyxy": bbox,
                        "mask_path": f"{object_id}_{frame_index}.png",
                        "time_value": frame_index / 20.0,
                    }
                    for frame_index in (0, 10, 20)
                ],
            }

        candidates = [
            {
                "id": 3,
                "stage1_object_id": 11,
                "stage1_instance_group_id": "instance_group_000",
                "stage1_instance_index": 0,
                "instance_selection_policy": "relation_disambiguation",
                "proposal_alias": "part instance 1",
                "proposal_phrase": "part",
                "static_text": "part",
                "quality": 0.99,
                "support_segments_test": [[0, 9]],
            },
            {
                "id": 4,
                "stage1_object_id": 12,
                "stage1_instance_group_id": "instance_group_000",
                "stage1_instance_index": 1,
                "instance_selection_policy": "relation_disambiguation",
                "proposal_alias": "part instance 2",
                "proposal_phrase": "part",
                "static_text": "part",
                "quality": 0.10,
                "support_segments_test": [[0, 9]],
            },
            {
                "id": 5,
                "stage1_object_id": 13,
                "proposal_alias": "device",
                "proposal_phrase": "device",
                "quality": 0.8,
                "support_segments_test": [[0, 9]],
            },
        ]
        payload = _compose_phrase_grounded_selection(
            query="The part moving the device.",
            query_plan_payload={
                "query_subject_phrases": ["part"],
                "relation_context_phrases": ["device"],
                "action_window_hint": "during the movement",
            },
            candidates=candidates,
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload={
                "instance_candidate_groups": [
                    {
                        "semantic_phrase": "part",
                        "object_ids": [11, 12],
                        "selection_policy": "relation_disambiguation",
                    }
                ],
                "tracks": [
                    track(11, "part", [300, 300, 340, 340]),
                    track(12, "part", [100, 100, 140, 140]),
                    track(13, "device", [105, 100, 165, 160]),
                ],
            },
            raw_phrase_payload={
                "subject_phrases": ["part instance 1", "device"],
                "successor_phrases": [],
            },
        )

        self.assertEqual([item["id"] for item in payload["selected"]], [4])
        self.assertTrue(payload["relation_disambiguation"]["applied"])
        self.assertEqual(payload["relation_disambiguation"]["winner_object_id"], 12)

    def test_relation_disambiguation_exposes_stable_instance_aliases(self) -> None:
        candidates = [
            {
                "id": 3,
                "stage1_object_id": 11,
                "stage1_instance_index": 0,
                "proposal_alias": "part",
                "proposal_phrase": "part",
            },
            {
                "id": 4,
                "stage1_object_id": 12,
                "stage1_instance_index": 1,
                "proposal_alias": "part",
                "proposal_phrase": "part",
            },
        ]
        decorated = _decorate_relation_disambiguation_candidates(
            candidates,
            {
                "instance_candidate_groups": [
                    {
                        "object_ids": [11, 12],
                        "selection_policy": "relation_disambiguation",
                    }
                ]
            },
        )

        self.assertEqual(
            [row["proposal_alias"] for row in decorated],
            ["part instance 1", "part instance 2"],
        )
        self.assertTrue(all(row["instance_selection_policy"] == "relation_disambiguation" for row in decorated))

    def test_multi_hypothesis_uses_each_instance_synchronized_stage1_support(self) -> None:
        test_times = np.linspace(0.0, 1.0, num=10)
        candidates = [
            {
                "id": 3,
                "stage1_object_id": 11,
                "proposal_alias": "left hand",
                "proposal_phrase": "left hand",
                "static_text": "left hand",
                "quality": 1.0,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 1]],
                "moving_segments_test": [[5, 9]],
            },
            {
                "id": 4,
                "stage1_object_id": 12,
                "proposal_alias": "left hand",
                "proposal_phrase": "left hand",
                "static_text": "left hand",
                "quality": 1.0,
                "support_segments_test": [[0, 9]],
                "query_relevant_segments_test": [[0, 1]],
                "moving_segments_test": [[5, 9]],
            },
        ]
        tracks = []
        for object_id, indices in ((11, range(5, 9)), (12, range(6, 10))):
            tracks.append(
                {
                    "object_id": object_id,
                    "frames": [
                        {
                            "active": True,
                            "mask_path": f"instance_{object_id}_{index}.png",
                            "time_value": float(test_times[index]),
                        }
                        for index in indices
                    ],
                }
            )

        payload = _compose_phrase_grounded_selection(
            query="The left hand while it is moving.",
            query_plan_payload={"query_subject_phrases": ["left hand"]},
            candidates=candidates,
            pair_candidates=[],
            test_times=test_times,
            tracks_payload={
                "tracks": tracks,
                "instance_candidate_groups": [
                    {
                        "semantic_phrase": "left hand",
                        "object_ids": [11, 12],
                        "selection_policy": "multi_hypothesis",
                    }
                ],
            },
            raw_phrase_payload={"subject_phrases": ["left hand"], "successor_phrases": []},
        )

        self.assertEqual([item["segments"] for item in payload["selected"]], [[[5, 8]], [[6, 9]]])

    def test_lifecycle_mode_does_not_bypass_static_set_membership(self) -> None:
        candidates = [
            {
                "id": index,
                "proposal_alias": phrase,
                "proposal_phrase": phrase,
                "static_text": phrase,
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "stationary_segments_test": [[0, 9]],
                "moving_segments_test": [],
            }
            for index, phrase in enumerate(("object one", "object two", "object three", "moving object"), start=1)
        ]
        candidates[-1]["stationary_segments_test"] = [[0, 2]]
        candidates[-1]["moving_segments_test"] = [[3, 9]]

        with patch.dict(os.environ, {"QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT": "1"}, clear=False):
            payload = _compose_phrase_grounded_selection(
                query="All objects that remain physically stationary throughout the video.",
                query_plan_payload={
                    "query_subject_phrases": [item["proposal_phrase"] for item in candidates],
                    "query_semantic_profile": {"asks_set": True},
                },
                candidates=candidates,
                pair_candidates=[],
                test_times=np.linspace(0.0, 1.0, num=10),
                tracks_payload=None,
                raw_phrase_payload=None,
            )

        self.assertEqual(payload["selection_mode"], "qwen_plan_static_full_video")
        self.assertEqual([item["id"] for item in payload["selected"]], [1, 2, 3])

    def test_static_set_keeps_visual_member_not_listed_in_plan_subjects(self) -> None:
        candidates = [
            {
                "id": 1,
                "proposal_alias": "primary object",
                "proposal_phrase": "primary object",
                "static_text": "primary object",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "stationary_segments_test": [],
                "moving_segments_test": [[0, 9]],
            },
            {
                "id": 2,
                "proposal_alias": "compound foreground region",
                "proposal_phrase": "compound foreground region",
                "static_text": "compound foreground region",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "stationary_segments_test": [],
                "moving_segments_test": [[0, 9]],
            },
        ]

        with patch.dict(os.environ, {"QUERY_ENTITY_LIFECYCLE_TEMPORAL_OUTPUT": "1"}, clear=False):
            payload = _compose_phrase_grounded_selection(
                query="All objects that remain physically stationary throughout the video.",
                query_plan_payload={
                    "query_subject_phrases": ["primary object"],
                    "query_semantic_profile": {"asks_set": True},
                },
                candidates=candidates,
                pair_candidates=[],
                test_times=np.linspace(0.0, 1.0, num=10),
                tracks_payload=None,
                raw_phrase_payload={
                    "subject_phrases": ["primary object", "compound foreground region"],
                    "successor_phrases": [],
                },
            )

        self.assertEqual(payload["selection_mode"], "qwen_visual_static_set")
        self.assertEqual([item["id"] for item in payload["selected"]], [1, 2])

    def test_static_set_filters_unrequested_scene_spanning_support_proxy(self) -> None:
        candidates = [
            {
                "id": 1,
                "proposal_alias": "foreground object",
                "proposal_phrase": "foreground object",
                "static_text": "foreground object",
                "entity_type": "object",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "stationary_segments_test": [[0, 9]],
                "moving_segments_test": [],
            },
            {
                "id": 2,
                "proposal_alias": "large support surface",
                "proposal_phrase": "large support surface",
                "static_text": "large support surface",
                "entity_type": "support_surface",
                "quality": 0.9,
                "support_segments_test": [[0, 9]],
                "stationary_segments_test": [[0, 9]],
                "moving_segments_test": [],
                "static_set_mask_geometry": {"median_area_fraction": 0.42},
            },
        ]

        payload = _compose_phrase_grounded_selection(
            query="All objects that remain physically stationary throughout the video.",
            query_plan_payload={
                "query_subject_phrases": ["foreground object", "large support surface"],
                "query_semantic_profile": {"asks_set": True},
            },
            candidates=candidates,
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload=None,
            raw_phrase_payload={
                "subject_phrases": ["foreground object", "large support surface"],
                "successor_phrases": [],
            },
        )

        self.assertEqual([item["id"] for item in payload["selected"]], [1])
        self.assertEqual(
            payload["selection_filters"]["excluded_scene_spanning_support_proxies"],
            [{"id": 2, "alias": "large support surface", "reason": "scene_spanning_support_proxy"}],
        )

    def test_static_set_keeps_a_scene_spanning_support_when_explicitly_named(self) -> None:
        candidate = {
            "id": 2,
            "proposal_alias": "large support surface",
            "proposal_phrase": "large support surface",
            "static_text": "large support surface",
            "entity_type": "support_surface",
            "quality": 0.9,
            "support_segments_test": [[0, 9]],
            "stationary_segments_test": [[0, 9]],
            "moving_segments_test": [],
            "static_set_mask_geometry": {"median_area_fraction": 0.42},
        }

        payload = _compose_phrase_grounded_selection(
            query="The large support surface that remains physically stationary throughout the video.",
            query_plan_payload={"query_subject_phrases": ["large support surface"]},
            candidates=[candidate],
            pair_candidates=[],
            test_times=np.linspace(0.0, 1.0, num=10),
            tracks_payload=None,
            raw_phrase_payload={"subject_phrases": ["large support surface"], "successor_phrases": []},
        )

        self.assertEqual([item["id"] for item in payload["selected"]], [2])
        self.assertEqual(payload["selection_filters"]["excluded_scene_spanning_support_proxies"], [])

    def test_qwen_budget_uses_available_large_gpu_memory_without_a_hidden_16gib_cap(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REFERGAUSSIAN_QWEN_GPU_RESERVE_GIB", None)
            os.environ.pop("REFERGAUSSIAN_QWEN_GPU_MAX_GIB", None)
            self.assertEqual(_qwen_gpu_memory_budget_gib(32), 28)
            self.assertEqual(_qwen_gpu_memory_budget_gib(12), 8)
        with patch.dict(
            os.environ,
            {
                "REFERGAUSSIAN_QWEN_GPU_RESERVE_GIB": "3",
                "REFERGAUSSIAN_QWEN_GPU_MAX_GIB": "20",
            },
            clear=False,
        ):
            self.assertEqual(_qwen_gpu_memory_budget_gib(32), 20)

    def test_qwen_generation_budget_defaults_to_existing_limit_and_is_bounded(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("REFERGAUSSIAN_QWEN_MAX_NEW_TOKENS", None)
            self.assertEqual(_qwen_max_new_tokens(), 1024)
        with patch.dict(
            os.environ,
            {"REFERGAUSSIAN_QWEN_MAX_NEW_TOKENS": "640"},
            clear=False,
        ):
            self.assertEqual(_qwen_max_new_tokens(), 640)

    def test_identity_attribute_contract_rejects_a_broad_category_substitute(self) -> None:
        candidates = [
            {
                "id": 7,
                "proposal_alias": "chocolate bar",
                "static_text": "dark brown elongated object",
                "global_desc": "A dark brown object on a tray.",
                "concept_tags": ["dark", "brown"],
            }
        ]
        with patch.dict(os.environ, {"QUERY_STRICT_ATTRIBUTE_EMPTY_ON_MISMATCH": "1"}, clear=False):
            verification = _identity_attribute_verification(
                query_plan_payload={"required_identity_attributes": ["white"]},
                raw_phrase_payload={"subject_phrases": ["chocolate bar"], "verified_identity_attributes": []},
                selection_payload={"selected": [{"id": 7}]},
                candidates=candidates,
            )

        self.assertTrue(verification["enabled"])
        self.assertTrue(verification["evaluated"])
        self.assertFalse(verification["passed"])
        self.assertEqual(verification["missing_attributes"], ["white"])

    def test_identity_attribute_contract_accepts_visual_selector_verification(self) -> None:
        candidates = [
            {
                "id": 7,
                "proposal_alias": "chocolate bar",
                "static_text": "dark brown elongated object",
                "global_desc": "A dark brown object on a tray.",
                "concept_tags": ["dark", "brown"],
            }
        ]
        with patch.dict(os.environ, {"QUERY_STRICT_ATTRIBUTE_EMPTY_ON_MISMATCH": "1"}, clear=False):
            verification = _identity_attribute_verification(
                query_plan_payload={"required_identity_attributes": ["white"]},
                raw_phrase_payload={"subject_phrases": ["chocolate bar"], "verified_identity_attributes": ["white"]},
                selection_payload={"selected": [{"id": 7}]},
                candidates=candidates,
            )

        self.assertTrue(verification["passed"])
        self.assertEqual(verification["verification_sources"], {"white": ["selector_visual"]})


if __name__ == "__main__":
    unittest.main()
