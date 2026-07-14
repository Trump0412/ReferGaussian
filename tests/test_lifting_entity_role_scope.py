"""Regression coverage for planner-role-scoped Gaussian lifting."""

from __future__ import annotations

import ast
import os
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
LIFTING_PATH = ROOT / "refergaussian" / "semantics" / "mask_supported_lifting.py"


def _load_role_scope_filter():
    module = ast.parse(LIFTING_PATH.read_text(encoding="utf-8"))
    names = {
        "_entity_role_scope",
        "_canonical_role_phrase",
        "_query_subject_role_phrases",
        "_query_plan_requires_entity_set",
        "_filter_tracks_for_entity_roles",
    }
    nodes = [
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"Any": Any, "os": os, "re": __import__("re")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(LIFTING_PATH), "exec"), namespace)
    return namespace["_filter_tracks_for_entity_roles"]


class LiftingEntityRoleScopeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.filter_tracks = _load_role_scope_filter()
        self.plan = {
            "primary_subject_phrases": ["chocolate bar"],
            "query_subject_phrases": ["chocolate bar"],
            "must_track_phrases": ["chocolate bar"],
        }

    def test_default_scope_retains_every_detector_track(self) -> None:
        tracks = [
            {"phrase": "chocolate bar"},
            {"phrase": "burnt chocolate residue"},
        ]
        with patch.dict("os.environ", {}, clear=False):
            selected, info = self.filter_tracks(tracks, self.plan)

        self.assertEqual(selected, tracks)
        self.assertEqual(info["match_mode"], "all_tracks")
        self.assertEqual(info["output_track_count"], 2)

    def test_subject_scope_drops_state_expansion_tracks(self) -> None:
        tracks = [
            {"phrase": "chocolate bar"},
            {"phrase": "burnt chocolate residue"},
            {"phrase": "melting chocolate bar"},
        ]
        with patch.dict(
            "os.environ",
            {"QUERY_LIFT_ENTITY_ROLE_SCOPE": "primary_subject"},
            clear=False,
        ):
            selected, info = self.filter_tracks(tracks, self.plan)

        self.assertEqual(selected, [tracks[0]])
        self.assertEqual(info["match_mode"], "direct_subject_phrase")
        self.assertEqual(info["skipped_track_phrases"], ["burnt chocolate residue", "melting chocolate bar"])

    def test_subject_scope_retains_all_same_noun_instance_hypotheses(self) -> None:
        tracks = [
            {"phrase": "hand", "object_id": 1},
            {"phrase": "hand", "object_id": 2},
            {"phrase": "blowtorch", "object_id": 3},
        ]
        plan = {"primary_subject_phrases": ["hand"]}
        with patch.dict(
            "os.environ",
            {"QUERY_LIFT_ENTITY_ROLE_SCOPE": "primary_subject"},
            clear=False,
        ):
            selected, info = self.filter_tracks(tracks, plan)

        self.assertEqual(selected, tracks[:2])
        self.assertEqual(info["match_mode"], "direct_subject_phrase")

    def test_subject_scope_retains_all_tracks_for_a_planner_declared_set(self) -> None:
        tracks = [
            {"phrase": "object a"},
            {"phrase": "object b"},
            {"phrase": "object c"},
        ]
        plan = {
            "primary_subject_phrases": ["object a"],
            "query_semantic_profile": {"asks_set": True},
        }
        with patch.dict(
            "os.environ",
            {"QUERY_LIFT_ENTITY_ROLE_SCOPE": "primary_subject"},
            clear=False,
        ):
            selected, info = self.filter_tracks(tracks, plan)

        self.assertEqual(selected, tracks)
        self.assertEqual(info["match_mode"], "set_query_all_tracks")

    def test_subject_scope_relaxes_only_when_direct_phrase_is_unavailable(self) -> None:
        tracks = [
            {"phrase": "solid chocolate bar"},
            {"phrase": "burnt chocolate residue"},
        ]
        with patch.dict(
            "os.environ",
            {"QUERY_LIFT_ENTITY_ROLE_SCOPE": "primary_subject"},
            clear=False,
        ):
            selected, info = self.filter_tracks(tracks, self.plan)

        self.assertEqual(selected, [tracks[0]])
        self.assertEqual(info["match_mode"], "relaxed_subject_phrase")

    def test_unresolved_subject_scope_never_silently_drops_every_track(self) -> None:
        tracks = [{"phrase": "cup"}, {"phrase": "table"}]
        with patch.dict(
            "os.environ",
            {"QUERY_LIFT_ENTITY_ROLE_SCOPE": "primary_subject"},
            clear=False,
        ):
            selected, info = self.filter_tracks(tracks, self.plan)

        self.assertEqual(selected, tracks)
        self.assertEqual(info["match_mode"], "scope_unresolved_no_matching_track")


if __name__ == "__main__":
    unittest.main()
