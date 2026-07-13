"""Tests for the declared Stage-1 multi-instance fast-path contract."""

from __future__ import annotations

import unittest

from refergaussian.semantics.instance_contract import declared_multi_instance_group


def _plan(**overrides: object) -> dict:
    payload = {
        "query_subject_phrases": ["both hands"],
        "query_successor_phrases": [],
    }
    payload.update(overrides)
    return payload


def _tracks(**overrides: object) -> dict:
    payload = {
        "instance_candidate_groups": [
            {
                "semantic_phrase": "both hands",
                "selection_policy": "multi_hypothesis",
                "object_ids": [4, 9],
            }
        ]
    }
    payload.update(overrides)
    return payload


def _entitybank(*object_ids: int) -> dict:
    return {
        "entities": [
            {"id": index, "stage1_object_id": object_id}
            for index, object_id in enumerate(object_ids)
        ]
    }


class DeclaredInstanceContractTest(unittest.TestCase):
    def test_accepts_matching_fully_lifted_declared_group(self) -> None:
        group = declared_multi_instance_group(_plan(), _tracks(), _entitybank(4, 9))
        self.assertEqual(
            group,
            {
                "semantic_phrase": "both hands",
                "object_ids": [4, 9],
                "selection_policy": "multi_hypothesis",
            },
        )

    def test_rejects_group_with_a_different_semantic_phrase(self) -> None:
        group = declared_multi_instance_group(
            _plan(),
            _tracks(instance_candidate_groups=[{
                "semantic_phrase": "both cups",
                "selection_policy": "multi_hypothesis",
                "object_ids": [4, 9],
            }]),
            _entitybank(4, 9),
        )
        self.assertIsNone(group)

    def test_rejects_partially_lifted_group(self) -> None:
        self.assertIsNone(declared_multi_instance_group(_plan(), _tracks(), _entitybank(4)))

    def test_rejects_successor_or_multiple_subject_plan(self) -> None:
        self.assertIsNone(
            declared_multi_instance_group(
                _plan(query_successor_phrases=["hand fragments"]), _tracks(), _entitybank(4, 9)
            )
        )
        self.assertIsNone(
            declared_multi_instance_group(
                _plan(query_subject_phrases=["left hand", "right hand"]), _tracks(), _entitybank(4, 9)
            )
        )
