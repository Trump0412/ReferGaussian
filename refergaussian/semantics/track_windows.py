"""Small geometry-free helpers for Stage-1 temporal track windows."""

from __future__ import annotations

import math
from collections.abc import Iterable


def required_anchor_window_radius(
    anchor_frame_indices: Iterable[int],
    *,
    first_frame_index: int,
    last_frame_index: int,
) -> int:
    """Return the smallest symmetric radius that covers the full frame range.

    Grounded-SAM2 may propagate from several independently selected anchors.
    A nominal radius can leave a gap between two anchor windows, which makes a
    later boundary gate see no synchronized Stage-1 mask.  This computes the
    minimum radius that prevents such gaps without changing the anchors,
    detector prompts, or semantic selection.
    """
    first = int(first_frame_index)
    last = int(last_frame_index)
    if last < first:
        raise ValueError(f"Invalid frame range [{first}, {last}]")

    anchors = sorted({int(value) for value in anchor_frame_indices})
    if not anchors:
        raise ValueError("At least one anchor frame is required")
    if anchors[0] < first or anchors[-1] > last:
        raise ValueError(
            f"Anchor frames {anchors[0]}..{anchors[-1]} lie outside [{first}, {last}]"
        )

    edge_radius = max(anchors[0] - first, last - anchors[-1])
    bridge_radius = max(
        (int(math.ceil((right - left) / 2.0)) for left, right in zip(anchors, anchors[1:])),
        default=0,
    )
    return int(max(0, edge_radius, bridge_radius))
