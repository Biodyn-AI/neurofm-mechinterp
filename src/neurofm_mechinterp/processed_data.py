"""Helpers for loading locally preprocessed BrainOmni-style `.pt` segments."""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import torch

from .data import BrainBatch


def load_metadata(metadata_path: Path) -> list[dict]:
    """Load a metadata JSON list produced by the upstream dataset-maker scripts."""
    with metadata_path.open("r") as handle:
        return json.load(handle)


def choose_examples(
    metadata: list[dict],
    max_segments: int,
    channel_count: int | None = None,
) -> list[dict]:
    """
    Select a fixed number of examples, optionally constraining the channel count.

    Keeping a consistent channel count makes batching straightforward.
    """
    if channel_count is None and metadata:
        channel_count = metadata[0]["channels"]
    filtered = [row for row in metadata if row["channels"] == channel_count]
    if len(filtered) <= max_segments:
        return filtered

    if not filtered or "subject" not in filtered[0]:
        return filtered[:max_segments]

    by_subject: dict[str, deque[dict]] = {}
    for row in filtered:
        by_subject.setdefault(row["subject"], deque()).append(row)

    selected = []
    subject_order = list(by_subject.keys())
    while len(selected) < max_segments:
        advanced = False
        for subject in subject_order:
            queue = by_subject[subject]
            if not queue:
                continue
            selected.append(queue.popleft())
            advanced = True
            if len(selected) >= max_segments:
                break
        if not advanced:
            break
    return selected


def load_batch_from_examples(examples: list[dict], root: Path | None = None) -> BrainBatch:
    """Load a list of `.pt` files into one batched tensor container."""
    if not examples:
        raise ValueError("No examples were provided.")

    xs, poses, sensor_types = [], [], []
    for row in examples:
        path = Path(row["path"])
        if root is not None and not path.is_absolute():
            path = root / path
        payload = torch.load(path, map_location="cpu", weights_only=True)
        xs.append(payload["x"])
        poses.append(payload["pos"])
        sensor_types.append(payload["sensor_type"])

    return BrainBatch(
        x=torch.stack(xs, dim=0),
        pos=torch.stack(poses, dim=0),
        sensor_type=torch.stack(sensor_types, dim=0),
    )
