"""Offline crop-v2 candidate generation.

This module is intentionally separate from the production renderer while the
heuristic is evaluated against a real photo corpus.
"""

from __future__ import annotations

import math
from typing import Any

Box = tuple[float, float, float, float]


def _valid_boxes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for item in items:
        box = item.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        try:
            x1, y1, x2, y2 = (float(value) for value in box)
        except (TypeError, ValueError):
            continue
        clipped = [
            min(1.0, max(0.0, x1)),
            min(1.0, max(0.0, y1)),
            min(1.0, max(0.0, x2)),
            min(1.0, max(0.0, y2)),
        ]
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            continue
        normalized = dict(item)
        normalized["box"] = clipped
        valid.append(normalized)
    return valid


def _box_area(item: dict[str, Any]) -> float:
    x1, y1, x2, y2 = item["box"]
    return (x2 - x1) * (y2 - y1)


def _box_center(item: dict[str, Any]) -> tuple[float, float]:
    x1, y1, x2, y2 = item["box"]
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _intersection_area(first: dict[str, Any], second: dict[str, Any]) -> float:
    first_x1, first_y1, first_x2, first_y2 = first["box"]
    second_x1, second_y1, second_x2, second_y2 = second["box"]
    width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    return width * height


def _deduplicate_boxes(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collapse repeated detections without merging a child held by an adult."""

    def confidence(item: dict[str, Any]) -> float:
        try:
            return float(item.get("score", 0.0))
        except (TypeError, ValueError):
            return 0.0

    kept: list[dict[str, Any]] = []
    ranked = sorted(
        items,
        key=lambda item: (confidence(item), _box_area(item)),
        reverse=True,
    )
    for candidate in ranked:
        candidate_area = _box_area(candidate)
        duplicate = False
        for existing in kept:
            existing_area = _box_area(existing)
            size_similarity = min(candidate_area, existing_area) / max(
                candidate_area, existing_area
            )
            overlap = _intersection_area(candidate, existing) / min(candidate_area, existing_area)
            if size_similarity >= 0.55 and overlap >= 0.80:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)
    return kept


def _choose_primary(
    items: list[dict[str, Any]],
    *,
    area_ratio: float,
    nearby_distance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not items:
        return [], []
    largest_area = max(_box_area(item) for item in items)
    primary = [item for item in items if _box_area(item) >= largest_area * area_ratio]
    total_weight = sum(math.sqrt(_box_area(item)) for item in primary)
    center_x = (
        sum(_box_center(item)[0] * math.sqrt(_box_area(item)) for item in primary) / total_weight
    )
    center_y = (
        sum(_box_center(item)[1] * math.sqrt(_box_area(item)) for item in primary) / total_weight
    )
    for item in items:
        if item in primary:
            continue
        item_x, item_y = _box_center(item)
        if math.hypot(item_x - center_x, item_y - center_y) <= nearby_distance:
            primary.append(item)
    dropped = [item for item in items if item not in primary]
    return primary, dropped


def _expanded_bounds(
    items: list[dict[str, Any]],
    *,
    horizontal_margin: float,
    vertical_margin: float,
) -> Box:
    expanded: list[Box] = []
    for item in items:
        x1, y1, x2, y2 = item["box"]
        width = x2 - x1
        height = y2 - y1
        expanded.append(
            (
                max(0.0, x1 - width * horizontal_margin),
                max(0.0, y1 - height * vertical_margin),
                min(1.0, x2 + width * horizontal_margin),
                min(1.0, y2 + height * vertical_margin),
            )
        )
    return (
        min(box[0] for box in expanded),
        min(box[1] for box in expanded),
        max(box[2] for box in expanded),
        max(box[3] for box in expanded),
    )


def _person_context_bounds(items: list[dict[str, Any]]) -> Box:
    """Keep full seated people and the useful upper body of standing people."""
    context_items: list[dict[str, Any]] = []
    for item in items:
        x1, y1, x2, y2 = item["box"]
        width = x2 - x1
        height = y2 - y1
        # Feet are rarely the subject on a landscape photo frame. Letting the
        # crop trim lower legs creates materially closer portraits while still
        # retaining full seated, lying, and close-up subjects.
        context_bottom = y1 + height * 0.76 if height / width >= 1.45 else y2
        context_item = dict(item)
        context_item["box"] = [x1, y1, x2, context_bottom]
        context_items.append(context_item)
    return _expanded_bounds(
        context_items,
        horizontal_margin=0.08,
        vertical_margin=0.08,
    )


def _contains_center(container: dict[str, Any], item: dict[str, Any]) -> bool:
    x1, y1, x2, y2 = container["box"]
    center_x, center_y = _box_center(item)
    return x1 <= center_x <= x2 and y1 <= center_y <= y2


def _place_crop(
    crop_width: float,
    crop_height: float,
    safe_bounds: Box,
    focus_x: float,
    focus_y: float,
    vertical_focus: float = 0.38,
) -> list[float]:
    safe_left, safe_top, safe_right, safe_bottom = safe_bounds
    left = min(max(focus_x - crop_width / 2, 0.0), 1.0 - crop_width)
    # Put faces above center, leaving more visual room below them.
    top = min(max(focus_y - crop_height * vertical_focus, 0.0), 1.0 - crop_height)
    left = min(left, safe_left)
    left = max(left, safe_right - crop_width)
    top = min(top, safe_top)
    top = max(top, safe_bottom - crop_height)
    left = min(max(left, 0.0), 1.0 - crop_width)
    top = min(max(top, 0.0), 1.0 - crop_height)
    return [left, top, left + crop_width, top + crop_height]


def calculate_baseline_crop(
    image_width: int,
    image_height: int,
    faces: list[dict[str, Any]],
    target_width: int,
    target_height: int,
    *,
    margin_ratio: float = 0.75,
    min_retained_fraction: float = 0.75,
    min_face_retained_fraction: float = 0.50,
) -> dict[str, Any]:
    """Mirror the production aspect-only face crop for offline comparisons."""
    valid_faces = _valid_boxes(faces)
    source_aspect = image_width / image_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    retained_fraction = crop_width * crop_height
    retained_threshold = min_face_retained_fraction if valid_faces else min_retained_fraction
    if retained_fraction < retained_threshold:
        return {
            "mode": "contain-aspect-fallback",
            "box": [0.0, 0.0, 1.0, 1.0],
            "retained_fraction": retained_fraction,
        }
    if not valid_faces:
        left = (1.0 - crop_width) / 2
        top = (1.0 - crop_height) / 2
        return {
            "mode": "cover-center",
            "box": [left, top, left + crop_width, top + crop_height],
        }

    safe_bounds = _expanded_bounds(
        valid_faces,
        horizontal_margin=margin_ratio,
        vertical_margin=margin_ratio,
    )
    safe_width = safe_bounds[2] - safe_bounds[0]
    safe_height = safe_bounds[3] - safe_bounds[1]
    if safe_width > crop_width or safe_height > crop_height:
        return {"mode": "contain-fallback", "box": [0.0, 0.0, 1.0, 1.0]}
    focus_x = (safe_bounds[0] + safe_bounds[2]) / 2
    focus_y = (safe_bounds[1] + safe_bounds[3]) / 2
    return {
        "mode": "cover-face-safe",
        "box": _place_crop(
            crop_width,
            crop_height,
            safe_bounds,
            focus_x,
            focus_y,
            vertical_focus=0.42,
        ),
    }


def calculate_subject_zoom_crop(
    image_width: int,
    image_height: int,
    faces: list[dict[str, Any]],
    target_width: int,
    target_height: int,
    *,
    people: list[dict[str, Any]] | None = None,
    max_zoom: float = 3.0,
    target_single_face_height: float = 0.24,
    target_group_face_height: float = 0.20,
    dominant_face_area_ratio: float = 0.20,
    dominant_person_area_ratio: float = 0.18,
    minimum_person_area: float = 0.004,
) -> dict[str, Any]:
    """Return a tighter landscape crop around primary people and faces.

    Face boxes establish identity-safe bounds. Optional person boxes preserve
    bodies and improve cases where faces are small, obscured, or turned away.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("source dimensions must be positive")
    if target_width <= 0 or target_height <= 0:
        raise ValueError("target dimensions must be positive")
    if max_zoom < 1:
        raise ValueError("maximum zoom must be at least 1")

    valid_faces = _valid_boxes(faces)
    candidate_people = _deduplicate_boxes(_valid_boxes(people or []))
    valid_people = [
        person for person in candidate_people if _box_area(person) >= minimum_person_area
    ]
    primary_faces, dropped_faces = _choose_primary(
        valid_faces,
        area_ratio=dominant_face_area_ratio,
        nearby_distance=0.28,
    )
    primary_people, dropped_people = _choose_primary(
        valid_people,
        area_ratio=dominant_person_area_ratio,
        nearby_distance=0.20,
    )
    # A small baby/child box can legitimately sit inside a larger adult box.
    # Conversely, a person detection containing a primary face should never be
    # discarded merely because their visible body is small.
    face_linked_people = [
        person
        for person in dropped_people
        if any(_contains_center(person, face) for face in primary_faces)
    ]
    if face_linked_people:
        primary_people.extend(face_linked_people)
        dropped_people = [person for person in dropped_people if person not in face_linked_people]

    if not primary_faces and not primary_people:
        if candidate_people:
            baseline = calculate_baseline_crop(
                image_width,
                image_height,
                valid_faces,
                target_width,
                target_height,
            )
            return {
                **baseline,
                "mode": f"smart-baseline-{baseline['mode']}",
                "zoom": 1.0,
                "primary_face_count": 0,
                "dropped_face_count": 0,
                "primary_person_count": 0,
                "dropped_person_count": 0,
            }
        return {
            "mode": "smart-contain-no-faces",
            "box": [0.0, 0.0, 1.0, 1.0],
            "zoom": 1.0,
            "primary_face_count": 0,
            "dropped_face_count": 0,
            "primary_person_count": 0,
            "dropped_person_count": 0,
        }

    source_aspect = image_width / image_height
    target_aspect = target_width / target_height
    normalized_aspect = target_aspect / source_aspect
    maximum_crop_height = 1.0 / normalized_aspect if normalized_aspect >= 1 else 1.0

    person_bounds = _person_context_bounds(primary_people) if primary_people else None
    if primary_faces:
        face_bounds = _expanded_bounds(
            primary_faces,
            horizontal_margin=0.45,
            vertical_margin=0.45,
        )
    else:
        face_bounds = None

    bounds = [box for box in (person_bounds, face_bounds) if box is not None]
    safe_bounds = (
        min(box[0] for box in bounds),
        min(box[1] for box in bounds),
        max(box[2] for box in bounds),
        max(box[3] for box in bounds),
    )
    safe_width = safe_bounds[2] - safe_bounds[0]
    safe_height = safe_bounds[3] - safe_bounds[1]
    required_crop_height = max(safe_height, safe_width / normalized_aspect)
    if required_crop_height > maximum_crop_height + 1e-9:
        baseline = calculate_baseline_crop(
            image_width,
            image_height,
            valid_faces,
            target_width,
            target_height,
        )
        if baseline["mode"].startswith("cover"):
            return {
                **baseline,
                "mode": f"smart-baseline-{baseline['mode']}",
                "zoom": 1.0,
                "primary_face_count": len(primary_faces),
                "dropped_face_count": len(dropped_faces),
                "primary_person_count": len(primary_people),
                "dropped_person_count": len(dropped_people),
            }
        return {
            "mode": "smart-contain-group",
            "box": [0.0, 0.0, 1.0, 1.0],
            "zoom": 1.0,
            "primary_face_count": len(primary_faces),
            "dropped_face_count": len(dropped_faces),
            "primary_person_count": len(primary_people),
            "dropped_person_count": len(dropped_people),
        }

    desired_crop_height = required_crop_height
    if primary_faces:
        maximum_face_height = max(
            float(face["box"][3]) - float(face["box"][1]) for face in primary_faces
        )
        target_face_height = (
            target_single_face_height if len(primary_faces) == 1 else target_group_face_height
        )
        desired_crop_height = max(
            desired_crop_height,
            maximum_face_height / target_face_height,
        )
    effective_max_zoom = min(max_zoom, 1.75) if primary_faces and not primary_people else max_zoom
    desired_crop_height = max(
        desired_crop_height,
        maximum_crop_height / effective_max_zoom,
    )
    crop_height = min(maximum_crop_height, desired_crop_height)
    crop_width = crop_height * normalized_aspect

    focus_items = primary_faces or primary_people
    focus_weights = [math.sqrt(_box_area(item)) for item in focus_items]
    focus_weight = sum(focus_weights)
    focus_x = (
        sum(
            _box_center(item)[0] * weight
            for item, weight in zip(focus_items, focus_weights, strict=True)
        )
        / focus_weight
    )
    focus_y = (
        sum(
            _box_center(item)[1] * weight
            for item, weight in zip(focus_items, focus_weights, strict=True)
        )
        / focus_weight
    )
    box = _place_crop(crop_width, crop_height, safe_bounds, focus_x, focus_y)
    zoom = maximum_crop_height / crop_height
    mode = "smart-person-zoom" if primary_people else "smart-face-zoom"
    return {
        "mode": mode,
        "box": box,
        "zoom": zoom,
        "primary_face_count": len(primary_faces),
        "dropped_face_count": len(dropped_faces),
        "primary_person_count": len(primary_people),
        "dropped_person_count": len(dropped_people),
    }
