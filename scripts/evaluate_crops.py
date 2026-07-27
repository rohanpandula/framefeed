#!/usr/bin/env python3
"""Compare production crops with offline crop-v2 candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from framefeed.crop_v2 import calculate_baseline_crop, calculate_subject_zoom_crop

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CELL_SIZE = (400, 300)
SHEET_ROWS = 6


def inventory(photo_dir: Path) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in photo_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        stat = path.stat()
        relative = path.relative_to(photo_dir).as_posix()
        stable_id = hashlib.sha256(
            f"{relative}\0{stat.st_size}".encode("utf-8", "surrogateescape")
        ).hexdigest()
        items.append({"id": stable_id, "path": path})
    return sorted(items, key=lambda item: item["id"])


def load_analysis(path: Path, version: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if value.get("analysis_version") == version:
                records[str(value["image_id"])] = value
    return records


def load_people(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            records[str(value["image_id"])] = value
    return records


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def oriented_image(path: Path) -> Image.Image:
    with Image.open(path) as original:
        return ImageOps.exif_transpose(original).convert("RGB")


def render_contain(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    background = ImageOps.fit(
        image,
        size,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5),
    ).filter(ImageFilter.GaussianBlur(18))
    background = Image.blend(background, Image.new("RGB", size, "black"), 0.18)
    foreground = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    background.paste(
        foreground,
        ((size[0] - foreground.width) // 2, (size[1] - foreground.height) // 2),
    )
    return background


def render_decision(
    image: Image.Image, decision: dict[str, Any], size: tuple[int, int]
) -> Image.Image:
    if "contain" in decision["mode"]:
        return render_contain(image, size)
    left, top, right, bottom = (float(value) for value in decision["box"])
    width, height = image.size
    pixel_box = (
        max(0, min(width - 1, round(left * width))),
        max(0, min(height - 1, round(top * height))),
        max(1, min(width, round(right * width))),
        max(1, min(height, round(bottom * height))),
    )
    return image.crop(pixel_box).resize(size, Image.Resampling.LANCZOS)


def draw_box(
    draw: ImageDraw.ImageDraw,
    box: list[float],
    source_size: tuple[int, int],
    offset: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(
        (
            offset[0] + round(x1 * source_size[0]),
            offset[1] + round(y1 * source_size[1]),
            offset[0] + round(x2 * source_size[0]),
            offset[1] + round(y2 * source_size[1]),
        ),
        outline=color,
        width=3,
    )


def labeled_cell(image: Image.Image, label: str) -> Image.Image:
    cell = image.copy()
    draw = ImageDraw.Draw(cell)
    font = ImageFont.load_default()
    bounds = draw.textbbox((0, 0), label, font=font)
    draw.rectangle((4, 4, 12 + bounds[2], 10 + bounds[3]), fill=(0, 0, 0))
    draw.text((8, 7), label, fill=(255, 255, 255), font=font)
    return cell


def original_cell(image: Image.Image, entry: dict[str, Any], code: str) -> Image.Image:
    source = image.copy()
    source.thumbnail(CELL_SIZE, Image.Resampling.LANCZOS)
    cell = Image.new("RGB", CELL_SIZE, (15, 15, 15))
    offset = ((CELL_SIZE[0] - source.width) // 2, (CELL_SIZE[1] - source.height) // 2)
    cell.paste(source, offset)
    draw = ImageDraw.Draw(cell)
    for face in entry["faces"]:
        draw_box(draw, face["box"], source.size, offset, (50, 255, 80))
    for person in entry["people"]:
        draw_box(draw, person["box"], source.size, offset, (60, 210, 255))
    return labeled_cell(
        cell,
        f"{code} source faces={len(entry['faces'])} people={len(entry['people'])}",
    )


def make_sheet(
    output: Path,
    name: str,
    entries: list[dict[str, Any]],
    sample_records: dict[str, dict[str, Any]],
) -> None:
    chosen = entries[:SHEET_ROWS]
    if not chosen:
        return
    sheet = Image.new("RGB", (CELL_SIZE[0] * 3, CELL_SIZE[1] * len(chosen)), (20, 20, 20))
    for row, entry in enumerate(chosen):
        code = f"{name[:2].upper()}{row + 1:02d}"
        source = oriented_image(entry["path"])
        baseline = render_decision(source, entry["baseline"], CELL_SIZE)
        smart = render_decision(source, entry["smart"], CELL_SIZE)
        sheet.paste(original_cell(source, entry, code), (0, row * CELL_SIZE[1]))
        sheet.paste(
            labeled_cell(baseline, f"current {entry['baseline']['mode']}"),
            (CELL_SIZE[0], row * CELL_SIZE[1]),
        )
        sheet.paste(
            labeled_cell(
                smart,
                f"v2 {entry['smart']['mode']} zoom={entry['smart']['zoom']:.2f}x",
            ),
            (CELL_SIZE[0] * 2, row * CELL_SIZE[1]),
        )
        sample_records[code] = {
            "category": name,
            "image_id_prefix": entry["id"][:12],
            "dimensions": list(entry["size"]),
            "faces": entry["faces"],
            "people": entry["people"],
            "baseline": entry["baseline"],
            "smart": entry["smart"],
        }
    sheet.save(output / f"{name}.jpg", quality=92, subsampling=0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--analysis-jsonl", type=Path, required=True)
    parser.add_argument("--analysis-version", required=True)
    parser.add_argument("--person-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1200)
    args = parser.parse_args()

    if args.output.exists():
        raise RuntimeError(f"output already exists: {args.output}")
    args.output.mkdir(parents=True)
    analyses = load_analysis(args.analysis_jsonl, args.analysis_version)
    people_records = load_people(args.person_jsonl)
    entries: list[dict[str, Any]] = []
    for item in inventory(args.photo_dir):
        analysis = analyses.get(item["id"])
        if not analysis:
            continue
        source = oriented_image(item["path"])
        source_size = source.size
        people = list((people_records.get(item["id"]) or {}).get("people") or [])
        faces = list(analysis.get("faces") or [])
        baseline = calculate_baseline_crop(
            source.width,
            source.height,
            faces,
            args.width,
            args.height,
        )
        smart = calculate_subject_zoom_crop(
            source.width,
            source.height,
            faces,
            args.width,
            args.height,
            people=people,
        )
        source.close()
        entries.append(
            {
                "id": item["id"],
                "path": item["path"],
                "size": source_size,
                "faces": faces,
                "people": people,
                "baseline": baseline,
                "smart": smart,
            }
        )

    zoomed = [entry for entry in entries if entry["smart"]["zoom"] > 1.05]
    baseline_contains = [
        entry for entry in entries if entry["baseline"]["mode"].startswith("contain")
    ]
    recovered = [
        entry
        for entry in baseline_contains
        if entry["smart"]["mode"] in {"smart-face-zoom", "smart-person-zoom"}
    ]
    dropped = [
        entry
        for entry in entries
        if entry["smart"]["dropped_face_count"] or entry["smart"]["dropped_person_count"]
    ]
    no_face_people = [entry for entry in entries if not entry["faces"] and entry["people"]]
    ambiguous = [
        entry
        for entry in entries
        if entry["smart"]["mode"] == "smart-contain-group"
        or entry["smart"]["zoom"] >= 2.75
        or len(entry["people"]) >= 5
    ]
    portraits = [entry for entry in entries if entry["size"][1] > entry["size"][0]]

    categories = {
        "largest_zoom": sorted(
            zoomed,
            key=lambda entry: (-entry["smart"]["zoom"], entry["id"]),
        ),
        "fallback_recovered": sorted(
            recovered,
            key=lambda entry: (-entry["smart"]["zoom"], entry["id"]),
        ),
        "background_dropped": sorted(
            dropped,
            key=lambda entry: (
                -entry["smart"]["dropped_face_count"] - entry["smart"]["dropped_person_count"],
                entry["id"],
            ),
        ),
        "no_face_people": sorted(no_face_people, key=lambda entry: entry["id"]),
        "ambiguous": sorted(
            ambiguous,
            key=lambda entry: (-len(entry["people"]), -entry["smart"]["zoom"], entry["id"]),
        ),
        "portrait_sample": sorted(portraits, key=lambda entry: entry["id"]),
    }
    samples: dict[str, dict[str, Any]] = {}
    for name, category in categories.items():
        make_sheet(args.output, name, category, samples)

    zoom_values = [float(entry["smart"]["zoom"]) for entry in zoomed]
    metrics = {
        "evaluated_photos": len(entries),
        "photos_with_person_boxes": sum(bool(entry["people"]) for entry in entries),
        "photos_without_faces_but_with_people": len(no_face_people),
        "baseline_modes": Counter(entry["baseline"]["mode"] for entry in entries),
        "smart_modes": Counter(entry["smart"]["mode"] for entry in entries),
        "baseline_contain_recovered_by_v2": len(recovered),
        "photos_zoomed_over_1_05x": len(zoomed),
        "photos_dropping_background_boxes": len(dropped),
        "ambiguous_review_queue": len(ambiguous),
        "zoom_percentiles_for_zoomed_photos": {
            "p10": percentile(zoom_values, 0.10),
            "p25": percentile(zoom_values, 0.25),
            "p50": percentile(zoom_values, 0.50),
            "p75": percentile(zoom_values, 0.75),
            "p90": percentile(zoom_values, 0.90),
        },
    }
    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output / "samples.json").write_text(
        json.dumps(samples, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
