#!/usr/bin/env python3
"""Run an offline GPU person-detection pass for crop evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def normalized_boxes(result: Any) -> list[dict[str, Any]]:
    height, width = (int(value) for value in result.orig_shape)
    detections: list[dict[str, Any]] = []
    if result.boxes is None:
        return detections
    boxes = result.boxes.xyxy.detach().cpu().tolist()
    scores = result.boxes.conf.detach().cpu().tolist()
    for box, score in zip(boxes, scores, strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        normalized = [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
        if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
            detections.append({"box": normalized, "score": round(float(score), 6)})
    detections.sort(key=lambda detection: tuple(detection["box"]))
    return detections


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
        items.append({"id": stable_id, "path": path, "relative": relative})
    return sorted(items, key=lambda item: item["id"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    items = inventory(args.photo_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    detected_people = 0
    with temporary.open("w", encoding="utf-8") as handle:
        for offset in range(0, len(items), args.batch_size):
            batch = items[offset : offset + args.batch_size]
            pixels: list[np.ndarray] = []
            dimensions: list[tuple[int, int]] = []
            for item in batch:
                with Image.open(item["path"]) as original:
                    image = ImageOps.exif_transpose(original).convert("RGB")
                    dimensions.append(image.size)
                    pixels.append(np.asarray(image)[:, :, ::-1])
            results = model.predict(
                source=pixels,
                classes=[0],
                conf=args.confidence,
                imgsz=args.image_size,
                device=args.device,
                verbose=False,
            )
            for item, size, result in zip(batch, dimensions, results, strict=True):
                people = normalized_boxes(result)
                detected_people += len(people)
                record = {
                    "image_id": item["id"],
                    "width": size[0],
                    "height": size[1],
                    "model": args.model,
                    "confidence": args.confidence,
                    "people": people,
                }
                handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
                handle.write("\n")
            print(f"analyzed {min(offset + len(batch), len(items))}/{len(items)}")
    temporary.replace(args.output)
    print(f"saved {len(items)} photos with {detected_people} person boxes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
