#!/usr/bin/env python3
"""Benchmark YOLO person detection on CUDA and Intel OpenVINO backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


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


def normalized_people(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    people: list[dict[str, Any]] = []
    for box, score in zip(boxes.tolist(), scores.tolist(), strict=True):
        x1, y1, x2, y2 = (float(value) for value in box)
        normalized = [
            max(0.0, min(1.0, x1 / width)),
            max(0.0, min(1.0, y1 / height)),
            max(0.0, min(1.0, x2 / width)),
            max(0.0, min(1.0, y2 / height)),
        ]
        if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
            people.append({"box": normalized, "score": round(float(score), 6)})
    people.sort(key=lambda detection: tuple(detection["box"]))
    return people


def load_rgb(path: Path) -> tuple[np.ndarray, int, int]:
    with Image.open(path) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    width, height = image.size
    return np.asarray(image), width, height


def letterbox_bgr(
    bgr: np.ndarray,
    image_size: int,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    import cv2

    height, width = bgr.shape[:2]
    ratio = min(image_size / height, image_size / width)
    resized_width = round(width * ratio)
    resized_height = round(height * ratio)
    horizontal_padding = round((image_size - resized_width) / 2 - 0.1)
    vertical_padding = round((image_size - resized_height) / 2 - 0.1)
    if (resized_width, resized_height) != (width, height):
        bgr = cv2.resize(
            bgr,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )
    right = image_size - resized_width - horizontal_padding
    bottom = image_size - resized_height - vertical_padding
    prepared = cv2.copyMakeBorder(
        bgr,
        vertical_padding,
        bottom,
        horizontal_padding,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    return prepared, ratio, (horizontal_padding, vertical_padding)


def restore_boxes(
    boxes: np.ndarray,
    ratio: float,
    padding: tuple[int, int],
    width: int,
    height: int,
) -> np.ndarray:
    restored = boxes.copy()
    restored[:, [0, 2]] = (restored[:, [0, 2]] - padding[0]) / ratio
    restored[:, [1, 3]] = (restored[:, [1, 3]] - padding[1]) / ratio
    restored[:, [0, 2]] = restored[:, [0, 2]].clip(0, width)
    restored[:, [1, 3]] = restored[:, [1, 3]].clip(0, height)
    return restored


def cuda_runner(
    model_path: str,
    device: str,
    image_size: int,
    confidence: float,
    rect: bool,
) -> tuple[Callable[[np.ndarray], tuple[list[dict[str, Any]], float]], float, str]:
    from ultralytics import YOLO

    started = time.perf_counter()
    model = YOLO(model_path)
    load_seconds = time.perf_counter() - started

    def predict(rgb: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        height, width = rgb.shape[:2]
        result = model.predict(
            source=rgb[:, :, ::-1],
            classes=[0],
            conf=confidence,
            imgsz=image_size,
            device=device,
            rect=rect,
            verbose=False,
        )[0]
        if result.boxes is None:
            return [], float(result.speed.get("inference") or 0.0)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        scores = result.boxes.conf.detach().cpu().numpy()
        return (
            normalized_people(boxes, scores, width, height),
            float(result.speed.get("inference") or 0.0),
        )

    return predict, load_seconds, "CUDA"


def openvino_runner(
    model_path: str,
    device: str,
    image_size: int,
    confidence: float,
    _rect: bool,
) -> tuple[Callable[[np.ndarray], tuple[list[dict[str, Any]], float]], float, str]:
    import openvino as ov

    started = time.perf_counter()
    core = ov.Core()
    model_file = next(Path(model_path).glob("*.xml"))
    model = core.read_model(model=model_file)
    compiled = core.compile_model(
        model,
        device_name=device,
        config={"PERFORMANCE_HINT": "LATENCY"},
    )
    input_name = compiled.input().get_any_name()
    execution_device = ", ".join(compiled.get_property("EXECUTION_DEVICES"))
    load_seconds = time.perf_counter() - started

    def predict(rgb: np.ndarray) -> tuple[list[dict[str, Any]], float]:
        height, width = rgb.shape[:2]
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        prepared, ratio, padding = letterbox_bgr(bgr, image_size)
        tensor = prepared[:, :, ::-1].transpose(2, 0, 1)
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)[None] / 255.0
        infer_started = time.perf_counter()
        output = next(iter(compiled({input_name: tensor}).values()))
        inference_ms = (time.perf_counter() - infer_started) * 1000
        predictions = np.asarray(output)[0]
        selected = predictions[
            (predictions[:, 4] >= confidence) & (predictions[:, 5].astype(int) == 0)
        ]
        if not len(selected):
            return [], inference_ms
        boxes = restore_boxes(
            selected[:, :4].copy(),
            ratio,
            padding,
            width,
            height,
        )
        return (
            normalized_people(boxes, selected[:, 4], width, height),
            inference_ms,
        )

    return predict, load_seconds, execution_device


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices={"cuda", "openvino"}, required=True)
    parser.add_argument("--photo-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--confidence", type=float, default=0.18)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument(
        "--rect",
        action="store_true",
        help="Use rectangular CUDA preprocessing; OpenVINO export is fixed-square.",
    )
    args = parser.parse_args()

    if args.repeats < 1:
        raise ValueError("repeats must be at least one")
    items = inventory(args.photo_dir)
    if not items:
        raise RuntimeError("photo corpus is empty")

    factory = cuda_runner if args.backend == "cuda" else openvino_runner
    predict, load_seconds, execution_device = factory(
        args.model,
        args.device,
        args.image_size,
        args.confidence,
        args.rect,
    )
    first_rgb, _, _ = load_rgb(items[0]["path"])
    predict(first_rgb)

    records: list[dict[str, Any]] = []
    total_times: list[float] = []
    inference_times: list[float] = []
    pass_times: list[float] = []
    for repeat in range(args.repeats):
        pass_started = time.perf_counter()
        for index, item in enumerate(items, start=1):
            started = time.perf_counter()
            rgb, width, height = load_rgb(item["path"])
            people, inference_ms = predict(rgb)
            total_ms = (time.perf_counter() - started) * 1000
            total_times.append(total_ms)
            inference_times.append(inference_ms)
            if repeat == 0:
                records.append(
                    {
                        "image_id": item["id"],
                        "width": width,
                        "height": height,
                        "model": args.model,
                        "backend": args.backend,
                        "confidence": args.confidence,
                        "people": people,
                    }
                )
            if index % 50 == 0 or index == len(items):
                print(f"{args.backend} pass {repeat + 1}: {index}/{len(items)}")
        pass_times.append(time.perf_counter() - pass_started)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, separators=(",", ":"), sort_keys=True))
            handle.write("\n")
    metrics = {
        "backend": args.backend,
        "device_requested": args.device,
        "device_used": execution_device,
        "model": args.model,
        "photos": len(items),
        "repeats": args.repeats,
        "model_load_seconds": load_seconds,
        "pass_seconds": pass_times,
        "photos_per_second": [len(items) / pass_seconds for pass_seconds in pass_times],
        "total_latency_ms": {
            "p50": percentile(total_times, 0.50),
            "p90": percentile(total_times, 0.90),
            "p95": percentile(total_times, 0.95),
            "max": max(total_times),
        },
        "inference_latency_ms": {
            "p50": percentile(inference_times, 0.50),
            "p90": percentile(inference_times, 0.90),
            "p95": percentile(inference_times, 0.95),
            "max": max(inference_times),
        },
        "person_boxes": sum(len(record["people"]) for record in records),
    }
    args.metrics.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
