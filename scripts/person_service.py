#!/usr/bin/env python3
"""Small private HTTP service for GPU-backed foreground-person detection."""

from __future__ import annotations

import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image, ImageOps
from ultralytics import YOLO

MODEL_PATH = os.getenv("PERSON_MODEL_PATH", "/models/yolo26s.pt")
CONFIDENCE = float(os.getenv("PERSON_CONFIDENCE", "0.18"))
IMAGE_SIZE = int(os.getenv("PERSON_IMAGE_SIZE", "1280"))
MAX_BODY_BYTES = 25 * 1024 * 1024
MODEL = YOLO(MODEL_PATH)
MODEL_LOCK = threading.Lock()


def detect_people(body: bytes) -> dict[str, object]:
    with Image.open(io.BytesIO(body)) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
    width, height = image.size
    with MODEL_LOCK:
        result = MODEL.predict(
            source=np.asarray(image),
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            classes=[0],
            device=0,
            verbose=False,
        )[0]
    people: list[dict[str, object]] = []
    if result.boxes is not None:
        boxes = result.boxes.xyxy.detach().cpu().tolist()
        scores = result.boxes.conf.detach().cpu().tolist()
        for box, score in zip(boxes, scores, strict=True):
            x1, y1, x2, y2 = (float(value) for value in box)
            people.append(
                {
                    "box": [
                        max(0.0, min(1.0, x1 / width)),
                        max(0.0, min(1.0, y1 / height)),
                        max(0.0, min(1.0, x2 / width)),
                        max(0.0, min(1.0, y2 / height)),
                    ],
                    "score": round(float(score), 6),
                }
            )
    return {
        "model": os.path.basename(MODEL_PATH),
        "confidence": CONFIDENCE,
        "width": width,
        "height": height,
        "people": people,
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "FrameFeedPersonDetector/1"

    def _json(self, status: int, value: dict[str, object]) -> None:
        encoded = json.dumps(value, separators=(",", ":")).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json(200, {"status": "ok", "model": os.path.basename(MODEL_PATH)})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/predict":
            self._json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if length <= 0 or length > MAX_BODY_BYTES:
            self._json(413, {"error": "invalid image size"})
            return
        try:
            result = detect_people(self.rfile.read(length))
        except Exception as exc:
            self._json(422, {"error": type(exc).__name__})
            return
        self._json(200, result)

    def log_message(self, message: str, *args: object) -> None:
        print(f"{self.client_address[0]} {message % args}", flush=True)


def main() -> None:
    port = int(os.getenv("PORT", "8000"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    print(f"person detector ready on port {port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
