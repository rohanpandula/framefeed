#!/usr/bin/env python3
"""Build a private, static e-paper photo frame from a photo folder.

The worker optionally mirrors an Apple iCloud Shared Album that has its
"Public Website" switch enabled. It never exposes the album URL, original
filenames, or source directory through the generated site.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import logging
import math
import os
import re
import shutil
import signal
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import cv2
import numpy as np
import requests
from PIL import Image, ImageFilter, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

LOG = logging.getLogger("framefeed")
USER_AGENT = "FrameFeed/0.1 (+https://github.com/rohanpandula/framefeed)"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp"}
SECRET_RE = re.compile(r"^[A-Za-z0-9_-]{48,128}$")
ALBUM_ID_RE = re.compile(r"^[A-Za-z0-9_-]{5,128}$")
ASSET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,200}$")
MAX_ASSET_BYTES = 100 * 1024 * 1024


def utc_now() -> float:
    return time.time()


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.chmod(temporary, 0o600)
    temporary.replace(path)


def atomic_text(path: Path, value: str, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(value)
        temporary = Path(handle.name)
    os.chmod(temporary, mode)
    temporary.replace(path)


def load_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else default
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def read_secret(path: Path) -> str:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"cannot read secret file: {path}") from exc
    if not SECRET_RE.fullmatch(value):
        raise RuntimeError("frame path secret must be 48-128 URL-safe characters")
    return value


def read_optional_secret(path: Path) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(f"cannot read optional secret file: {path}") from exc
    return value or None


@dataclass(frozen=True)
class Config:
    photo_dir: Path
    site_root: Path
    state_dir: Path
    frame_secret_file: Path
    album_url_file: Path
    width: int = 1200
    height: int = 1600
    rotation_seconds: int = 3600
    new_photo_hold_seconds: int = 3900
    scan_interval_seconds: int = 300
    album_sync_interval_seconds: int = 900
    jpeg_quality: int = 88
    blur_radius: int = 56
    face_detector: str = "yunet"
    face_model_path: Path = Path("/app/models/face_detection_yunet_2023mar.onnx")
    face_analysis_url: str = ""
    face_model: str = "buffalo_l"
    face_min_score: float = 0.30
    face_margin_ratio: float = 0.75
    min_crop_retained_fraction: float = 0.0
    analysis_batch_size: int = 8

    @classmethod
    def from_environment(cls) -> Config:
        return cls(
            photo_dir=Path(os.getenv("PHOTO_DIR", "/photos")),
            site_root=Path(os.getenv("SITE_ROOT", "/site")),
            state_dir=Path(os.getenv("STATE_DIR", "/state")),
            frame_secret_file=Path(os.getenv("FRAME_SECRET_FILE", "/run/secrets/frame_path")),
            album_url_file=Path(
                os.getenv(
                    "ICLOUD_SHARED_ALBUM_URL_FILE",
                    "/run/secrets/icloud_shared_album_url",
                )
            ),
            width=int(os.getenv("FRAME_WIDTH", "1200")),
            height=int(os.getenv("FRAME_HEIGHT", "1600")),
            rotation_seconds=int(os.getenv("ROTATION_SECONDS", "3600")),
            new_photo_hold_seconds=int(os.getenv("NEW_PHOTO_HOLD_SECONDS", "3900")),
            scan_interval_seconds=int(os.getenv("SCAN_INTERVAL_SECONDS", "300")),
            album_sync_interval_seconds=int(os.getenv("ALBUM_SYNC_INTERVAL_SECONDS", "900")),
            jpeg_quality=int(os.getenv("JPEG_QUALITY", "88")),
            blur_radius=int(os.getenv("BLUR_RADIUS", "56")),
            face_detector=os.getenv("FACE_DETECTOR", "yunet").strip().lower(),
            face_model_path=Path(
                os.getenv(
                    "FACE_MODEL_PATH",
                    "/app/models/face_detection_yunet_2023mar.onnx",
                )
            ),
            face_analysis_url=os.getenv("FACE_ANALYSIS_URL", "").strip(),
            face_model=os.getenv("FACE_MODEL", "buffalo_l").strip(),
            face_min_score=float(os.getenv("FACE_MIN_SCORE", "0.30")),
            face_margin_ratio=float(os.getenv("FACE_MARGIN_RATIO", "0.75")),
            min_crop_retained_fraction=float(os.getenv("MIN_CROP_RETAINED_FRACTION", "0.0")),
            analysis_batch_size=int(os.getenv("ANALYSIS_BATCH_SIZE", "8")),
        )

    def validate(self) -> None:
        if not 100 <= self.width <= 10000 or not 100 <= self.height <= 10000:
            raise RuntimeError("frame dimensions must be between 100 and 10000 pixels")
        if self.rotation_seconds < 300:
            raise RuntimeError("rotation interval must be at least 5 minutes")
        if self.new_photo_hold_seconds < 300:
            raise RuntimeError("new-photo hold must be at least 5 minutes")
        if self.scan_interval_seconds < 30:
            raise RuntimeError("scan interval must be at least 30 seconds")
        if self.album_sync_interval_seconds < 60:
            raise RuntimeError("album sync interval must be at least 60 seconds")
        if not 50 <= self.jpeg_quality <= 95:
            raise RuntimeError("JPEG quality must be between 50 and 95")
        if not 0.05 <= self.face_min_score <= 1:
            raise RuntimeError("face detection score must be between 0.05 and 1")
        if not 0 <= self.face_margin_ratio <= 3:
            raise RuntimeError("face margin ratio must be between 0 and 3")
        if not 0 <= self.min_crop_retained_fraction <= 1:
            raise RuntimeError("minimum retained crop fraction must be between 0 and 1")
        if not 0 <= self.analysis_batch_size <= 100:
            raise RuntimeError("analysis batch size must be between 0 and 100")
        if self.face_detector not in {"yunet", "immich", "none"}:
            raise RuntimeError("face detector must be yunet, immich, or none")
        if self.face_detector == "immich" and not self.face_analysis_url:
            raise RuntimeError("FACE_ANALYSIS_URL is required with FACE_DETECTOR=immich")


def parse_shared_album_id(url: str) -> str:
    parsed = urlparse(url.strip())
    if parsed.scheme != "https" or parsed.hostname not in {
        "www.icloud.com",
        "icloud.com",
    }:
        raise ValueError("expected an https://www.icloud.com/sharedalbum/... URL")
    if "sharedalbum" not in parsed.path.lower():
        raise ValueError("URL is not an iCloud Shared Album public website")
    album_id = parsed.fragment or parsed.path.rstrip("/").split("/")[-1]
    album_id = unquote(album_id).strip()
    if not ALBUM_ID_RE.fullmatch(album_id):
        raise ValueError("iCloud Shared Album URL does not contain a valid album ID")
    return album_id


def _post_json(session: requests.Session, url: str, body: dict[str, Any]) -> dict[str, Any]:
    response = session.post(
        url,
        json=body,
        timeout=(10, 45),
        headers={"Accept": "application/json", "User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    value = response.json()
    if not isinstance(value, dict):
        raise RuntimeError("unexpected non-object response from iCloud")
    return value


def _best_derivative(photo: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    candidates: list[tuple[int, int, str, dict[str, Any]]] = []
    derivatives = photo.get("derivatives") or {}
    if not isinstance(derivatives, dict):
        return None
    for derivative in derivatives.values():
        if not isinstance(derivative, dict):
            continue
        checksum = str(derivative.get("checksum") or "")
        if not checksum:
            continue
        try:
            size = int(derivative.get("fileSize") or 0)
            pixels = int(derivative.get("width") or 0) * int(derivative.get("height") or 0)
        except (TypeError, ValueError):
            size, pixels = 0, 0
        candidates.append((size, pixels, checksum, derivative))
    if not candidates:
        return None
    _, _, checksum, derivative = max(candidates)
    return checksum, derivative


def _asset_url_map(value: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    items = value.get("items") or {}
    if isinstance(items, dict):
        iterable = items.items()
    elif isinstance(items, list):
        iterable = ((str(index), item) for index, item in enumerate(items))
    else:
        iterable = []
    for key, item in iterable:
        if not isinstance(item, dict):
            continue
        location = str(item.get("url_location") or "").lower()
        path = str(item.get("url_path") or "")
        trusted_host = location.endswith((".icloud.com", ".icloud-content.com"))
        trusted_host = trusted_host or location in {"icloud.com", "icloud-content.com"}
        if trusted_host and re.fullmatch(r"[a-z0-9.-]+", location) and path.startswith("/"):
            result[str(key)] = f"https://{location}{path}"
    return result


def sync_shared_album(
    album_url: str,
    destination: Path,
    manifest_path: Path,
    session: requests.Session | None = None,
) -> dict[str, Any]:
    """Mirror still images from an iCloud Shared Album public website."""
    album_id = parse_shared_album_id(album_url)
    destination.mkdir(parents=True, exist_ok=True)
    session = session or requests.Session()
    base = f"https://p01-sharedstreams.icloud.com/{album_id}/sharedstreams"
    stream = _post_json(session, f"{base}/webstream", {"streamCtag": None})
    redirected_host = stream.get("X-Apple-MMe-Host")
    if redirected_host:
        host = str(redirected_host).strip()
        if not re.fullmatch(r"[A-Za-z0-9.-]+\.icloud\.com", host):
            raise RuntimeError("iCloud returned an invalid redirect host")
        base = f"https://{host}/{album_id}/sharedstreams"
        stream = _post_json(session, f"{base}/webstream", {"streamCtag": None})

    photos = stream.get("photos") or []
    if not isinstance(photos, list):
        raise RuntimeError("iCloud response did not contain a photo list")

    photo_by_guid: dict[str, dict[str, Any]] = {}
    wanted_checksum: dict[str, str] = {}
    for photo in photos:
        if not isinstance(photo, dict):
            continue
        guid = str(photo.get("photoGuid") or "")
        best = _best_derivative(photo)
        if ASSET_ID_RE.fullmatch(guid) and best:
            checksum, _ = best
            photo_by_guid[guid] = photo
            wanted_checksum[guid] = checksum

    urls_by_checksum: dict[str, str] = {}
    guids = list(photo_by_guid)
    for offset in range(0, len(guids), 100):
        batch = guids[offset : offset + 100]
        response = _post_json(session, f"{base}/webasseturls", {"photoGuids": batch})
        urls_by_checksum.update(_asset_url_map(response))

    previous = load_json(manifest_path, {"assets": {}})
    previous_assets = previous.get("assets") or {}
    current_assets: dict[str, dict[str, Any]] = {}
    downloaded = 0
    for guid, photo in photo_by_guid.items():
        checksum = wanted_checksum[guid]
        url = urls_by_checksum.get(checksum)
        if not url:
            # Some API variants use a URL-map key containing the checksum.
            url = next(
                (candidate for key, candidate in urls_by_checksum.items() if checksum in key),
                None,
            )
        if not url:
            LOG.warning("Skipping an album item whose download URL is unavailable")
            continue

        old_candidate = previous_assets.get(guid) if isinstance(previous_assets, dict) else None
        old_asset = old_candidate if isinstance(old_candidate, dict) else None
        old_filename = str((old_asset or {}).get("filename") or "")
        old_path = (
            destination / old_filename
            if old_filename and Path(old_filename).name == old_filename
            else None
        )
        if old_asset and old_asset.get("checksum") == checksum and old_path and old_path.is_file():
            filename = old_filename
        else:
            response = session.get(
                url,
                stream=True,
                timeout=(10, 120),
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].lower()
            extension = {
                "image/jpeg": ".jpg",
                "image/png": ".png",
                "image/heic": ".heic",
                "image/heif": ".heif",
                "image/webp": ".webp",
            }.get(content_type, ".jpg")
            if content_type and not content_type.startswith("image/"):
                LOG.info("Skipping a non-image asset from the shared album")
                continue
            filename = f"{guid}{extension}"
            target = destination / filename
            with tempfile.NamedTemporaryFile(dir=destination, delete=False) as handle:
                temporary = Path(handle.name)
                total = 0
                try:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        total += len(chunk)
                        if total > MAX_ASSET_BYTES:
                            raise RuntimeError("iCloud image exceeds the 100 MB safety limit")
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                with Image.open(temporary) as downloaded_image:
                    downloaded_image.verify()
                os.chmod(temporary, 0o640)
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
            downloaded += 1

        current_assets[guid] = {
            "checksum": checksum,
            "filename": filename,
            "date_created": photo.get("dateCreated"),
        }

    # Removed album items are archived, never destructively deleted.
    stale = (
        set(previous_assets) - set(current_assets) if isinstance(previous_assets, dict) else set()
    )
    if stale:
        archive = destination / ".removed" / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        archive.mkdir(parents=True, exist_ok=True)
        for guid in stale:
            stale_candidate = previous_assets.get(guid)
            stale_asset = stale_candidate if isinstance(stale_candidate, dict) else {}
            filename = str(stale_asset.get("filename") or "")
            source = destination / filename
            if filename and Path(filename).name == filename and source.is_file():
                shutil.move(str(source), archive / source.name)

    manifest = {
        "album_id_hash": hashlib.sha256(album_id.encode()).hexdigest(),
        "assets": current_assets,
        "last_sync": utc_now(),
    }
    atomic_json(manifest_path, manifest)
    return {"assets": len(current_assets), "downloaded": downloaded, "removed": len(stale)}


def image_inventory(photo_dir: Path) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    if not photo_dir.exists():
        return result
    for path in photo_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        if ".removed" in path.parts:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = path.relative_to(photo_dir).as_posix()
        stable_id = hashlib.sha256(
            f"{relative}\0{stat.st_size}".encode("utf-8", "surrogateescape")
        ).hexdigest()
        result.append(
            {
                "id": stable_id,
                "path": path,
                "mtime": stat.st_mtime,
                "relative": relative,
            }
        )
    return result


def add_shared_album_dates(inventory: list[dict[str, Any]], manifest_path: Path) -> None:
    manifest = load_json(manifest_path, {})
    assets = manifest.get("assets") or {}
    dates_by_filename: dict[str, float] = {}
    if isinstance(assets, dict):
        for asset in assets.values():
            if not isinstance(asset, dict):
                continue
            filename = str(asset.get("filename") or "")
            raw_date = str(asset.get("date_created") or "")
            if not filename or not raw_date:
                continue
            try:
                parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
            except ValueError:
                continue
            dates_by_filename[filename] = parsed.timestamp()
    for item in inventory:
        timestamp = dates_by_filename.get(item["relative"])
        if timestamp is not None:
            item["date_created"] = timestamp


def update_inventory_state(
    state: dict[str, Any], inventory: list[dict[str, Any]], now: float, hold_seconds: int
) -> dict[str, Any]:
    seen = state.setdefault("seen", {})
    current_ids = {item["id"] for item in inventory}
    is_bootstrap = not seen
    new_items = [item for item in inventory if item["id"] not in seen]
    for item in inventory:
        seen.setdefault(item["id"], now)
    # Bound state growth without losing the current inventory.
    for stale_id in list(seen):
        if stale_id not in current_ids and now - float(seen[stale_id]) > 90 * 86400:
            del seen[stale_id]

    if new_items:
        newest = max(new_items, key=lambda item: (item["mtime"], item["id"]))
        state["featured"] = {
            "id": newest["id"],
            "until": now + hold_seconds,
            "bootstrap": is_bootstrap,
        }
    return state


def choose_photo(
    inventory: list[dict[str, Any]],
    state: dict[str, Any],
    now: float,
    rotation_seconds: int,
    history_size: int = 24,
) -> tuple[dict[str, Any], str]:
    if not inventory:
        raise RuntimeError("no supported photos are available")
    by_id = {item["id"]: item for item in inventory}
    featured = state.get("featured") or {}
    featured_id = str(featured.get("id") or "")
    if featured_id in by_id and float(featured.get("until") or 0) > now:
        return by_id[featured_id], "new"

    slot = int(now // rotation_seconds)
    published_id = str(state.get("published_id") or "")
    if (
        published_id in by_id
        and state.get("published_mode") == "rotation"
        and int(state.get("published_slot") or -1) == slot
    ):
        return by_id[published_id], "rotation"

    history = state.get("display_history") or []
    recent_history = history[-history_size:] if history_size > 0 else []
    excluded = set(str(value) for value in recent_history)
    last_featured = str((state.get("last_featured") or {}).get("id") or featured_id)
    if last_featured:
        excluded.add(last_featured)
    candidates = [item for item in inventory if item["id"] not in excluded]
    if not candidates:
        candidates = inventory

    ranked: list[tuple[float, str, dict[str, Any]]] = []
    for item in candidates:
        age_days = effective_recency_age_days(item, state, now)
        weight = recency_weight(age_days)
        digest = hashlib.sha256(f"framefeed-recency-playlist:{slot}:{item['id']}".encode()).digest()
        integer = int.from_bytes(digest[:8], "big")
        uniform = (integer + 1) / (2**64 + 1)
        # An exponential race gives deterministic weighted sampling per hour.
        priority = -math.log(uniform) / weight
        ranked.append((priority, item["id"], item))
    chosen = min(ranked, key=lambda value: (value[0], value[1]))[2]
    return chosen, "rotation"


def effective_recency_age_days(item: dict[str, Any], state: dict[str, Any], now: float) -> float:
    seen = state.get("seen") or {}
    first_seen = float(seen.get(item["id"]) or 0)
    baseline_cutoff = float(
        state.get("recency_baseline_cutoff")
        or max((float(value) for value in seen.values()), default=0)
    )
    candidates: list[float] = []
    date_created = float(item.get("date_created") or 0)
    if date_created:
        candidates.append(max(0.0, (now - date_created) / 86400))
    # Initial-import photos use Apple's date. Later additions also use the
    # exact time this worker first observed them, including old photos added now.
    if first_seen and (first_seen > baseline_cutoff or not candidates):
        candidates.append(max(0.0, (now - first_seen) / 86400))
    return min(candidates) if candidates else 3650.0


def recency_weight(age_days: float) -> float:
    return 1.0 + 6.0 * math.exp(-max(0.0, age_days) / 21.0)


def analysis_version(config: Config) -> str:
    model = config.face_model if config.face_detector == "immich" else "yunet-2023mar"
    return (
        f"{config.face_detector}-{model}-v1"
        f":score={config.face_min_score:.3f}"
        f":margin={config.face_margin_ratio:.3f}"
    )


def load_analysis_jsonl(path: Path, version: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(value, dict) or value.get("analysis_version") != version:
                    continue
                image_id = str(value.get("image_id") or "")
                if image_id:
                    records[image_id] = value
    except FileNotFoundError:
        pass
    return records


def append_analysis_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":"), sort_keys=True))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o600)


def calculate_face_safe_crop(
    image_width: int,
    image_height: int,
    faces: list[dict[str, Any]],
    target_width: int,
    target_height: int,
    margin_ratio: float,
    min_retained_fraction: float = 0.0,
) -> dict[str, Any]:
    """Return a normalized crop box, or a contain fallback for unsafe groups."""
    source_aspect = image_width / image_height
    target_aspect = target_width / target_height
    if source_aspect > target_aspect:
        crop_width = target_aspect / source_aspect
        crop_height = 1.0
    else:
        crop_width = 1.0
        crop_height = source_aspect / target_aspect

    retained_fraction = crop_width * crop_height
    if retained_fraction < min_retained_fraction:
        return {
            "mode": "contain-aspect-fallback",
            "box": [0.0, 0.0, 1.0, 1.0],
            "retained_fraction": retained_fraction,
        }

    if not faces:
        left = (1.0 - crop_width) / 2
        top = (1.0 - crop_height) / 2
        return {
            "mode": "cover-center",
            "box": [left, top, left + crop_width, top + crop_height],
        }

    expanded: list[tuple[float, float, float, float]] = []
    for face in faces:
        box = face.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        face_width = max(0.0, x2 - x1)
        face_height = max(0.0, y2 - y1)
        if face_width <= 0 or face_height <= 0:
            continue
        x_margin = face_width * margin_ratio
        y_margin = face_height * margin_ratio
        expanded.append(
            (
                max(0.0, x1 - x_margin),
                max(0.0, y1 - y_margin),
                min(1.0, x2 + x_margin),
                min(1.0, y2 + y_margin),
            )
        )

    if not expanded:
        left = (1.0 - crop_width) / 2
        top = (1.0 - crop_height) / 2
        return {
            "mode": "cover-center",
            "box": [left, top, left + crop_width, top + crop_height],
        }

    safe_left = min(box[0] for box in expanded)
    safe_top = min(box[1] for box in expanded)
    safe_right = max(box[2] for box in expanded)
    safe_bottom = max(box[3] for box in expanded)
    if safe_right - safe_left > crop_width or safe_bottom - safe_top > crop_height:
        return {"mode": "contain-fallback", "box": [0.0, 0.0, 1.0, 1.0]}

    focus_x = (safe_left + safe_right) / 2
    focus_y = (safe_top + safe_bottom) / 2
    left = min(max(focus_x - crop_width / 2, 0.0), 1.0 - crop_width)
    # Faces generally compose better slightly above the vertical midpoint.
    top = min(max(focus_y - crop_height * 0.42, 0.0), 1.0 - crop_height)

    if left > safe_left:
        left = safe_left
    if left + crop_width < safe_right:
        left = safe_right - crop_width
    if top > safe_top:
        top = safe_top
    if top + crop_height < safe_bottom:
        top = safe_bottom - crop_height
    left = min(max(left, 0.0), 1.0 - crop_width)
    top = min(max(top, 0.0), 1.0 - crop_height)
    return {
        "mode": "cover-face-safe",
        "box": [left, top, left + crop_width, top + crop_height],
    }


def detect_faces_yunet(source: Path, config: Config) -> tuple[int, int, list[dict[str, Any]]]:
    """Detect faces locally with OpenCV YuNet and return normalized boxes."""
    if not config.face_model_path.is_file():
        raise RuntimeError(f"YuNet model is missing: {config.face_model_path}")

    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image_width, image_height = image.size
        preview = image.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    pixels = cv2.cvtColor(np.asarray(preview), cv2.COLOR_RGB2BGR)
    detector = cv2.FaceDetectorYN.create(
        str(config.face_model_path),
        "",
        preview.size,
        config.face_min_score,
        0.3,
        5000,
    )
    detector.setInputSize(preview.size)
    _, detections = detector.detect(pixels)
    faces: list[dict[str, Any]] = []
    if detections is not None:
        for detection in detections:
            x, y, width, height = (float(value) for value in detection[:4])
            score = float(detection[-1])
            normalized = [
                max(0.0, min(1.0, x / preview.width)),
                max(0.0, min(1.0, y / preview.height)),
                max(0.0, min(1.0, (x + width) / preview.width)),
                max(0.0, min(1.0, (y + height) / preview.height)),
            ]
            if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
                faces.append({"box": normalized, "score": round(score, 6)})
    faces.sort(key=lambda face: tuple(face["box"]))
    return image_width, image_height, faces


def detect_faces_immich(source: Path, config: Config) -> tuple[int, int, list[dict[str, Any]]]:
    """Use an existing Immich machine-learning service as an optional detector."""
    if not config.face_analysis_url:
        raise RuntimeError("face analysis URL is not configured")

    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        image_width, image_height = image.size
        preview = image.copy()
    preview.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    encoded = io.BytesIO()
    preview.save(encoded, format="JPEG", quality=88, optimize=True)

    entries = {
        "facial-recognition": {
            "detection": {
                "modelName": config.face_model,
                "options": {"minScore": config.face_min_score},
            },
            # Immich's recognition stage converts detector arrays to its stable,
            # serializable face response. Embeddings are discarded immediately.
            "recognition": {"modelName": config.face_model, "options": {}},
        }
    }
    response = requests.post(
        config.face_analysis_url,
        data={"entries": json.dumps(entries, separators=(",", ":"))},
        files={"image": ("analysis.jpg", encoded.getvalue(), "image/jpeg")},
        timeout=(5, 120),
        headers={"User-Agent": USER_AGENT},
    )
    response.raise_for_status()
    payload = response.json()
    response_width = int(payload.get("imageWidth") or preview.width)
    response_height = int(payload.get("imageHeight") or preview.height)
    raw_faces = payload.get("facial-recognition") or []
    faces: list[dict[str, Any]] = []
    if isinstance(raw_faces, list):
        for raw_face in raw_faces:
            if not isinstance(raw_face, dict):
                continue
            box = raw_face.get("boundingBox") or {}
            try:
                normalized = [
                    max(0.0, min(1.0, float(box["x1"]) / response_width)),
                    max(0.0, min(1.0, float(box["y1"]) / response_height)),
                    max(0.0, min(1.0, float(box["x2"]) / response_width)),
                    max(0.0, min(1.0, float(box["y2"]) / response_height)),
                ]
                score = float(raw_face.get("score") or 0)
            except (KeyError, TypeError, ValueError, ZeroDivisionError):
                continue
            if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
                faces.append({"box": normalized, "score": round(score, 6)})
    faces.sort(key=lambda face: tuple(face["box"]))
    return image_width, image_height, faces


def detect_faces(source: Path, config: Config) -> tuple[int, int, list[dict[str, Any]]]:
    if config.face_detector == "yunet":
        return detect_faces_yunet(source, config)
    if config.face_detector == "immich":
        return detect_faces_immich(source, config)
    raise RuntimeError("face detection is disabled")


def analyze_photo(item: dict[str, Any], config: Config) -> dict[str, Any]:
    width, height, faces = detect_faces(item["path"], config)
    crop = calculate_face_safe_crop(
        width,
        height,
        faces,
        config.width,
        config.height,
        config.face_margin_ratio,
        config.min_crop_retained_fraction,
    )
    face_count = len(faces)
    return {
        "analysis_version": analysis_version(config),
        "image_id": item["id"],
        "relative": item["relative"],
        "analyzed_at": datetime.now(UTC).isoformat(),
        "width": width,
        "height": height,
        "faces": faces,
        "crop": crop,
        "description": (
            f"{face_count} face{'s' if face_count != 1 else ''} detected; layout {crop['mode']}"
        ),
    }


def analysis_for_layout(analysis: dict[str, Any], config: Config) -> dict[str, Any]:
    """Recompute only layout geometry while reusing cached face detection."""
    adjusted = dict(analysis)
    faces = analysis.get("faces") or []
    crop = calculate_face_safe_crop(
        int(analysis["width"]),
        int(analysis["height"]),
        faces,
        config.width,
        config.height,
        config.face_margin_ratio,
        config.min_crop_retained_fraction,
    )
    adjusted["crop"] = crop
    face_count = len(faces)
    adjusted["description"] = (
        f"{face_count} face{'s' if face_count != 1 else ''} detected; layout {crop['mode']}"
    )
    return adjusted


def render_contained_blur(source: Path, target: Path, config: Config) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        background = ImageOps.fit(
            image,
            (config.width, config.height),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        ).filter(ImageFilter.GaussianBlur(config.blur_radius))
        # Slightly dim the blur so the uncropped foreground reads as intentional.
        background = Image.blend(background, Image.new("RGB", background.size, "black"), 0.18)
        foreground = ImageOps.contain(
            image,
            (config.width, config.height),
            method=Image.Resampling.LANCZOS,
        )
        left = (config.width - foreground.width) // 2
        top = (config.height - foreground.height) // 2
        background.paste(foreground, (left, top))

        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            background.save(
                temporary,
                format="JPEG",
                quality=config.jpeg_quality,
                optimize=True,
                progressive=False,
                subsampling=0,
            )
            os.chmod(temporary, 0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def render_analyzed_photo(
    source: Path,
    target: Path,
    config: Config,
    analysis: dict[str, Any] | None,
) -> str:
    crop = (analysis or {}).get("crop") or {}
    crop_mode = str(crop.get("mode") or "contain-unanalysed")
    box = crop.get("box")
    if crop_mode not in {"cover-face-safe", "cover-center"} or not isinstance(box, list):
        render_contained_blur(source, target, config)
        return crop_mode

    target.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as original:
        image = ImageOps.exif_transpose(original).convert("RGB")
        width, height = image.size
        left, top, right, bottom = (float(value) for value in box)
        pixel_box = (
            max(0, min(width - 1, round(left * width))),
            max(0, min(height - 1, round(top * height))),
            max(1, min(width, round(right * width))),
            max(1, min(height, round(bottom * height))),
        )
        cropped = image.crop(pixel_box).resize(
            (config.width, config.height), Image.Resampling.LANCZOS
        )
        with tempfile.NamedTemporaryFile(dir=target.parent, suffix=".jpg", delete=False) as handle:
            temporary = Path(handle.name)
        try:
            cropped.save(
                temporary,
                format="JPEG",
                quality=config.jpeg_quality,
                optimize=True,
                progressive=False,
                subsampling=0,
            )
            os.chmod(temporary, 0o644)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    return crop_mode


def frame_html(frame_filename: str, width: int, height: int) -> str:
    safe_name = html.escape(frame_filename, quote=True)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1">
  <meta name="robots" content="noindex,nofollow,noarchive,nosnippet,noimageindex">
  <meta http-equiv="Cache-Control" content="no-store, no-cache, must-revalidate, max-age=0">
  <title>FrameFeed</title>
  <style>
    html,body{{margin:0;width:100%;height:100%;overflow:hidden;background:#000}}
    body{{display:grid;place-items:center}}
    img{{display:block;width:100vw;height:100vh;object-fit:contain}}
    @page{{size:{width}px {height}px;margin:0}}
  </style>
</head>
<body><img src="frames/{safe_name}" width="{width}" height="{height}" alt=""></body>
</html>
"""


class Worker:
    def __init__(self, config: Config, clock=utc_now):
        self.config = config
        self.clock = clock
        self.stop_requested = False
        self.state_path = config.state_dir / "state.json"
        self.health_path = config.state_dir / "health.json"
        self.album_manifest_path = config.state_dir / "icloud-shared-manifest.json"
        self.analysis_path = config.state_dir / "image-analysis.jsonl"
        self.analysis_version = analysis_version(config)
        self.analysis_cache = load_analysis_jsonl(self.analysis_path, self.analysis_version)
        self.state = load_json(self.state_path, {})
        self.secret = read_secret(config.frame_secret_file)
        self.public_dir = config.site_root / self.secret

    def analysis_for(self, item: dict[str, Any]) -> dict[str, Any] | None:
        cached = self.analysis_cache.get(item["id"])
        if cached is not None:
            return analysis_for_layout(cached, self.config)
        if self.config.face_detector == "none":
            return None
        try:
            value = analyze_photo(item, self.config)
        except Exception as exc:
            LOG.warning(
                "Face analysis unavailable for one photo; using uncropped fallback: %s",
                type(exc).__name__,
            )
            return None
        append_analysis_jsonl(self.analysis_path, value)
        self.analysis_cache[item["id"]] = value
        LOG.info(
            "Cached face-safe crop metadata for one photo (%d faces, %s)",
            len(value["faces"]),
            value["crop"]["mode"],
        )
        return analysis_for_layout(value, self.config)

    def analyze_pending(self, inventory: list[dict[str, Any]], exclude_id: str = "") -> None:
        remaining = self.config.analysis_batch_size
        if remaining <= 0 or self.config.face_detector == "none":
            return
        for item in sorted(inventory, key=lambda candidate: candidate["id"]):
            if remaining <= 0:
                break
            if item["id"] == exclude_id or item["id"] in self.analysis_cache:
                continue
            before = len(self.analysis_cache)
            self.analysis_for(item)
            if len(self.analysis_cache) == before:
                # Avoid hammering a detector that is currently unavailable.
                break
            remaining -= 1

    def request_stop(self, *_: Any) -> None:
        self.stop_requested = True

    def sync_if_due(self, now: float, force: bool = False) -> None:
        album_url = read_optional_secret(self.config.album_url_file)
        if not album_url:
            return
        last_sync = float(self.state.get("last_album_sync") or 0)
        if not force and now - last_sync < self.config.album_sync_interval_seconds:
            return
        result = sync_shared_album(
            album_url,
            self.config.photo_dir,
            self.album_manifest_path,
        )
        self.state["last_album_sync"] = now
        self.state["last_album_sync_result"] = result
        LOG.info(
            "Shared Album sync complete: %d active, %d new",
            result["assets"],
            result["downloaded"],
        )

    def publish(self, now: float) -> dict[str, Any]:
        inventory = image_inventory(self.config.photo_dir)
        add_shared_album_dates(inventory, self.album_manifest_path)
        update_inventory_state(self.state, inventory, now, self.config.new_photo_hold_seconds)
        if not float(self.state.get("recency_baseline_cutoff") or 0):
            self.state["recency_baseline_cutoff"] = max(
                (float(value) for value in (self.state.get("seen") or {}).values()),
                default=now,
            )
        history = self.state.setdefault("display_history", [])
        previously_published = str(self.state.get("published_id") or "")
        if previously_published and not history:
            history.append(previously_published)
        chosen, mode = choose_photo(inventory, self.state, now, self.config.rotation_seconds)
        analysis = self.analysis_for(chosen)
        analysis_signature = hashlib.sha256(
            json.dumps(
                {
                    "version": (analysis or {}).get("analysis_version"),
                    "crop": (analysis or {}).get("crop"),
                    "frame": [self.config.width, self.config.height],
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:16]

        previous_id = str(self.state.get("published_id") or "")
        previous_mode = str(self.state.get("published_mode") or "")
        previous_analysis = str(self.state.get("published_analysis") or "")
        current_frame = self.public_dir / str(self.state.get("published_frame") or "")
        needs_render = (
            chosen["id"] != previous_id
            or mode != previous_mode
            or analysis_signature != previous_analysis
            or not current_frame.is_file()
        )
        if needs_render:
            revision = hashlib.sha256(
                f"{chosen['id']}:{mode}:{analysis_signature}:{int(now)}".encode()
            ).hexdigest()[:16]
            filename = f"frame-{revision}.jpg"
            frame_path = self.public_dir / "frames" / filename
            layout = render_analyzed_photo(chosen["path"], frame_path, self.config, analysis)
            atomic_text(
                self.public_dir / "index.html",
                frame_html(filename, self.config.width, self.config.height),
            )
            self.state["published_id"] = chosen["id"]
            self.state["published_mode"] = mode
            self.state["published_frame"] = f"frames/{filename}"
            self.state["published_at"] = now
            self.state["published_slot"] = int(now // self.config.rotation_seconds)
            self.state["published_analysis"] = analysis_signature
            self.state["published_layout"] = layout
            if chosen["id"] != previous_id:
                history.append(chosen["id"])
                self.state["display_history"] = history[-24:]
            if mode == "new":
                self.state["last_featured"] = {"id": chosen["id"], "at": now}
            self._prune_public_frames(keep=6)
            self._prune_stale_public_paths()
            LOG.info(
                "Published a %s photo using %s (%d photos available)",
                mode,
                layout,
                len(inventory),
            )

        self.state["last_success"] = now
        self.state["photo_count"] = len(inventory)
        atomic_json(self.state_path, self.state)
        health = {
            "heartbeat": now,
            "last_success": now,
            "photo_count": len(inventory),
            "status": "ok",
        }
        atomic_json(self.health_path, health)
        self.analyze_pending(inventory, exclude_id=chosen["id"])
        return health

    def _prune_public_frames(self, keep: int) -> None:
        frames = self.public_dir / "frames"
        if not frames.exists():
            return
        candidates = sorted(
            (path for path in frames.glob("frame-*.jpg") if path.is_file()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale in candidates[keep:]:
            stale.unlink(missing_ok=True)

    def _prune_stale_public_paths(self) -> None:
        """Remove generated output left behind after a frame-secret rotation."""
        if not self.config.site_root.exists():
            return
        for candidate in self.config.site_root.iterdir():
            if candidate == self.public_dir or not candidate.is_dir():
                continue
            if SECRET_RE.fullmatch(candidate.name) and (candidate / "index.html").is_file():
                shutil.rmtree(candidate)

    def heartbeat_error(self, now: float, error: Exception) -> None:
        health = load_json(self.health_path, {})
        health.update(
            {
                "heartbeat": now,
                "last_error_at": now,
                "last_error": type(error).__name__,
                "status": "degraded",
            }
        )
        atomic_json(self.health_path, health)

    def run_once(self, force_sync: bool = False) -> dict[str, Any]:
        now = self.clock()
        self.sync_if_due(now, force=force_sync)
        return self.publish(now)

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, self.request_stop)
        signal.signal(signal.SIGINT, self.request_stop)
        while not self.stop_requested:
            started = self.clock()
            try:
                self.run_once()
            except Exception as exc:  # Keep last good image during upstream outages.
                LOG.exception("Frame update failed; preserving the last published image")
                self.heartbeat_error(started, exc)
            elapsed = max(0, self.clock() - started)
            remaining = max(1, self.config.scan_interval_seconds - elapsed)
            deadline = self.clock() + remaining
            while not self.stop_requested and self.clock() < deadline:
                time.sleep(min(1, deadline - self.clock()))


def healthcheck(config: Config) -> int:
    health = load_json(config.state_dir / "health.json", {})
    heartbeat = float(health.get("heartbeat") or 0)
    maximum_age = max(config.scan_interval_seconds * 3, 900)
    if utc_now() - heartbeat > maximum_age:
        print("frame worker heartbeat is stale", file=sys.stderr)
        return 1
    if not float(health.get("last_success") or 0):
        print("frame worker has never published successfully", file=sys.stderr)
        return 1
    return 0


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--force-sync", action="store_true")
    parser.add_argument("--healthcheck", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    config = Config.from_environment()
    config.validate()
    if args.healthcheck:
        return healthcheck(config)
    worker = Worker(config)
    if args.once:
        worker.run_once(force_sync=args.force_sync)
    else:
        worker.run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
