import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from framefeed.app import (
    Config,
    Worker,
    _asset_url_map,
    add_shared_album_dates,
    analysis_version,
    append_analysis_jsonl,
    calculate_face_safe_crop,
    choose_photo,
    effective_recency_age_days,
    frame_html,
    load_analysis_jsonl,
    parse_shared_album_id,
    recency_weight,
    render_analyzed_photo,
    render_contained_blur,
    sync_shared_album,
    update_inventory_state,
)


class FrameWorkerTests(unittest.TestCase):
    def test_asset_urls_are_restricted_to_apple_content_hosts(self):
        value = {
            "items": {
                "safe": {
                    "url_location": "content.icloud.com",
                    "url_path": "/photo.jpg",
                },
                "unsafe": {
                    "url_location": "attacker.example",
                    "url_path": "/photo.jpg",
                },
            }
        }
        self.assertEqual(
            _asset_url_map(value),
            {"safe": "https://content.icloud.com/photo.jpg"},
        )

    def test_shared_album_url_parser(self):
        self.assertEqual(
            parse_shared_album_id("https://www.icloud.com/sharedalbum/#ExampleAlbum123"),
            "ExampleAlbum123",
        )
        with self.assertRaises(ValueError):
            parse_shared_album_id("https://attacker.example/sharedalbum/#abc")

    def test_new_photo_is_featured_then_rotation_resumes(self):
        inventory = [
            {"id": "old", "mtime": 1, "path": Path("old.jpg")},
            {"id": "new", "mtime": 2, "path": Path("new.jpg")},
        ]
        state = {"seen": {"old": 10}}
        update_inventory_state(state, inventory, now=100, hold_seconds=60)
        selected, mode = choose_photo(inventory, state, now=120, rotation_seconds=60)
        self.assertEqual((selected["id"], mode), ("new", "new"))
        selected, mode = choose_photo(inventory, state, now=161, rotation_seconds=60)
        self.assertEqual(mode, "rotation")
        self.assertNotEqual(selected["id"], "new")

    def test_recency_weight_decays_to_uniform_baseline(self):
        self.assertAlmostEqual(recency_weight(0), 7.0)
        self.assertGreater(recency_weight(7), recency_weight(30))
        self.assertGreater(recency_weight(30), recency_weight(90))
        self.assertAlmostEqual(recency_weight(365), 1.0, places=6)

    def test_newly_added_old_photo_uses_first_seen_age(self):
        now = 10_000_000.0
        old_capture = now - 365 * 86400
        baseline = {"id": "baseline", "date_created": old_capture}
        newly_added = {"id": "newly-added", "date_created": old_capture}
        state = {
            "seen": {"baseline": 100.0, "newly-added": now - 86400},
            "recency_baseline_cutoff": 100.0,
        }
        self.assertAlmostEqual(effective_recency_age_days(baseline, state, now), 365)
        self.assertAlmostEqual(effective_recency_age_days(newly_added, state, now), 1)

    def test_hourly_weighted_choice_is_stable_and_honors_history(self):
        now = 20_000_000.0
        inventory = [
            {"id": f"photo-{index}", "date_created": now - index * 86400} for index in range(30)
        ]
        state = {
            "seen": {item["id"]: 100.0 for item in inventory},
            "recency_baseline_cutoff": 100.0,
            "display_history": [],
        }
        first, mode = choose_photo(inventory, state, now, 3600)
        repeated, _ = choose_photo(inventory, state, now + 120, 3600)
        self.assertEqual(mode, "rotation")
        self.assertEqual(first["id"], repeated["id"])
        state["display_history"] = [first["id"]]
        next_choice, _ = choose_photo(inventory, state, now, 3600)
        self.assertNotEqual(first["id"], next_choice["id"])

    def test_published_rotation_is_held_for_the_entire_slot(self):
        now = 20_000_000.0
        inventory = [
            {"id": "held", "date_created": now},
            {"id": "other", "date_created": now},
        ]
        slot = int(now // 3600)
        state = {
            "seen": {"held": 100.0, "other": 100.0},
            "display_history": ["held"],
            "published_id": "held",
            "published_mode": "rotation",
            "published_slot": slot,
        }
        selected, mode = choose_photo(inventory, state, now + 120, 3600)
        self.assertEqual((selected["id"], mode), ("held", "rotation"))

    def test_shared_album_dates_are_attached_by_anonymous_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "assets": {
                            "guid": {
                                "filename": "GUID.jpg",
                                "date_created": "2026-07-01T12:00:00Z",
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            inventory = [{"id": "one", "relative": "GUID.jpg"}]
            add_shared_album_dates(inventory, manifest)
            self.assertGreater(inventory[0]["date_created"], 0)

    def test_contain_with_blurred_background_keeps_full_foreground(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wide.png"
            target = root / "frame.jpg"
            image = Image.new("RGB", (800, 400), (230, 20, 20))
            image.save(source)
            config = Config(
                photo_dir=root,
                site_root=root,
                state_dir=root,
                frame_secret_file=root / "secret",
                album_url_file=root / "album",
                width=120,
                height=160,
                blur_radius=8,
            )
            render_contained_blur(source, target, config)
            with Image.open(target) as rendered:
                self.assertEqual(rendered.size, (120, 160))
                center = rendered.getpixel((60, 80))
                background = rendered.getpixel((60, 10))
                self.assertGreater(center[0], 200)
                self.assertGreater(center[0], background[0])

    def test_face_safe_crop_fills_portrait_frame_and_contains_face_margin(self):
        crop = calculate_face_safe_crop(
            6000,
            4000,
            [{"box": [0.72, 0.20, 0.82, 0.38], "score": 0.9}],
            1200,
            1600,
            margin_ratio=0.75,
        )
        self.assertEqual(crop["mode"], "cover-face-safe")
        left, top, right, bottom = crop["box"]
        self.assertAlmostEqual(((right - left) * 1.5) / (bottom - top), 1200 / 1600, places=6)
        self.assertLessEqual(left, 0.72 - 0.10 * 0.75)
        self.assertGreaterEqual(right, 0.82 + 0.10 * 0.75)

    def test_wide_face_group_uses_uncropped_fallback(self):
        crop = calculate_face_safe_crop(
            6000,
            4000,
            [
                {"box": [0.03, 0.2, 0.13, 0.4], "score": 0.9},
                {"box": [0.87, 0.2, 0.97, 0.4], "score": 0.9},
            ],
            1200,
            1600,
            margin_ratio=0.75,
        )
        self.assertEqual(crop["mode"], "contain-fallback")

    def test_landscape_camera_photo_uses_small_aspect_crop(self):
        crop = calculate_face_safe_crop(
            6000,
            4000,
            [],
            1600,
            1200,
            margin_ratio=0.75,
            min_retained_fraction=0.75,
        )
        self.assertEqual(crop["mode"], "cover-center")
        left, top, right, bottom = crop["box"]
        self.assertAlmostEqual((right - left) * (bottom - top), 8 / 9, places=6)

    def test_portrait_photo_on_landscape_frame_is_not_destructively_cropped(self):
        crop = calculate_face_safe_crop(
            4000,
            6000,
            [{"box": [0.35, 0.1, 0.65, 0.3], "score": 0.9}],
            1600,
            1200,
            margin_ratio=0.75,
            min_retained_fraction=0.75,
        )
        self.assertEqual(crop["mode"], "contain-aspect-fallback")
        self.assertAlmostEqual(crop["retained_fraction"], 0.5)

    def test_analyzed_crop_preserves_ratio_without_squishing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wide.png"
            target = root / "frame.jpg"
            image = Image.new("RGB", (600, 400), "blue")
            # Simulated face region near the right side.
            for x in range(430, 500):
                for y in range(100, 200):
                    image.putpixel((x, y), (240, 30, 30))
            image.save(source)
            config = Config(
                photo_dir=root,
                site_root=root,
                state_dir=root,
                frame_secret_file=root / "secret",
                album_url_file=root / "album",
                width=120,
                height=160,
            )
            analysis = {
                "crop": calculate_face_safe_crop(
                    600,
                    400,
                    [{"box": [430 / 600, 0.25, 500 / 600, 0.5], "score": 0.9}],
                    120,
                    160,
                    margin_ratio=0.5,
                )
            }
            mode = render_analyzed_photo(source, target, config, analysis)
            with Image.open(target) as rendered:
                self.assertEqual(mode, "cover-face-safe")
                self.assertEqual(rendered.size, (120, 160))
                red_pixels = sum(
                    1
                    for red, green, blue in np.asarray(rendered).reshape(-1, 3)
                    if red > 180 and green < 90 and blue < 90
                )
                self.assertGreater(red_pixels, 100)

    def test_analysis_jsonl_is_private_and_versioned(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                photo_dir=root,
                site_root=root,
                state_dir=root,
                frame_secret_file=root / "secret",
                album_url_file=root / "album",
            )
            version = analysis_version(config)
            path = root / "image-analysis.jsonl"
            append_analysis_jsonl(
                path,
                {"analysis_version": version, "image_id": "one", "faces": []},
            )
            append_analysis_jsonl(
                path,
                {"analysis_version": "old", "image_id": "two", "faces": []},
            )
            self.assertEqual(set(load_analysis_jsonl(path, version)), {"one"})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)

    def test_generated_html_hides_source_and_blocks_indexing(self):
        document = frame_html("frame-abc.jpg", 1200, 1600)
        self.assertIn("noindex,nofollow", document)
        self.assertIn("frames/frame-abc.jpg", document)
        self.assertIn("object-fit:contain", document)
        self.assertNotIn("object-fit:fill", document)
        self.assertNotIn("icloud", document.lower())

    def test_shared_album_sync_uses_largest_derivative_and_anonymous_filename(self):
        class FakeResponse:
            def __init__(self, value=None, content=b"", content_type="application/json"):
                self.value = value
                self.content = content
                self.headers = {"Content-Type": content_type}

            def raise_for_status(self):
                return None

            def json(self):
                return self.value

            def iter_content(self, _size):
                yield self.content

        class FakeSession:
            def __init__(self, jpeg):
                self.jpeg = jpeg
                self.webstream_calls = 0

            def post(self, url, **_kwargs):
                if url.endswith("/webstream"):
                    self.webstream_calls += 1
                    if self.webstream_calls == 1:
                        return FakeResponse({"X-Apple-MMe-Host": "p42-sharedstreams.icloud.com"})
                    return FakeResponse(
                        {
                            "photos": [
                                {
                                    "photoGuid": "GUID-ONE",
                                    "dateCreated": "2026-07-20T00:00:00Z",
                                    "derivatives": {
                                        "small": {"checksum": "small", "fileSize": "10"},
                                        "large": {"checksum": "large", "fileSize": "100"},
                                    },
                                }
                            ]
                        }
                    )
                return FakeResponse(
                    {
                        "items": {
                            "large": {
                                "url_location": "content.icloud.com",
                                "url_path": "/asset-large",
                            }
                        }
                    }
                )

            def get(self, url, **_kwargs):
                if url != "https://content.icloud.com/asset-large":
                    raise AssertionError(url)
                return FakeResponse(content=self.jpeg, content_type="image/jpeg")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            buffer = io.BytesIO()
            Image.new("RGB", (40, 30), "purple").save(buffer, format="JPEG")
            result = sync_shared_album(
                "https://www.icloud.com/sharedalbum/#ExampleAlbum123",
                root / "photos",
                root / "manifest.json",
                session=FakeSession(buffer.getvalue()),
            )
            self.assertEqual(result, {"assets": 1, "downloaded": 1, "removed": 0})
            downloaded = root / "photos" / "GUID-ONE.jpg"
            self.assertTrue(downloaded.is_file())
            with Image.open(downloaded) as image:
                self.assertEqual(image.size, (40, 30))

    def test_worker_publishes_without_exposing_original_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            site = root / "site"
            state = root / "state"
            photos.mkdir()
            secret_file = root / "frame-secret"
            secret = "a" * 64
            secret_file.write_text(secret, encoding="utf-8")
            private_name = "private-family-filename.jpg"
            Image.new("RGB", (80, 60), "green").save(photos / private_name)
            config = Config(
                photo_dir=photos,
                site_root=site,
                state_dir=state,
                frame_secret_file=secret_file,
                album_url_file=root / "missing-album-url",
                width=120,
                height=160,
                rotation_seconds=3600,
                new_photo_hold_seconds=3900,
            )
            worker = Worker(config, clock=lambda: 10_000)
            result = worker.run_once()
            index = (site / secret / "index.html").read_text(encoding="utf-8")
            self.assertEqual(result["photo_count"], 1)
            self.assertNotIn(private_name, index)
            persisted = json.loads((state / "state.json").read_text(encoding="utf-8"))
            self.assertEqual(persisted["photo_count"], 1)

    def test_secret_rotation_removes_previous_generated_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            photos = root / "photos"
            site = root / "site"
            state = root / "state"
            photos.mkdir()
            Image.new("RGB", (80, 60), "green").save(photos / "one.jpg")
            old_secret = "a" * 64
            old_path = site / old_secret
            old_path.mkdir(parents=True)
            (old_path / "index.html").write_text("old", encoding="utf-8")
            secret_file = root / "frame-secret"
            new_secret = "b" * 64
            secret_file.write_text(new_secret, encoding="utf-8")
            config = Config(
                photo_dir=photos,
                site_root=site,
                state_dir=state,
                frame_secret_file=secret_file,
                album_url_file=root / "missing-album-url",
                width=120,
                height=160,
                face_detector="none",
            )
            Worker(config, clock=lambda: 10_000).run_once()
            self.assertFalse(old_path.exists())
            self.assertTrue((site / new_secret / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
