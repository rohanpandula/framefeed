import unittest

from framefeed.app import calculate_face_safe_crop
from framefeed.crop_v2 import calculate_baseline_crop, calculate_subject_zoom_crop


class SubjectZoomCropTests(unittest.TestCase):
    def test_tiny_face_is_zoomed_beyond_aspect_only_crop(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [{"box": [0.44, 0.42, 0.50, 0.50], "score": 0.9}],
            1600,
            1200,
        )
        self.assertEqual(crop["mode"], "smart-face-zoom")
        left, top, right, bottom = crop["box"]
        self.assertAlmostEqual((right - left) / (bottom - top), 1.0)
        self.assertGreater(crop["zoom"], 1.5)
        self.assertLessEqual(left, 0.44)
        self.assertGreaterEqual(right, 0.50)
        self.assertLessEqual(top, 0.42)
        self.assertGreaterEqual(bottom, 0.50)

    def test_nearby_smaller_baby_face_is_kept_with_adults(self):
        crop = calculate_subject_zoom_crop(
            3000,
            4000,
            [
                {"box": [0.25, 0.20, 0.40, 0.35], "score": 0.9},
                {"box": [0.58, 0.20, 0.73, 0.35], "score": 0.9},
                {"box": [0.46, 0.36, 0.52, 0.42], "score": 0.8},
            ],
            1600,
            1200,
        )
        self.assertEqual(crop["primary_face_count"], 3)
        self.assertEqual(crop["dropped_face_count"], 0)
        left, top, right, bottom = crop["box"]
        self.assertLessEqual(left, 0.46)
        self.assertGreaterEqual(right, 0.52)
        self.assertLessEqual(top, 0.36)
        self.assertGreaterEqual(bottom, 0.42)

    def test_tiny_distant_background_face_can_be_dropped(self):
        crop = calculate_subject_zoom_crop(
            3000,
            4000,
            [
                {"box": [0.35, 0.55, 0.55, 0.75], "score": 0.95},
                {"box": [0.05, 0.05, 0.08, 0.08], "score": 0.6},
            ],
            1600,
            1200,
        )
        self.assertEqual(crop["primary_face_count"], 1)
        self.assertEqual(crop["dropped_face_count"], 1)

    def test_equal_size_wide_group_falls_back_instead_of_cutting_people(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [
                {"box": [0.02, 0.30, 0.18, 0.50], "score": 0.9},
                {"box": [0.82, 0.30, 0.98, 0.50], "score": 0.9},
            ],
            1200,
            1600,
        )
        self.assertEqual(crop["mode"], "smart-contain-group")
        self.assertEqual(crop["box"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(crop["dropped_face_count"], 0)

    def test_no_faces_uses_contain_fallback(self):
        crop = calculate_subject_zoom_crop(4000, 3000, [], 1600, 1200)
        self.assertEqual(crop["mode"], "smart-contain-no-faces")
        self.assertEqual(crop["box"], [0.0, 0.0, 1.0, 1.0])

    def test_person_detection_can_zoom_without_a_visible_face(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [],
            1600,
            1200,
            people=[{"box": [0.40, 0.25, 0.60, 0.80], "score": 0.8}],
        )
        self.assertEqual(crop["mode"], "smart-person-zoom")
        self.assertEqual(crop["primary_person_count"], 1)
        self.assertGreater(crop["zoom"], 1.2)
        left, top, right, bottom = crop["box"]
        self.assertLessEqual(left, 0.40)
        self.assertGreaterEqual(right, 0.60)
        self.assertLessEqual(top, 0.25)
        # A landscape frame should preserve the person's upper body without
        # requiring every pixel around their feet.
        self.assertGreaterEqual(bottom, 0.68)

    def test_overlapping_person_detections_are_deduplicated(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [],
            1600,
            1200,
            people=[
                {"box": [0.35, 0.25, 0.65, 0.82], "score": 0.92},
                {"box": [0.36, 0.26, 0.64, 0.81], "score": 0.88},
                {"box": [0.34, 0.24, 0.66, 0.83], "score": 0.75},
            ],
        )
        self.assertEqual(crop["primary_person_count"], 1)
        self.assertEqual(crop["dropped_person_count"], 0)
        self.assertGreater(crop["zoom"], 1.2)

    def test_tiny_distant_people_do_not_block_main_subject_zoom(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [{"box": [0.43, 0.30, 0.52, 0.41], "score": 0.95}],
            1600,
            1200,
            people=[
                {"box": [0.35, 0.25, 0.62, 0.85], "score": 0.94},
                {"box": [0.03, 0.20, 0.07, 0.34], "score": 0.72},
                {"box": [0.88, 0.16, 0.92, 0.30], "score": 0.68},
                {"box": [0.76, 0.65, 0.79, 0.76], "score": 0.60},
            ],
        )
        self.assertEqual(crop["primary_person_count"], 1)
        self.assertEqual(crop["dropped_person_count"], 2)
        self.assertGreater(crop["zoom"], 1.2)

    def test_only_tiny_background_people_are_not_treated_as_the_subject(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [],
            1600,
            1200,
            people=[
                {"box": [0.03, 0.20, 0.05, 0.28], "score": 0.72},
                {"box": [0.88, 0.16, 0.90, 0.24], "score": 0.68},
                {"box": [0.76, 0.65, 0.78, 0.73], "score": 0.60},
            ],
        )
        self.assertEqual(crop["mode"], "smart-baseline-cover-center")
        self.assertEqual(crop["box"], [0.0, 0.0, 1.0, 1.0])
        self.assertEqual(crop["zoom"], 1.0)

    def test_face_only_zoom_is_capped_to_preserve_missed_people(self):
        crop = calculate_subject_zoom_crop(
            4000,
            3000,
            [{"box": [0.65, 0.45, 0.71, 0.53], "score": 0.92}],
            1600,
            1200,
        )
        self.assertEqual(crop["mode"], "smart-face-zoom")
        self.assertLessEqual(crop["zoom"], 1.75 + 1e-9)

    def test_offline_baseline_matches_production_crop(self):
        faces = [
            {"box": [0.25, 0.20, 0.40, 0.35], "score": 0.9},
            {"box": [0.58, 0.20, 0.73, 0.35], "score": 0.9},
        ]
        expected = calculate_face_safe_crop(
            3000,
            4000,
            faces,
            1600,
            1200,
            margin_ratio=0.75,
            min_retained_fraction=0.75,
            min_face_retained_fraction=0.50,
        )
        actual = calculate_baseline_crop(
            3000,
            4000,
            faces,
            1600,
            1200,
        )
        self.assertEqual(actual, expected)

    def test_crop_box_stays_in_bounds_and_matches_target_aspect(self):
        crop = calculate_subject_zoom_crop(
            1536,
            2048,
            [{"box": [0.01, 0.01, 0.20, 0.20], "score": 0.9}],
            1600,
            1200,
        )
        left, top, right, bottom = crop["box"]
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, 1)
        self.assertLessEqual(bottom, 1)
        normalized_aspect = ((right - left) * 1536) / ((bottom - top) * 2048)
        self.assertAlmostEqual(normalized_aspect, 1600 / 1200, places=6)


if __name__ == "__main__":
    unittest.main()
