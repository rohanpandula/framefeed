import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.benchmark_people import normalized_people, percentile, restore_boxes


class BenchmarkPeopleTests(unittest.TestCase):
    def test_percentile_interpolates_between_observations(self):
        self.assertEqual(percentile([10, 20, 30], 0.25), 15)
        self.assertEqual(percentile([], 0.50), 0)

    def test_normalized_people_clips_and_sorts_boxes(self):
        people = normalized_people(
            np.asarray(
                [
                    [80.0, 20.0, 110.0, 90.0],
                    [-10.0, 10.0, 30.0, 60.0],
                ]
            ),
            np.asarray([0.8, 0.9]),
            width=100,
            height=100,
        )
        self.assertEqual(
            people,
            [
                {"box": [0.0, 0.1, 0.3, 0.6], "score": 0.9},
                {"box": [0.8, 0.2, 1.0, 0.9], "score": 0.8},
            ],
        )

    def test_restore_boxes_removes_letterbox_padding(self):
        restored = restore_boxes(
            np.asarray([[320.0, 240.0, 960.0, 1040.0]]),
            ratio=0.5,
            padding=(100, 200),
            width=1600,
            height=1800,
        )
        np.testing.assert_allclose(
            restored,
            np.asarray([[440.0, 80.0, 1600.0, 1680.0]]),
        )


if __name__ == "__main__":
    unittest.main()
