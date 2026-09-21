"""Dependency-light checks for Stage 2 annotation time helpers."""

import unittest

from src.stage2.labeler_widget import frame_number_to_time, tolerance_radius_frames


class Stage2LabelerHelperTests(unittest.TestCase):
    def test_one_based_aihub_frame_time(self):
        self.assertEqual(frame_number_to_time(1, 15), 0.0)
        self.assertAlmostEqual(frame_number_to_time(101, 15), 100 / 15)

    def test_tolerance_radius_uses_only_valid_integer_offsets(self):
        self.assertEqual(tolerance_radius_frames(15), 4)
        self.assertEqual(tolerance_radius_frames(30), 9)
        self.assertEqual(tolerance_radius_frames(60), 18)

    def test_bad_values_raise(self):
        with self.assertRaises(ValueError):
            frame_number_to_time(0, 15)
        with self.assertRaises(ValueError):
            frame_number_to_time(1, 0)
        with self.assertRaises(ValueError):
            tolerance_radius_frames(0)


if __name__ == "__main__":
    unittest.main()

