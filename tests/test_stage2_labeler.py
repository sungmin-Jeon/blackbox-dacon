"""Dependency-light checks for Stage 2 annotation time helpers."""

import unittest
from types import SimpleNamespace

from src.stage2.labeler_widget import Stage2Labeler, frame_number_to_time, tolerance_radius_frames


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

    def test_collision_can_be_excluded_from_validation(self):
        labeler = Stage2Labeler.__new__(Stage2Labeler)
        labeler.include_collision = False
        labeler.status = SimpleNamespace(value="KEEP")
        labeler._marks = {
            "entry_min": 20,
            "entry_frame": 21,
            "entry_max": 22,
            "collision_min": None,
            "collision_frame": None,
            "collision_max": None,
        }
        labeler.entry_side = SimpleNamespace(value="LEFT")
        labeler.evasion_space = SimpleNamespace(value=1)
        labeler.entry_confidence = SimpleNamespace(value="HIGH")
        labeler.collision_confidence = SimpleNamespace(value="")
        labeler.lane_basis = SimpleNamespace(value="VISIBLE")

        self.assertIsNone(labeler._validation_error())

    def test_collision_is_still_required_by_default_mode(self):
        labeler = Stage2Labeler.__new__(Stage2Labeler)
        labeler.include_collision = True
        labeler.status = SimpleNamespace(value="KEEP")
        labeler._marks = {
            "entry_min": 20,
            "entry_frame": 21,
            "entry_max": 22,
            "collision_min": None,
            "collision_frame": None,
            "collision_max": None,
        }
        labeler.entry_side = SimpleNamespace(value="LEFT")
        labeler.evasion_space = SimpleNamespace(value=1)
        labeler.entry_confidence = SimpleNamespace(value="HIGH")
        labeler.collision_confidence = SimpleNamespace(value="")
        labeler.lane_basis = SimpleNamespace(value="VISIBLE")

        self.assertEqual(
            labeler._validation_error(),
            "충돌의 min/best/max를 모두 지정하세요. 판단 불가 영상은 SKIP으로 저장하세요.",
        )


if __name__ == "__main__":
    unittest.main()
