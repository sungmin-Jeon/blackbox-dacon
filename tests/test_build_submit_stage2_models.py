"""Checks legacy and final Stage 2 submission model layouts."""

import tempfile
import unittest
from pathlib import Path

from build_submit import FINAL_STAGE2_MODELS, LEGACY_STAGE2_MODELS, required_models


class BuildSubmitStage2ModelTests(unittest.TestCase):
    def test_final_layout_is_selected_when_all_four_models_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for relative in FINAL_STAGE2_MODELS:
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            selected = required_models(root)
            self.assertTrue(FINAL_STAGE2_MODELS.issubset(selected))
            self.assertTrue(LEGACY_STAGE2_MODELS.isdisjoint(selected))

    def test_incomplete_final_layout_falls_back_to_legacy_requirements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / next(iter(FINAL_STAGE2_MODELS))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
            selected = required_models(root)
            self.assertTrue(LEGACY_STAGE2_MODELS.issubset(selected))


if __name__ == "__main__":
    unittest.main()
