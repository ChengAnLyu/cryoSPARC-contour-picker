"""Default launch must preserve the screen workflow, without starting a desktop here."""
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import Mock, patch

import app


class LaunchModeTests(unittest.TestCase):
    def launch(self, arguments):
        root = Mock()
        with patch.dict(sys.modules, {"tkinter": SimpleNamespace(Tk=Mock(return_value=root))}), \
             patch.object(app, "setup_logging", return_value=Path("test.log")), \
             patch.object(app, "resolve_checkpoint", return_value=Path("bundled-model.pth")), \
             patch.object(app, "enable_dpi_awareness"), \
             patch.object(app, "ScreenPickerApp") as screen, \
             patch.object(app, "PickerApp") as review:
            self.assertEqual(app.main(arguments), 0)
            root.mainloop.assert_called_once()
            return root, screen, review

    def test_normal_shortcut_launch_uses_compact_screen_application(self):
        root, screen, review = self.launch([])
        screen.assert_called_once()
        self.assertIs(screen.call_args.args[0], root)
        self.assertEqual(screen.call_args.args[3], Path("bundled-model.pth"))
        self.assertEqual(screen.call_args.args[1].min_capsid_area, 3500)
        self.assertEqual(screen.call_args.args[1].backend, "sam")
        self.assertEqual(screen.call_args.args[1].automatic_profile, "capsid")
        review.assert_not_called()

    def test_explicit_sam_source_backend_preserves_screen_workflow(self):
        _, screen, review = self.launch(["--backend", "sam"])
        self.assertEqual(screen.call_args.args[1].backend, "sam")
        self.assertEqual(screen.call_args.args[1].automatic_profile, "capsid")
        review.assert_not_called()

    def test_size_filter_override_is_available_without_changing_launch_mode(self):
        _, screen, review = self.launch(["--min-capsid-area", "8000"])
        self.assertEqual(screen.call_args.args[1].min_capsid_area, 8000)
        review.assert_not_called()

    def test_snapshot_editor_is_explicitly_optional(self):
        _, screen, review = self.launch(["--review"])
        review.assert_called_once()
        screen.assert_not_called()

    def test_opened_image_is_routed_to_offline_review(self):
        _, screen, review = self.launch(["--image", "micrograph.png"])
        review.assert_called_once()
        self.assertEqual(review.call_args.args[1].image, "micrograph.png")
        screen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
