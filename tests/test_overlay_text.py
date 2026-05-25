import unittest

from utils.overlay_text import overlay_fish_name, overlay_text, to_ascii


class OverlayTextTests(unittest.TestCase):
    def test_to_ascii_strips_unicode_without_question_marks(self):
        self.assertEqual(to_ascii("🐟 小游戏 F0001"), "F0001")
        self.assertEqual(to_ascii("旋转 12°…"), "12 deg...")

    def test_overlay_debug_labels_are_english_ascii(self):
        self.assertEqual(overlay_text("debug.rotation", angle=12.5), "Rotation: 12.5 deg")
        self.assertEqual(overlay_text("debug.noFishBar"), "X no fish+bar")

    def test_overlay_fish_names_use_english_strings(self):
        self.assertEqual(overlay_fish_name("fish_black"), "Black Fish")
        self.assertEqual(overlay_fish_name("fish_clover"), "Clover")


if __name__ == "__main__":
    unittest.main()
