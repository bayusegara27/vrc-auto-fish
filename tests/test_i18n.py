import os
import tempfile
import unittest
from unittest import mock

import config
from utils.i18n import (
    TRANSLATIONS,
    detect_system_language,
    fish_name,
    get_language,
    normalize_language,
    read_persisted_language,
    set_language,
    t,
    write_persisted_language,
)


class I18nTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.old_settings_file = config.SETTINGS_FILE
        self.old_language = config.LANGUAGE
        config.SETTINGS_FILE = os.path.join(self.tmpdir.name, "settings.json")
        set_language("zh-CN")

    def tearDown(self):
        config.SETTINGS_FILE = self.old_settings_file
        set_language(self.old_language)
        self.tmpdir.cleanup()

    def test_set_language_translates_basic_keys(self):
        set_language("en-US")
        self.assertEqual(get_language(), "en-US")
        self.assertEqual(t("status.ready"), "Ready")

        set_language("ja-JP")
        self.assertEqual(get_language(), "ja-JP")
        self.assertEqual(t("status.ready"), "準備完了")

        set_language("zh-CN")
        self.assertEqual(t("status.ready"), "就绪")

    def test_write_and_read_persisted_language(self):
        write_persisted_language("en-US")
        self.assertEqual(read_persisted_language(), "en-US")

        write_persisted_language("ja")
        self.assertEqual(read_persisted_language(), "ja-JP")

    def test_normalize_language_supports_japanese_aliases(self):
        self.assertEqual(normalize_language("ja"), "ja-JP")
        self.assertEqual(normalize_language("ja-jp"), "ja-JP")
        self.assertEqual(normalize_language("jp"), "ja-JP")

    def test_normalize_language_supports_auto(self):
        with mock.patch("utils.i18n.detect_system_language", return_value="en-US"):
            self.assertEqual(normalize_language("auto"), "en-US")

    def test_detect_system_language_falls_back_from_windows_locale(self):
        with (
            mock.patch("utils.i18n._read_windows_ui_language", return_value="ja-JP"),
            mock.patch("utils.i18n.locale.getlocale", return_value=(None, None)),
        ):
            self.assertEqual(detect_system_language(), "ja-JP")

    def test_read_persisted_language_uses_system_language_when_auto(self):
        config.LANGUAGE = "auto"
        with mock.patch("utils.i18n.detect_system_language", return_value="en-US"):
            self.assertEqual(read_persisted_language(), "en-US")

    def test_fish_teal_name_is_available_in_all_languages(self):
        set_language("zh-CN")
        self.assertEqual(fish_name("fish_teal"), "四叶草")

        set_language("en-US")
        self.assertEqual(fish_name("fish_teal"), "Clover")

        set_language("ja-JP")
        self.assertEqual(fish_name("fish_teal"), "クローバー")

    def test_renamed_fish_names_are_available_in_all_languages(self):
        set_language("zh-CN")
        self.assertEqual(fish_name("fish_green"), "绿鱼")
        self.assertEqual(fish_name("fish_clover"), "四叶草")
        self.assertEqual(fish_name("fish_relic"), "遗物")
        self.assertEqual(fish_name("fish_black"), "黑鱼")
        self.assertEqual(fish_name("fish_question"), "问号鱼")

        set_language("en-US")
        self.assertEqual(fish_name("fish_green"), "Green Fish")
        self.assertEqual(fish_name("fish_clover"), "Clover")
        self.assertEqual(fish_name("fish_relic"), "Relic")
        self.assertEqual(fish_name("fish_black"), "Black Fish")
        self.assertEqual(fish_name("fish_question"), "Question Fish")

        set_language("ja-JP")
        self.assertEqual(fish_name("fish_green"), "緑魚")
        self.assertEqual(fish_name("fish_clover"), "クローバー")
        self.assertEqual(fish_name("fish_relic"), "遺物")
        self.assertEqual(fish_name("fish_black"), "黒魚")
        self.assertEqual(fish_name("fish_question"), "はてな魚")

    def test_translation_languages_have_identical_key_sets(self):
        key_sets = {lang: set(entries.keys()) for lang, entries in TRANSLATIONS.items()}
        self.assertEqual(key_sets["zh-CN"], key_sets["en-US"])
        self.assertEqual(key_sets["zh-CN"], key_sets["ja-JP"])

    def test_missing_translation_key_uses_default_or_key(self):
        set_language("en-US")
        self.assertEqual(t("missing.translation", default="Fallback"), "Fallback")
        self.assertEqual(t("missing.translation"), "missing.translation")


if __name__ == "__main__":
    unittest.main()
