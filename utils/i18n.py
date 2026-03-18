"""
Internationalization helpers.

Provide a small key-based translation layer shared by the main GUI,
runtime logs, and the fish_trainer GUI.
"""

from __future__ import annotations

import ctypes
import json
import locale
import os
from typing import Any

import config


DEFAULT_LANGUAGE = "zh-CN"
SUPPORTED_LANGUAGES = ("zh-CN", "en-US", "ja-JP")
LANGUAGE_NAMES = {
    "zh-CN": "简体中文",
    "en-US": "English",
    "ja-JP": "日本語",
}
TRANSLATION_RESOURCE = os.path.join("utils", "i18n.json")

WINDOWS_UI_LANGUAGE_MAP = {
    0x04: "zh-CN",
    0x09: "en-US",
    0x11: "ja-JP",
}


def _translation_file_path() -> str:
    if hasattr(config, "resolve_resource_path"):
        return config.resolve_resource_path(TRANSLATION_RESOURCE)
    return os.path.join(os.path.dirname(__file__), "i18n.json")


def _fallback_translations() -> dict[str, dict[str, str]]:
    return {
        "zh-CN": {
            "language.zh-CN": "简体中文",
            "language.en-US": "English",
            "language.ja-JP": "日本語",
            "status.ready": "就绪",
        },
        "en-US": {
            "language.zh-CN": "Simplified Chinese",
            "language.en-US": "English",
            "language.ja-JP": "Japanese",
            "status.ready": "Ready",
        },
        "ja-JP": {
            "language.zh-CN": "中国語(簡体字)",
            "language.en-US": "English",
            "language.ja-JP": "日本語",
            "status.ready": "準備完了",
        },
    }


def _load_translations() -> dict[str, dict[str, str]]:
    path = _translation_file_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except Exception:
        return _fallback_translations()

    if not isinstance(loaded, dict):
        return _fallback_translations()

    translations: dict[str, dict[str, str]] = {}
    for lang, mapping in loaded.items():
        if isinstance(lang, str) and isinstance(mapping, dict):
            translations[lang] = {
                str(key): str(value)
                for key, value in mapping.items()
            }

    for lang, fallback_name in LANGUAGE_NAMES.items():
        translations.setdefault(lang, {})
        translations[lang].setdefault(f"language.{lang}", fallback_name)
    return translations or _fallback_translations()


TRANSLATIONS = _load_translations()


def _normalize_language_code(lang: str | None) -> str | None:
    if lang in SUPPORTED_LANGUAGES:
        return lang
    if isinstance(lang, str):
        low = lang.lower()
        if low.startswith("zh"):
            return "zh-CN"
        if low.startswith("en"):
            return "en-US"
        if low.startswith("ja") or low.startswith("jp"):
            return "ja-JP"
    return None


def _read_windows_ui_language() -> str | None:
    if os.name != "nt":
        return None
    try:
        lang_id = ctypes.windll.kernel32.GetUserDefaultUILanguage()
    except Exception:
        return None
    normalized = _normalize_language_code(locale.windows_locale.get(lang_id, ""))
    if normalized:
        return normalized
    return WINDOWS_UI_LANGUAGE_MAP.get(lang_id & 0xFF)


def detect_system_language() -> str:
    candidates = [
        _read_windows_ui_language(),
        locale.getlocale()[0],
        os.environ.get("LC_ALL"),
        os.environ.get("LC_MESSAGES"),
        os.environ.get("LANG"),
        os.environ.get("LANGUAGE"),
    ]
    for candidate in candidates:
        normalized = _normalize_language_code(candidate)
        if normalized:
            return normalized
    return DEFAULT_LANGUAGE


_current_language = detect_system_language()


def normalize_language(lang: str | None) -> str:
    if isinstance(lang, str) and lang.lower() == "auto":
        return detect_system_language()
    normalized = _normalize_language_code(lang)
    if normalized:
        return normalized
    return DEFAULT_LANGUAGE


def available_languages() -> list[tuple[str, str]]:
    return [(lang, t(f"language.{lang}", lang=lang)) for lang in SUPPORTED_LANGUAGES]


def get_language() -> str:
    return _current_language


def set_language(lang: str | None) -> str:
    global _current_language
    normalized = normalize_language(lang)
    _current_language = normalized
    config.LANGUAGE = normalized
    return normalized


def t(key: str, default: str | None = None, **kwargs: Any) -> str:
    lang = get_language()
    template = TRANSLATIONS.get(lang, {}).get(key)
    if template is None:
        template = TRANSLATIONS.get(DEFAULT_LANGUAGE, {}).get(key)
    if template is None:
        template = default if default is not None else key
    if kwargs:
        try:
            return template.format(**kwargs)
        except Exception:
            return template
    return template


def fish_name(key: str) -> str:
    return t(f"fish.{key}", default=key)


def read_persisted_language() -> str:
    path = getattr(config, "SETTINGS_FILE", "")
    if not path or not os.path.exists(path):
        return normalize_language(getattr(config, "LANGUAGE", DEFAULT_LANGUAGE))
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return normalize_language(getattr(config, "LANGUAGE", DEFAULT_LANGUAGE))

    if isinstance(data, dict):
        current = data.get("current")
        if isinstance(current, dict) and isinstance(current.get("LANGUAGE"), str):
            return normalize_language(current["LANGUAGE"])
        if isinstance(data.get("LANGUAGE"), str):
            return normalize_language(data["LANGUAGE"])
    return normalize_language(getattr(config, "LANGUAGE", DEFAULT_LANGUAGE))


def write_persisted_language(lang: str | None):
    normalized = normalize_language(lang)
    path = getattr(config, "SETTINGS_FILE", "")
    if not path:
        return
    raw: dict[str, Any] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
            if isinstance(loaded, dict):
                raw = loaded
        except Exception:
            raw = {}

    if "current" in raw or "presets" in raw:
        current = raw.setdefault("current", {})
        if isinstance(current, dict):
            current["LANGUAGE"] = normalized
    else:
        raw["LANGUAGE"] = normalized

    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(raw, handle, indent=2, ensure_ascii=False)


def init_language() -> str:
    return set_language(read_persisted_language())
