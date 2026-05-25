"""
ASCII-safe helpers for OpenCV overlay text.
"""

from __future__ import annotations

import unicodedata

from utils.i18n import TRANSLATIONS


OVERLAY_LANGUAGE = "en-US"
ASCII_REPLACEMENTS = {
    "°": " deg",
    "…": "...",
    "–": "-",
    "—": "-",
    "→": " -> ",
    "←": " <- ",
    "✓": "OK",
    "✗": "X",
}


def to_ascii(text) -> str:
    normalized = str(text or "")
    for src, dst in ASCII_REPLACEMENTS.items():
        normalized = normalized.replace(src, dst)
    normalized = (
        unicodedata.normalize("NFKD", normalized)
        .encode("ascii", errors="ignore")
        .decode("ascii")
    )
    return " ".join(normalized.split())


def overlay_text(key: str, default: str | None = None, **kwargs) -> str:
    template = TRANSLATIONS.get(OVERLAY_LANGUAGE, {}).get(key)
    if template is None:
        template = default if default is not None else key
    if kwargs:
        try:
            template = template.format(**kwargs)
        except Exception:
            pass
    return to_ascii(template)


def overlay_fish_name(key: str) -> str:
    default = key.replace("_", " ")
    return overlay_text(f"fish.{key}", default=default)
