"""Vocabulary box parsing and user-facing error text. Pure Python, no Gradio."""

from __future__ import annotations

import re
from typing import List

_SPLIT = re.compile(r"[,\n\r]+")
_MAX_DETAIL = 300


def parse_terms(text: str | None) -> List[str]:
    """Split on commas and new lines; trim, collapse inner spaces, drop empty
    entries and case-insensitive duplicates. Multi-word entries stay intact:
    "conveyor belt" is one term."""
    seen, out = set(), []
    for raw in _SPLIT.split(text or ""):
        term = " ".join(raw.split())
        key = term.lower()
        if term and key not in seen:
            seen.add(key)
            out.append(term)
    return out


def describe_error(exc: BaseException) -> tuple[str, str]:
    """(plain-English explanation, technical detail) for a failed update."""
    detail = f"{type(exc).__name__}: {exc}".strip()
    if len(detail) > _MAX_DETAIL:
        detail = detail[:_MAX_DETAIL - 3] + "..."
    low = detail.lower()
    if "prompt-free model does not support setting classes" in low:
        hint = ("This detector cannot take a custom object list, and the "
                "text-prompt detector could not be loaded.")
    elif "zerogpu" in low or "gpu quota" in low or "no gpu" in low:
        hint = ("GPU unavailable on this link. Open the Space on "
                "huggingface.co, or try again shortly.")
    elif any(k in low for k in ("download", "connection", "timed out",
                                "urlopen", "resolve host", "http error")):
        hint = ("A model file could not be downloaded. Check the Space's "
                "internet access, then try again.")
    elif "out of memory" in low or "memoryerror" in low:
        hint = "The server ran out of memory. Try a shorter list, or restart."
    elif "cuda" in low or "device" in low:
        hint = "A model is on the wrong device (CPU or GPU). Restart the app."
    else:
        hint = "The model rejected this update."
    return hint, detail
