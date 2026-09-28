"""Brand constants and static assets for the demo UI.

Everything a client rebrand touches lives here: product name, company name,
logo, favicon and the bundled sample image. Override the product name without
editing code by setting the PRODUCT_NAME environment variable.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"

PRODUCT_NAME = os.environ.get("PRODUCT_NAME", "Aether Scene Intelligence")
COMPANY_NAME = "Aether AI"
PAGE_HEADING = "Live scene analysis"
TAGLINE = ("Point a camera or upload an image to see which objects are in the "
           "scene and how they relate to each other.")

# Logo slot. SVG is inlined so its `currentColor` text follows light/dark mode;
# a PNG/JPG at the same stem is used if no SVG is present.
LOGO_PATH = ASSETS / "aether-logo.svg"
FAVICON_PATH = ASSETS / "aether-favicon.svg"

# Rendered on load so the page never opens on an empty canvas. CC0 (see
# assets/reel/credits.json).
SAMPLE_IMAGE = ASSETS / "reel" / "images" / "bicycle.jpg"
SAMPLE_CREDIT = ("Sample image: \"Man riding bicycle\" by Clem Onojeghuo, "
                 "CC0, via Wikimedia Commons.")


def logo_markup() -> str:
    """Header logo as HTML: inline SVG, an <img> data URI, or a text mark."""
    if LOGO_PATH.is_file():
        svg = LOGO_PATH.read_text(encoding="utf-8")
        svg = re.sub(r"<!--.*?-->", "", svg, flags=re.S).strip()
        return f'<span class="ae-logo" aria-label="{COMPANY_NAME}">{svg}</span>'
    for ext, mime in ((".png", "image/png"), (".jpg", "image/jpeg")):
        p = LOGO_PATH.with_suffix(ext)
        if p.is_file():
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            return (f'<img class="ae-logo" alt="{COMPANY_NAME}" '
                    f'src="data:{mime};base64,{b64}">')
    return f'<span class="ae-logo ae-logo--text">{COMPANY_NAME}</span>'
