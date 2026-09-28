"""Brand constants and static assets for the demo UI.

Everything a client rebrand touches lives here: product name, company name,
logo, favicon and the bundled sample image. Override the product name without
editing code by setting the PRODUCT_NAME environment variable.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS = REPO_ROOT / "assets"

PRODUCT_NAME = os.environ.get("PRODUCT_NAME", "RelayAI")
COMPANY_NAME = "Aether AI"
PAGE_HEADING = "Live scene analysis"
TAGLINE = ("Point a camera or upload an image to see which objects are in the "
           "scene and how they relate to each other.")

# Official logo on a transparent background: black for light mode, white for
# dark mode (1043x250, shown 24px tall). The favicon is the mark alone.
LOGO_LIGHT = ASSETS / "aether-logo-black.png"
LOGO_DARK = ASSETS / "aether-logo-white.png"
FAVICON_PATH = ASSETS / "aether-favicon.png"

# Rendered on load so the page never opens on an empty canvas. CC0 (see
# assets/reel/credits.json).
SAMPLE_IMAGE = ASSETS / "reel" / "images" / "bicycle.jpg"
SAMPLE_CREDIT = ("Sample image: \"Man riding bicycle\" by Clem Onojeghuo, "
                 "CC0, via Wikimedia Commons.")


def _data_uri(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def logo_markup() -> str:
    """Header logo: light and dark variants, CSS shows the one that fits."""
    if not (LOGO_LIGHT.is_file() and LOGO_DARK.is_file()):
        return f'<span class="ae-logo ae-logo--text">{COMPANY_NAME}</span>'
    return (f'<span class="ae-logo" role="img" aria-label="{COMPANY_NAME}">'
            f'<img class="ae-logo__img ae-logo__img--light" alt="" '
            f'src="{_data_uri(LOGO_LIGHT)}">'
            f'<img class="ae-logo__img ae-logo__img--dark" alt="" '
            f'src="{_data_uri(LOGO_DARK)}"></span>')
