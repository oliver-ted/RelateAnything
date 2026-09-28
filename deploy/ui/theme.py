"""Aether theme for the Gradio demo.

The design tokens (colours, radii, spacing, type scale) are defined ONCE, as
CSS custom properties at the top of styles.css, for both light and dark mode.
The Gradio theme below points its variables at those tokens, so a colour
change is a one-line edit in the stylesheet. The hue scales exist only so no
stock Gradio indigo can leak into a component the tokens do not reach.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import gradio as gr

from .branding import FAVICON_PATH

_HERE = Path(__file__).resolve().parent
CSS_PATH = _HERE / "styles.css"
HEAD_JS_PATH = _HERE / "head.js"

GRADIO_MAJOR = int(gr.__version__.split(".")[0])

# Aether green, anchored on #1B3A2B (brand, 800) and #2F6B4F (accent, 600).
AETHER_GREEN = gr.themes.Color(
    name="aether_green",
    c50="#EEF5F1", c100="#D5E7DD", c200="#AFCFBE", c300="#80B39A",
    c400="#56967A", c500="#3E8563", c600="#2F6B4F", c700="#265840",
    c800="#1B3A2B", c900="#142C21", c950="#0C1C15",
)
AETHER_GREY = gr.themes.Color(
    name="aether_grey",
    c50="#F6F7F6", c100="#EFF1F0", c200="#E3E6E4", c300="#CDD2CF",
    c400="#A3A9A6", c500="#7B817E", c600="#666666", c700="#3F4542",
    c800="#232926", c900="#161B18", c950="#0F1311",
)


def _v(name: str) -> str:
    return f"var(--ae-{name})"


# Gradio variable -> token. Each applies to light and dark alike: the token
# itself switches value under `.dark` in styles.css.
_TOKEN_MAP = {
    "body_background_fill": _v("bg"),
    "body_text_color": _v("text"),
    "body_text_color_subdued": _v("text-muted"),
    "body_text_size": _v("fs-body"),
    "background_fill_primary": _v("surface"),
    "background_fill_secondary": _v("surface-2"),
    "border_color_primary": _v("border"),
    "border_color_accent": _v("accent"),
    "border_color_accent_subdued": _v("accent-soft"),
    "color_accent": _v("accent"),
    "color_accent_soft": _v("accent-soft"),
    "link_text_color": _v("accent"),
    "link_text_color_hover": _v("brand"),
    "link_text_color_active": _v("brand"),
    "link_text_color_visited": _v("accent"),
    "loader_color": _v("accent"),
    "shadow_drop": "none",
    "shadow_drop_lg": _v("shadow"),
    "shadow_spread": "0px",
    # blocks and panels
    "block_background_fill": _v("surface"),
    "block_border_color": _v("border"),
    "block_border_width": "1px",
    "block_radius": _v("radius-lg"),
    "block_shadow": "none",
    "block_padding": _v("space-2"),
    "block_label_background_fill": "transparent",
    "block_label_border_width": "0px",
    "block_label_shadow": "none",
    "block_label_text_color": _v("text-muted"),
    "block_label_text_size": _v("fs-caption"),
    "block_label_text_weight": "500",
    "block_label_padding": "0px",
    "block_title_background_fill": "transparent",
    "block_title_border_width": "0px",
    "block_title_text_color": _v("text-muted"),
    "block_title_text_size": _v("fs-caption"),
    "block_title_text_weight": "500",
    "block_title_padding": "0px",
    "block_title_radius": "0px",
    "container_radius": _v("radius-lg"),
    "panel_background_fill": _v("surface"),
    "panel_border_color": _v("border"),
    "panel_border_width": "1px",
    "layout_gap": _v("space-2"),
    "form_gap_width": "0px",
    "section_header_text_size": _v("fs-section"),
    "section_header_text_weight": "600",
    # inputs
    "input_background_fill": _v("surface"),
    "input_background_fill_focus": _v("surface"),
    "input_background_fill_hover": _v("surface"),
    "input_border_color": _v("border-strong"),
    "input_border_color_hover": _v("text-subtle"),
    "input_border_color_focus": _v("accent"),
    "input_border_width": "1px",
    "input_radius": _v("radius"),
    "input_shadow": "none",
    "input_shadow_focus": f"0 0 0 3px {_v('accent-soft')}",
    "input_placeholder_color": _v("text-subtle"),
    "input_text_size": _v("fs-body"),
    "input_padding": _v("space-1") + " 12px",
    # sliders, checkboxes, radios: neutral, accent only when active
    "slider_color": _v("accent"),
    "checkbox_background_color": _v("surface"),
    "checkbox_background_color_hover": _v("surface"),
    "checkbox_background_color_focus": _v("surface"),
    "checkbox_background_color_selected": _v("accent"),
    "checkbox_border_color": _v("border-strong"),
    "checkbox_border_color_hover": _v("text-subtle"),
    "checkbox_border_color_focus": _v("accent"),
    "checkbox_border_color_selected": _v("accent"),
    "checkbox_border_radius": _v("radius-sm"),
    "checkbox_shadow": "none",
    "checkbox_label_background_fill": "transparent",
    "checkbox_label_background_fill_hover": "transparent",
    "checkbox_label_background_fill_selected": "transparent",
    "checkbox_label_border_color": "transparent",
    "checkbox_label_border_color_hover": "transparent",
    "checkbox_label_border_color_selected": "transparent",
    "checkbox_label_border_width": "0px",
    "checkbox_label_shadow": "none",
    "checkbox_label_padding": _v("space-1") + " 0px",
    "checkbox_label_text_color": _v("text"),
    "checkbox_label_text_color_selected": _v("text"),
    "checkbox_label_text_size": _v("fs-body"),
    "checkbox_label_text_weight": "400",
    # tables
    "table_border_color": _v("border"),
    "table_even_background_fill": _v("surface"),
    "table_odd_background_fill": _v("surface-2"),
    "table_radius": _v("radius"),
    "table_row_focus": _v("accent-soft"),
    "table_text_color": _v("text"),
    "stat_background_fill": _v("accent-soft"),
    "error_background_fill": _v("danger-soft"),
    "error_border_color": _v("danger"),
    # buttons: exactly one filled primary on screen
    "button_primary_background_fill": _v("accent"),
    "button_primary_background_fill_hover": _v("accent-hover"),
    "button_primary_border_color": _v("accent"),
    "button_primary_border_color_hover": _v("accent-hover"),
    "button_primary_text_color": "#FFFFFF",
    "button_primary_text_color_hover": "#FFFFFF",
    "button_primary_shadow": "none",
    "button_primary_shadow_hover": "none",
    "button_primary_shadow_active": "none",
    "button_secondary_background_fill": _v("surface"),
    "button_secondary_background_fill_hover": _v("surface-2"),
    "button_secondary_border_color": _v("border-strong"),
    "button_secondary_border_color_hover": _v("text-subtle"),
    "button_secondary_text_color": _v("text"),
    "button_secondary_text_color_hover": _v("text"),
    "button_secondary_shadow": "none",
    "button_secondary_shadow_hover": "none",
    "button_secondary_shadow_active": "none",
    "button_cancel_background_fill": _v("surface"),
    "button_cancel_background_fill_hover": _v("surface-2"),
    "button_cancel_border_color": _v("border-strong"),
    "button_cancel_shadow": "none",
}


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Base(
        primary_hue=AETHER_GREEN,
        secondary_hue=AETHER_GREY,
        neutral_hue=AETHER_GREY,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        text_size=gr.themes.sizes.text_md,
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"],
    )
    # Gradio versions differ in which variables exist; set what this one has.
    known = set(inspect.signature(theme.set).parameters)
    values = {}
    for key, val in _TOKEN_MAP.items():
        if key in known:
            values[key] = val
        if f"{key}_dark" in known:
            values[f"{key}_dark"] = val
    return theme.set(**values)


def load_css() -> str:
    return CSS_PATH.read_text(encoding="utf-8")


def load_head() -> str:
    return f"<script>\n{HEAD_JS_PATH.read_text(encoding='utf-8')}\n</script>"


def _accepted(fn, kwargs: dict) -> dict:
    params = inspect.signature(fn).parameters
    return {k: v for k, v in kwargs.items() if k in params}


def blocks_kwargs(title: str) -> dict:
    """Kwargs for gr.Blocks(). Gradio 5 takes theme/css/head here, 6 in launch."""
    kw = {"title": title, "fill_width": True}
    if GRADIO_MAJOR < 6:
        kw.update(theme=build_theme(), css=load_css(), head=load_head())
    return _accepted(gr.Blocks.__init__, kw)


def launch_kwargs() -> dict:
    """Presentation kwargs for Blocks.launch(): favicon, footer, theme on 6+."""
    kw = {"favicon_path": str(FAVICON_PATH), "show_api": False,
          "footer_links": []}
    if GRADIO_MAJOR >= 6:
        kw.update(theme=build_theme(), css=load_css(), head=load_head())
    return _accepted(gr.Blocks.launch, kw)


def component_kwargs(cls, **kwargs) -> dict:
    """Drop kwargs this Gradio version's component does not accept."""
    return _accepted(cls.__init__, kwargs)
