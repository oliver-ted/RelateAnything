"""Presentation of a SceneResult: results table rows and status strip HTML.

Pure Python (no Gradio, no torch) so it can be unit tested on its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import List, Optional, Sequence

TABLE_HEADERS = ["Subject", "Relationship", "Object", "Confidence"]

# graph view value -> which triplet lists feed the table
_STREAMS = {
    "merged": ("triplets",),
    "spatial": ("triplets_spatial",),
    "semantic": ("triplets_semantic",),
    "both": ("triplets_spatial", "triplets_semantic"),
}


def object_name(labels: Sequence[str], i: int) -> str:
    """Instance name as shown in the table and on the image, e.g. 'person #2'."""
    lab = labels[i] if i < len(labels) else "object"
    return f"{lab} #{i + 1}"


def relationship_rows(res, mode: str) -> List[list]:
    """[Subject, Relationship, Object, Confidence] rows, highest confidence first."""
    rows = []
    for attr in _STREAMS.get(mode, _STREAMS["merged"]):
        for s, pred, o, sc in getattr(res, attr, None) or []:
            rows.append([object_name(res.labels, s), str(pred),
                         object_name(res.labels, o), round(float(sc), 2)])
    rows.sort(key=lambda r: r[3], reverse=True)
    return rows


@dataclass
class Metrics:
    latency_ms: Optional[float] = None
    fps: Optional[float] = None
    objects: Optional[int] = None
    relationships: Optional[int] = None
    breakdown: str = ""          # tooltip for the latency figure
    warning: str = ""            # short caveat, shown as a muted pill

    @classmethod
    def from_result(cls, res, n_relationships: int, warning: str = ""):
        t = res.timing
        breakdown = (f"Detection {t.det:.1f} ms, scene encoding "
                     f"{t.backbone:.1f} ms, relationships {t.relation:.1f} ms")
        return cls(latency_ms=float(t.total), fps=float(t.fps),
                   objects=int(len(res.boxes_xyxy)),
                   relationships=int(n_relationships),
                   breakdown=breakdown, warning=warning)


# state -> (dot modifier, label)
_STATES = {
    "ready": ("is-ready", "Model ready"),
    "processing": ("is-busy", "Processing"),
    "updating": ("is-busy", "Updating vocabulary"),
    "loading": ("is-busy", "Model loading"),
    "idle": ("is-idle", "Waiting for input"),
    "error": ("is-error", "Something went wrong"),
}


def _stat(label: str, value: str, title: str = "") -> str:
    tip = f' title="{escape(title)}"' if title else ""
    return (f'<div class="ae-stat"{tip}><span class="ae-stat__label">'
            f'{escape(label)}</span><span class="ae-stat__value">'
            f'{escape(value)}</span></div>')


def status_strip(state: str, m: Optional[Metrics] = None,
                 message: str = "") -> str:
    """Status strip under the scene graph: model state plus key figures."""
    m = m or Metrics()
    dot, label = _STATES.get(state, _STATES["ready"])
    dash = "–"
    latency = (f"{m.latency_ms:.0f} ms" if m.latency_ms is not None else dash)
    fps = f"{m.fps:.1f}" if m.fps is not None else dash
    objects = str(m.objects) if m.objects is not None else dash
    rels = str(m.relationships) if m.relationships is not None else dash
    parts = [
        f'<div class="ae-status__state"><span class="ae-dot {dot}"></span>'
        f'<span>{escape(label)}</span></div>',
        _stat("Latency", latency, m.breakdown),
        _stat("Frames per second", fps),
        _stat("Objects", objects),
        _stat("Relationships", rels),
    ]
    extra = message or m.warning
    if extra:
        parts.append(f'<div class="ae-status__note">{escape(extra)}</div>')
    return f'<div class="ae-status" role="status">{"".join(parts)}</div>'


def notice(kind: str, title: str, body: str = "", detail: str = "") -> str:
    """Inline state. kind: 'empty' | 'error' | 'info' | 'success'.
    `detail` is the technical error text, shown verbatim in a muted line."""
    body_html = f'<p class="ae-empty__body">{escape(body)}</p>' if body else ""
    detail_html = (f'<p class="ae-empty__detail">Details: '
                   f'<code>{escape(detail)}</code></p>' if detail else "")
    return (f'<div class="ae-empty ae-empty--{kind}">'
            f'<p class="ae-empty__title">{escape(title)}</p>{body_html}'
            f'{detail_html}</div>')
