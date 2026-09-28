"""Demo UI presentation helpers (deploy/ui/results.py): no Gradio, no torch."""
from types import SimpleNamespace

import numpy as np

from deploy.ui.results import (Metrics, notice, object_name,
                               relationship_rows, status_strip)


def _res():
    return SimpleNamespace(
        labels=["person", "bicycle", "helmet"],
        boxes_xyxy=np.zeros((3, 4)),
        triplets=[(0, "wearing", 2, 0.64), (0, "riding", 1, 0.81)],
        triplets_spatial=[(2, "on", 0, 0.55)],
        triplets_semantic=[(0, "riding", 1, 0.81)],
        timing=SimpleNamespace(det=12.0, backbone=15.0, relation=9.0,
                               total=37.4, fps=26.7),
    )


def test_object_name_is_one_based_and_tolerates_missing_labels():
    assert object_name(["person"], 0) == "person #1"
    assert object_name([], 3) == "object #4"


def test_rows_sorted_by_confidence_for_each_graph_view():
    res = _res()
    merged = relationship_rows(res, "merged")
    assert merged == [["person #1", "riding", "bicycle #2", 0.81],
                      ["person #1", "wearing", "helmet #3", 0.64]]
    both = relationship_rows(res, "both")
    assert [r[3] for r in both] == [0.81, 0.55]
    assert relationship_rows(res, "spatial") == [
        ["helmet #3", "on", "person #1", 0.55]]


def test_status_strip_shows_metrics_and_escapes_text():
    res = _res()
    html = status_strip("ready", Metrics.from_result(res, 2))
    assert "Model ready" in html and "37 ms" in html and "26.7" in html
    assert "Detection 12.0 ms" in html          # latency tooltip
    assert "&lt;b&gt;" in status_strip("error", message="<b>")
    assert "Waiting for input" in status_strip("idle")


def test_notice_escapes_user_visible_text():
    assert "&lt;script&gt;" in notice("error", "<script>")
