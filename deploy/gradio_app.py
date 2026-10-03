"""RelateAnything live demo — YOLOE-11m (masks) + open-vocabulary relations.

Presented as a branded client demo (RelayAI by Aether AI). Theme, styles,
branding and results formatting live in deploy/ui/; this file holds the layout
and the event wiring. Model calls and default knob values are unchanged.

    python deploy/gradio_app.py                    # auto GPU, opens in browser
    python deploy/gradio_app.py --device cpu       # no GPU needed
    python deploy/gradio_app.py --share            # public link (SME demo)
    python deploy/gradio_app.py --port 7860

Everything runs server-side; the browser only ships webcam frames, so the demo
works on a laptop with the model on a remote GPU. On CPU it still runs, just
slower — use the Image tab rather than the live webcam there.

BOTH models are re-parameterizable live, which is the point of the demo:
  * object classes  -> YOLOE text prompts (`set_classes` + `get_text_pe`)
  * predicates      -> relation head vocabulary, encoded by the checkpoint's
                       own text encoder; inference stays pure-vision afterwards
Type any words into either box, hit Apply, and the pipeline is re-targeted
without touching the weights.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr                                        # noqa: E402
from deploy.pipeline import (ParallelScenePipeline,        # noqa: E402
                             PipelineConfig, _default_predicates)
from deploy.ui import branding                             # noqa: E402
from deploy.ui.results import (TABLE_HEADERS, Metrics,     # noqa: E402
                               notice, object_name,
                               relationship_rows, status_strip)
from deploy.ui.vocab_input import describe_error, parse_terms  # noqa: E402
from deploy.ui.theme import (blocks_kwargs,                # noqa: E402
                             component_kwargs, launch_kwargs)

log = logging.getLogger("demo")

DEFAULT_CLASSES = [
    "person", "face", "hand", "laptop", "keyboard", "mouse", "monitor", "cup",
    "bottle", "phone", "book", "chair", "desk", "backpack", "hat", "glasses",
    "plant", "door", "window", "picture", "lamp", "bag",
]

# Distinct hues per object instance, fixed by index so a box keeps its colour
# across frames (colour follows the entity, never its rank).
PALETTE = [(42, 120, 214), (235, 104, 52), (27, 175, 122), (138, 99, 210),
           (214, 168, 42), (52, 187, 235), (200, 62, 120), (120, 160, 60)]

PIPE: ParallelScenePipeline | None = None


def _color(i: int):
    return PALETTE[i % len(PALETTE)]


# Two-graph edge colours: layout (spatial) vs content (semantic). Fixed by
# meaning, not by rank — a predicate keeps its colour across frames.
SPATIAL_EDGE = (120, 200, 255)    # BGR: warm-cyan  → "where things are"
SEMANTIC_EDGE = (140, 255, 170)   # BGR: green      → "what things do"
MERGED_EDGE = (255, 255, 255)


def _edges_for(res, mode: str):
    """(list_of_triplets, colour) pairs for the requested graph mode."""
    if mode == "merged":
        return [(res.triplets, MERGED_EDGE)]
    if mode == "spatial":
        return [(res.triplets_spatial, SPATIAL_EDGE)]
    if mode == "semantic":
        return [(res.triplets_semantic, SEMANTIC_EDGE)]
    return [(res.triplets_spatial, SPATIAL_EDGE),
            (res.triplets_semantic, SEMANTIC_EDGE)]     # both


def render(res, show_masks: bool, show_labels: bool,
           mode: str = "merged", hud: bool = True) -> np.ndarray:
    """Draw masks, boxes and relation arrows onto the frame (BGR in, RGB out).

    `hud=False` omits the latency bar; the web UI shows it in its status strip.
    """
    img = res.frame.copy()
    H, W = img.shape[:2]

    if show_masks and res.masks is not None and len(res.masks):
        overlay = img.copy()
        for i, m in enumerate(res.masks[:len(res.boxes_xyxy)]):
            if m.shape[:2] != (H, W):
                m = cv2.resize(m.astype(np.uint8), (W, H),
                               interpolation=cv2.INTER_NEAREST).astype(bool)
            overlay[m] = _color(i)
        img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)

    centers = []
    for i, bb in enumerate(res.boxes_xyxy):
        x1, y1, x2, y2 = [int(v) for v in bb]
        centers.append(((x1 + x2) // 2, (y1 + y2) // 2))
        cv2.rectangle(img, (x1, y1), (x2, y2), _color(i), 2)
        if show_labels:
            sc = float(res.scores[i]) if i < len(res.scores) else 0.0
            txt = f"{object_name(res.labels, i)} {sc:.2f}"
            (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1),
                          _color(i), -1)
            cv2.putText(img, txt, (x1 + 2, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                        cv2.LINE_AA)

    # relation arrows, thickness ~ score; colour ~ graph (layout vs content).
    # In "both" mode the two streams are drawn with a small perpendicular
    # offset so a pair carrying one edge of each type stays readable.
    streams = _edges_for(res, mode)
    for si, (trips, edge_col) in enumerate(streams):
        off = 0 if len(streams) == 1 else (6 if si == 0 else -6)
        for s, pred, o, sc in trips:
            if s >= len(centers) or o >= len(centers):
                continue
            p1, p2 = centers[s], centers[o]
            dx, dy = p2[0] - p1[0], p2[1] - p1[1]
            n = max(1.0, (dx * dx + dy * dy) ** 0.5)
            ox, oy = int(-dy / n * off), int(dx / n * off)
            q1, q2 = (p1[0] + ox, p1[1] + oy), (p2[0] + ox, p2[1] + oy)
            cv2.arrowedLine(img, q1, q2, edge_col, max(1, int(1 + 3 * sc)),
                            cv2.LINE_AA, tipLength=0.03)
            mid = ((q1[0] + q2[0]) // 2, (q1[1] + q2[1]) // 2)
            (tw, th), _ = cv2.getTextSize(pred, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img, (mid[0] - 2, mid[1] - th - 4),
                          (mid[0] + tw + 4, mid[1] + 3), (30, 30, 30), -1)
            cv2.putText(img, pred, (mid[0] + 1, mid[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, edge_col, 1, cv2.LINE_AA)

    if not hud:
        return img[:,:,::-1]
    t = res.timing
    hud = (f"{t.fps:5.1f} FPS | det {t.det:5.1f} | backbone {t.backbone:5.1f} "
           f"| rel {t.relation:5.1f} | total {t.total:5.1f} ms")
    cv2.rectangle(img, (0, 0), (W, 24), (20, 20, 20), -1)
    cv2.putText(img, hud, (8, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (120, 230, 160), 1, cv2.LINE_AA)
    return img[:,:,::-1]


# --------------------------------------------------------------------------
# Inference (unchanged pipeline call) and its presentation
# --------------------------------------------------------------------------

GRAPH_VIEWS = [("Combined", "merged"), ("Both", "both"),
               ("Spatial", "spatial"), ("Semantic", "semantic")]
SOURCES = ["Image", "Camera"]
_NO_CHANGE = 5      # number of outputs `analyze` writes


def run_pipeline(frame_rgb, conf, top_k, score_thr, mode, spatial_raw):
    """The model call, exactly as the original demo made it."""
    PIPE.cfg.det_conf = float(conf)
    return PIPE(frame_rgb[:,:,::-1].copy(), top_k=int(top_k),
                score_thr=float(score_thr), decompose=(mode != "merged"),
                spatial_drop_pair=bool(spatial_raw))


def analyze(frame_rgb, conf, top_k, score_thr, show_masks, show_labels, mode,
            spatial_raw):
    """-> (scene graph image, table rows, status html, note html, metrics)."""
    if PIPE is None:
        return (None, [], status_strip("loading"),
                notice("info", "The model is starting up",
                       "Results will appear here in a moment."), None)
    if frame_rgb is None:
        return (None, [], status_strip("idle"),
                notice("empty", "No image yet",
                       "Upload an image, or switch to Camera for live "
                       "analysis."), None)
    try:
        res = run_pipeline(frame_rgb, conf, top_k, score_thr, mode,
                           spatial_raw)
        rows = relationship_rows(res, mode)
        warning = ""
        if mode != "merged" and not PIPE.has_dual_head:
            warning = "Graph view split is approximate for this model"
        metrics = Metrics.from_result(res, len(rows), warning)
        image = render(res, show_masks, show_labels, mode, hud=False)
    except Exception:
        log.exception("inference failed")
        return (gr.update(), gr.update(),
                status_strip("error", message="Could not analyze this frame"),
                notice("error", "We could not analyze this image",
                       "Try a different image, or refresh the page if the "
                       "problem continues."), None)
    if rows:
        note = ""
    elif metrics.objects:
        note = notice("empty", "No relationships found",
                      "Try lowering Relationship confidence, or add "
                      "relationships to the detection vocabulary.")
    else:
        note = notice("empty", "No objects detected",
                      "Try lowering Detection sensitivity, or check the "
                      "objects in the detection vocabulary.")
    return image, rows, status_strip("ready", metrics), note, metrics


def analyze_still(source, frame_rgb, *knobs):
    """Re-run the still image after a knob change; no-op in camera mode.

    `frame_rgb` comes from server-side session state, not from the Image
    component: re-reading the component makes Gradio reopen its temp-file
    copy, which may no longer exist (it broke the hosted Space).
    """
    if source != "Image":
        return (gr.update(),) * _NO_CHANGE
    return analyze(frame_rgb, *knobs)


def analyze_input(frame_rgb, *knobs):
    """New upload, paste or clear: remember the frame, then analyze it."""
    return (frame_rgb,) + analyze(frame_rgb, *knobs)


def load_sample(*knobs):
    """Page load: show and analyze the bundled sample, straight from memory."""
    frame = _load_sample()
    return (frame, frame) + analyze(frame, *knobs)


def mark_processing(metrics):
    return status_strip("processing", metrics)


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'s' if n != 1 else ''}"


def apply_vocab(classes_txt, preds_txt):
    """Re-target both vocabularies and report what actually happened.

    Every failure is logged with its full traceback and shown in the page with
    its real error text; nothing escapes to Gradio's generic error toast.
    """
    if PIPE is None:
        return notice("info", "The model is still starting up",
                      "Try again in a moment.")
    done, errors = [], []
    cls, prs = parse_terms(classes_txt), parse_terms(preds_txt)
    try:
        if cls:
            PIPE.set_object_classes(cls)
            done.append(_count(len(cls), "object"))
        elif PIPE.reset_object_classes():
            done.append("any object")
    except Exception as e:
        log.exception("set_object_classes failed for %r", cls)
        errors.append(("objects", e))
    try:
        if prs:
            PIPE.set_predicates(prs)
            done.append(_count(len(prs), "relationship"))
    except Exception as e:
        log.exception("set_predicates failed for %r", prs)
        errors.append(("relationships", e))
    if errors:
        what = " and ".join(name for name, _ in errors)
        hint, detail = describe_error(errors[0][1])
        if len(errors) > 1:
            detail += f" | {describe_error(errors[1][1])[1]}"
        updated = f" Updated: {' and '.join(done)}." if done else ""
        return notice("error", f"Could not update {what}", hint + updated,
                      detail=detail)
    if not done:
        return notice("info", "Nothing to apply",
                      "Add at least one object or relationship, then "
                      "select Apply.")
    return notice("success", "Vocabulary updated",
                  f"Now detecting {' and '.join(done)}.")


def _apply_busy():
    return (gr.update(value="Applying", interactive=False),
            notice("info", "Updating vocabulary",
                   "The first update can take up to a minute while models "
                   "load."))


def _apply_idle():
    return gr.update(value="Apply", interactive=True)


def switch_source(source):
    camera = source == "Camera"
    note = (notice("info", "Camera selected",
                   "Start the camera in the input panel to begin live "
                   "analysis.") if camera else gr.update())
    return gr.update(visible=not camera), gr.update(visible=camera), note


def _load_sample():
    img = cv2.imread(str(branding.SAMPLE_IMAGE))
    return None if img is None else img[:,:,::-1].copy()


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------

def _html(value: str = "", **kw) -> gr.HTML:
    """Static/structural HTML block without Gradio's own padding and frame."""
    return gr.HTML(value, **kw, **component_kwargs(gr.HTML, padding=False,
                                                    container=False))


def _section(title: str, caption: str = "") -> str:
    cap = f'<p class="ae-section__caption">{caption}</p>' if caption else ""
    return f'<div class="ae-section"><h2 class="ae-section__title">{title}</h2>{cap}</div>'


def _header() -> str:
    return (
        '<header class="ae-header">'
        '<div class="ae-header__brand">'
        f'{branding.logo_markup()}'
        '<span class="ae-header__divider" aria-hidden="true"></span>'
        f'<span class="ae-header__product">{branding.PRODUCT_NAME}</span>'
        '</div>'
        f'<div class="ae-header__meta">Powered by <strong>{branding.COMPANY_NAME}</strong></div>'
        '</header>'
        '<div class="ae-intro">'
        f'<h1 class="ae-intro__title">{branding.PAGE_HEADING}</h1>'
        f'<p class="ae-intro__subtitle">{branding.TAGLINE}</p>'
        '</div>')


def _about(runtime: dict) -> str:
    return f"""
**How it works.** An object detector (YOLOE-11m with segmentation masks) finds
the objects in each frame. An open-vocabulary relationship model then scores
how every pair of objects relates. Both vocabularies can be changed at any
time, with no retraining.

**Graph view.** *Combined* shows a single ranked list of relationships.
*Spatial* shows where things are relative to each other (amber arrows) and
*Semantic* shows what things are doing (green arrows). *Both* shows the two
side by side. All views come from the same analysis pass, ranked separately.
In benchmark testing, the split views beat a single combined graph twice their
size on all 6 test settings.

**Strict spatial scoring.** Scores spatial relationships on geometry alone.
This is the most accurate setting for judging spatial truth (+0.068 macro AUC
on SpatialSense), but it can surface duplicate detections as relationships,
so it is off by default.

**Runtime.** Compute: {runtime.get("device", "unknown")}.
Detector: `{runtime.get("detector", "unknown")}`.
Relationship model: `{runtime.get("relation_model", "unknown")}`.

{branding.SAMPLE_CREDIT}
"""


def build_ui(runtime: dict | None = None):
    runtime = runtime or {}
    prompt_free = runtime.get("prompt_free", True)

    with gr.Blocks(**blocks_kwargs(f"{branding.PRODUCT_NAME} | "
                                   f"{branding.COMPANY_NAME}")) as demo:
        _html(_header(), elem_classes="ae-header-host")

        with gr.Row(elem_classes="ae-main", equal_height=False):
            # ---------------- main column: output first ----------------
            with gr.Column(scale=8, min_width=560, elem_classes="ae-col"):
                with gr.Column(elem_classes="ae-card ae-hero"):
                    _html(_section("Scene graph"))
                    out = gr.Image(
                        label="Scene graph", show_label=False, height=520,
                        interactive=False, elem_classes="ae-hero__image",
                        **component_kwargs(
                            gr.Image, buttons=["download", "fullscreen"],
                            show_share_button=False,
                            show_fullscreen_button=True))
                    status = _html(status_strip("loading"),
                                   elem_classes="ae-status-host")

                with gr.Row(equal_height=False, elem_classes="ae-subrow"):
                    with gr.Column(scale=2, min_width=280,
                                   elem_classes="ae-card"):
                        _html(_section("Input"))
                        source = gr.Radio(SOURCES, value="Image",
                                          show_label=False, container=False,
                                          elem_classes="ae-segmented")
                        still = gr.Image(sources=["upload", "clipboard"],
                                         type="numpy", label="Image",
                                         show_label=False, height=240)
                        cam = gr.Image(sources=["webcam"], streaming=True,
                                       type="numpy", label="Camera",
                                       show_label=False, height=240,
                                       visible=False)
                        _html('<div class="ae-notice" role="alert">'
                              '<span class="ae-notice__text"></span></div>',
                              elem_id="ae-cam-notice")

                    with gr.Column(scale=3, min_width=320,
                                   elem_classes="ae-card"):
                        _html(_section("Detected relationships",
                                       "Sorted by confidence"))
                        results_note = _html(elem_classes="ae-collapsible")
                        table = gr.Dataframe(
                            headers=TABLE_HEADERS,
                            datatype=["str", "str", "str", "number"],
                            value=[], interactive=False, wrap=True,
                            column_widths=["27%", "25%", "26%", "22%"],
                            show_label=False, elem_classes="ae-table",
                            **component_kwargs(gr.Dataframe, max_height=320,
                                               buttons=[]))
                        with gr.Row(elem_classes="ae-actions"):
                            csv_btn = gr.Button("Export CSV", size="sm",
                                                variant="secondary")
                            json_btn = gr.Button("Export JSON", size="sm",
                                                 variant="secondary")

            # ---------------- side column: controls ----------------
            with gr.Column(scale=4, min_width=300, elem_classes="ae-col"):
                with gr.Column(elem_classes="ae-card"):
                    _html(_section("Detection vocabulary",
                                   "Type any words, separated by commas, "
                                   "then select Apply."))
                    classes_txt = gr.Textbox(
                        label="Objects to detect", lines=3, value="",
                        placeholder="e.g. forklift, pallet, safety vest",
                        info=("Leave empty to detect anything" if prompt_free
                              else "Leave empty to keep the current list"))
                    preds_txt = gr.Textbox(
                        label="Relationships to detect", lines=3,
                        value=", ".join(_default_predicates()))
                    apply_btn = gr.Button("Apply", variant="primary",
                                          elem_classes="ae-apply")
                    vocab_msg = _html(elem_classes="ae-collapsible")

                with gr.Column(elem_classes="ae-card"):
                    _html(_section("Display"))
                    conf = gr.Slider(0.05, 0.9, 0.25, step=0.05,
                                     label="Detection sensitivity",
                                     info="Minimum confidence for an object "
                                          "to be shown")
                    score_thr = gr.Slider(0.0, 0.95, 0.30, step=0.05,
                                          label="Relationship confidence",
                                          info="Minimum confidence for a "
                                               "relationship to be shown")
                    top_k = gr.Slider(1, 30, 12, step=1,
                                      label="Max relationships shown")

                with gr.Accordion("Advanced settings", open=False,
                                  elem_classes="ae-card ae-accordion"):
                    mode = gr.Radio(GRAPH_VIEWS, value="merged",
                                    label="Graph view",
                                    info="How relationships are grouped. "
                                         "See About for details.",
                                    elem_classes="ae-segmented")
                    show_masks = gr.Checkbox(True, label="Show object masks")
                    show_labels = gr.Checkbox(True, label="Show object labels")
                    spatial_raw = gr.Checkbox(
                        False, label="Strict spatial scoring",
                        info="Judges spatial relationships on geometry "
                             "alone. May show duplicate objects.")

                with gr.Accordion("About", open=False,
                                  elem_classes="ae-card ae-accordion"):
                    gr.Markdown(_about(runtime), elem_classes="ae-about")

        # ---------------- events ----------------
        knobs = [conf, top_k, score_thr, show_masks, show_labels, mode,
                 spatial_raw]
        outputs = [out, table, status, results_note]
        metrics = gr.State(None)
        outs = outputs + [metrics]

        # The still image being analyzed, kept in session state so knob
        # changes never depend on Gradio's temp file for the Image component.
        frame = gr.State(None)

        cam.stream(analyze, [cam] + knobs, outs, stream_every=0.12,
                   concurrency_limit=1, show_progress="hidden")
        # `input` fires on user upload/paste/clear only, not when load_sample
        # sets the value, so the sample is not analyzed twice.
        still.input(mark_processing, [metrics], [status], queue=False,
                    show_progress="hidden").then(
            analyze_input, [still] + knobs, [frame] + outs,
            show_progress="hidden")

        rerun = dict(fn=analyze_still, inputs=[source, frame] + knobs,
                     outputs=outs, show_progress="hidden")
        for slider in (conf, top_k, score_thr):
            slider.release(**rerun)
        for toggle in (show_masks, show_labels, mode, spatial_raw):
            toggle.input(**rerun)

        source.change(switch_source, [source], [still, cam, results_note],
                      js="(s) => window.aeSourceChanged(s)",
                      queue=False, show_progress="hidden").then(**rerun)

        apply_btn.click(_apply_busy, None, [apply_btn, vocab_msg],
                        queue=False, show_progress="hidden").then(
            apply_vocab, [classes_txt, preds_txt], vocab_msg,
            show_progress="hidden").then(
            _apply_idle, None, apply_btn, queue=False,
            show_progress="hidden").then(**rerun)

        csv_btn.click(None, [table], None,
                      js="(df) => window.aeExport('csv', df)")
        json_btn.click(None, [table], None,
                       js="(df) => window.aeExport('json', df)")

        # never open on an empty canvas: render the bundled sample
        demo.load(load_sample, knobs, [still, frame] + outs,
                  show_progress="hidden")
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=PipelineConfig.ckpt)
    ap.add_argument("--det", default="checkpoints/detectors/yoloe-11m-seg-pf.pt",
                    help="'-pf' = prompt-free YOLOE (works with no class list); "
                         "yoloe-11m-seg.pt = text-prompt (needs classes set)")
    ap.add_argument("--det_text", default="",
                    help="text-prompt YOLOE used when objects are typed on a "
                         "prompt-free --det (default: --det without '-pf')")
    ap.add_argument("--no_preload", action="store_true",
                    help="skip loading the text-prompt detector at startup")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_objects", type=int, default=16)
    ap.add_argument("--final_budget", type=int, default=64)
    ap.add_argument("--no_overlap", action="store_true",
                    help="disable detector‖backbone overlap (A/B the pipeline)")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    import torch
    dev = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    if dev != a.device:
        print("[demo] CUDA unavailable — falling back to CPU")

    global PIPE
    prompt_free = "-pf" in a.det
    cfg = PipelineConfig(ckpt=a.ckpt, det_weights=a.det, device=dev,
                         max_objects=a.max_objects,
                         final_budget=a.final_budget,
                         overlap=(dev == "cuda" and not a.no_overlap),
                         default_classes=None if prompt_free else DEFAULT_CLASSES,
                         det_text_weights=a.det_text)
    print(f"[demo] loading pipeline on {dev} …")
    PIPE = ParallelScenePipeline(cfg)
    note = (f"Running on {dev.upper()}"
            + (" with detector‖backbone overlap." if cfg.overlap else "."))
    print(f"[demo] ready — {note}")
    if prompt_free and not a.no_preload:
        # Typing objects switches to the text-prompt detector; fetch it and its
        # text encoder now so the first Apply does not wait on downloads.
        def _preload():
            try:
                PIPE.preload_text_detector()
                log.info("text-prompt detector ready")
            except Exception:
                log.exception("text-prompt detector preload failed; the first "
                              "Apply with objects will retry and report it")
        threading.Thread(target=_preload, daemon=True).start()
    runtime = {
        "device": dev.upper() + (", parallel detection and encoding"
                                 if cfg.overlap else ""),
        "detector": os.path.basename(a.det),
        "relation_model": os.path.basename(a.ckpt),
        "prompt_free": prompt_free,
    }
    # Errors are caught and shown as friendly in-page states; raw tracebacks
    # stay in the server log.
    build_ui(runtime).queue(max_size=4).launch(
        server_name="0.0.0.0", server_port=a.port, share=a.share,
        show_error=False, **launch_kwargs())


if __name__ == "__main__":
    main()
