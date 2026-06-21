"""
Visualization utilities for TrafficSentinel AI.

Provides drawing functions for rendering detections, violation annotations,
scene graph overlays, evidence collages, and HUD elements onto images.
All functions operate on numpy arrays (BGR format, as used by OpenCV) and
return new annotated copies — originals are never mutated.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from config.settings import (
    EntityClass,
    ViolationType,
    VIOLATION_DISPLAY_NAMES,
)

if TYPE_CHECKING:
    from core.entity_detector import Detection
    from core.violation_engine import ViolationRecord
    from core.scene_graph import SceneGraph

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════
# COLOR SCHEME (BGR tuples for OpenCV)
# ══════════════════════════════════════════════════════════════

ENTITY_COLORS: Dict[EntityClass, Tuple[int, int, int]] = {
    # Vehicles — blue / teal family
    EntityClass.CAR: (200, 150, 50),
    EntityClass.MOTORCYCLE: (200, 100, 0),
    EntityClass.AUTO_RICKSHAW: (0, 180, 180),
    EntityClass.BUS: (180, 120, 40),
    EntityClass.TRUCK: (140, 100, 60),
    EntityClass.BICYCLE: (200, 200, 0),
    EntityClass.TEMPO: (160, 140, 80),
    # Road users — green family
    EntityClass.PERSON: (0, 220, 0),
    EntityClass.RIDER: (0, 200, 100),
    EntityClass.DRIVER: (50, 200, 50),
    EntityClass.PEDESTRIAN: (100, 255, 100),
    # Safety gear
    EntityClass.HELMET: (0, 255, 0),
    EntityClass.NO_HELMET: (0, 0, 255),
    EntityClass.SEATBELT: (0, 255, 0),
    EntityClass.NO_SEATBELT: (0, 0, 255),
    # Infrastructure — grey / cyan
    EntityClass.TRAFFIC_LIGHT_RED: (0, 0, 255),
    EntityClass.TRAFFIC_LIGHT_GREEN: (0, 255, 0),
    EntityClass.TRAFFIC_LIGHT_YELLOW: (0, 255, 255),
    EntityClass.STOP_LINE: (255, 255, 255),
    EntityClass.LANE_MARKING: (200, 200, 200),
    # License plate
    EntityClass.LICENSE_PLATE: (255, 180, 0),
}

VIOLATION_COLORS: Dict[ViolationType, Tuple[int, int, int]] = {
    ViolationType.HELMET_NON_COMPLIANCE: (0, 0, 220),
    ViolationType.SEATBELT_NON_COMPLIANCE: (0, 40, 220),
    ViolationType.TRIPLE_RIDING: (0, 80, 255),
    ViolationType.WRONG_SIDE_DRIVING: (20, 0, 200),
    ViolationType.STOP_LINE_VIOLATION: (0, 100, 255),
    ViolationType.RED_LIGHT_VIOLATION: (0, 0, 255),
    ViolationType.ILLEGAL_PARKING: (50, 50, 200),
    ViolationType.MISSING_LICENSE_PLATE: (0, 69, 255),
}

# Default colour when a key is missing from the above dictionaries.
_DEFAULT_ENTITY_COLOR: Tuple[int, int, int] = (180, 180, 180)
_DEFAULT_VIOLATION_COLOR: Tuple[int, int, int] = (0, 0, 255)

# Font settings
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE_LABEL = 0.50
_FONT_SCALE_BADGE = 0.55
_FONT_THICKNESS = 1
_BADGE_FONT_THICKNESS = 2


# ══════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def _clamp(val: int, lo: int, hi: int) -> int:
    """Clamp *val* to [lo, hi]."""
    return max(lo, min(val, hi))


def _overlay_rect(
    image: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    alpha: float = 0.35,
) -> np.ndarray:
    """Draw a semi-transparent filled rectangle on *image*.

    Args:
        image: Input BGR image (modified in-place for performance;
               callers should pass a copy if the original must stay clean).
        pt1: Top-left corner ``(x, y)``.
        pt2: Bottom-right corner ``(x, y)``.
        color: BGR colour tuple.
        alpha: Opacity of the overlay (0 = invisible, 1 = opaque).

    Returns:
        The image with the overlay applied.
    """
    overlay = image.copy()
    cv2.rectangle(overlay, pt1, pt2, color, cv2.FILLED)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, dst=image)
    return image


def _put_text_with_bg(
    image: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    font_scale: float = _FONT_SCALE_LABEL,
    color: Tuple[int, int, int] = (255, 255, 255),
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    thickness: int = _FONT_THICKNESS,
    alpha: float = 0.6,
) -> np.ndarray:
    """Render *text* at *origin* with a semi-transparent background pill.

    Args:
        image: BGR image.
        text: String to render.
        origin: Bottom-left point ``(x, y)`` of the text.
        font_scale: OpenCV font scale.
        color: Text colour (BGR).
        bg_color: Background pill colour (BGR).
        thickness: Font thickness.
        alpha: Background pill opacity.

    Returns:
        Image with the text rendered.
    """
    (tw, th), baseline = cv2.getTextSize(text, _FONT, font_scale, thickness)
    pad = 4
    x, y = origin
    pt1 = (x - pad, y - th - pad)
    pt2 = (x + tw + pad, y + baseline + pad)
    _overlay_rect(image, pt1, pt2, bg_color, alpha)
    cv2.putText(image, text, (x, y), _FONT, font_scale, color, thickness, cv2.LINE_AA)
    return image


# ══════════════════════════════════════════════════════════════
# CORE DRAWING FUNCTIONS
# ══════════════════════════════════════════════════════════════

def draw_detections(
    image: np.ndarray,
    detections: List["Detection"],
    show_labels: bool = True,
    show_confidence: bool = True,
) -> np.ndarray:
    """Draw bounding boxes and labels for every detection.

    Each detection is expected to expose at minimum:
    ``entity_class`` (:class:`EntityClass`), ``bbox`` (x1, y1, x2, y2),
    and ``confidence`` (float 0-1).

    Args:
        image: BGR image to annotate.
        detections: Sequence of ``Detection`` objects.
        show_labels: If ``True``, render the class label above the box.
        show_confidence: If ``True``, append the confidence score to the
            label text.

    Returns:
        A new annotated copy of the image.
    """
    canvas = image.copy()
    if not detections:
        return canvas

    try:
        for det in detections:
            entity_cls: EntityClass = det.entity_class
            color = ENTITY_COLORS.get(entity_cls, _DEFAULT_ENTITY_COLOR)
            x1, y1, x2, y2 = (int(c) for c in det.bbox)
            h_img, w_img = canvas.shape[:2]
            x1, y1 = _clamp(x1, 0, w_img), _clamp(y1, 0, h_img)
            x2, y2 = _clamp(x2, 0, w_img), _clamp(y2, 0, h_img)

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Label
            if show_labels:
                label = entity_cls.value if entity_cls is not None else det.class_name
                if show_confidence:
                    label = f"{label} {det.confidence:.0%}"
                _put_text_with_bg(
                    canvas,
                    label,
                    (x1, y1 - 4),
                    color=(255, 255, 255),
                    bg_color=color,
                )
    except Exception:
        logger.exception("Error drawing detections — returning partially annotated image")

    return canvas


def draw_violations(
    image: np.ndarray,
    violations: List["ViolationRecord"],
    detections: Optional[List["Detection"]] = None,
) -> np.ndarray:
    """Draw thick violation bounding boxes with type labels.

    Violation boxes are drawn with a bold red/orange border and a
    semi-transparent fill to make them stand out.  If *detections* are
    supplied they are drawn first as a softer underlay.

    Args:
        image: BGR image.
        violations: Violation records to annotate.  Each is expected to
            expose ``violation_type`` (:class:`ViolationType`),
            ``bbox`` (x1, y1, x2, y2), and ``confidence`` (float).
        detections: Optional list of base detections drawn as context.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()

    # Draw base detections first (muted)
    if detections:
        canvas = draw_detections(canvas, detections, show_labels=True, show_confidence=False)

    if not violations:
        return canvas

    try:
        for viol in violations:
            v_type: ViolationType = viol.violation_type
            color = VIOLATION_COLORS.get(v_type, _DEFAULT_VIOLATION_COLOR)
            x1, y1, x2, y2 = (int(c) for c in viol.bbox)
            h_img, w_img = canvas.shape[:2]
            x1, y1 = _clamp(x1, 0, w_img), _clamp(y1, 0, h_img)
            x2, y2 = _clamp(x2, 0, w_img), _clamp(y2, 0, h_img)

            # Semi-transparent fill
            _overlay_rect(canvas, (x1, y1), (x2, y2), color, alpha=0.20)

            # Thick border
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)

            # Violation label
            display_name = VIOLATION_DISPLAY_NAMES.get(v_type, v_type.value)
            label = f"[!] {display_name} ({viol.confidence:.0%})"
            _put_text_with_bg(
                canvas,
                label,
                (x1, y1 - 6),
                font_scale=_FONT_SCALE_BADGE,
                color=(255, 255, 255),
                bg_color=color,
                thickness=_BADGE_FONT_THICKNESS,
                alpha=0.75,
            )
    except Exception:
        logger.exception("Error drawing violations — returning partially annotated image")

    return canvas


def draw_scene_graph_overlay(
    image: np.ndarray,
    scene_graph: "SceneGraph",
) -> np.ndarray:
    """Render a scene graph as coloured node boxes connected by edges.

    Nodes are drawn as labelled coloured rectangles at the centre of their
    bounding boxes.  Edges are rendered as lines between node centres with
    relation labels at their midpoints.

    Args:
        image: BGR image.
        scene_graph: ``SceneGraph`` object containing ``nodes`` and
            ``edges`` attributes.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()

    try:
        node_centres: Dict[str, Tuple[int, int]] = {}

        # ── Draw nodes ──────────────────────────────────────────
        for node in scene_graph.nodes:
            entity_cls: EntityClass = node.entity_class
            color = ENTITY_COLORS.get(entity_cls, _DEFAULT_ENTITY_COLOR)
            x1, y1, x2, y2 = (int(c) for c in node.bbox)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            node_centres[node.node_id] = (cx, cy)

            # Semi-transparent filled box
            _overlay_rect(canvas, (x1, y1), (x2, y2), color, alpha=0.25)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            # Node label
            label = f"{node.node_id}: {entity_cls.value}"
            _put_text_with_bg(canvas, label, (x1, y1 - 4), bg_color=color)

        # ── Draw edges ──────────────────────────────────────────
        for edge in scene_graph.edges:
            src = node_centres.get(edge.source_id)
            dst = node_centres.get(edge.target_id)
            if src is None or dst is None:
                continue

            # Edge line
            cv2.line(canvas, src, dst, (0, 255, 255), 2, cv2.LINE_AA)

            # Relation label at midpoint
            mx = (src[0] + dst[0]) // 2
            my = (src[1] + dst[1]) // 2
            _put_text_with_bg(
                canvas,
                edge.relation,
                (mx, my),
                font_scale=0.40,
                color=(0, 255, 255),
                bg_color=(40, 40, 40),
            )
    except Exception:
        logger.exception("Error drawing scene graph overlay")

    return canvas


# ══════════════════════════════════════════════════════════════
# EVIDENCE & HUD UTILITIES
# ══════════════════════════════════════════════════════════════

def create_evidence_collage(
    original: np.ndarray,
    annotated: np.ndarray,
    vehicle_crop: np.ndarray,
    plate_crop: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build a 2×2 evidence collage from the provided images.

    Layout::

        ┌─────────────┬─────────────┐
        │  Original   │  Annotated  │
        ├─────────────┬─────────────┤
        │ Vehicle Crop│  Plate Crop │
        └─────────────┴─────────────┘

    All four panels are resized to uniform dimensions (half the target
    width × half the target height) so the collage is always square-ish.
    If *plate_crop* is ``None`` a dark placeholder is used.

    Args:
        original: Original captured frame (BGR).
        annotated: Frame with annotations drawn.
        vehicle_crop: Cropped vehicle region.
        plate_crop: Optional cropped license plate region.

    Returns:
        A single BGR image containing the 2×2 collage.
    """
    try:
        cell_w, cell_h = 400, 300

        def _resize_cell(img: np.ndarray) -> np.ndarray:
            """Resize an image to the fixed cell dimensions."""
            if img is None or img.size == 0:
                return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            return cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)

        top_left = _resize_cell(original)
        top_right = _resize_cell(annotated)
        bot_left = _resize_cell(vehicle_crop)

        if plate_crop is not None and plate_crop.size > 0:
            bot_right = _resize_cell(plate_crop)
        else:
            bot_right = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            _put_text_with_bg(
                bot_right,
                "Plate N/A",
                (cell_w // 2 - 40, cell_h // 2),
                color=(120, 120, 120),
                bg_color=(30, 30, 30),
            )

        # Add panel captions
        for panel, caption in [
            (top_left, "Original"),
            (top_right, "Annotated"),
            (bot_left, "Vehicle"),
            (bot_right, "Plate"),
        ]:
            _put_text_with_bg(panel, caption, (8, 22), font_scale=0.45, alpha=0.50)

        top_row = np.hstack([top_left, top_right])
        bot_row = np.hstack([bot_left, bot_right])
        collage = np.vstack([top_row, bot_row])

        return collage

    except Exception:
        logger.exception("Error creating evidence collage — returning blank image")
        return np.zeros((600, 800, 3), dtype=np.uint8)


def draw_violation_badge(
    image: np.ndarray,
    violation_type: ViolationType,
    position: Tuple[int, int],
) -> np.ndarray:
    """Draw a coloured violation badge / tag at the given position.

    The badge is a rounded-look rectangle with the violation name in
    bold white text.

    Args:
        image: BGR image to annotate.
        violation_type: Violation enum value.
        position: Top-left ``(x, y)`` of the badge.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()
    try:
        display_name = VIOLATION_DISPLAY_NAMES.get(violation_type, violation_type.value)
        color = VIOLATION_COLORS.get(violation_type, _DEFAULT_VIOLATION_COLOR)

        (tw, th), baseline = cv2.getTextSize(
            display_name, _FONT, _FONT_SCALE_BADGE, _BADGE_FONT_THICKNESS
        )
        pad_x, pad_y = 10, 6
        x, y = position
        x2 = x + tw + 2 * pad_x
        y2 = y + th + 2 * pad_y + baseline

        # Rounded-look badge (filled rect + small circles at corners)
        radius = 8
        cv2.rectangle(canvas, (x + radius, y), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(canvas, (x, y + radius), (x2, y2 - radius), color, cv2.FILLED)
        cv2.circle(canvas, (x + radius, y + radius), radius, color, cv2.FILLED)
        cv2.circle(canvas, (x2 - radius, y + radius), radius, color, cv2.FILLED)
        cv2.circle(canvas, (x + radius, y2 - radius), radius, color, cv2.FILLED)
        cv2.circle(canvas, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)

        # Text
        text_x = x + pad_x
        text_y = y + pad_y + th
        cv2.putText(
            canvas,
            display_name,
            (text_x, text_y),
            _FONT,
            _FONT_SCALE_BADGE,
            (255, 255, 255),
            _BADGE_FONT_THICKNESS,
            cv2.LINE_AA,
        )
    except Exception:
        logger.exception("Error drawing violation badge")

    return canvas


def add_timestamp_watermark(
    image: np.ndarray,
    timestamp: str,
    camera_id: str = "CAM-01",
) -> np.ndarray:
    """Add a timestamp and camera ID watermark to the bottom of the image.

    Renders a semi-transparent dark bar across the bottom with the
    timestamp on the left and the camera ID on the right.

    Args:
        image: BGR image.
        timestamp: Human-readable timestamp string (e.g. ``2026-06-16 12:30:45``).
        camera_id: Camera identifier displayed on the right.

    Returns:
        Annotated copy of the image.
    """
    canvas = image.copy()
    try:
        h, w = canvas.shape[:2]
        bar_height = 32
        bar_y = h - bar_height

        # Semi-transparent dark bar
        _overlay_rect(canvas, (0, bar_y), (w, h), (0, 0, 0), alpha=0.55)

        # Timestamp — bottom-left
        ts_text = f"  {timestamp}"
        cv2.putText(
            canvas, ts_text, (4, h - 10),
            _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )

        # Camera ID — bottom-right
        (cw, _), _ = cv2.getTextSize(camera_id, _FONT, 0.45, 1)
        cv2.putText(
            canvas, camera_id, (w - cw - 8, h - 10),
            _FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )
    except Exception:
        logger.exception("Error adding timestamp watermark")

    return canvas


def resize_for_display(
    image: np.ndarray,
    max_width: int = 800,
) -> np.ndarray:
    """Resize an image for display while preserving the aspect ratio.

    If the image width is already ≤ *max_width* it is returned unchanged.

    Args:
        image: BGR image.
        max_width: Maximum width in pixels.

    Returns:
        Resized (or original) image.
    """
    try:
        h, w = image.shape[:2]
        if w <= max_width:
            return image
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    except Exception:
        logger.exception("Error resizing image for display")
        return image
