"""
BoT-SORT Multi-Object Tracking Module — Layer 2 of HSUP Pipeline.

Wraps the Ultralytics built-in BoT-SORT / ByteTrack tracker to provide
persistent object tracking across video frames.  On top of raw track IDs
the module maintains per-object trajectories and exposes helpers for
direction estimation and speed calculation — both critical inputs for
wrong-side-driving and stop-line violation rules.

Usage::

    from core.multi_tracker import MultiTracker
    from core.entity_detector import Detection

    tracker = MultiTracker()
    detections = tracker.track(yolo_model, frame)
    speed = tracker.get_speed_estimate(track_id=5, fps=30.0)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config.settings import Settings, SETTINGS, TrackerConfig
from core.entity_detector import Detection

logger = logging.getLogger(__name__)


class MultiTracker:
    """BoT-SORT / ByteTrack multi-object tracker with trajectory analysis.

    The tracker delegates frame-level association to the Ultralytics
    ``model.track()`` API and internally accumulates centre-point
    trajectories keyed by track ID.

    Args:
        settings: Project-wide ``Settings`` instance.  Falls back to the
            global ``SETTINGS`` singleton when *None*.

    Example::

        tracker = MultiTracker()
        dets = tracker.track(model, frame)
        for d in dets:
            traj = tracker.get_trajectory(d.track_id)
            print(f"Track {d.track_id}: {len(traj)} points")
    """

    def __init__(self, settings: Settings = None) -> None:
        self._settings: Settings = settings or SETTINGS
        self._tracker_cfg: TrackerConfig = self._settings.tracker

        # track_id → list of (centre_x, centre_y, frame_number)
        self._trajectories: Dict[int, List[Tuple[float, float, int]]] = {}

        # Monotonically increasing frame counter (auto-incremented in
        # ``track()`` if the caller does not supply a frame number).
        self._frame_counter: int = 0

        logger.info(
            "MultiTracker initialised (type=%s, high=%.2f, low=%.2f, "
            "new=%.2f, buffer=%d, match=%.2f).",
            self._tracker_cfg.tracker_type,
            self._tracker_cfg.track_high_thresh,
            self._tracker_cfg.track_low_thresh,
            self._tracker_cfg.new_track_thresh,
            self._tracker_cfg.track_buffer,
            self._tracker_cfg.match_thresh,
        )

    # ──────────────────────────────────────────────────────
    # Primary tracking API
    # ──────────────────────────────────────────────────────

    def track(
        self,
        model,
        frame: np.ndarray,
        *,
        frame_num: Optional[int] = None,
    ) -> List[Detection]:
        """Run YOLO tracking on a single video frame.

        Calls ``model.track()`` with the BoT-SORT (or ByteTrack) tracker
        and converts results into ``Detection`` objects that carry
        ``track_id``.  Trajectories are updated automatically.

        Args:
            model: An Ultralytics ``YOLO`` model instance.
            frame: BGR numpy array of shape ``(H, W, C)``.
            frame_num: Optional explicit frame number.  If *None* an
                internal counter is used.

        Returns:
            List of ``Detection`` objects with ``track_id`` populated.
        """
        if frame is None or frame.size == 0:
            logger.warning("track() received empty frame.")
            return []

        if model is None:
            logger.warning("track() received None model — returning empty.")
            return []

        if frame_num is None:
            frame_num = self._frame_counter
        self._frame_counter = frame_num + 1

        cfg = self._settings.vehicle_detector

        try:
            results = model.track(
                source=frame,
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                imgsz=cfg.image_size,
                tracker=f"{self._tracker_cfg.tracker_type}.yaml",
                persist=True,
                verbose=False,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("model.track() failed: %s", exc)
            return []

        detections = self._parse_track_results(results)

        # Update internal trajectories
        self.update(detections, frame_num)

        logger.debug(
            "Frame %d: %d tracked detections, %d active trajectories.",
            frame_num,
            len(detections),
            len(self._trajectories),
        )
        return detections

    # ──────────────────────────────────────────────────────
    # Trajectory management
    # ──────────────────────────────────────────────────────

    def update(self, detections: List[Detection], frame_num: int) -> None:
        """Manually update trajectories from a list of detections.

        Only detections whose ``track_id`` is not *None* are recorded.

        Args:
            detections: Detections to incorporate.
            frame_num: The frame number these detections belong to.
        """
        for det in detections:
            if det.track_id is None:
                continue
            x1, y1, x2, y2 = det.bbox
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            self._trajectories.setdefault(det.track_id, []).append(
                (cx, cy, frame_num)
            )

    def get_trajectory(self, track_id: int) -> List[Tuple[float, float]]:
        """Return the centre-point trajectory for *track_id*.

        Args:
            track_id: The tracker-assigned object ID.

        Returns:
            List of ``(cx, cy)`` centre positions in chronological order.
            Returns an empty list if the track ID is unknown.
        """
        points = self._trajectories.get(track_id, [])
        return [(cx, cy) for cx, cy, _ in points]

    def get_direction_vector(
        self, track_id: int
    ) -> Optional[Tuple[float, float]]:
        """Compute a smoothed average direction vector for *track_id*.

        The direction is computed over the last *N* trajectory points
        where *N* = ``scene_graph.direction_smoothing_window`` from
        settings.  The returned vector is **unit-normalised**.

        Args:
            track_id: The tracker-assigned object ID.

        Returns:
            ``(dx, dy)`` unit vector, or ``None`` if the trajectory is
            too short (fewer than ``min_trajectory_length`` points).
        """
        sg_cfg = self._settings.scene_graph
        points = self._trajectories.get(track_id, [])

        if len(points) < sg_cfg.min_trajectory_length:
            return None

        window = sg_cfg.direction_smoothing_window
        # Use at least the last *window* segments
        recent = points[-max(window + 1, 2):]

        dx_sum = 0.0
        dy_sum = 0.0
        count = 0
        for i in range(1, len(recent)):
            dx_sum += recent[i][0] - recent[i - 1][0]
            dy_sum += recent[i][1] - recent[i - 1][1]
            count += 1

        if count == 0:
            return None

        dx_avg = dx_sum / count
        dy_avg = dy_sum / count
        magnitude = math.hypot(dx_avg, dy_avg)

        if magnitude < 1e-6:
            return None

        return (dx_avg / magnitude, dy_avg / magnitude)

    def get_speed_estimate(
        self,
        track_id: int,
        fps: float = 30.0,
        pixels_per_meter: float = 10.0,
    ) -> float:
        """Estimate object speed in **km/h** from its trajectory.

        Speed is derived from the average pixel displacement per frame,
        converted to metres via *pixels_per_meter* and then to km/h
        using the frame rate.

        Args:
            track_id: The tracker-assigned object ID.
            fps: Video frame rate (frames per second).
            pixels_per_meter: Calibration constant mapping pixels to
                real-world metres.

        Returns:
            Estimated speed in km/h.  Returns ``0.0`` when the
            trajectory is too short or inputs are invalid.
        """
        points = self._trajectories.get(track_id, [])
        if len(points) < 2:
            return 0.0

        if fps <= 0 or pixels_per_meter <= 0:
            logger.warning(
                "Invalid fps (%.2f) or pixels_per_meter (%.2f); "
                "returning 0 speed.",
                fps,
                pixels_per_meter,
            )
            return 0.0

        total_distance_px = 0.0
        total_frames = 0

        for i in range(1, len(points)):
            cx1, cy1, f1 = points[i - 1]
            cx2, cy2, f2 = points[i]
            dist = math.hypot(cx2 - cx1, cy2 - cy1)
            total_distance_px += dist
            total_frames += max(f2 - f1, 1)

        if total_frames == 0:
            return 0.0

        # pixels per frame → metres per second → km/h
        px_per_frame = total_distance_px / total_frames
        metres_per_second = (px_per_frame * fps) / pixels_per_meter
        kmph = metres_per_second * 3.6

        logger.debug(
            "Track %d speed: %.1f px/frame → %.1f m/s → %.1f km/h",
            track_id,
            px_per_frame,
            metres_per_second,
            kmph,
        )
        return kmph

    # ──────────────────────────────────────────────────────
    # State management
    # ──────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all tracking state and trajectories.

        Call this between video files or when the camera viewpoint
        changes to avoid stale trajectory data.
        """
        num_tracks = len(self._trajectories)
        self._trajectories.clear()
        self._frame_counter = 0
        logger.info("Tracker reset — cleared %d trajectories.", num_tracks)

    @property
    def active_track_ids(self) -> List[int]:
        """Return a list of all track IDs with stored trajectories.

        Returns:
            Sorted list of integer track IDs.
        """
        return sorted(self._trajectories.keys())

    @property
    def trajectory_count(self) -> int:
        """Number of active trajectories.

        Returns:
            Integer count of stored trajectories.
        """
        return len(self._trajectories)

    # ──────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────

    def _parse_track_results(self, results) -> List[Detection]:
        """Convert Ultralytics tracking results to ``Detection`` objects.

        Args:
            results: Raw results list returned by ``YOLO.track()``.

        Returns:
            List of ``Detection`` with ``track_id`` populated.
        """
        from config.settings import COCO_TO_ENTITY

        detections: List[Detection] = []

        try:
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1, y1, x2, y2 = (
                        int(xyxy[0]),
                        int(xyxy[1]),
                        int(xyxy[2]),
                        int(xyxy[3]),
                    )

                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    class_name = (
                        result.names.get(cls_id, f"class_{cls_id}")
                        if hasattr(result, "names") and result.names
                        else f"class_{cls_id}"
                    )

                    entity_class = COCO_TO_ENTITY.get(cls_id)

                    track_id: Optional[int] = None
                    if boxes.id is not None:
                        track_id = int(boxes.id[i].cpu().numpy())

                    detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=class_name,
                            entity_class=entity_class,
                            confidence=conf,
                            track_id=track_id,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            logger.error("Error parsing tracking results: %s", exc)

        return detections
