"""
Violation Engine — Layer 4 of the HSUP Pipeline.

Queries the :class:`SceneGraph` to detect traffic violations by reasoning
over the relational structure of detected entities.  Each violation type
has a dedicated detection method that encodes Indian traffic-law logic.

Supported Violations
────────────────────
- **Helmet non-compliance** — rider on two-wheeler without helmet.
- **Triple riding** — three or more persons on a single two-wheeler.
- **Seatbelt non-compliance** — driver/passenger without seatbelt.
- **Wrong-side driving** — vehicle moving against expected traffic flow.
- **Stop-line violation** — vehicle crossing stop line while signal is red.
- **Red-light violation** — vehicle proceeding through red signal.
- **Illegal parking** — vehicle stationary in a no-parking zone.

Architecture
────────────
    SceneGraph (Layer 3)
          │
          ▼
    ViolationEngine.analyze()
      ├── _check_helmet_violations()
      ├── _check_triple_riding()
      ├── _check_seatbelt_violations()
      ├── _check_wrong_side()
      ├── _check_stop_line()
      ├── _check_red_light()
      └── _check_illegal_parking()
          │
          ▼
    List[ViolationRecord]  →  Evidence Generator (Layer 5)
"""

from __future__ import annotations

import datetime
import logging
import threading
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config.settings import (
    EntityClass,
    Settings,
    SETTINGS,
    TWO_WHEELER_CLASSES,
    FOUR_WHEELER_CLASSES,
    ViolationType,
    VIOLATION_DISPLAY_NAMES,
)
from core.scene_graph import (
    DRIVES,
    HAS_PLATE,
    NOT_WEARS,
    RIDES,
    WEARS,
    NEAR_SIGNAL,
    CROSSED_LINE,
    SceneEdge,
    SceneGraph,
    SceneNode,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# VIOLATION RECORD
# ══════════════════════════════════════════════════════════════

@dataclass
class ViolationRecord:
    """
    Immutable record of a single detected traffic violation.

    Produced by :meth:`ViolationEngine.analyze` and consumed downstream by
    the evidence generator (Layer 5) and the dashboard.

    Attributes:
        violation_id:   Unique identifier (e.g. ``"VIO-20260616-001"``).
        violation_type: Semantic type from :class:`ViolationType`.
        confidence:     Aggregate confidence ``[0, 1]``.
        involved_nodes: ``node_id`` values of all participating entities.
        description:    Human-readable violation description.
        bbox:           Union bounding box ``(x1, y1, x2, y2)`` that
                        encompasses all involved entities.
        timestamp:      ISO-8601 timestamp of detection.
        metadata:       Free-form extras — plate text, fine amount, section
                        of law, etc.
    """

    violation_id: str
    violation_type: ViolationType
    confidence: float
    involved_nodes: List[str]
    description: str
    bbox: Tuple[int, int, int, int]
    timestamp: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)


# ══════════════════════════════════════════════════════════════
# VIOLATION ENGINE
# ══════════════════════════════════════════════════════════════

class ViolationEngine:
    """
    Rule-based violation reasoning engine.

    Inspects a :class:`SceneGraph` and emits :class:`ViolationRecord`
    instances for every detected infraction.  Designed to be called
    once per frame.

    Usage::

        engine = ViolationEngine()
        violations = engine.analyze(scene_graph, frame=current_frame)
        for v in violations:
            print(v.violation_type, v.description)
    """

    _global_counter: int = -1
    _global_counter_lock = threading.Lock()
    _global_date_str: str = ""

    def __init__(self, settings: Settings | None = None) -> None:
        """
        Initialise the violation engine.

        Args:
            settings: Global settings instance.  Falls back to
                      :data:`SETTINGS` if ``None``.
        """
        self._settings: Settings = settings or SETTINGS
        today = datetime.date.today().strftime("%Y%m%d")

        # Check if running within a Streamlit session
        self._use_session_state = False
        try:
            import streamlit as st
            # Confirm we are in a Streamlit thread/execution context
            if hasattr(st, "session_state"):
                self._use_session_state = True
        except Exception:
            pass

        if self._use_session_state:
            import streamlit as st
            if "global_violation_counter" not in st.session_state or st.session_state.get("global_violation_date") != today:
                max_seq = 0
                try:
                    import re
                    evidence_dir = self._settings.evidence_dir
                    if evidence_dir.exists():
                        pattern = re.compile(rf"^VIO-{today}-(\d+)$")
                        for path in evidence_dir.iterdir():
                            if path.is_dir():
                                match = pattern.match(path.name)
                                if match:
                                    seq_val = int(match.group(1))
                                    if seq_val > max_seq:
                                        max_seq = seq_val
                except Exception as e:
                    logger.warning("Failed to scan evidence directory: %s", e)
                st.session_state["global_violation_counter"] = max_seq
                st.session_state["global_violation_date"] = today
        else:
            with ViolationEngine._global_counter_lock:
                # If the class-level counter is not initialized or the date changed:
                if ViolationEngine._global_counter == -1 or ViolationEngine._global_date_str != today:
                    ViolationEngine._global_date_str = today
                    max_seq = 0
                    try:
                        import re
                        evidence_dir = self._settings.evidence_dir
                        if evidence_dir.exists():
                            pattern = re.compile(rf"^VIO-{today}-(\d+)$")
                            for path in evidence_dir.iterdir():
                                if path.is_dir():
                                    match = pattern.match(path.name)
                                    if match:
                                        seq_val = int(match.group(1))
                                        if seq_val > max_seq:
                                            max_seq = seq_val
                    except Exception as e:
                        logger.warning("Failed to scan evidence directory: %s", e)
                    ViolationEngine._global_counter = max_seq

        # Load all registered violation detectors explicitly to handle Streamlit caching/reloading issues
        from violations.helmet import HelmetViolationDetector
        from violations.seatbelt import SeatbeltViolationDetector
        from violations.triple_riding import TripleRidingViolationDetector
        from violations.wrong_side import WrongSideViolationDetector
        from violations.stop_line import StopLineViolationDetector
        from violations.red_light import RedLightViolationDetector
        from violations.illegal_parking import IllegalParkingViolationDetector
        from violations.missing_plate import MissingPlateViolationDetector

        detector_map = {
            ViolationType.HELMET_NON_COMPLIANCE: HelmetViolationDetector,
            ViolationType.SEATBELT_NON_COMPLIANCE: SeatbeltViolationDetector,
            ViolationType.TRIPLE_RIDING: TripleRidingViolationDetector,
            ViolationType.WRONG_SIDE_DRIVING: WrongSideViolationDetector,
            ViolationType.STOP_LINE_VIOLATION: StopLineViolationDetector,
            ViolationType.RED_LIGHT_VIOLATION: RedLightViolationDetector,
            ViolationType.ILLEGAL_PARKING: IllegalParkingViolationDetector,
            ViolationType.MISSING_LICENSE_PLATE: MissingPlateViolationDetector,
        }

        self._detectors = {}
        for vtype, detector_cls in detector_map.items():
            self._detectors[vtype] = detector_cls(
                enabled=True,
                min_confidence=self._settings.min_violation_confidence
            )

        logger.info(
            "ViolationEngine initialised with %d detectors: %s",
            len(self._detectors),
            [d.__class__.__name__ for d in self._detectors.values()]
        )

    # ──────────────────────────────────────────────────────────
    # PUBLIC API
    # ──────────────────────────────────────────────────────────

    def analyze(
        self,
        scene_graph: SceneGraph,
        frame: np.ndarray | None = None,
    ) -> List[ViolationRecord]:
        """
        Run all registered violation detectors on a scene graph.

        Args:
            scene_graph: A fully-built :class:`SceneGraph` for the current
                         frame.
            frame:       Optional raw frame.

        Returns:
            List of :class:`ViolationRecord` instances.
        """
        violations: List[ViolationRecord] = []

        for vtype, detector in self._detectors.items():
            try:
                results = detector.run(scene_graph, frame)
                for vr in results:
                    # Standardise violation ID
                    vr.violation_id = self._generate_violation_id()
                    violations.append(vr)
            except Exception:
                logger.error(
                    "Detector %s failed during run().",
                    detector.__class__.__name__,
                    exc_info=True
                )

        # De-duplicate: remove duplicate violations for the same vehicle
        seen = set()
        unique_violations = []
        for v in violations:
            # Create a key from violation type + primary vehicle node
            vehicle_nodes = [n for n in v.involved_nodes if any(vt in n for vt in ['car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck', 'tempo'])]
            key = (v.violation_type.value, tuple(sorted(vehicle_nodes)))
            if key not in seen:
                # Filter out background/small vehicle violations
                img_h, img_w = None, None
                if frame is not None:
                    img_h, img_w = frame.shape[:2]
                elif getattr(scene_graph, "frame_shape", None) is not None:
                    img_h, img_w = scene_graph.frame_shape[:2]

                if img_h is not None and img_w is not None:
                    total_area = img_h * img_w
                    
                    # Find the primary vehicle involved in this violation
                    primary_veh_node = None
                    for node_id in v.involved_nodes:
                        node = scene_graph.nodes.get(node_id)
                        if node and node.entity_class.value in ['car', 'motorcycle', 'auto_rickshaw', 'bus', 'truck', 'tempo']:
                            primary_veh_node = node
                            break
                    
                    if primary_veh_node is not None:
                        vx1, vy1, vx2, vy2 = primary_veh_node.bbox
                        v_area = (vx2 - vx1) * (vy2 - vy1)
                        # Filter if vehicle area is less than configured threshold (focus on near objects)
                        if (v_area / total_area) < self._settings.min_vehicle_area_ratio:
                            logger.info(
                                "Filtering out background vehicle violation: %s (area ratio: %.4f < %.4f)",
                                primary_veh_node.node_id, v_area / total_area, self._settings.min_vehicle_area_ratio
                            )
                            continue
                seen.add(key)
                unique_violations.append(v)
            else:
                logger.debug("De-duplicated violation: %s for %s", v.violation_type.value, vehicle_nodes)
        violations = unique_violations

        logger.info(
            "ViolationEngine found %d violation(s) in current frame.",
            len(violations),
        )
        return violations


    # ──────────────────────────────────────────────────────────
    # VIOLATION CHECKS
    # ──────────────────────────────────────────────────────────

    def _check_helmet_violations(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect helmet non-compliance on two-wheelers.

        Logic:
        1. Find all ``RIDES`` edges linking a person to a two-wheeler.
        2. For each rider, query ``get_helmet_status()``.
        3. If status is ``"no_helmet"`` → emit a violation.
        4. If status is ``None`` (no helmet detection at all) → skip
           (ambiguous — the detector may not have picked it up).

        Returns:
            List of helmet-violation records.
        """
        violations: List[ViolationRecord] = []

        # Query all person → two-wheeler relationships
        for ec in TWO_WHEELER_CLASSES:
            triplets = sg.query(
                source_class=None, relation=RIDES, target_class=ec
            )
            for rider_node, edge, vehicle_node in triplets:
                status = sg.get_helmet_status(rider_node.node_id)
                if status != "no_helmet":
                    continue

                # Build violation record
                involved = [rider_node.node_id, vehicle_node.node_id]
                plate_node = sg.get_plate_of(vehicle_node.node_id)
                plate_text = ""
                if plate_node is not None:
                    involved.append(plate_node.node_id)
                    plate_text = plate_node.attributes.get("plate_text", "")

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    ViolationType.HELMET_NON_COMPLIANCE,
                    "Helmet Non-Compliance",
                )

                violations.append(
                    ViolationRecord(
                        violation_id=self._generate_violation_id(),
                        violation_type=ViolationType.HELMET_NON_COMPLIANCE,
                        confidence=min(
                            rider_node.confidence, vehicle_node.confidence
                        ),
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: Rider on {vehicle_node.entity_class.value} "
                            f"without helmet."
                        ),
                        bbox=self._merge_bbox(
                            [rider_node, vehicle_node]
                        ),
                        timestamp=self._now_iso(),
                        metadata={
                            "plate_text": plate_text,
                            "vehicle_type": vehicle_node.entity_class.value,
                            "rider_node": rider_node.node_id,
                        },
                    )
                )

        return violations

    def _check_triple_riding(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect three or more persons on a single two-wheeler.

        Logic:
        1. For each two-wheeler node, call ``get_passenger_count()``.
        2. If count ≥ ``triple_riding_min_persons`` → violation.

        Returns:
            List of triple-riding violation records.
        """
        violations: List[ViolationRecord] = []
        min_persons = self._settings.triple_riding_min_persons

        all_two_wheelers = [
            n
            for n in sg.nodes.values()
            if n.entity_class in TWO_WHEELER_CLASSES
        ]

        for vehicle in all_two_wheelers:
            count = sg.get_passenger_count(vehicle.node_id)
            if count < min_persons:
                continue

            riders = sg.get_riders_of(vehicle.node_id)
            involved = [vehicle.node_id] + [r.node_id for r in riders]

            plate_node = sg.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                ViolationType.TRIPLE_RIDING, "Triple Riding"
            )

            all_involved_nodes = [vehicle] + riders
            violations.append(
                ViolationRecord(
                    violation_id=self._generate_violation_id(),
                    violation_type=ViolationType.TRIPLE_RIDING,
                    confidence=min(
                        n.confidence for n in all_involved_nodes
                    ),
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {count} persons detected on "
                        f"{vehicle.entity_class.value}."
                    ),
                    bbox=self._merge_bbox(all_involved_nodes),
                    timestamp=self._now_iso(),
                    metadata={
                        "passenger_count": count,
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                    },
                )
            )

        return violations

    def _check_seatbelt_violations(
        self,
        sg: SceneGraph,
        frame: np.ndarray | None = None,
    ) -> List[ViolationRecord]:
        """
        Detect seatbelt non-compliance in four-wheelers.

        Logic:
        1. Find all ``DRIVES`` edges linking a person to a four-wheeler.
        2. Check for ``NOT_WEARS`` edges targeting ``NO_SEATBELT`` on that
           person.
        3. If the frame is available, a future enhancement can crop the
           windshield region for CNN-based seatbelt verification.

        Returns:
            List of seatbelt-violation records.
        """
        violations: List[ViolationRecord] = []

        for ec in FOUR_WHEELER_CLASSES:
            triplets = sg.query(
                source_class=None, relation=DRIVES, target_class=ec
            )
            for driver_node, edge, vehicle_node in triplets:
                # Check for explicit no-seatbelt detection
                has_no_seatbelt = any(
                    e.source_id == driver_node.node_id
                    and e.relation == NOT_WEARS
                    and sg.nodes.get(e.target_id, SceneNode(
                        node_id="", entity_class=EntityClass.PERSON,
                        bbox=(0, 0, 0, 0), confidence=0
                    )).entity_class == EntityClass.NO_SEATBELT
                    for e in sg.edges
                )

                if not has_no_seatbelt:
                    continue

                involved = [driver_node.node_id, vehicle_node.node_id]
                plate_node = sg.get_plate_of(vehicle_node.node_id)
                plate_text = ""
                if plate_node is not None:
                    involved.append(plate_node.node_id)
                    plate_text = plate_node.attributes.get("plate_text", "")

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    ViolationType.SEATBELT_NON_COMPLIANCE,
                    "Seatbelt Non-Compliance",
                )

                violations.append(
                    ViolationRecord(
                        violation_id=self._generate_violation_id(),
                        violation_type=ViolationType.SEATBELT_NON_COMPLIANCE,
                        confidence=min(
                            driver_node.confidence, vehicle_node.confidence
                        ),
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: Driver in {vehicle_node.entity_class.value} "
                            f"without seatbelt."
                        ),
                        bbox=self._merge_bbox([driver_node, vehicle_node]),
                        timestamp=self._now_iso(),
                        metadata={
                            "plate_text": plate_text,
                            "vehicle_type": vehicle_node.entity_class.value,
                            "frame_available": frame is not None,
                        },
                    )
                )

        return violations

    def _check_wrong_side(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect wrong-side driving.

        Logic:
        1. For each vehicle with a direction attribute, check if the
           direction matches the expected traffic flow for its lane.
        2. Indian traffic drives on the LEFT.  In a typical camera view:
           - Vehicles in left lanes should move in one direction (e.g. SOUTH).
           - Vehicles in right lanes should move the opposite (e.g. NORTH).
        3. If a vehicle moves in the *opposite* expected direction for its
           lane → violation.

        This is a heuristic — production deployment would calibrate the
        expected direction per camera installation.

        Returns:
            List of wrong-side violation records.
        """
        violations: List[ViolationRecord] = []

        # Default expected directions by lane half (left / right of frame)
        # This should be configured per-camera in production.
        expected_left_half = {"SOUTH", "SOUTHWEST", "SOUTHEAST"}
        expected_right_half = {"NORTH", "NORTHWEST", "NORTHEAST"}

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for node in sg.nodes.values():
            if node.entity_class not in all_vehicle_classes:
                continue

            direction = sg.get_direction(node.node_id)
            if direction is None or direction == "STATIONARY":
                continue

            lane_index = node.attributes.get("lane_index")
            if lane_index is None:
                continue

            # Determine which half of the road
            is_left_half = lane_index < 2  # lanes 0, 1 = left half

            if is_left_half and direction in expected_right_half:
                wrong_side = True
            elif not is_left_half and direction in expected_left_half:
                wrong_side = True
            else:
                wrong_side = False

            if not wrong_side:
                continue

            involved = [node.node_id]
            plate_node = sg.get_plate_of(node.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                ViolationType.WRONG_SIDE_DRIVING, "Wrong-Side Driving"
            )

            violations.append(
                ViolationRecord(
                    violation_id=self._generate_violation_id(),
                    violation_type=ViolationType.WRONG_SIDE_DRIVING,
                    confidence=node.confidence * 0.8,  # penalise heuristic
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {node.entity_class.value} moving "
                        f"{direction} in lane {lane_index} (expected opposite flow)."
                    ),
                    bbox=node.bbox,
                    timestamp=self._now_iso(),
                    metadata={
                        "direction": direction,
                        "lane_index": lane_index,
                        "plate_text": plate_text,
                        "vehicle_type": node.entity_class.value,
                    },
                )
            )

        return violations

    def _check_stop_line(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect vehicles crossing a stop line while the signal is red.

        Logic:
        1. Find all ``STOP_LINE`` nodes in the graph.
        2. Find all ``TRAFFIC_LIGHT_RED`` nodes in the graph.
        3. For each vehicle, check if it overlaps with / has crossed past
           the stop-line bbox while a red signal is visible.

        Returns:
            List of stop-line violation records.
        """
        violations: List[ViolationRecord] = []

        stop_lines = [
            n for n in sg.nodes.values()
            if n.entity_class == EntityClass.STOP_LINE
        ]
        red_lights = [
            n for n in sg.nodes.values()
            if n.entity_class == EntityClass.TRAFFIC_LIGHT_RED
        ]

        # No stop line or no red light → no violation possible
        if not stop_lines or not red_lights:
            return violations

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for vehicle in sg.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            for stop_line in stop_lines:
                # Vehicle has crossed stop line if its bottom edge is
                # below the stop line's y-centre (vehicle went past it)
                vehicle_bottom = vehicle.bbox[3]
                stop_line_y = (stop_line.bbox[1] + stop_line.bbox[3]) / 2.0

                # Check horizontal overlap (vehicle is in the same lane)
                v_cx = (vehicle.bbox[0] + vehicle.bbox[2]) / 2.0
                sl_x1 = stop_line.bbox[0]
                sl_x2 = stop_line.bbox[2]

                if not (sl_x1 <= v_cx <= sl_x2):
                    continue

                if vehicle_bottom <= stop_line_y:
                    continue  # vehicle has NOT crossed

                # Vehicle has crossed the stop line while signal is red
                involved = [vehicle.node_id, stop_line.node_id]
                plate_node = sg.get_plate_of(vehicle.node_id)
                plate_text = ""
                if plate_node is not None:
                    involved.append(plate_node.node_id)
                    plate_text = plate_node.attributes.get("plate_text", "")

                # Include the nearest red light
                if red_lights:
                    involved.append(red_lights[0].node_id)

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    ViolationType.STOP_LINE_VIOLATION,
                    "Stop-Line Violation",
                )

                violations.append(
                    ViolationRecord(
                        violation_id=self._generate_violation_id(),
                        violation_type=ViolationType.STOP_LINE_VIOLATION,
                        confidence=min(
                            vehicle.confidence, stop_line.confidence
                        ),
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: {vehicle.entity_class.value} "
                            f"crossed stop line during red signal."
                        ),
                        bbox=self._merge_bbox(
                            [vehicle, stop_line]
                        ),
                        timestamp=self._now_iso(),
                        metadata={
                            "plate_text": plate_text,
                            "vehicle_type": vehicle.entity_class.value,
                            "signal_state": "red",
                        },
                    )
                )
                break  # one violation per vehicle per frame

        return violations

    def _check_red_light(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect vehicles running a red light.

        This differs from stop-line violation in that it focuses on vehicles
        that are *actively proceeding through* an intersection during a red
        signal (i.e. moving past the stop line area).

        Logic:
        1. Find red-light signals.
        2. Find vehicles with direction data that are **beyond** the stop-line
           zone and still moving.
        3. A vehicle that was already past the stop line before the light
           turned red is not a violator (would need temporal state —
           simplified here).

        Returns:
            List of red-light violation records.
        """
        violations: List[ViolationRecord] = []

        red_lights = [
            n for n in sg.nodes.values()
            if n.entity_class == EntityClass.TRAFFIC_LIGHT_RED
        ]
        stop_lines = [
            n for n in sg.nodes.values()
            if n.entity_class == EntityClass.STOP_LINE
        ]

        if not red_lights:
            return violations

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for vehicle in sg.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            direction = sg.get_direction(vehicle.node_id)
            if direction is None or direction == "STATIONARY":
                continue  # not moving → not running a red light

            # Check if vehicle is in the intersection zone (past stop lines)
            crossed_stop = False
            for sl in stop_lines:
                vehicle_bottom = vehicle.bbox[3]
                sl_y = (sl.bbox[1] + sl.bbox[3]) / 2.0

                v_cx = (vehicle.bbox[0] + vehicle.bbox[2]) / 2.0
                if sl.bbox[0] <= v_cx <= sl.bbox[2] and vehicle_bottom > sl_y:
                    crossed_stop = True
                    break

            if not crossed_stop and stop_lines:
                continue  # vehicle hasn't crossed into intersection

            # If no stop lines are detected, use proximity to red light
            if not stop_lines:
                from core.scene_graph import _bbox_center_distance

                min_dist = min(
                    _bbox_center_distance(vehicle.bbox, rl.bbox)
                    for rl in red_lights
                )
                # Only flag if vehicle is reasonably close to the signal
                frame_diag = max(vehicle.bbox[2], vehicle.bbox[3]) * 5
                if min_dist > frame_diag:
                    continue

            involved = [vehicle.node_id, red_lights[0].node_id]
            plate_node = sg.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                ViolationType.RED_LIGHT_VIOLATION, "Red-Light Violation"
            )

            violations.append(
                ViolationRecord(
                    violation_id=self._generate_violation_id(),
                    violation_type=ViolationType.RED_LIGHT_VIOLATION,
                    confidence=min(
                        vehicle.confidence, red_lights[0].confidence
                    ) * 0.9,  # slight discount for heuristic
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {vehicle.entity_class.value} "
                        f"proceeding through red signal (direction: {direction})."
                    ),
                    bbox=self._merge_bbox(
                        [vehicle, red_lights[0]]
                    ),
                    timestamp=self._now_iso(),
                    metadata={
                        "direction": direction,
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                        "signal_state": "red",
                    },
                )
            )

        return violations

    def _check_illegal_parking(
        self, sg: SceneGraph
    ) -> List[ViolationRecord]:
        """
        Detect illegally parked vehicles.

        Logic:
        1. Find vehicles with direction ``"STATIONARY"`` or no direction
           data and no ``track_id`` (untracked ≈ stationary for a while).
        2. In production this would also consult a no-parking zone map.
           For now, we flag stationary vehicles that have been detected
           consistently (placeholder for temporal accumulation).

        Note:
            Full illegal-parking detection requires multi-frame temporal
            analysis (duration threshold from ``Settings.illegal_parking_duration_sec``).
            This single-frame check only flags *candidates*.

        Returns:
            List of illegal-parking violation records.
        """
        violations: List[ViolationRecord] = []

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for vehicle in sg.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            direction = vehicle.attributes.get("direction")
            # Only consider stationary vehicles
            if direction is not None and direction != "STATIONARY":
                continue

            # Require parking flag from temporal analysis (set externally)
            is_parked_flag = vehicle.attributes.get("is_parked", False)
            if not is_parked_flag:
                continue

            involved = [vehicle.node_id]
            plate_node = sg.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                ViolationType.ILLEGAL_PARKING, "Illegal Parking"
            )

            violations.append(
                ViolationRecord(
                    violation_id=self._generate_violation_id(),
                    violation_type=ViolationType.ILLEGAL_PARKING,
                    confidence=vehicle.confidence * 0.7,  # low conf for single-frame
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {vehicle.entity_class.value} appears "
                        f"to be illegally parked."
                    ),
                    bbox=vehicle.bbox,
                    timestamp=self._now_iso(),
                    metadata={
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                        "stationary": True,
                    },
                )
            )

        return violations

    # ──────────────────────────────────────────────────────────
    # INTERNAL HELPERS
    # ──────────────────────────────────────────────────────────

    def _generate_violation_id(self) -> str:
        """
        Generate a unique, human-readable violation ID.

        Format: ``VIO-YYYYMMDD-NNN`` where NNN is a zero-padded sequence
        number that resets daily.

        Returns:
            Unique violation ID string.
        """
        today = datetime.date.today().strftime("%Y%m%d")

        if self._use_session_state:
            import streamlit as st
            if "global_violation_date" not in st.session_state or st.session_state.get("global_violation_date") != today:
                st.session_state["global_violation_date"] = today
                st.session_state["global_violation_counter"] = 0
            st.session_state["global_violation_counter"] += 1
            seq = st.session_state["global_violation_counter"]
        else:
            with ViolationEngine._global_counter_lock:
                if today != ViolationEngine._global_date_str:
                    ViolationEngine._global_date_str = today
                    ViolationEngine._global_counter = 0

                ViolationEngine._global_counter += 1
                seq = ViolationEngine._global_counter

        return f"VIO-{today}-{seq:03d}"

    def _merge_bbox(
        self, nodes: List[SceneNode]
    ) -> Tuple[int, int, int, int]:
        """
        Compute the union bounding box that encompasses all *nodes*.

        Args:
            nodes: List of :class:`SceneNode` instances whose bounding
                   boxes should be merged.

        Returns:
            ``(x1, y1, x2, y2)`` — the smallest axis-aligned box that
            contains all input boxes.  Returns ``(0, 0, 0, 0)`` if the
            list is empty.
        """
        if not nodes:
            return (0, 0, 0, 0)

        x1 = min(n.bbox[0] for n in nodes)
        y1 = min(n.bbox[1] for n in nodes)
        x2 = max(n.bbox[2] for n in nodes)
        y2 = max(n.bbox[3] for n in nodes)
        return (x1, y1, x2, y2)

    @staticmethod
    def _now_iso() -> str:
        """Return the current UTC time as an ISO-8601 string."""
        return datetime.datetime.now(datetime.timezone.utc).isoformat()

    def __repr__(self) -> str:  # pragma: no cover
        return f"ViolationEngine(counter={ViolationEngine._global_counter})"
