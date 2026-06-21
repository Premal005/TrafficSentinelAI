"""
Red-Light Violation Detector.

Detects vehicles actively proceeding through an intersection during a red traffic signal.
"""

import logging
from typing import List, Optional
import numpy as np

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
from config.settings import EntityClass
from core.violation_engine import ViolationRecord
from core.scene_graph import SceneGraph
from violations.base import BaseViolationDetector

logger = logging.getLogger(__name__)


class RedLightViolationDetector(BaseViolationDetector):
    """
    Detector for red-light violations.
    """

    violation_type = ViolationType.RED_LIGHT_VIOLATION

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for vehicles running red traffic lights.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Red-Light violations.
        """
        violations: List[ViolationRecord] = []
        import datetime

        red_lights = [
            n for n in scene_graph.nodes.values()
            if n.entity_class == EntityClass.TRAFFIC_LIGHT_RED
        ]
        stop_lines = [
            n for n in scene_graph.nodes.values()
            if n.entity_class == EntityClass.STOP_LINE
        ]

        if not red_lights:
            return violations

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for vehicle in scene_graph.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            direction = scene_graph.get_direction(vehicle.node_id)
            if direction == "STATIONARY":
                continue  # Skip only if verified stationary (to handle static frames where direction is None)

            # Check if vehicle has crossed a stop line
            crossed_stop = False
            for sl in stop_lines:
                vehicle_bottom = vehicle.bbox[3]
                sl_y = (sl.bbox[1] + sl.bbox[3]) / 2.0

                v_cx = (vehicle.bbox[0] + vehicle.bbox[2]) / 2.0
                if sl.bbox[0] <= v_cx <= sl.bbox[2] and vehicle_bottom > sl_y:
                    crossed_stop = True
                    break

            if not crossed_stop and stop_lines:
                continue

            # If no stop line is detected, check signal proximity as fallback
            if not stop_lines:
                # Approximate distance from vehicle to red light
                vx, vy = vehicle.center
                lx, ly = red_lights[0].center
                dist = np.sqrt((vx - lx) ** 2 + (vy - ly) ** 2)
                
                # Exclude if too far (e.g. > 12x size of vehicle box, or > 1200 pixels)
                v_size = max(vehicle.bbox[2] - vehicle.bbox[0], vehicle.bbox[3] - vehicle.bbox[1])
                if dist > max(1200.0, v_size * 12.0):
                    continue

            involved = [vehicle.node_id, red_lights[0].node_id]
            plate_node = scene_graph.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                self.violation_type, "Red-Light Violation"
            )

            x1 = min(vehicle.bbox[0], red_lights[0].bbox[0])
            y1 = min(vehicle.bbox[1], red_lights[0].bbox[1])
            x2 = max(vehicle.bbox[2], red_lights[0].bbox[2])
            y2 = max(vehicle.bbox[3], red_lights[0].bbox[3])

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            unique_suffix = f"{vehicle.node_id[-3:]}-{red_lights[0].node_id[-3:]}"
            temp_id = f"VIO-REDLIGHT-{unique_suffix}"

            chain = [
                f"Detected traffic_light_red_{red_lights[0].node_id} (conf={red_lights[0].confidence:.2f})",
                f"Detected {vehicle.entity_class.value}_{vehicle.node_id} (conf={vehicle.confidence:.2f}) past stop line",
                f"Vehicle moving {direction} — crossed intersection during red signal",
                f"VIOLATION: {display_name} — {vehicle.entity_class.value}_{vehicle.node_id}",
            ]

            violations.append(
                ViolationRecord(
                    violation_id=temp_id,
                    violation_type=self.violation_type,
                    confidence=min(vehicle.confidence, red_lights[0].confidence) * 0.9,  # Heuristic penalty
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {vehicle.entity_class.value} "
                        f"proceeding through red signal (direction: {direction})."
                    ),
                    bbox=(x1, y1, x2, y2),
                    timestamp=timestamp,
                    metadata={
                        "direction": direction,
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                        "signal_state": "red",
                    },
                    reasoning_chain=chain,
                )
            )

        return violations
