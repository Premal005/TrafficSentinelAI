"""
Wrong-Side Driving Detector.

Detects vehicles travelling against the expected flow of traffic in their assigned lane.
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


class WrongSideViolationDetector(BaseViolationDetector):
    """
    Detector for wrong-side driving violations.
    """

    violation_type = ViolationType.WRONG_SIDE_DRIVING

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for vehicles driving in the wrong direction.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Wrong-Side driving violations.
        """
        violations: List[ViolationRecord] = []
        import datetime

        # Expected directions by lane half (left / right of frame).
        # In India, traffic flows left (Left-Hand Driving). We assume:
        # - Left lanes (indices < 2) should move NORTH (away from camera).
        # - Right lanes (indices >= 2) should move SOUTH (oncoming/towards camera).
        expected_left_half = {"NORTH", "NORTHWEST", "NORTHEAST"}
        expected_right_half = {"SOUTH", "SOUTHWEST", "SOUTHEAST"}

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for node in scene_graph.nodes.values():
            if node.entity_class not in all_vehicle_classes:
                continue

            direction = scene_graph.get_direction(node.node_id)
            if direction is None or direction == "STATIONARY":
                continue

            # Skip nodes with negligible movement to avoid false flags from tracking jitter
            import math
            vector = node.attributes.get("direction_vector")
            if vector is not None:
                dx, dy = vector
                magnitude = math.hypot(dx, dy)
                if magnitude < 2.0:
                    continue

            lane_index = node.attributes.get("lane_index")
            if lane_index is None:
                continue

            is_left_half = lane_index < 2
            wrong_side = False

            if is_left_half and direction in expected_right_half:
                wrong_side = True
            elif not is_left_half and direction in expected_left_half:
                wrong_side = True

            if not wrong_side:
                continue

            involved = [node.node_id]
            plate_node = scene_graph.get_plate_of(node.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                self.violation_type, "Wrong-Side Driving"
            )

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            unique_suffix = f"{node.node_id[-3:]}-{direction}"
            temp_id = f"VIO-WRONGSIDE-{unique_suffix}"

            expected_dir = "SOUTH-bound" if is_left_half else "NORTH-bound"
            chain = [
                f"Detected {node.entity_class.value}_{node.node_id} (conf={node.confidence:.2f}) with trajectory data",
                f"Computed direction: {direction} in lane {lane_index}",
                f"Lane {lane_index} expected {expected_dir} flow, vehicle moving {direction} (against flow)",
                f"VIOLATION: {display_name} — {node.entity_class.value}_{node.node_id}",
            ]

            violations.append(
                ViolationRecord(
                    violation_id=temp_id,
                    violation_type=self.violation_type,
                    confidence=node.confidence * 0.8,  # Heuristic penalty
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {node.entity_class.value} moving "
                        f"{direction} in lane {lane_index} (expected opposite flow)."
                    ),
                    bbox=node.bbox,
                    timestamp=timestamp,
                    metadata={
                        "direction": direction,
                        "lane_index": lane_index,
                        "plate_text": plate_text,
                        "vehicle_type": node.entity_class.value,
                    },
                    reasoning_chain=chain,
                )
            )

        return violations
