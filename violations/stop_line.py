"""
Stop-Line Violation Detector.

Detects vehicles crossing the stop line at an intersection when the traffic signal is red.
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


class StopLineViolationDetector(BaseViolationDetector):
    """
    Detector for stop-line violations.
    """

    violation_type = ViolationType.STOP_LINE_VIOLATION

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for vehicles crossing stop lines on red light.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Stop-Line violations.
        """
        violations: List[ViolationRecord] = []
        import datetime

        stop_lines = [
            n for n in scene_graph.nodes.values()
            if n.entity_class == EntityClass.STOP_LINE
        ]
        red_lights = [
            n for n in scene_graph.nodes.values()
            if n.entity_class == EntityClass.TRAFFIC_LIGHT_RED
        ]

        if not red_lights:
            return violations

        if not stop_lines:
            # Fallback to a virtual stop line running horizontally at 70% of frame height
            frame_height = 1080
            frame_width = 1920
            if frame is not None:
                frame_height, frame_width = frame.shape[:2]
            else:
                # Infer resolution from scene graph node bounds
                max_y = 0
                max_x = 0
                for node in scene_graph.nodes.values():
                    max_x = max(max_x, node.bbox[2])
                    max_y = max(max_y, node.bbox[3])
                if max_y > 0:
                    frame_height = max_y
                    frame_width = max_x

            virtual_y = int(frame_height * 0.70)
            from core.scene_graph import SceneNode
            virtual_stop_line = SceneNode(
                node_id="stop_line_virtual",
                entity_class=EntityClass.STOP_LINE,
                bbox=(0, virtual_y - 2, frame_width, virtual_y + 2),
                confidence=1.0,
                attributes={"virtual": True}
            )
            stop_lines = [virtual_stop_line]

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        for vehicle in scene_graph.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            for stop_line in stop_lines:
                # Vehicle crossed stop line if bottom is below the stop line center-y
                vehicle_bottom = vehicle.bbox[3]
                stop_line_y = (stop_line.bbox[1] + stop_line.bbox[3]) / 2.0

                # Horizontal alignment check (in the same lane horizontal span)
                v_cx = (vehicle.bbox[0] + vehicle.bbox[2]) / 2.0
                sl_x1 = stop_line.bbox[0]
                sl_x2 = stop_line.bbox[2]

                if not (sl_x1 <= v_cx <= sl_x2):
                    continue

                if vehicle_bottom <= stop_line_y:
                    continue

                involved = [vehicle.node_id, stop_line.node_id]
                plate_node = scene_graph.get_plate_of(vehicle.node_id)
                plate_text = ""
                if plate_node is not None:
                    involved.append(plate_node.node_id)
                    plate_text = plate_node.attributes.get("plate_text", "")

                involved.append(red_lights[0].node_id)

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    self.violation_type,
                    "Stop-Line Violation",
                )

                x1 = min(vehicle.bbox[0], stop_line.bbox[0])
                y1 = min(vehicle.bbox[1], stop_line.bbox[1])
                x2 = max(vehicle.bbox[2], stop_line.bbox[2])
                y2 = max(vehicle.bbox[3], stop_line.bbox[3])

                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                unique_suffix = f"{vehicle.node_id[-3:]}-{stop_line.node_id[-3:]}"
                temp_id = f"VIO-STOPLINE-{unique_suffix}"

                chain = [
                    f"Detected {stop_line.entity_class.value}_{stop_line.node_id} at y={stop_line_y:.0f}",
                    f"Detected {vehicle.entity_class.value}_{vehicle.node_id} (conf={vehicle.confidence:.2f})",
                    f"Vehicle bbox bottom ({vehicle_bottom}) crosses stop_line y-position ({stop_line_y:.0f})",
                    f"VIOLATION: {display_name} — {vehicle.entity_class.value}_{vehicle.node_id} past stop line during red",
                ]

                violations.append(
                    ViolationRecord(
                        violation_id=temp_id,
                        violation_type=self.violation_type,
                        confidence=min(vehicle.confidence, stop_line.confidence),
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: {vehicle.entity_class.value} "
                            f"crossed stop line during red signal."
                        ),
                        bbox=(x1, y1, x2, y2),
                        timestamp=timestamp,
                        metadata={
                            "plate_text": plate_text,
                            "vehicle_type": vehicle.entity_class.value,
                            "signal_state": "red",
                        },
                        reasoning_chain=chain,
                    )
                )
                break  # one stop line violation check per vehicle is enough

        return violations
