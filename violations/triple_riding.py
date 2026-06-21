"""
Triple Riding Detector.

Detects three or more persons riding on a single two-wheeler (motorcycle, bicycle).
"""

import logging
from typing import List, Optional
import numpy as np

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, TWO_WHEELER_CLASSES, SETTINGS
from core.violation_engine import ViolationRecord
from core.scene_graph import SceneGraph
from violations.base import BaseViolationDetector

logger = logging.getLogger(__name__)


class TripleRidingViolationDetector(BaseViolationDetector):
    """
    Detector for triple riding violations on two-wheelers.
    """

    violation_type = ViolationType.TRIPLE_RIDING

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for two-wheelers carrying three or more people.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Triple Riding violations.
        """
        violations: List[ViolationRecord] = []
        import datetime
        
        # Use settings limit (default = 3)
        min_persons = getattr(self, "triple_riding_min_persons", None)
        if min_persons is None:
            min_persons = SETTINGS.triple_riding_min_persons

        all_two_wheelers = [
            n for n in scene_graph.nodes.values()
            if n.entity_class in TWO_WHEELER_CLASSES
        ]

        for vehicle in all_two_wheelers:
            count = scene_graph.get_passenger_count(vehicle.node_id)
            if count < min_persons:
                continue

            riders = scene_graph.get_riders_of(vehicle.node_id)
            involved = [vehicle.node_id] + [r.node_id for r in riders]

            plate_node = scene_graph.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                self.violation_type, "Triple Riding"
            )

            all_involved_nodes = [vehicle] + riders
            
            # Find union bounding box
            x1 = min(n.bbox[0] for n in all_involved_nodes)
            y1 = min(n.bbox[1] for n in all_involved_nodes)
            x2 = max(n.bbox[2] for n in all_involved_nodes)
            y2 = max(n.bbox[3] for n in all_involved_nodes)

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            unique_suffix = f"{vehicle.node_id[-3:]}-{count}"
            temp_id = f"VIO-TRIPLE-{unique_suffix}"

            rider_ids = [r.node_id for r in riders]
            chain = [
                f"Detected {vehicle.entity_class.value}_{vehicle.node_id} (conf={vehicle.confidence:.2f})",
                f"Counted {count} persons RIDES {vehicle.node_id}: {rider_ids}",
                f"{count} >= {min_persons} (triple riding threshold)",
                f"VIOLATION: {display_name} on {vehicle.entity_class.value}_{vehicle.node_id}",
            ]

            violations.append(
                ViolationRecord(
                    violation_id=temp_id,
                    violation_type=self.violation_type,
                    confidence=sum(n.confidence for n in all_involved_nodes) / len(all_involved_nodes),
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {count} persons detected on "
                        f"{vehicle.entity_class.value}."
                    ),
                    bbox=(x1, y1, x2, y2),
                    timestamp=timestamp,
                    metadata={
                        "passenger_count": count,
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                    },
                    reasoning_chain=chain,
                )
            )

        return violations
