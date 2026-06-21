"""
Missing License Plate / HSRP Detector.

Detects when a vehicle (car, motorcycle, auto_rickshaw, bus, truck, tempo)
is operating without a license plate detected, querying the Scene Graph relations.
"""

import logging
from typing import List, Optional
import numpy as np

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, SETTINGS, EntityClass
from core.violation_engine import ViolationRecord
from core.scene_graph import SceneGraph
from violations.base import BaseViolationDetector

logger = logging.getLogger(__name__)


class MissingPlateViolationDetector(BaseViolationDetector):
    """
    Detector for missing license plates on vehicles.
    """

    violation_type = ViolationType.MISSING_LICENSE_PLATE

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for vehicles lacking an associated license plate.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Missing Plate violations.
        """
        violations: List[ViolationRecord] = []
        import datetime

        # Bicycles do not require license plates, exclude them
        PLATE_REQUIRED_CLASSES = {
            EntityClass.CAR,
            EntityClass.MOTORCYCLE,
            EntityClass.AUTO_RICKSHAW,
            EntityClass.BUS,
            EntityClass.TRUCK,
            EntityClass.TEMPO,
        }

        all_vehicles = [
            n for n in scene_graph.nodes.values()
            if n.entity_class in PLATE_REQUIRED_CLASSES
        ]

        for vehicle in all_vehicles:
            # Check if there is a license plate associated with the vehicle
            plate_node = scene_graph.get_plate_of(vehicle.node_id)
            
            # If no plate is associated, it's a violation
            if plate_node is None:
                # 1. Require high vehicle confidence (>= 0.70) to prevent false-alarm dispatches
                if vehicle.confidence < 0.70:
                    logger.info(
                        "Skipping missing plate check for %s: low confidence (%.2f < 0.70)",
                        vehicle.node_id,
                        vehicle.confidence
                    )
                    continue

                # 2. Skip check if the vehicle is in side view (where plate is not expected to be visible)
                x1, y1, x2, y2 = vehicle.bbox
                w = x2 - x1
                h = y2 - y1
                aspect_ratio = w / h if h > 0 else 0
                
                is_side_view = False
                if vehicle.entity_class == EntityClass.MOTORCYCLE and aspect_ratio > 0.9:
                    is_side_view = True
                elif vehicle.entity_class in {EntityClass.CAR, EntityClass.TRUCK, EntityClass.BUS, EntityClass.TEMPO} and aspect_ratio > 1.7:
                    is_side_view = True
                
                if is_side_view:
                    logger.info(
                        "Skipping missing plate check for %s (detected as side view, aspect_ratio=%.2f)",
                        vehicle.node_id,
                        aspect_ratio
                    )
                    continue

                # 3. Spatial Occlusion Check: If another vehicle or person overlaps more than 10% of the vehicle's area,
                # the plate might be blocked/hidden by traffic. Skip check to maintain extreme precision.
                is_occluded = False
                veh_area = w * h
                for other_node in scene_graph.nodes.values():
                    if other_node.node_id == vehicle.node_id:
                        continue
                    if other_node.entity_class in (PLATE_REQUIRED_CLASSES | {EntityClass.PERSON, EntityClass.RIDER}):
                        # Ignore persons/riders who are driving or riding THIS vehicle
                        is_occupant = False
                        for edge in scene_graph.edges:
                            if edge.source_id == other_node.node_id and edge.target_id == vehicle.node_id:
                                is_occupant = True
                                break
                        if is_occupant:
                            continue
                            
                        ox1, oy1, ox2, oy2 = other_node.bbox
                        ix1 = max(x1, ox1)
                        iy1 = max(y1, oy1)
                        ix2 = min(x2, ox2)
                        iy2 = min(y2, oy2)
                        
                        inter_w = ix2 - ix1
                        inter_h = iy2 - iy1
                        if inter_w > 0 and inter_h > 0:
                            inter_area = inter_w * inter_h
                            overlap_ratio = inter_area / veh_area if veh_area > 0 else 0
                            if overlap_ratio > 0.10:
                                is_occluded = True
                                logger.info(
                                    "Skipping missing plate check for %s: occluded by %s (overlap=%.2f)",
                                    vehicle.node_id,
                                    other_node.node_id,
                                    overlap_ratio
                                )
                                break
                
                if is_occluded:
                    continue

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    self.violation_type, "Missing License Plate / HSRP Violation"
                )

                involved = [vehicle.node_id]

                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                unique_suffix = f"{vehicle.node_id[-3:]}-noplate"
                temp_id = f"VIO-PLATE-{unique_suffix}"

                chain = [
                    f"Detected vehicle {vehicle.entity_class.value}_{vehicle.node_id} (conf={vehicle.confidence:.2f})",
                    f"Queried scene graph for license plate linked to {vehicle.node_id}",
                    f"No license plate associated with {vehicle.node_id} (HAS_PLATE edge missing)",
                    f"VIOLATION: {display_name} on {vehicle.entity_class.value}_{vehicle.node_id}",
                ]

                violations.append(
                    ViolationRecord(
                        violation_id=temp_id,
                        violation_type=self.violation_type,
                        confidence=vehicle.confidence,
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: {vehicle.entity_class.value.capitalize()} "
                            f"operating without visible number plate."
                        ),
                        bbox=(x1, y1, x2, y2),
                        timestamp=timestamp,
                        metadata={
                            "vehicle_type": vehicle.entity_class.value,
                            "vehicle_node": vehicle.node_id,
                        },
                        reasoning_chain=chain,
                    )
                )

        return violations
