"""
Helmet Non-Compliance Detector.

Detects when a rider or pillion passenger on a two-wheeler (motorcycle, bicycle)
is not wearing a helmet, querying the Scene Graph relations.
"""

import logging
from typing import List, Optional
import numpy as np

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, TWO_WHEELER_CLASSES, SETTINGS, EntityClass
from core.violation_engine import ViolationRecord
from core.scene_graph import SceneGraph, RIDES, DRIVES, NOT_WEARS
from violations.base import BaseViolationDetector

logger = logging.getLogger(__name__)


class HelmetViolationDetector(BaseViolationDetector):
    """
    Detector for helmet non-compliance on two-wheelers.
    """

    violation_type = ViolationType.HELMET_NON_COMPLIANCE

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for two-wheelers with riders lacking helmets.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Helmet non-compliance violations.
        """
        violations: List[ViolationRecord] = []
        import datetime
        # Group riders by vehicle node ID
        vehicle_riders = {}
        vehicle_nodes = {}
        for ec in TWO_WHEELER_CLASSES:
            triplets = scene_graph.query(
                source_class=None, relation=RIDES, target_class=ec
            )
            for rider_node, edge, vehicle_node in triplets:
                vid = vehicle_node.node_id
                if vid not in vehicle_riders:
                    vehicle_riders[vid] = []
                    vehicle_nodes[vid] = vehicle_node
                vehicle_riders[vid].append(rider_node)

        for vid, riders in vehicle_riders.items():
            vehicle_node = vehicle_nodes[vid]
            
            # Determine which rider is the driver (closest to the front)
            driver_node = None
            if riders:
                if len(riders) == 1:
                    driver_node = riders[0]
                else:
                    direction = vehicle_node.attributes.get("direction", "")
                    dx, dy = vehicle_node.attributes.get("direction_vector", (0.0, 0.0))
                    # In India, traffic drives on the left.
                    # If vehicle is moving EAST (right), front is right -> max(x).
                    # If vehicle is moving WEST (left), front is left -> min(x).
                    if "EAST" in direction or direction == "EAST" or dx > 0:
                        driver_node = max(riders, key=lambda r: r.center[0])
                    elif "WEST" in direction or direction == "WEST" or dx < 0:
                        driver_node = min(riders, key=lambda r: r.center[0])
                    else:
                        # Heuristic based on plate position
                        plate_node = scene_graph.get_plate_of(vid)
                        if plate_node is not None:
                            px = plate_node.center[0]
                            vx = vehicle_node.center[0]
                            # Assuming rear plate: if plate is to the left, front is right -> max(x)
                            if px < vx:
                                driver_node = max(riders, key=lambda r: r.center[0])
                            else:
                                driver_node = min(riders, key=lambda r: r.center[0])
                        else:
                            # Fallback: assume facing right
                            driver_node = max(riders, key=lambda r: r.center[0])

            # Determine who we must check
            if not SETTINGS.pillion_helmet_required:
                riders_to_check = [driver_node] if driver_node else []
            else:
                riders_to_check = riders

            for rider_node in riders_to_check:
                status = scene_graph.get_helmet_status(rider_node.node_id)
                if status != "no_helmet":
                    continue

                # Build the list of involved nodes
                involved = [rider_node.node_id, vehicle_node.node_id]
                
                # Check for an associated license plate
                plate_node = scene_graph.get_plate_of(vehicle_node.node_id)
                plate_text = ""
                if plate_node is not None:
                    involved.append(plate_node.node_id)
                    plate_text = plate_node.attributes.get("plate_text", "")

                display_name = VIOLATION_DISPLAY_NAMES.get(
                    self.violation_type,
                    "Helmet Non-Compliance",
                )

                # Find the bounding box that covers both the rider and vehicle
                x1 = min(rider_node.bbox[0], vehicle_node.bbox[0])
                y1 = min(rider_node.bbox[1], vehicle_node.bbox[1])
                x2 = max(rider_node.bbox[2], vehicle_node.bbox[2])
                y2 = max(rider_node.bbox[3], vehicle_node.bbox[3])

                # Generate a temporary ID that the ViolationEngine can standardise if needed
                timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                unique_suffix = f"{rider_node.node_id[-3:]}-{vehicle_node.node_id[-3:]}"
                temp_id = f"VIO-HELMET-{unique_suffix}"

                role_str = "Driver" if rider_node == driver_node else "Pillion"
                chain = [
                    f"Detected {rider_node.entity_class.value}_{rider_node.node_id} (conf={rider_node.confidence:.2f}) near {vehicle_node.entity_class.value}_{vehicle_node.node_id} (conf={vehicle_node.confidence:.2f})",
                    f"{rider_node.node_id} RIDES {vehicle_node.node_id} (scene graph edge)",
                    f"Role: {role_str} (front-most rider check)",
                    f"Helmet scan on {rider_node.node_id} head region → NO_HELMET detected",
                    f"{rider_node.node_id} NOT_WEARS Helmet → VIOLATION: {display_name}",
                ]

                # Retrieve the confidence of the associated NO_HELMET node if available
                violation_conf = min(rider_node.confidence, vehicle_node.confidence)
                for edge in scene_graph.edges:
                    if edge.source_id == rider_node.node_id and edge.relation == NOT_WEARS:
                        target_node = scene_graph.nodes.get(edge.target_id)
                        if target_node and target_node.entity_class == EntityClass.NO_HELMET:
                            violation_conf = target_node.confidence
                            break

                violations.append(
                    ViolationRecord(
                        violation_id=temp_id,
                        violation_type=self.violation_type,
                        confidence=violation_conf,
                        involved_nodes=involved,
                        description=(
                            f"{display_name}: {role_str} on {vehicle_node.entity_class.value} "
                            f"without helmet."
                        ),
                        bbox=(x1, y1, x2, y2),
                        timestamp=timestamp,
                        metadata={
                            "plate_text": plate_text,
                            "vehicle_type": vehicle_node.entity_class.value,
                            "rider_node": rider_node.node_id,
                            "rider_role": role_str.lower(),
                        },
                        reasoning_chain=chain,
                    )
                )

        # Fallback for cases where a person has no_helmet but is not associated with any two-wheeler.
        # This occurs on pre-cropped snippet datasets or when the vehicle detector misses the two-wheeler.
        for node_id, node in scene_graph.nodes.items():
            if node.entity_class in {EntityClass.PERSON, EntityClass.RIDER}:
                # Check if this person is already associated with any vehicle (e.g. via RIDES or DRIVES)
                is_associated = False
                for edge in scene_graph.edges:
                    if edge.source_id == node_id and edge.relation in {RIDES, DRIVES}:
                        is_associated = True
                        break
                
                if not is_associated:
                    status = scene_graph.get_helmet_status(node_id)
                    if status == "no_helmet":
                        display_name = VIOLATION_DISPLAY_NAMES.get(
                            self.violation_type,
                            "Helmet Non-Compliance",
                        )
                        
                        involved = [node_id]
                        x1, y1, x2, y2 = node.bbox
                        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                        temp_id = f"VIO-HELMET-FALLBACK-{node_id[-3:]}"
                        
                        chain = [
                            f"Detected {node.entity_class.value}_{node.node_id} (conf={node.confidence:.2f}) with no motorcycle associated",
                            f"Helmet scan on {node.node_id} head region → NO_HELMET detected",
                            f"{node.node_id} NOT_WEARS Helmet → VIOLATION: {display_name} (Fallback)",
                        ]
                        
                        # Retrieve the confidence of the associated NO_HELMET node if available
                        fallback_conf = node.confidence
                        for edge in scene_graph.edges:
                            if edge.source_id == node_id and edge.relation == NOT_WEARS:
                                target_node = scene_graph.nodes.get(edge.target_id)
                                if target_node and target_node.entity_class == EntityClass.NO_HELMET:
                                    fallback_conf = target_node.confidence
                                    break

                        violations.append(
                            ViolationRecord(
                                violation_id=temp_id,
                                violation_type=self.violation_type,
                                confidence=fallback_conf,
                                involved_nodes=involved,
                                description=f"{display_name}: Rider without helmet (vehicle detection fallback).",
                                bbox=(x1, y1, x2, y2),
                                timestamp=timestamp,
                                metadata={
                                    "plate_text": "",
                                    "vehicle_type": "motorcycle",
                                    "rider_node": node_id,
                                    "rider_role": "driver",
                                    "fallback_mode": True
                                },
                                reasoning_chain=chain,
                            )
                        )

        return violations
