"""
Seatbelt Non-Compliance Detector.

Detects drivers or front-seat passengers in four-wheelers who are not wearing seatbelts.
"""

import logging
from typing import List, Optional
import numpy as np

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, FOUR_WHEELER_CLASSES
from core.violation_engine import ViolationRecord
from core.scene_graph import SceneGraph, DRIVES, NOT_WEARS, WEARS, SceneNode
from config.settings import EntityClass
from violations.base import BaseViolationDetector

logger = logging.getLogger(__name__)


class SeatbeltViolationDetector(BaseViolationDetector):
    """
    Detector for seatbelt non-compliance in four-wheelers.
    """

    violation_type = ViolationType.SEATBELT_NON_COMPLIANCE

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for four-wheeler drivers without seatbelts.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Seatbelt non-compliance violations.
        """
        violations: List[ViolationRecord] = []
        import datetime

        for ec in FOUR_WHEELER_CLASSES:
            triplets = scene_graph.query(
                source_class=None, relation=DRIVES, target_class=ec
            )
            
            # Group driver/occupant nodes by vehicle node
            veh_to_drivers = {}
            for driver_node, edge, vehicle_node in triplets:
                if vehicle_node.node_id not in veh_to_drivers:
                    veh_to_drivers[vehicle_node.node_id] = []
                veh_to_drivers[vehicle_node.node_id].append((driver_node, vehicle_node))

            for veh_id, drivers in veh_to_drivers.items():
                # Check if ANY driver in this vehicle is wearing a seatbelt
                any_wearing = False
                for d_node, v_node in drivers:
                    wears_seatbelt = any(
                        e.source_id == d_node.node_id
                        and e.relation == WEARS
                        and scene_graph.nodes.get(e.target_id, SceneNode(
                            node_id="", entity_class=EntityClass.PERSON,
                            bbox=(0, 0, 0, 0), confidence=0
                        )).entity_class == EntityClass.SEATBELT
                        for e in scene_graph.edges
                    )
                    if not wears_seatbelt:
                        wears_seatbelt = d_node.attributes.get("seatbelt_status") == "seatbelt"
                    if wears_seatbelt:
                        any_wearing = True
                        break

                if any_wearing:
                    logger.info(f"Seatbelt detected on occupant of {veh_id}. Suppressing violation.")
                    continue

                # None are wearing, let's flag the first driver node that has an explicit no_seatbelt detection
                for d_node, v_node in drivers:
                    has_no_seatbelt = any(
                        e.source_id == d_node.node_id
                        and e.relation == NOT_WEARS
                        and scene_graph.nodes.get(e.target_id, SceneNode(
                            node_id="", entity_class=EntityClass.PERSON,
                            bbox=(0, 0, 0, 0), confidence=0
                        )).entity_class == EntityClass.NO_SEATBELT
                        for e in scene_graph.edges
                    )
                    if not has_no_seatbelt:
                        has_no_seatbelt = d_node.attributes.get("seatbelt_status") == "no_seatbelt"

                    if not has_no_seatbelt:
                        continue

                    # Generate violation record
                    involved = [d_node.node_id, v_node.node_id]
                    plate_node = scene_graph.get_plate_of(v_node.node_id)
                    plate_text = ""
                    if plate_node is not None:
                        involved.append(plate_node.node_id)
                        plate_text = plate_node.attributes.get("plate_text", "")

                    display_name = VIOLATION_DISPLAY_NAMES.get(
                        self.violation_type,
                        "Seatbelt Non-Compliance",
                    )

                    x1 = min(d_node.bbox[0], v_node.bbox[0])
                    y1 = min(d_node.bbox[1], v_node.bbox[1])
                    x2 = max(d_node.bbox[2], v_node.bbox[2])
                    y2 = max(d_node.bbox[3], v_node.bbox[3])

                    timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
                    unique_suffix = f"{d_node.node_id[-3:]}-{v_node.node_id[-3:]}"
                    temp_id = f"VIO-SEATBELT-{unique_suffix}"

                    chain = [
                        f"Detected {d_node.entity_class.value}_{d_node.node_id} (conf={d_node.confidence:.2f}) near {v_node.entity_class.value}_{v_node.node_id} (conf={v_node.confidence:.2f})",
                        f"{d_node.node_id} DRIVES {v_node.node_id} (scene graph edge)",
                        f"Seatbelt scan on {d_node.node_id} → NO_SEATBELT detected",
                        f"{d_node.node_id} NOT_WEARS Seatbelt → VIOLATION: {display_name}",
                    ]

                    violations.append(
                        ViolationRecord(
                            violation_id=temp_id,
                            violation_type=self.violation_type,
                            confidence=min(d_node.confidence, v_node.confidence),
                            involved_nodes=involved,
                            description=(
                                f"{display_name}: Driver in {v_node.entity_class.value} "
                                f"without seatbelt."
                            ),
                            bbox=(x1, y1, x2, y2),
                            timestamp=timestamp,
                            metadata={
                                "plate_text": plate_text,
                                "vehicle_type": v_node.entity_class.value,
                                "driver_node": d_node.node_id,
                                "frame_available": frame is not None,
                            },
                            reasoning_chain=chain,
                        )
                    )
                    # Stop after generating one violation per vehicle
                    break

        return violations
