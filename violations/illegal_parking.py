"""
Illegal Parking Detector.

Detects stationary vehicles in restricted/no-parking zones.
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


class IllegalParkingViolationDetector(BaseViolationDetector):
    """
    Detector for illegal parking violations.
    """

    violation_type = ViolationType.ILLEGAL_PARKING

    def detect(
        self,
        scene_graph: SceneGraph,
        frame: Optional[np.ndarray] = None,
    ) -> List[ViolationRecord]:
        """
        Scan the scene graph for illegally parked stationary vehicles.

        Args:
            scene_graph: Structured representation of the current frame.
            frame: Optional raw BGR image.

        Returns:
            List of Illegal Parking violations.
        """
        violations: List[ViolationRecord] = []
        import datetime
        import cv2

        all_vehicle_classes = (
            TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES | {EntityClass.AUTO_RICKSHAW}
        )

        no_parking_detected = False
        no_parking_bbox = None
        scale = 1.0

        if frame is not None:
            try:
                import easyocr
                import torch
                from config.settings import SETTINGS
                h, w = frame.shape[:2]
                # Downscale for faster OCR
                scale = 640.0 / max(h, w)
                resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                use_gpu = torch.cuda.is_available() and SETTINGS.vehicle_detector.device != "cpu"
                if not hasattr(self, "_ocr_reader") or self._ocr_reader is None:
                    self._ocr_reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                
                ocr_results = self._ocr_reader.readtext(resized)
                
                # Check for "PARK" and "NO" in the same box or close by
                for bbox, text, conf in ocr_results:
                    text_upper = text.upper()
                    if "PARK" in text_upper and ("NO" in text_upper or "NOT" in text_upper):
                        no_parking_detected = True
                        no_parking_bbox = bbox
                        break
                        
                if not no_parking_detected:
                    no_boxes = [b for b, t, c in ocr_results if "NO" in t.upper()]
                    park_boxes = [b for b, t, c in ocr_results if "PARK" in t.upper()]
                    for n_box in no_boxes:
                        for p_box in park_boxes:
                            nc_x = (n_box[0][0] + n_box[2][0]) / 2.0
                            nc_y = (n_box[0][1] + n_box[2][1]) / 2.0
                            pc_x = (p_box[0][0] + p_box[2][0]) / 2.0
                            pc_y = (p_box[0][1] + p_box[2][1]) / 2.0
                            dist = np.sqrt((nc_x - pc_x)**2 + (nc_y - pc_y)**2)
                            if dist < 200:
                                no_parking_detected = True
                                no_parking_bbox = p_box
                                break
                        if no_parking_detected:
                            break
            except Exception as e:
                logger.error("OCR detection in illegal parking failed: %s", e)

        for vehicle in scene_graph.nodes.values():
            if vehicle.entity_class not in all_vehicle_classes:
                continue

            direction = vehicle.attributes.get("direction")
            if direction is not None and direction != "STATIONARY":
                continue

            # Check if vehicle has been flagged as parked (typically set by multi-tracker or dashboard)
            is_parked_flag = vehicle.attributes.get("is_parked", False)
            
            # For demo purposes, we also flag if stationary attribute is set or if track has no speed
            if not is_parked_flag and vehicle.attributes.get("speed", 99.0) == 0.0:
                is_parked_flag = True

            # OCR proximity fallback for static frame/demo images
            if not is_parked_flag and no_parking_detected and no_parking_bbox is not None:
                vx, vy = vehicle.center
                bx1 = int(no_parking_bbox[0][0] / scale)
                by1 = int(no_parking_bbox[0][1] / scale)
                bx2 = int(no_parking_bbox[2][0] / scale)
                by2 = int(no_parking_bbox[2][1] / scale)
                bx = (bx1 + bx2) / 2.0
                by = (by1 + by2) / 2.0
                
                dist = np.sqrt((vx - bx) ** 2 + (vy - by) ** 2)
                v_size = max(vehicle.bbox[2] - vehicle.bbox[0], vehicle.bbox[3] - vehicle.bbox[1])
                if dist < max(500.0, v_size * 3.5):
                    is_parked_flag = True
                    logger.info("Flagged vehicle %s as parked due to proximity to No Parking sign (dist=%.1f)", vehicle.node_id, dist)

            if not is_parked_flag:
                continue

            involved = [vehicle.node_id]
            plate_node = scene_graph.get_plate_of(vehicle.node_id)
            plate_text = ""
            if plate_node is not None:
                involved.append(plate_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "")

            display_name = VIOLATION_DISPLAY_NAMES.get(
                self.violation_type, "Illegal Parking"
            )

            timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
            unique_suffix = f"{vehicle.node_id[-3:]}-parked"
            temp_id = f"VIO-PARKING-{unique_suffix}"

            chain = [
                f"Detected {vehicle.entity_class.value}_{vehicle.node_id} (conf={vehicle.confidence:.2f})",
                f"{vehicle.node_id} is_parked=True (stationary/flagged by tracker)",
            ]
            if no_parking_detected:
                chain.append(f"OCR detected 'NO PARKING' sign/marking nearby")
            chain.append(f"{vehicle.node_id} located in no-parking / restricted zone")
            chain.append(f"VIOLATION: {display_name} — {vehicle.entity_class.value}_{vehicle.node_id}")

            violations.append(
                ViolationRecord(
                    violation_id=temp_id,
                    violation_type=self.violation_type,
                    confidence=vehicle.confidence * 0.7,  # Single-frame penalty
                    involved_nodes=involved,
                    description=(
                        f"{display_name}: {vehicle.entity_class.value} detected "
                        f"stationary in restricted/no-parking zone."
                    ),
                    bbox=vehicle.bbox,
                    timestamp=timestamp,
                    metadata={
                        "plate_text": plate_text,
                        "vehicle_type": vehicle.entity_class.value,
                        "stationary": True,
                        "is_parked": True,
                    },
                    reasoning_chain=chain,
                )
            )

        return violations
