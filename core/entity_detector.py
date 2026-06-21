"""
Multi-Scale Entity Detection Module — Layer 2 of HSUP Pipeline.

Provides high-accuracy object detection for traffic scenes using YOLOv11
with optional SAHI (Slicing Aided Hyper Inference) for small-object
enhancement. Supports three specialized detection heads:

    1. Vehicle/Person detection (COCO-pretrained YOLOv11m)
    2. Helmet detection (custom-trained YOLOv11s)
    3. License plate detection (custom-trained YOLOv11s)

Usage::

    from core.entity_detector import EntityDetector, Detection

    detector = EntityDetector()
    detections = detector.detect(frame, use_sahi=True)
    plates = detector.detect_plates(frame)
"""

from __future__ import annotations

import logging
import cv2
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

from config.settings import (
    COCO_TO_ENTITY,
    EntityClass,
    ModelConfig,
    SAHIConfig,
    Settings,
    SETTINGS,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# DETECTION DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class Detection:
    """A single detected entity in a frame.

    Attributes:
        bbox: Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        class_id: Integer class ID from the underlying model.
        class_name: Human-readable class name string.
        entity_class: Mapped ``EntityClass`` enum value, or ``None`` if
            the COCO class has no traffic-scene mapping.
        confidence: Detection confidence score in [0, 1].
        track_id: Object-tracking ID assigned by a tracker, or ``None``
            if tracking has not been applied.
    """

    bbox: Tuple[int, int, int, int]
    class_id: int
    class_name: str
    entity_class: Optional[EntityClass] = None
    confidence: float = 0.0
    track_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════
# ENTITY DETECTOR
# ══════════════════════════════════════════════════════════════

class EntityDetector:
    """Multi-scale entity detector using YOLOv11 and SAHI.

    Wraps Ultralytics YOLO models and an optional SAHI slicing pipeline
    to detect vehicles, persons, helmets, and license plates in traffic
    surveillance imagery.

    Args:
        settings: Project-wide ``Settings`` instance.  Falls back to the
            global ``SETTINGS`` singleton when *None*.

    Example::

        detector = EntityDetector()
        dets = detector.detect(frame)
        for d in dets:
            print(f"{d.class_name} @ {d.bbox}  conf={d.confidence:.2f}")
    """

    def __init__(self, settings: Settings = None) -> None:
        self._settings: Settings = settings or SETTINGS
        self._vehicle_model = None
        self._helmet_model = None
        self._plate_model = None
        self._sahi_model = None
        self._helmet_classifier = None
        self._helmet_device = None
        self._classifier_transforms = None

        # Eagerly load the primary vehicle/person model
        self._load_vehicle_model()

    # ──────────────────────────────────────────────────────
    # Model loading helpers
    # ──────────────────────────────────────────────────────

    def _load_yolo(self, config: ModelConfig, label: str):
        """Load a YOLO model from *config*, returning it or ``None``.

        Args:
            config: ``ModelConfig`` with weights path and thresholds.
            label: Human-readable label used in log messages.

        Returns:
            A ``YOLO`` model instance, or ``None`` if loading fails.
        """
        try:
            from ultralytics import YOLO  # type: ignore[import-untyped]
        except ImportError:
            logger.error(
                "ultralytics package is not installed. "
                "Run: pip install ultralytics"
            )
            return None

        model_path = self._settings.get_model_path(config)
        try:
            model = YOLO(str(model_path))
            logger.info(
                "Loaded %s model from %s (conf=%.2f, iou=%.2f)",
                label,
                model_path,
                config.confidence_threshold,
                config.iou_threshold,
            )
            return model
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to load %s model from %s: %s. "
                "Detection will return empty results.",
                label,
                model_path,
                exc,
            )
            return None

    def _load_vehicle_model(self) -> None:
        """Load the primary vehicle/person YOLO model."""
        self._vehicle_model = self._load_yolo(
            self._settings.vehicle_detector, "vehicle/person"
        )

    def _load_helmet_model(self) -> None:
        """Lazy-load the helmet detection model on first use."""
        if self._helmet_model is None:
            self._helmet_model = self._load_yolo(
                self._settings.helmet_detector, "helmet"
            )

    def _load_plate_model(self) -> None:
        """Lazy-load the license-plate detection model on first use."""
        if self._plate_model is None:
            self._plate_model = self._load_yolo(
                self._settings.plate_detector, "plate"
            )

    def _load_custom_helmet_classifier(self) -> None:
        """Load the custom MobileNetV3 helmet classifier."""
        if self._helmet_classifier is not None:
            return

        model_path = self._settings.models_dir / "custom_helmet_classifier.pth"
        if not model_path.exists():
            return

        try:
            import torch
            from torchvision import models, transforms

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Construct MobileNetV3 Architecture
            model = models.mobilenet_v3_small(weights=None)
            in_features = model.classifier[3].in_features
            model.classifier[3] = torch.nn.Linear(in_features, 2)
            
            # Load state dict
            state_dict = torch.load(str(model_path), map_location=device)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            
            self._helmet_classifier = model
            self._helmet_device = device
            self._classifier_transforms = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            logger.info("Custom MobileNetV3 helmet classifier loaded successfully.")
        except Exception as e:
            logger.error("Failed to load custom helmet classifier: %s", e)

    # ──────────────────────────────────────────────────────
    # Image utilities
    # ──────────────────────────────────────────────────────

    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Down-scale *image* so its longest side ≤ ``max_image_dimension``.

        The aspect ratio is preserved.  If the image is already within
        limits it is returned unchanged (no copy).

        Args:
            image: BGR/RGB numpy array of shape ``(H, W, C)``.

        Returns:
            Possibly resized copy of *image*.
        """
        max_dim = self._settings.max_image_dimension
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image

        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        try:
            import cv2  # type: ignore[import-untyped]

            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            logger.debug(
                "Resized image from (%d, %d) → (%d, %d)",
                w, h, new_w, new_h,
            )
            return resized
        except ImportError:
            logger.warning(
                "cv2 not available; returning original image without resize."
            )
            return image

    def _compute_box_sharpness(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> float:
        """Compute the sharpness (Laplacian variance) of the bounding box crop."""
        try:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            
            if x2 <= x1 or y2 <= y1:
                return 0.0
                
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return 0.0
                
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            return float(sharpness)
        except Exception as e:
            logger.debug("Failed to compute sharpness for box %s: %s", bbox, e)
            return 0.0

    # ──────────────────────────────────────────────────────
    # Core detection
    # ──────────────────────────────────────────────────────

    def detect(
        self,
        image: np.ndarray,
        use_sahi: bool = True,
    ) -> List[Detection]:
        """Run entity detection on *image*.

        When *use_sahi* is ``True`` **and** SAHI is enabled in settings,
        the image is processed through sliced inference for improved
        small-object recall.  Otherwise, standard single-pass YOLO
        inference is used.

        Args:
            image: BGR numpy array of shape ``(H, W, C)``.
            use_sahi: Whether to attempt SAHI sliced inference.

        Returns:
            List of ``Detection`` objects (may be empty).
        """
        if image is None or image.size == 0:
            logger.warning("detect() received empty image.")
            return []

        image = self._resize_if_needed(image)

        if use_sahi and self._settings.sahi.enabled:
            detections = self._detect_with_sahi(image)
            if not detections:
                detections = self._detect_direct(image)
        else:
            detections = self._detect_direct(image)

        # Merge split vehicle detections (e.g. front/rear halves from SAHI)
        detections = self._merge_split_vehicles(detections)
        detections = self._apply_class_wise_nms(detections)



        # ──────────────────────────────────────────────────────
        # Foreground Focus: Filter out background/distant entities
        # ──────────────────────────────────────────────────────
        if not detections:
            return []

        img_h, img_w = image.shape[:2]
        total_area = img_h * img_w
        
        filtered_dets = []
        vehicles_kept = []
        
        vehicle_class_values = {"car", "motorcycle", "auto_rickshaw", "bus", "truck", "tempo", "bicycle"}
        
        for det in detections:
            # If it is a vehicle, check its area ratio and focal sharpness
            if det.entity_class is not None and det.entity_class.value in vehicle_class_values:
                x1, y1, x2, y2 = det.bbox
                area = (x2 - x1) * (y2 - y1)
                area_ratio = area / total_area
                
                # Compute local sharpness and store it in metadata
                sharpness = self._compute_box_sharpness(image, det.bbox)
                det.metadata["sharpness"] = sharpness
                
                # Decide if foreground
                is_foreground_area = area_ratio >= self._settings.min_vehicle_area_ratio
                
                # Check focal focus sharpness
                is_in_focus = True
                if self._settings.enable_focal_pruning:
                    # Ignore sharpness pruning only if the vehicle is exceptionally close/large (area_ratio >= 0.15)
                    if area_ratio < 0.15 and sharpness < self._settings.min_vehicle_sharpness:
                        is_in_focus = False
                
                if is_foreground_area and is_in_focus:
                    filtered_dets.append(det)
                    vehicles_kept.append(det)
                else:
                    reason = "area ratio" if not is_foreground_area else "focal sharpness blur"
                    logger.info(
                        "Foreground Focus: Filtered out background vehicle %s (%s, area ratio %.4f, sharpness %.1f)",
                        det.class_name, reason, area_ratio, sharpness
                    )
            else:
                # Keep other classes for now, we will filter persons next
                filtered_dets.append(det)
                
        # 2. Filter out background/distant pedestrians
        final_dets = []
        
        # Local helper for IoU and overlap calculation to avoid circular imports
        def get_box_metrics(box1, box2):
            # IoU
            ix1 = max(box1[0], box2[0])
            iy1 = max(box1[1], box2[1])
            ix2 = min(box1[2], box2[2])
            iy2 = min(box1[3], box2[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            
            area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
            area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
            
            union = area1 + area2 - inter
            iou_val = inter / union if union > 0 else 0.0
            overlap_val = inter / area1 if area1 > 0 else 0.0
            return max(iou_val, overlap_val)
        
        for det in filtered_dets:
            if det.entity_class == EntityClass.PERSON:
                # Check if this person is near any of the kept foreground vehicles
                is_near_vehicle = False
                for v in vehicles_kept:
                    from config.settings import TWO_WHEELER_CLASSES
                    if v.entity_class in TWO_WHEELER_CLASSES:
                        score = get_box_metrics(det.bbox, v.bbox)
                        if score >= 0.08: # Keep if they have any reasonable association
                            is_near_vehicle = True
                            break
                    else:
                        from core.scene_graph import is_four_wheeler_occupant
                        if is_four_wheeler_occupant(det.bbox, v.bbox):
                            is_near_vehicle = True
                            break
                
                if is_near_vehicle:
                    final_dets.append(det)
                else:
                    px1, py1, px2, py2 = det.bbox
                    p_area = (px2 - px1) * (py2 - py1)
                    p_height = py2 - py1
                    logger.info(
                        "Foreground Focus: Filtered out background/isolated person (height=%d, area=%d, is_near=%s)",
                        p_height, p_area, is_near_vehicle
                    )
            else:
                # Keep plates, helmets, traffic signals, etc.
                final_dets.append(det)
                
        return final_dets


    def _detect_direct(self, image: np.ndarray) -> List[Detection]:
        """Standard single-pass YOLO inference.

        Args:
            image: Pre-processed BGR numpy array.

        Returns:
            List of ``Detection`` objects.
        """
        if self._vehicle_model is None:
            logger.warning("Vehicle model not loaded — returning empty.")
            return []

        cfg = self._settings.vehicle_detector
        # Use a lower floor threshold (0.15) for the YOLO model call so that
        # traffic lights (conf >= 0.20) and persons (conf >= conf_thresh - 0.10) are not discarded.
        model_conf_floor = min(0.15, cfg.confidence_threshold)
        try:
            results = self._vehicle_model(
                image,
                conf=model_conf_floor,
                iou=cfg.iou_threshold,
                imgsz=cfg.image_size,
                verbose=False,
                device=cfg.device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("YOLO inference failed: %s", exc)
            return []

        return self._parse_results(results, image=image)

    def _detect_with_sahi(self, image: np.ndarray) -> List[Detection]:
        """SAHI sliced-inference for enhanced small-object detection.

        Uses the ``sahi`` library to slice the image into overlapping
        tiles, run YOLO on each, and merge predictions via NMS.

        Args:
            image: BGR numpy array.

        Returns:
            List of ``Detection`` objects, or empty list on failure.
        """
        try:
            from sahi import AutoDetectionModel  # type: ignore[import-untyped]
            from sahi.predict import get_sliced_prediction  # type: ignore[import-untyped]
        except ImportError:
            logger.warning(
                "sahi package is not installed. "
                "Run: pip install sahi — falling back to direct inference."
            )
            return []

        if self._vehicle_model is None:
            logger.warning("Vehicle model not loaded — SAHI skipped.")
            return []

        cfg = self._settings.vehicle_detector
        sahi_cfg = self._settings.sahi

        try:
            # Build the SAHI detection-model wrapper (lazy, cached)
            if self._sahi_model is None:
                model_path = self._settings.get_model_path(cfg)
                # Use a lower floor threshold (0.15) for the model wrapper
                self._sahi_model = AutoDetectionModel.from_pretrained(
                    model_type="yolov8",  # ultralytics family
                    model_path=str(model_path),
                    confidence_threshold=0.15,
                    device=cfg.device if cfg.device != "auto" else "",
                )

            result = get_sliced_prediction(
                image=image,
                detection_model=self._sahi_model,
                slice_height=sahi_cfg.slice_height,
                slice_width=sahi_cfg.slice_width,
                overlap_height_ratio=sahi_cfg.overlap_ratio,
                overlap_width_ratio=sahi_cfg.overlap_ratio,
                postprocess_type=sahi_cfg.postprocess_type,
                postprocess_match_threshold=sahi_cfg.postprocess_match_threshold,
                verbose=0,
            )

            detections: List[Detection] = []
            for pred in result.object_prediction_list:
                bbox_xyxy = pred.bbox.to_xyxy()
                x1, y1, x2, y2 = (
                    int(bbox_xyxy[0]),
                    int(bbox_xyxy[1]),
                    int(bbox_xyxy[2]),
                    int(bbox_xyxy[3]),
                )
                class_id = pred.category.id
                class_name = pred.category.name
                entity_class = COCO_TO_ENTITY.get(class_id)
                if class_id == 9 or (class_name and class_name.lower() == "traffic light"):
                    entity_class = self._classify_traffic_light_color(image, (x1, y1, x2, y2))
                
                confidence = float(pred.score.value)
                # Apply class-specific confidence filtering:
                is_person = entity_class in [EntityClass.PERSON, EntityClass.RIDER]
                is_light = (entity_class in (EntityClass.TRAFFIC_LIGHT_RED, EntityClass.TRAFFIC_LIGHT_GREEN, EntityClass.TRAFFIC_LIGHT_YELLOW))
                
                if is_person:
                    required_conf = max(0.20, cfg.confidence_threshold - 0.10)
                elif is_light:
                    required_conf = 0.20
                else:
                    required_conf = cfg.confidence_threshold
                    
                if confidence < required_conf:
                    continue

                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=class_id,
                        class_name=class_name,
                        entity_class=entity_class,
                        confidence=confidence,
                    )
                )

            logger.info(
                "SAHI detection: %d objects found (slices %dx%d, overlap %.1f).",
                len(detections),
                sahi_cfg.slice_width,
                sahi_cfg.slice_height,
                sahi_cfg.overlap_ratio,
            )
            return detections

        except Exception as exc:  # noqa: BLE001
            logger.error("SAHI inference failed: %s", exc)
            return []

    # Alias so callers can use the public name documented in the spec.
    detect_with_sahi = _detect_with_sahi

    # ──────────────────────────────────────────────────────
    # Specialised detection heads
    # ──────────────────────────────────────────────────────

    def detect_helmets(
        self,
        image: np.ndarray,
        person_crops: Optional[List[Tuple[int, int, int, int]]] = None,
    ) -> List[Detection]:
        """Run helmet detection on the full frame (YOLO) or on crops (classifier).

        If the configured weights path is a YOLO detector (.pt), runs detection on the full frame.
        If the configured weights path is a classifier (.pth), runs classification on person crops.

        Args:
            image: Full-frame BGR numpy array.
            person_crops: Optional list of (x1, y1, x2, y2) bounding boxes.
                Required if using classifier mode.

        Returns:
            List of Detection objects with helmet/no-helmet labels.
        """
        if image is None or image.size == 0:
            logger.warning("detect_helmets() received empty image.")
            return []

        cfg = self._settings.helmet_detector

        # 1. Classifier Mode (.pth)
        if cfg.weights_path.endswith(".pth"):
            if not person_crops:
                logger.warning("Classifier mode requires person_crops, but none provided.")
                return []

            self._load_custom_helmet_classifier()
            if self._helmet_classifier is None:
                logger.error("Custom helmet classifier not loaded.")
                return []

            import torch
            all_detections = []
            for crop_box in person_crops:
                x1, y1, x2, y2 = crop_box
                h, w = image.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                # Crop head region (top 35% of expanded person)
                person_h = y2 - y1
                y1_expanded = max(0, y1 - int(person_h * 0.35))
                head_h = (y2 - y1_expanded) // 2
                y2_head = y1_expanded + head_h
                
                head_crop = image[y1_expanded:y2_head, x1:x2]
                if head_crop.size == 0 or head_crop.shape[0] < 16 or head_crop.shape[1] < 16:
                    continue

                try:
                    # Convert BGR to RGB for PIL/PyTorch classifier
                    crop_rgb = cv2.cvtColor(head_crop, cv2.COLOR_BGR2RGB)
                    input_tensor = self._classifier_transforms(crop_rgb).unsqueeze(0).to(self._helmet_device)
                    
                    with torch.no_grad():
                        outputs = self._helmet_classifier(input_tensor)
                        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                        confidence, predicted_class = torch.max(probabilities, 0)
                        
                    predicted_class = int(predicted_class.cpu().item())
                    confidence = float(confidence.cpu().item())
                    
                    # Map classes: 0 -> helmet, 1 -> no_helmet
                    if predicted_class == 0:
                        det_class = EntityClass.HELMET
                        det_name = "helmet"
                    else:
                        det_class = EntityClass.NO_HELMET
                        det_name = "no_helmet"

                    all_detections.append(
                        Detection(
                            bbox=(x1, y1_expanded, x2, y2_head),
                            class_id=predicted_class,
                            class_name=det_name,
                            entity_class=det_class,
                            confidence=confidence,
                        )
                    )
                except Exception as e:
                    logger.debug("Custom classifier inference failed: %s", e)

            logger.info(
                "Custom helmet classifier: %d detections across %d crops.",
                len(all_detections),
                len(person_crops),
            )
            return all_detections

        # 2. YOLO Detector Mode (.pt)
        self._load_helmet_model()
        if self._helmet_model is None:
            logger.error("Helmet model not loaded.")
            return []

        try:
            results = self._helmet_model(
                image,
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                verbose=False,
                device=cfg.device,
            )
        except Exception as exc:
            logger.error("Helmet inference failed on full frame: %s", exc)
            return []

        all_detections = []
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                
                class_name = result.names.get(cls_id, f"class_{cls_id}")
                
                name_lower = class_name.lower().replace("-", "").replace("_", "")
                if "nohard" in name_lower or "nohelmet" in name_lower or "without" in name_lower:
                    ent_class = EntityClass.NO_HELMET
                    det_name = "no_helmet"
                else:
                    ent_class = EntityClass.HELMET
                    det_name = "helmet"
                
                if conf >= cfg.confidence_threshold:
                    all_detections.append(
                        Detection(
                            bbox=(x1, y1, x2, y2),
                            class_id=cls_id,
                            class_name=det_name,
                            entity_class=ent_class,
                            confidence=conf
                        )
                    )

        logger.info(
            "Helmet detection: %d detections found on full frame.",
            len(all_detections),
        )
        return all_detections

    def detect_plates(
        self,
        image: np.ndarray,
        vehicle_detections: Optional[List[Detection]] = None,
    ) -> List[Detection]:
        """Run license-plate detection on the full frame.

        Args:
            image: BGR numpy array of shape ``(H, W, C)``.
            vehicle_detections: Optional list of previously detected entities in the frame.

        Returns:
            List of ``Detection`` objects for detected plates.
        """
        if image is None or image.size == 0:
            logger.warning("detect_plates() received empty image.")
            return []

        self._load_plate_model()
        if self._plate_model is None:
            logger.error(
                "Plate model not loaded. Download with: python models/download_hf_models.py"
            )
            return []


        image = self._resize_if_needed(image)
        cfg = self._settings.plate_detector

        try:
            results = self._plate_model(
                image,
                conf=cfg.confidence_threshold,
                iou=cfg.iou_threshold,
                imgsz=cfg.image_size,
                verbose=False,
                device=cfg.device,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Plate inference failed: %s", exc)
            return []

        raw_detections = self._parse_results(results, image=image)
        detections = []

        # Keep only license plate detections from the HuggingFace model
        # keremberke/yolov8s-license-plate-detection: class 0 = 'license-plate'
        for det in raw_detections:
            if "plate" in det.class_name.lower() or "license" in det.class_name.lower():
                det.entity_class = EntityClass.LICENSE_PLATE
                det.class_name = "license_plate"
                detections.append(det)

        # Fallback to OCR text detection on foreground vehicles if they have no YOLO plate detected
        if vehicle_detections is not None and len(vehicle_detections) > 0:
            logger.info("Running OCR fallback on vehicles missing YOLO plate detections...")
            try:
                import easyocr
                import re
                import torch
                
                use_gpu = torch.cuda.is_available() and self._settings.vehicle_detector.device != "cpu"
                reader = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
                
                img_h, img_w = image.shape[:2]
                total_area = img_h * img_w
                
                for v_det in vehicle_detections:
                    if v_det.entity_class is not None and v_det.entity_class.value in ["car", "motorcycle", "auto_rickshaw", "bus", "truck", "tempo"]:
                        vx1, vy1, vx2, vy2 = v_det.bbox
                        v_w = vx2 - vx1
                        v_h = vy2 - vy1
                        v_area = v_w * v_h
                        
                        # Only check foreground vehicles (area >= 3% of total image)
                        if (v_area / total_area) < 0.03:
                            continue
                            
                        # Check if this vehicle already has a YOLO plate
                        has_yolo_plate = False
                        for p_det in detections:
                            px1, py1, px2, py2 = p_det.bbox
                            pcx = (px1 + px2) / 2
                            pcy = (py1 + py2) / 2
                            if vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2:
                                has_yolo_plate = True
                                break
                        
                        if has_yolo_plate:
                            continue

                        # Crop the vehicle region
                        vx1_c, vy1_c = max(0, int(vx1)), max(0, int(vy1))
                        vx2_c, vy2_c = min(img_w, int(vx2)), min(img_h, int(vy2))
                        if vx2_c <= vx1_c or vy2_c <= vy1_c:
                            continue
                            
                        v_crop = image[vy1_c:vy2_c, vx1_c:vx2_c]
                        
                        # Preprocess crop by scaling up 4x for EasyOCR to detect mudguard plate
                        gray = cv2.cvtColor(v_crop, cv2.COLOR_BGR2GRAY)
                        resized = cv2.resize(gray, (0, 0), fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
                        
                        ocr_results = reader.readtext(resized, detail=1)
                        for bbox, text, conf in ocr_results:
                            cleaned = re.sub(r'[\s\-\.\,]', '', text.upper().strip())
                            
                            # Clean/correct plate prefix text first using a quick preview correction
                            if re.match(r'^[1IUTDUD09O]{2}H\d+$', cleaned):
                                cleaned = "TDH" + cleaned[3:]
                                
                            has_letter = any(c.isalpha() for c in cleaned)
                            has_digit = any(c.isdigit() for c in cleaned)
                            
                            if len(cleaned) >= 4 and has_letter and has_digit and conf >= 0.15:
                                # Translate coordinates from 4x scaled crop back to original image
                                x0_crop = min(bbox[0][0], bbox[3][0]) / 4.0
                                y0_crop = min(bbox[0][1], bbox[1][1]) / 4.0
                                x1_crop = max(bbox[1][0], bbox[2][0]) / 4.0
                                y1_crop = max(bbox[2][1], bbox[3][1]) / 4.0
                                
                                tx1 = int(vx1_c + x0_crop)
                                ty1 = int(vy1_c + y0_crop)
                                tx2 = int(vx1_c + x1_crop)
                                ty2 = int(vy1_c + y1_crop)
                                
                                logger.info(
                                    "OCR fallback found plate candidate: text='%s' (conf=%.2f) at absolute bbox=(%d,%d,%d,%d)",
                                    cleaned, conf, tx1, ty1, tx2, ty2
                                )
                                
                                detections.append(
                                    Detection(
                                        bbox=(tx1, ty1, tx2, ty2),
                                        class_id=0,
                                        class_name="license_plate",
                                        entity_class=EntityClass.LICENSE_PLATE,
                                        confidence=conf,
                                    )
                                )
                                # Break after finding the first plate on this vehicle to avoid duplicates
                                break
            except Exception as ocr_err:
                logger.error("OCR fallback failed: %s", ocr_err)

        logger.info("Plate detection: %d plates found.", len(detections))
        return detections

    def detect_seatbelts(
        self,
        image: np.ndarray,
        person_crops: List[Tuple[int, int, int, int]],
    ) -> List[Detection]:
        """Run seatbelt heuristic on cropped person regions."""
        if image is None or image.size == 0:
            return []

        import math
        import cv2
        detections = []
        for crop_box in person_crops:
            x1, y1, x2, y2 = crop_box
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue

            # Narrow down the horizontal crop to the center 55% of the person's bounding box
            # to filter out window frame, dashboard, steering wheel, and background clutter
            person_w = x2 - x1
            center_x = (x1 + x2) // 2
            new_w = int(person_w * 0.55)
            x1_crop = max(x1, center_x - new_w // 2)
            x2_crop = min(x2, center_x + new_w // 2)

            # Crop specifically to the torso region (height: 30% to 85% of person height)
            person_h = y2 - y1
            y1_torso = y1 + int(person_h * 0.30)
            y2_torso = y1 + int(person_h * 0.85)

            # Ensure coordinates are within image bounds
            y1_torso = max(y1, min(y2, y1_torso))
            y2_torso = max(y1, min(y2, y2_torso))

            if y2_torso <= y1_torso:
                y1_torso, y2_torso = y1, y2

            # Crop the torso region
            crop = image[y1_torso:y2_torso, x1_crop:x2_crop]
            if crop.size == 0:
                continue

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            # Enhance contrast for dark/light seatbelt straps
            gray_eq = cv2.equalizeHist(gray)
            blurred = cv2.GaussianBlur(gray_eq, (3, 3), 0)
            edges = cv2.Canny(blurred, 30, 100)
            
            # Dynamically scale minimum Hough line length based on torso crop height
            min_line_len = max(10, int(crop.shape[0] * 0.25))
            
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=min_line_len, maxLineGap=15)
            
            has_seatbelt = False
            if lines is not None:
                for line in lines:
                    lx1, ly1, lx2, ly2 = line[0]
                    angle = math.degrees(math.atan2(ly2 - ly1, lx2 - lx1))
                    if 20 < abs(angle) < 70:
                        has_seatbelt = True
                        break
            
            if not has_seatbelt:
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=0,
                        class_name="no_seatbelt",
                        entity_class=EntityClass.NO_SEATBELT,
                        confidence=0.75,
                    )
                )
            else:
                 detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        class_id=1,
                        class_name="seatbelt",
                        entity_class=EntityClass.SEATBELT,
                        confidence=0.75,
                    )
                )

        logger.info(
            "Seatbelt heuristic: processed %d crops.",
            len(person_crops),
        )
        return detections

    # ──────────────────────────────────────────────────────
    # Result parsing
    # ──────────────────────────────────────────────────────

    def _classify_traffic_light_color(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int]
    ) -> EntityClass:
        """Classify a traffic light detection box into Red, Yellow, or Green using HSV analysis."""
        try:
            x1, y1, x2, y2 = bbox
            h_img, w_img = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w_img, x2), min(h_img, y2)
            
            if x2 <= x1 or y2 <= y1:
                return EntityClass.TRAFFIC_LIGHT_RED  # fallback
                
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                return EntityClass.TRAFFIC_LIGHT_RED
                
            # Convert to HSV color space
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            
            # Divide crop into three vertical regions (top = Red, middle = Yellow, bottom = Green)
            h_crop = crop.shape[0]
            if h_crop < 3:
                return EntityClass.TRAFFIC_LIGHT_RED
                
            d = h_crop // 3
            top_v = v[0:d, :]
            mid_v = v[d:2*d, :]
            bot_v = v[2*d:, :]
            
            top_s = s[0:d, :]
            mid_s = s[d:2*d, :]
            bot_s = s[2*d:, :]
            
            # Calculate brightness * saturation metric to find the active glowing light
            top_intensity = np.mean(top_v.astype(float) * top_s.astype(float))
            mid_intensity = np.mean(mid_v.astype(float) * mid_s.astype(float))
            bot_intensity = np.mean(bot_v.astype(float) * bot_s.astype(float))
            
            # Select the region with the highest intensity
            max_intensity = max(top_intensity, mid_intensity, bot_intensity)
            
            if max_intensity < 100.0:  # low saturation / brightness (inactive light)
                # Fallback to checking pure brightness (V channel)
                top_brightness = np.mean(top_v)
                mid_brightness = np.mean(mid_v)
                bot_brightness = np.mean(bot_v)
                if top_brightness > mid_brightness and top_brightness > bot_brightness:
                    return EntityClass.TRAFFIC_LIGHT_RED
                elif bot_brightness > top_brightness and bot_brightness > mid_brightness:
                    return EntityClass.TRAFFIC_LIGHT_GREEN
                elif mid_brightness > top_brightness and mid_brightness > bot_brightness:
                    return EntityClass.TRAFFIC_LIGHT_YELLOW
                return EntityClass.TRAFFIC_LIGHT_RED
                
            if max_intensity == top_intensity:
                return EntityClass.TRAFFIC_LIGHT_RED
            elif max_intensity == bot_intensity:
                return EntityClass.TRAFFIC_LIGHT_GREEN
            else:
                return EntityClass.TRAFFIC_LIGHT_YELLOW
        except Exception as e:
            logger.error("Traffic light classification failed: %s", e)
            return EntityClass.TRAFFIC_LIGHT_RED  # Default fallback

    def _parse_results(
        self,
        results,
        offset: Tuple[int, int] = (0, 0),
        image: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """Convert Ultralytics results into ``Detection`` objects.

        Args:
            results: Raw results list returned by ``YOLO.__call__`` or
                ``YOLO.track``.
            offset: ``(ox, oy)`` pixel offset to add to all bounding-box
                coordinates (used when running on a crop).
            image: BGR frame to classify traffic light colors.

        Returns:
            List of ``Detection`` objects.
        """
        detections: List[Detection] = []
        ox, oy = offset

        try:
            for result in results:
                boxes = result.boxes
                if boxes is None or len(boxes) == 0:
                    continue

                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].cpu().numpy()
                    x1 = int(xyxy[0]) + ox
                    y1 = int(xyxy[1]) + oy
                    x2 = int(xyxy[2]) + ox
                    y2 = int(xyxy[3]) + oy

                    conf = float(boxes.conf[i].cpu().numpy())
                    cls_id = int(boxes.cls[i].cpu().numpy())

                    # Class name from model metadata
                    class_name = (
                        result.names.get(cls_id, f"class_{cls_id}")
                        if hasattr(result, "names") and result.names
                        else f"class_{cls_id}"
                    )

                    # Map to EntityClass (None if not traffic-relevant)
                    entity_class = COCO_TO_ENTITY.get(cls_id)
                    
                    if cls_id == 9 or class_name.lower() == "traffic light":
                        if image is not None:
                            entity_class = self._classify_traffic_light_color(image, (x1, y1, x2, y2))
                        else:
                            entity_class = EntityClass.TRAFFIC_LIGHT_RED  # fallback

                    # Apply class-specific confidence filtering:
                    cfg = self._settings.vehicle_detector
                    is_person = entity_class in [EntityClass.PERSON, EntityClass.RIDER]
                    is_light = (entity_class in (EntityClass.TRAFFIC_LIGHT_RED, EntityClass.TRAFFIC_LIGHT_GREEN, EntityClass.TRAFFIC_LIGHT_YELLOW))
                    
                    if is_person:
                        required_conf = max(0.20, cfg.confidence_threshold - 0.10)
                    elif is_light:
                        required_conf = 0.20  # Lowered to capture small/distant traffic lights
                    else:
                        required_conf = cfg.confidence_threshold
                        
                    if conf < required_conf:
                        continue

                    # Track ID (populated when model.track() is used)
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
            logger.error("Error parsing YOLO results: %s", exc)

        return detections

    def _apply_class_wise_nms(self, detections: List[Detection], iou_threshold: float = 0.45) -> List[Detection]:
        """Apply class-wise Non-Maximum Suppression to remove overlapping duplicates."""
        if not detections:
            return []

        # Group by class
        class_groups = {}
        for det in detections:
            cls = det.entity_class
            if cls not in class_groups:
                class_groups[cls] = []
            class_groups[cls].append(det)

        keep_detections = []

        for cls, dets in class_groups.items():
            if len(dets) <= 1:
                keep_detections.extend(dets)
                continue

            # Sort by confidence descending
            sorted_dets = sorted(dets, key=lambda x: x.confidence, reverse=True)
            while sorted_dets:
                best = sorted_dets.pop(0)
                keep_detections.append(best)
                
                # Filter out any overlapping detections of the same class
                remaining = []
                for d in sorted_dets:
                    x1 = max(best.bbox[0], d.bbox[0])
                    y1 = max(best.bbox[1], d.bbox[1])
                    x2 = min(best.bbox[2], d.bbox[2])
                    y2 = min(best.bbox[3], d.bbox[3])
                    
                    inter = max(0, x2 - x1) * max(0, y2 - y1)
                    if inter == 0:
                        remaining.append(d)
                        continue
                        
                    area1 = (best.bbox[2] - best.bbox[0]) * (best.bbox[3] - best.bbox[1])
                    area2 = (d.bbox[2] - d.bbox[0]) * (d.bbox[3] - d.bbox[1])
                    union = area1 + area2 - inter
                    iou = inter / union if union > 0 else 0.0
                    
                    if iou < iou_threshold:
                        remaining.append(d)
                sorted_dets = remaining

        return keep_detections

    def _merge_split_vehicles(self, detections: List[Detection]) -> List[Detection]:
        """
        Merge detections of the same vehicle class that are split (e.g. front and rear halves).
        This happens when SAHI slices a long vehicle.
        """
        # Only merge large vehicles that can be split by SAHI slicing (e.g. bus, truck, tempo).
        # Do not merge motorcycles, bicycles, auto-rickshaws or cars, as they are small and 
        # merging them creates false large boxes when they are side-by-side in traffic.
        VEHICLE_CLASSES = {EntityClass.TRUCK, EntityClass.BUS, EntityClass.TEMPO}
        
        # Filter vehicle detections
        vehicles = [d for d in detections if d.entity_class in VEHICLE_CLASSES]
        others = [d for d in detections if d.entity_class not in VEHICLE_CLASSES]
        
        if len(vehicles) <= 1:
            return detections
            
        merged_any = True
        while merged_any:
            merged_any = False
            to_remove = set()
            to_add = []
            
            for i in range(len(vehicles)):
                if i in to_remove:
                    continue
                for j in range(i + 1, len(vehicles)):
                    if j in to_remove:
                        continue
                    
                    v1 = vehicles[i]
                    v2 = vehicles[j]
                    
                    if v1.entity_class != v2.entity_class:
                        continue
                        
                    # Calculate bounding box properties
                    x1_a, y1_a, x2_a, y2_a = v1.bbox
                    x1_b, y1_b, x2_b, y2_b = v2.bbox
                    
                    h_a = y2_a - y1_a
                    h_b = y2_b - y1_b
                    
                    # 1. Vertical overlap fraction
                    y_overlap = max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
                    min_h = min(h_a, h_b)
                    v_overlap_ratio = y_overlap / min_h if min_h > 0 else 0
                    
                    # 2. Horizontal proximity/overlap
                    x_overlap = max(0, min(x2_a, x2_b) - max(x1_a, x1_b))
                    w_a = x2_a - x1_a
                    w_b = x2_b - x1_b
                    
                    horizontal_connected = False
                    if x_overlap > 0:
                        horizontal_connected = True
                    else:
                        # Distance between them
                        dist = max(x1_a, x1_b) - min(x2_a, x2_b)
                        if dist < 0.25 * max(w_a, w_b):
                            horizontal_connected = True
                            
                    # If they share high vertical overlap and are horizontally adjacent/overlapping, they are the same vehicle
                    if v_overlap_ratio > 0.5 and horizontal_connected:
                        # Merge them!
                        merged_bbox = (
                            min(x1_a, x1_b),
                            min(y1_a, y1_b),
                            max(x2_a, x2_b),
                            max(y2_a, y2_b)
                        )
                        merged_conf = max(v1.confidence, v2.confidence)
                        
                        # Create new merged detection
                        merged_det = Detection(
                            bbox=merged_bbox,
                            class_id=v1.class_id,
                            class_name=v1.class_name,
                            entity_class=v1.entity_class,
                            confidence=merged_conf,
                            track_id=v1.track_id or v2.track_id
                        )
                        
                        to_remove.add(i)
                        to_remove.add(j)
                        to_add.append(merged_det)
                        merged_any = True
                        break
                if merged_any:
                    # Rebuild vehicles list and loop again
                    vehicles = [v for idx, v in enumerate(vehicles) if idx not in to_remove] + to_add
                    break
                    
        return vehicles + others
