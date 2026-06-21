"""
Evaluation script to test TrafficSentinel HSUP on the provided dataset.
Runs inference on Test data folders and calculates actual Precision, Recall, and F1.
Saves results to data/eval_results.json.
"""

import os
import sys
import json
import time
from pathlib import Path
import cv2
import numpy as np
import logging

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass, ViolationType
from core.scene_conditioner import SceneConditioner
from core.entity_detector import EntityDetector, Detection
from core.scene_graph import SceneGraph
from core.violation_engine import ViolationEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def run_dataset_evaluation():
    dataset_dir = PROJECT_ROOT / "Traffic Violations Analysis Dataset" / "Test data"
    if not dataset_dir.exists():
        logger.error(f"Dataset directory not found at {dataset_dir}")
        return

    logger.info("Initializing models for evaluation...")
    conditioner = SceneConditioner(SETTINGS)
    detector = EntityDetector(SETTINGS)
    violation_engine = ViolationEngine(SETTINGS)

    # We will evaluate helmet detection (helmet vs no_helmet) and overloading
    results = {
        "helmet": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0},
        "overloading": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "total": 0}
    }

    # 1. Evaluate Helmet detection
    # Folder 'no_helmet': we expect to find helmet violations (TP)
    no_helmet_dir = dataset_dir / "no_helmet"
    if no_helmet_dir.exists():
        files = (list(no_helmet_dir.glob("*.jpg")) + list(no_helmet_dir.glob("*.png")))[:25]
        logger.info(f"Evaluating {len(files)} images in 'no_helmet' folder...")
        for idx, f_path in enumerate(files):
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
            
            # Run pipeline
            cond_res = conditioner.condition(frame)
            enhanced = cond_res.enhanced_image
            detections = detector.detect(enhanced, use_sahi=False)
            
            # Crop riders and detect helmets/seatbelts based on vehicle association
            rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
            if rider_nodes:
                from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
                from config.settings import TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
                
                helmet_crops = []
                seatbelt_crops = []
                
                vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]
                
                for r in rider_nodes:
                    is_two_wheeler_rider = False
                    is_occupant = False
                    
                    for v in vehicles:
                        iou = compute_iou(r.bbox, v.bbox)
                        overlap = compute_overlap_ratio(r.bbox, v.bbox)
                        score = max(iou, overlap)
                        
                        if v.entity_class in TWO_WHEELER_CLASSES:
                            if score >= 0.15:
                                is_two_wheeler_rider = True
                        elif v.entity_class in FOUR_WHEELER_CLASSES:
                            if is_four_wheeler_occupant(r.bbox, v.bbox):
                                is_occupant = True
                    
                    if is_two_wheeler_rider:
                        helmet_crops.append(r.bbox)
                    if is_occupant:
                        seatbelt_crops.append(r.bbox)
                        
                if helmet_crops:
                    helmet_detections = detector.detect_helmets(enhanced, helmet_crops)
                    detections.extend(helmet_detections)
                    
                if seatbelt_crops:
                    seatbelt_detections = detector.detect_seatbelts(enhanced, seatbelt_crops)
                    detections.extend(seatbelt_detections)
                
            sg = SceneGraph(SETTINGS)
            sg.build(detections, frame_shape=enhanced.shape[:2])
            violations = violation_engine.analyze(sg, enhanced)
            
            has_helmet_violation = any(v.violation_type == ViolationType.HELMET_NON_COMPLIANCE for v in violations)
            
            results["helmet"]["total"] += 1
            if has_helmet_violation:
                results["helmet"]["tp"] += 1
            else:
                results["helmet"]["fn"] += 1
                
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(files)} no_helmet images")

    # Folder 'helmet': we expect NO helmet violations (TN)
    helmet_dir = dataset_dir / "helmet"
    if helmet_dir.exists():
        files = (list(helmet_dir.glob("*.jpg")) + list(helmet_dir.glob("*.png")))[:25]
        logger.info(f"Evaluating {len(files)} images in 'helmet' folder...")
        for idx, f_path in enumerate(files):
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
            
            # Run pipeline
            cond_res = conditioner.condition(frame)
            enhanced = cond_res.enhanced_image
            detections = detector.detect(enhanced, use_sahi=False)
            
            # Crop riders and detect helmets/seatbelts based on vehicle association
            rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
            if rider_nodes:
                from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
                from config.settings import TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
                
                helmet_crops = []
                seatbelt_crops = []
                
                vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]
                
                for r in rider_nodes:
                    is_two_wheeler_rider = False
                    is_occupant = False
                    
                    for v in vehicles:
                        iou = compute_iou(r.bbox, v.bbox)
                        overlap = compute_overlap_ratio(r.bbox, v.bbox)
                        score = max(iou, overlap)
                        
                        if v.entity_class in TWO_WHEELER_CLASSES:
                            if score >= 0.15:
                                is_two_wheeler_rider = True
                        elif v.entity_class in FOUR_WHEELER_CLASSES:
                            if is_four_wheeler_occupant(r.bbox, v.bbox):
                                is_occupant = True
                    
                    if is_two_wheeler_rider:
                        helmet_crops.append(r.bbox)
                    if is_occupant:
                        seatbelt_crops.append(r.bbox)
                        
                if helmet_crops:
                    helmet_detections = detector.detect_helmets(enhanced, helmet_crops)
                    detections.extend(helmet_detections)
                    
                if seatbelt_crops:
                    seatbelt_detections = detector.detect_seatbelts(enhanced, seatbelt_crops)
                    detections.extend(seatbelt_detections)
                
            sg = SceneGraph(SETTINGS)
            sg.build(detections, frame_shape=enhanced.shape[:2])
            violations = violation_engine.analyze(sg, enhanced)
            
            has_helmet_violation = any(v.violation_type == ViolationType.HELMET_NON_COMPLIANCE for v in violations)
            
            results["helmet"]["total"] += 1
            if has_helmet_violation:
                results["helmet"]["fp"] += 1
            else:
                results["helmet"]["tn"] += 1
                
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(files)} helmet images")

    # 2. Evaluate Overloading detection
    # Folder 'overloading': we expect overloading/triple riding violations (TP)
    overloading_dir = dataset_dir / "overloading"
    if overloading_dir.exists():
        files = (list(overloading_dir.glob("*.jpg")) + list(overloading_dir.glob("*.png")))[:25]
        logger.info(f"Evaluating {len(files)} images in 'overloading' folder...")
        for idx, f_path in enumerate(files):
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
            
            # Run pipeline
            cond_res = conditioner.condition(frame)
            enhanced = cond_res.enhanced_image
            detections = detector.detect(enhanced, use_sahi=False)
            
            # Crop riders and detect helmets/seatbelts based on vehicle association
            rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
            if rider_nodes:
                from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
                from config.settings import TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
                
                helmet_crops = []
                seatbelt_crops = []
                
                vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]
                
                for r in rider_nodes:
                    is_two_wheeler_rider = False
                    is_occupant = False
                    
                    for v in vehicles:
                        iou = compute_iou(r.bbox, v.bbox)
                        overlap = compute_overlap_ratio(r.bbox, v.bbox)
                        score = max(iou, overlap)
                        
                        if v.entity_class in TWO_WHEELER_CLASSES:
                            if score >= 0.15:
                                is_two_wheeler_rider = True
                        elif v.entity_class in FOUR_WHEELER_CLASSES:
                            if is_four_wheeler_occupant(r.bbox, v.bbox):
                                is_occupant = True
                    
                    if is_two_wheeler_rider:
                        helmet_crops.append(r.bbox)
                    if is_occupant:
                        seatbelt_crops.append(r.bbox)
                        
                if helmet_crops:
                    helmet_detections = detector.detect_helmets(enhanced, helmet_crops)
                    detections.extend(helmet_detections)
                    
                if seatbelt_crops:
                    seatbelt_detections = detector.detect_seatbelts(enhanced, seatbelt_crops)
                    detections.extend(seatbelt_detections)
                
            sg = SceneGraph(SETTINGS)
            sg.build(detections, frame_shape=enhanced.shape[:2])
            violations = violation_engine.analyze(sg, enhanced)
            
            has_overloading = any(v.violation_type == ViolationType.TRIPLE_RIDING for v in violations)
            
            results["overloading"]["total"] += 1
            if has_overloading:
                results["overloading"]["tp"] += 1
            else:
                results["overloading"]["fn"] += 1
                
            if (idx + 1) % 20 == 0:
                logger.info(f"Processed {idx + 1}/{len(files)} overloading images")

        # For overloading FP/TN, we use the 'helmet' and 'no_helmet' folders as negative cases (normal capacity)
        # We check how many normal images got flagged as triple riding (FP) vs ignored (TN)
        for idx, f_path in enumerate(list(helmet_dir.glob("*.jpg"))[:12] + list(no_helmet_dir.glob("*.jpg"))[:13]):
            frame = cv2.imread(str(f_path))
            if frame is None:
                continue
            
            cond_res = conditioner.condition(frame)
            enhanced = cond_res.enhanced_image
            detections = detector.detect(enhanced, use_sahi=False)
            
            sg = SceneGraph(SETTINGS)
            sg.build(detections)
            violations = violation_engine.analyze(sg, enhanced)
            
            has_overloading = any(v.violation_type == ViolationType.TRIPLE_RIDING for v in violations)
            
            if has_overloading:
                results["overloading"]["fp"] += 1
            else:
                results["overloading"]["tn"] += 1

    # Calculate final metrics
    metrics = {}
    for task_name, counts in results.items():
        tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0.0
        
        metrics[task_name] = {
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "accuracy": float(accuracy)
        }
        logger.info(f"{task_name.upper()} Metrics: Precision={precision:.2%}, Recall={recall:.2%}, F1={f1:.2%}, Accuracy={accuracy:.2%}")

    # Save to JSON
    output_path = PROJECT_ROOT / "data" / "eval_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info(f"Saved evaluation results to {output_path}")

if __name__ == "__main__":
    run_dataset_evaluation()
