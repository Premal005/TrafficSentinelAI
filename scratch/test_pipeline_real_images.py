import os
import sys
import cv2
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path("E:/flipkart2r")
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS
from core.scene_conditioner import SceneConditioner
from core.entity_detector import EntityDetector
from core.scene_graph import SceneGraph
from core.violation_engine import ViolationEngine

def test_real_images():
    conditioner = SceneConditioner(SETTINGS)
    detector = EntityDetector(SETTINGS)
    violation_engine = ViolationEngine(SETTINGS)
    
    test_dir = PROJECT_ROOT / "data" / "test_violations"
    images = list(test_dir.glob("*.png")) + list(test_dir.glob("*.jpg"))
    
    print(f"Found {len(images)} test images in {test_dir}")
    
    for img_path in images:
        print(f"\n==========================================")
        print(f"Processing: {img_path.name}")
        print(f"==========================================")
        
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("  Error: Could not read image.")
            continue
            
        # 1. Conditioning
        cond_res = conditioner.condition(frame)
        enhanced = cond_res.enhanced_image
        print(f"  Conditioner: Degradation={cond_res.degradation_type.value}, Quality={cond_res.quality_score:.2f}")
        
        # 2. Entity Detection
        detections = detector.detect(enhanced, use_sahi=False)
        print(f"  Entity Detector: Found {len(detections)} primary detections.")
        for d in detections:
            print(f"    - {d.class_name} (conf={d.confidence:.2f}) at bbox={d.bbox}")
            
        # Get rider head crops and vehicle crops to lazy-load helmet/seatbelt models
        person_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON] if 'EntityClass' in globals() else [d for d in detections if d.class_name == "person"]
        
        # Crop riders for helmet and seatbelt checks
        if person_nodes:
            from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
            from config.settings import TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
            
            helmet_crops = []
            seatbelt_crops = []
            
            vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]
            
            for r in person_nodes:
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
                print(f"  Triggering helmet detection on {len(helmet_crops)} rider crops...")
                helmet_detections = detector.detect_helmets(enhanced, helmet_crops)
                print(f"    Found {len(helmet_detections)} helmet nodes.")
                for hd in helmet_detections:
                    print(f"      * {hd.class_name} (conf={hd.confidence:.2f}) at bbox={hd.bbox}")
                detections.extend(helmet_detections)
                
            if seatbelt_crops:
                print(f"  Triggering seatbelt detection on {len(seatbelt_crops)} driver crops...")
                seatbelt_detections = detector.detect_seatbelts(enhanced, seatbelt_crops)
                print(f"    Found {len(seatbelt_detections)} seatbelt/no_seatbelt nodes.")
                for sd in seatbelt_detections:
                    print(f"      * {sd.class_name} (conf={sd.confidence:.2f}) at bbox={sd.bbox}")
                detections.extend(seatbelt_detections)

        # Plates detection
        plate_detections = detector.detect_plates(enhanced, detections)
        print(f"  Plate Detector: Found {len(plate_detections)} plate detections.")
        for pd in plate_detections:
            print(f"    - {pd.class_name} (conf={pd.confidence:.2f}) at bbox={pd.bbox}")
        detections.extend(plate_detections)
        
        # 3. Scene Graph
        sg = SceneGraph(SETTINGS)
        sg.build(detections, frame_shape=enhanced.shape[:2])
        print(f"  Scene Graph: Built with {len(sg.nodes)} nodes, {len(sg.edges)} edges.")
        for edge in sg.edges:
            print(f"    - Edge: {edge.source_id} --({edge.relation})--> {edge.target_id} (conf={edge.confidence:.2f})")
            
        # 4. Violation Reasoning
        violations = violation_engine.analyze(sg, enhanced)
        print(f"  Violation Engine: Found {len(violations)} violation(s).")
        for v in violations:
            print(f"    - VIOLATION: {v.violation_type.value} - {v.description} (conf={v.confidence:.2f})")

if __name__ == "__main__":
    from config.settings import EntityClass # make sure EntityClass is imported in main
    test_real_images()
