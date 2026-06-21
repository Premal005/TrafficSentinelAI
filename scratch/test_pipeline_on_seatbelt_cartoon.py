import sys
from pathlib import Path
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass, ViolationType
from core.scene_conditioner import SceneConditioner
from core.entity_detector import EntityDetector
from core.scene_graph import SceneGraph, DRIVES
from core.violation_engine import ViolationEngine

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    image = cv2.imread(str(image_path))
    
    # Run full pipeline sequence
    conditioner = SceneConditioner(SETTINGS)
    detector = EntityDetector(SETTINGS)
    sg_builder = SceneGraph(SETTINGS)
    violation_engine = ViolationEngine(SETTINGS)
    
    # L1
    cond_res = conditioner.condition(image)
    enhanced = cond_res.enhanced_image
    
    # L2
    detections = detector.detect(enhanced, use_sahi=False)
    
    # Let's check for seatbelt crops
    rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
    from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
    from config.settings import FOUR_WHEELER_CLASSES
    
    vehicles = [d for d in detections if d.entity_class in FOUR_WHEELER_CLASSES]
    seatbelt_crops = []
    for r in rider_nodes:
        is_occ = False
        for v in vehicles:
            if is_four_wheeler_occupant(r.bbox, v.bbox):
                is_occ = True
                break
        if is_occ:
            seatbelt_crops.append(r.bbox)
            
    print(f"Seatbelt crops generated: {len(seatbelt_crops)}")
    
    seatbelt_dets = detector.detect_seatbelts(enhanced, seatbelt_crops)
    detections.extend(seatbelt_dets)
    
    # L3
    sg_builder.build(detections, frame_shape=enhanced.shape[:2])
    
    # Verify driver is present in nodes
    driver_node = None
    for nid, node in sg_builder.nodes.items():
        if node.entity_class == EntityClass.PERSON and node.bbox == (630, 320, 840, 590):
            driver_node = node
            print(f"Found driver node {nid}")
            
    # L4
    violations = violation_engine.analyze(sg_builder, enhanced)
    
    print("\n--- Detected Violations ---")
    seatbelt_violation = None
    for idx, v in enumerate(violations):
        print(f"{idx}: Type={v.violation_type.name}, Desc='{v.description}'")
        if v.violation_type == ViolationType.SEATBELT_NON_COMPLIANCE:
            seatbelt_violation = v
            
    if seatbelt_violation:
        print("\nSUCCESS: Seatbelt violation successfully detected!")
        print(f"  Violation ID: {seatbelt_violation.violation_id}")
        print(f"  Involved Nodes: {seatbelt_violation.involved_nodes}")
        print(f"  Reasoning Chain:")
        for step in seatbelt_violation.reasoning_chain:
            print(f"    - {step}")
    else:
        print("\nFAILURE: Seatbelt violation not detected.")

if __name__ == "__main__":
    main()
