import sys
from pathlib import Path
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass
from core.entity_detector import EntityDetector, Detection
from core.scene_graph import SceneGraph, is_four_wheeler_occupant

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    image = cv2.imread(str(image_path))
    detector = EntityDetector(SETTINGS)
    
    # 1. Run standard detections
    detections = detector.detect(image, use_sahi=False)
    
    # 2. Inject candidate driver box
    driver_box = (630, 320, 840, 590)  # candidate driver box
    driver_det = Detection(
        bbox=driver_box,
        class_id=0,
        class_name="person",
        entity_class=EntityClass.PERSON,
        confidence=0.88
    )
    detections.append(driver_det)
    
    # 3. Re-run seatbelt heuristic on the crop to see what it detects
    seatbelt_dets = detector.detect_seatbelts(image, [driver_box])
    print("\n--- Seatbelt Detection on Injected Driver ---")
    for det in seatbelt_dets:
        print(f"Class={det.class_name}, Conf={det.confidence:.3f}, BBox={det.bbox}")
        
    # 4. Check scene graph association
    sg = SceneGraph(SETTINGS)
    sg.build(detections)
    
    print("\n--- Driver Nodes in Scene Graph ---")
    driver_node = None
    for nid, node in sg.nodes.items():
        if node.entity_class == EntityClass.PERSON and node.bbox == driver_box:
            driver_node = node
            print(f"Found driver node: {nid}, BBox={node.bbox}")
            
    if driver_node:
        # Check drives edge
        drives_edges = [e for e in sg.edges if e.source_id == driver_node.node_id and e.relation == "DRIVES"]
        print(f"Drives edges for driver: {drives_edges}")
        
        # Check seatbelt status
        seatbelt_edges = [e for e in sg.edges if e.source_id == driver_node.node_id and e.relation in ("WEARS", "NOT_WEARS")]
        print(f"Seatbelt edges for driver: {seatbelt_edges}")
        print(f"Seatbelt attribute status: {driver_node.attributes.get('seatbelt_status')}")

if __name__ == "__main__":
    main()
