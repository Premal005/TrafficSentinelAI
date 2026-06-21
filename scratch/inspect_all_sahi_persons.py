import sys
from pathlib import Path
import cv2

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass
from core.entity_detector import EntityDetector
from core.scene_graph import is_four_wheeler_occupant, compute_overlap_ratio

def main():
    image_path = PROJECT_ROOT / "data" / "sample_images" / "2. seatbelt.png"
    if not image_path.exists():
        print(f"Error: {image_path} does not exist!")
        return

    image = cv2.imread(str(image_path))
    
    SETTINGS.sahi.enabled = True
    detector = EntityDetector(SETTINGS)
    
    # We want to get all raw detections before filtering
    print("Running detector.detect(..., use_sahi=True)...")
    detections = detector.detect(image, use_sahi=True)
    
    print("\n--- All Post-Processed Detections ---")
    for idx, det in enumerate(detections):
        print(f"{idx}: Class={det.class_name} ({det.entity_class}), Conf={det.confidence:.3f}, BBox={det.bbox}")
        
    print("\n--- Testing Occupant Association on all PERSON/RIDER detections ---")
    persons = [d for d in detections if d.entity_class in (EntityClass.PERSON, EntityClass.RIDER)]
    cars = [d for d in detections if d.entity_class == EntityClass.CAR]
    
    for p_idx, p in enumerate(persons):
        for c_idx, c in enumerate(cars):
            px1, py1, px2, py2 = p.bbox
            vx1, vy1, vx2, vy2 = c.bbox
            p_h = py2 - py1
            p_w = px2 - px1
            v_h = vy2 - vy1
            v_w = vx2 - vx1
            
            overlap = compute_overlap_ratio(p.bbox, c.bbox)
            h_ratio = p_h / v_h if v_h > 0 else 0
            w_ratio = p_w / v_w if v_w > 0 else 0
            
            # Check conditions
            cond_overlap = overlap >= 0.40
            cond_h = 0.15 <= h_ratio <= 0.85
            cond_w = 0.10 <= w_ratio <= 0.65
            cond_y1 = vy1 - 0.35 * v_h <= py1 <= vy1 + 0.65 * v_h
            cond_y2 = vy1 + 0.05 * v_h <= py2 <= vy1 + 0.95 * v_h
            cond_x = vx1 - 0.15 * v_w <= px1 and px2 <= vx2 + 0.15 * v_w
            
            is_occ = is_four_wheeler_occupant(p.bbox, c.bbox)
            print(f"Person {p.bbox} vs Car {c.bbox}:")
            print(f"  Overlap={overlap:.3f} (ok: {cond_overlap})")
            print(f"  Height ratio={h_ratio:.3f} (ok: {cond_h})")
            print(f"  Width ratio={w_ratio:.3f} (ok: {cond_w})")
            print(f"  Y1 pos check: {vy1 - 0.35 * v_h:.1f} <= {py1} <= {vy1 + 0.65 * v_h:.1f} (ok: {cond_y1})")
            print(f"  Y2 pos check: {vy1 + 0.05 * v_h:.1f} <= {py2} <= {vy1 + 0.95 * v_h:.1f} (ok: {cond_y2})")
            print(f"  X pos check: {vx1 - 0.15 * v_w:.1f} <= {px1} and {px2} <= {vx2 + 0.15 * v_w:.1f} (ok: {cond_x})")
            print(f"  Result matches = {is_occ}")

if __name__ == "__main__":
    main()
