"""
Unit test script for testing the new rider-to-rider propagation and transitive pruning heuristics.
"""

import sys
from pathlib import Path
import unittest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass, ViolationType
from core.scene_graph import SceneGraph, RIDES, WEARS, NOT_WEARS, HAS_PLATE
from core.entity_detector import Detection
from core.violation_engine import ViolationEngine

class TestTripleRidingAndPruning(unittest.TestCase):
    def test_triple_riding_with_propagation(self):
        """
        Verify that a passenger who does not directly overlap with the narrow motorcycle box
        but overlaps with the driver gets associated with the motorcycle via Pass 2,
        and that a third rider with only a head detection is synthesized.
        """
        sg = SceneGraph(SETTINGS)
        
        # BBoxes layout:
        # Motorcycle: (200, 400, 300, 500) -> Narrow, center
        # Rider 1 (Driver): (220, 300, 280, 420) -> Centered, direct overlap with motorcycle
        # Rider 2 (Pillion, sitting slightly right): (270, 300, 330, 420)
        #   -> Overlaps with Driver (Rider 1), but very minimal overlap with Motorcycle
        # Rider 3 (Occluded pillion, only head/no_helmet detected on the right):
        #   -> Head at (320, 270, 350, 300) -> lies outside original 15% span but matches expanded 65% span
        
        dets = [
            # Vehicle
            Detection(bbox=(200, 400, 300, 500), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.90),
            
            # Drivers / Riders
            Detection(bbox=(220, 300, 280, 420), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.85),
            Detection(bbox=(270, 300, 330, 420), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.80),
            
            # Heads/safety gear
            Detection(bbox=(235, 300, 265, 330), class_id=9, class_name="helmet", entity_class=EntityClass.HELMET, confidence=0.80),
            Detection(bbox=(285, 300, 315, 330), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.85),
            Detection(bbox=(320, 270, 350, 300), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.75), # Occluded third rider head
            
            # Pedestrian walking on the side (should be pruned)
            Detection(bbox=(500, 300, 550, 420), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.85),
            Detection(bbox=(515, 300, 535, 325), class_id=9, class_name="helmet", entity_class=EntityClass.HELMET, confidence=0.80),
        ]
        
        sg.build(dets)
        
        # Verify that only the relevant nodes are kept (Pedestrian and his helmet should be pruned)
        print("Active nodes in scene graph:", list(sg.nodes.keys()))
        for node_id, node in sg.nodes.items():
            # No node should be the isolated pedestrian at x >= 500
            self.assertLess(node.bbox[0], 450, f"Node {node_id} ({node.entity_class}) should have been pruned!")
            
        # Verify that Rider 2 (Pillion) is associated to the motorcycle
        motorcycle_nodes = [nid for nid, n in sg.nodes.items() if n.entity_class == EntityClass.MOTORCYCLE]
        self.assertEqual(len(motorcycle_nodes), 1)
        moto_id = motorcycle_nodes[0]
        
        riders = sg.get_riders_of(moto_id)
        # Should contain: Rider 1, Rider 2, and the synthesized Rider 3
        print("Associated riders count:", len(riders))
        self.assertEqual(len(riders), 3, "Failed to associate all 3 riders to the motorcycle!")
        
        # Run violation detector to see if triple riding is detected
        engine = ViolationEngine(SETTINGS)
        violations = engine.analyze(sg, None)
        
        triple_vios = [v for v in violations if v.violation_type == ViolationType.TRIPLE_RIDING]
        self.assertEqual(len(triple_vios), 1, "Failed to detect triple riding violation!")
        print("Violation detected successfully:", triple_vios[0].description)

if __name__ == "__main__":
    unittest.main()
