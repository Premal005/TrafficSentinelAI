"""
End-to-End Integration Test for HSUP Pipeline.

Runs a complete smoke test of the TrafficSentinel AI pipeline:
1. Scene Conditioner (Layer 1)
2. Entity Detector (Layer 2)
3. Multi-Tracker (Layer 2)
4. Scene Graph Builder (Layer 3)
5. Violation Engine (Layer 4)
6. Evidence Generator & PDF Challan (Layer 5)
"""

import os
import sys
import unittest
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import cv2


# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import SETTINGS, EntityClass, ViolationType
from core.scene_conditioner import SceneConditioner
from core.entity_detector import EntityDetector, Detection
from core.multi_tracker import MultiTracker
from core.scene_graph import SceneGraph, SceneNode, SceneEdge, RIDES, DRIVES, WEARS, NOT_WEARS, HAS_PLATE
from core.violation_engine import ViolationEngine, ViolationRecord
from core.plate_recognizer import PlateRecognizer
from core.evidence_generator import EvidenceGenerator


class TestHSUPPipeline(unittest.TestCase):
    """
    Smoke test checking the integration and data flow across all HSUP layers.
    """

    def setUp(self) -> None:
        # Create a dummy image for testing (640x640 RGB gradient)
        self.test_img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_img, (0, 0), (640, 320), (100, 50, 50), -1) # Mock road/sky division
        cv2.rectangle(self.test_img, (0, 320), (640, 640), (40, 40, 40), -1) # Mock road surface

    def test_layer1_scene_conditioner(self) -> None:
        """Test Layer 1: Scene conditioning."""
        conditioner = SceneConditioner(SETTINGS)
        # 1. Low light enhancement
        cond_res = conditioner.condition(self.test_img)
        enhanced = cond_res.enhanced_image
        deg_type = cond_res.degradation_type.value
        quality = cond_res.quality_score
        self.assertEqual(enhanced.shape, self.test_img.shape)

        self.assertIsInstance(deg_type, str)
        self.assertGreaterEqual(quality, 0.0)
        self.assertLessEqual(quality, 1.0)
        print(f"Layer 1 Test: Detected degradation = {deg_type}, Quality = {quality:.2f}")

    def test_layer2_entity_detector(self) -> None:
        """Test Layer 2: Detection and Model Loading (fails gracefully if no weights)."""
        detector = EntityDetector(SETTINGS)
        # Should execute without crashing even if weights don't exist yet
        dets = detector.detect(self.test_img, use_sahi=False)
        self.assertIsInstance(dets, list)
        print(f"Layer 2 Test: Direct detection completed. Mapped detections count: {len(dets)}")

    def test_layer2_multi_tracker(self) -> None:
        """Test Layer 2: Tracking trajectory updates."""
        tracker = MultiTracker(SETTINGS)
        
        # Mock some detections with track IDs
        dets = [
            Detection(bbox=(100, 200, 180, 280), class_id=2, class_name="car", entity_class=EntityClass.CAR, confidence=0.85, track_id=1),
            Detection(bbox=(300, 250, 350, 350), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.90, track_id=2),
        ]
        
        tracker.update(dets, frame_num=1)
        self.assertEqual(len(tracker._trajectories), 2)
        
        # Test trajectory fetching
        traj1 = tracker.get_trajectory(1)
        self.assertEqual(len(traj1), 1)
        self.assertAlmostEqual(traj1[0][0], 140.0) # Center x
        self.assertAlmostEqual(traj1[0][1], 240.0) # Center y
        print("Layer 2 Tracker Test: Trajectory updated successfully.")

    def test_layer3_scene_graph(self) -> None:
        """Test Layer 3: Building spatial relationships."""
        sg = SceneGraph(SETTINGS)
        
        # Construct mock detections mimicking:
        # - A motorcycle
        # - A person riding the motorcycle
        # - A license plate on the motorcycle
        # - The rider wearing NO helmet
        dets = [
            Detection(bbox=(200, 300, 300, 450), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.92, track_id=5),
            Detection(bbox=(210, 250, 290, 380), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.88, track_id=6),
            Detection(bbox=(240, 255, 270, 290), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.82),
            Detection(bbox=(230, 420, 280, 450), class_id=15, class_name="license_plate", entity_class=EntityClass.LICENSE_PLATE, confidence=0.87),
        ]
        
        sg.build(dets)
        
        # Check nodes are created
        self.assertGreaterEqual(len(sg.nodes), 4)
        
        # Check relations
        rides_relations = sg.query(relation=RIDES)
        self.assertGreaterEqual(len(rides_relations), 1, "Rides relation was not associated!")
        
        helmet_status = None
        for node_id, node in sg.nodes.items():
            if node.entity_class == EntityClass.PERSON:
                helmet_status = sg.get_helmet_status(node_id)
        
        self.assertEqual(helmet_status, "no_helmet", "Helmet status should be no_helmet!")
        print("Layer 3 Test: Scene Graph built and spatial relations validated.")

    def test_layer4_violation_engine(self) -> None:
        """Test Layer 4: Violation reasoning on Scene Graph."""
        sg = SceneGraph(SETTINGS)
        # Mock scene graph containing helmet violation
        dets = [
            Detection(bbox=(200, 300, 300, 450), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.95, track_id=5),
            Detection(bbox=(210, 250, 290, 380), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.91, track_id=6),
            Detection(bbox=(240, 255, 270, 290), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.89),
        ]
        sg.build(dets)
        
        engine = ViolationEngine(SETTINGS)
        violations = engine.analyze(sg, self.test_img)
        
        self.assertGreaterEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, ViolationType.HELMET_NON_COMPLIANCE)
        print(f"Layer 4 Test: Violation reasoning completed. Found violation: {violations[0].description}")

    def test_missing_plate_violation(self) -> None:
        """Test Missing License Plate detection on Scene Graph."""
        sg = SceneGraph(SETTINGS)
        # Mock scene graph with a car but NO license plate detection
        dets = [
            Detection(bbox=(100, 100, 300, 300), class_id=2, class_name="car", entity_class=EntityClass.CAR, confidence=0.95, track_id=10),
        ]
        sg.build(dets)
        
        engine = ViolationEngine(SETTINGS)
        violations = engine.analyze(sg, self.test_img)
        
        self.assertGreaterEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, ViolationType.MISSING_LICENSE_PLATE)
        print(f"Missing Plate Test: Found violation: {violations[0].description}")

    def test_layer5_evidence_and_pdf(self) -> None:
        """Test Layer 5: Evidence capture and e-Challan generation."""
        sg = SceneGraph(SETTINGS)
        # Set up a mock motorcycle + rider without helmet + plate
        dets = [
            Detection(bbox=(200, 300, 300, 450), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.95, track_id=5),
            Detection(bbox=(210, 250, 290, 380), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.91, track_id=6),
            Detection(bbox=(240, 255, 270, 290), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.89),
            Detection(bbox=(235, 420, 280, 445), class_id=15, class_name="license_plate", entity_class=EntityClass.LICENSE_PLATE, confidence=0.90)
        ]
        sg.build(dets)
        
        # Force set plate text in scene graph node attributes
        for node in sg.nodes.values():
            if node.entity_class == EntityClass.LICENSE_PLATE:
                node.attributes["plate_text"] = "KA-05-MN-1234"
                node.attributes["plate_state"] = "Karnataka"
                node.attributes["plate_type"] = "private"

                
        engine = ViolationEngine(SETTINGS)
        violations = engine.analyze(sg, self.test_img)
        self.assertGreaterEqual(len(violations), 1)
        
        generator = EvidenceGenerator(SETTINGS)
        annotated_img = self.setUp_annotated_frame(violations)
        
        # Generate JSON packet
        packet = generator.generate_evidence_packet(
            self.test_img, annotated_img, violations[0], sg, camera_id="CAM-TEST-99"
        )
        self.assertEqual(packet["vehicle"]["plate_number"], "KA-05-MN-1234")
        self.assertEqual(packet["fine_amount"], 1000)
        
        # Generate PDF Challan
        pdf_path = generator.generate_challan_pdf(packet)
        self.assertTrue(os.path.exists(pdf_path))
        print(f"Layer 5 Test: Evidence and PDF Challan generated at {pdf_path}")
        
        # Cleanup generated test evidence files
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            # Remove directory if empty
            os.rmdir(os.path.dirname(pdf_path))
        except Exception:
            pass

    def test_police_dispatch_evidence_and_pdf(self) -> None:
        """Test Layer 5: Evidence capture and e-Challan generation for missing plate/police dispatch."""
        sg = SceneGraph(SETTINGS)
        # Mock scene graph containing a car but NO license plate detection
        dets = [
            Detection(bbox=(100, 100, 300, 300), class_id=2, class_name="car", entity_class=EntityClass.CAR, confidence=0.95, track_id=10),
        ]
        sg.build(dets)
        
        engine = ViolationEngine(SETTINGS)
        violations = engine.analyze(sg, self.test_img)
        self.assertGreaterEqual(len(violations), 1)
        self.assertEqual(violations[0].violation_type, ViolationType.MISSING_LICENSE_PLATE)
        
        generator = EvidenceGenerator(SETTINGS)
        annotated_img = self.setUp_annotated_frame(violations)
        
        # Generate JSON packet (triggers police dispatch logic)
        packet = generator.generate_evidence_packet(
            self.test_img, annotated_img, violations[0], sg, camera_id="CAM-JCN-01"
        )
        self.assertTrue("police_dispatch" in packet)
        self.assertEqual(packet["police_dispatch"]["patrol_unit"], "MG Road Patrol 2")
        
        # Generate PDF Challan with police dispatch
        pdf_path = generator.generate_challan_pdf(packet)
        self.assertTrue(os.path.exists(pdf_path))
        print(f"Police Dispatch PDF Challan generated at {pdf_path}")
        
        # Cleanup generated test evidence files
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            # Remove directory if empty
            os.rmdir(os.path.dirname(pdf_path))
        except Exception:
            pass

    def test_aggregated_multiple_violations(self) -> None:
        """Test Layer 5: Consolidation of multiple violations for the same vehicle in one notice."""
        sg = SceneGraph(SETTINGS)
        # Mock scene graph containing a car and plate
        dets = [
            Detection(bbox=(100, 100, 300, 300), class_id=2, class_name="car", entity_class=EntityClass.CAR, confidence=0.95, track_id=10),
            Detection(bbox=(180, 260, 220, 290), class_id=15, class_name="license_plate", entity_class=EntityClass.LICENSE_PLATE, confidence=0.91)
        ]
        sg.build(dets)
        
        # Explicitly set plate details
        for node in sg.nodes.values():
            if node.entity_class == EntityClass.LICENSE_PLATE:
                node.attributes["plate_text"] = "KA-01-AB-1234"
                node.attributes["plate_state"] = "Karnataka"
                node.attributes["plate_type"] = "private"

        # Mock two different violation records for the same vehicle node ID
        v1 = ViolationRecord(
            violation_id="VIO-TEST-GRP-001",
            violation_type=ViolationType.STOP_LINE_VIOLATION,
            confidence=0.94,
            involved_nodes=["car_10", "license_plate_0"],
            description="Stop-Line Violation: Crossed stop line",
            bbox=(100, 100, 300, 300),
            timestamp="2026-06-21T05:00:00Z",
            metadata={}
        )
        
        v2 = ViolationRecord(
            violation_id="VIO-TEST-GRP-001",
            violation_type=ViolationType.RED_LIGHT_VIOLATION,
            confidence=0.88,
            involved_nodes=["car_10", "license_plate_0"],
            description="Red-Light Violation: Ran red light",
            bbox=(100, 100, 300, 300),
            timestamp="2026-06-21T05:00:00Z",
            metadata={}
        )
        
        generator = EvidenceGenerator(SETTINGS)
        # Generate consolidated packet
        packet = generator.generate_evidence_packet(
            self.test_img, self.test_img, [v1, v2], sg, camera_id="CAM-JCN-02"
        )
        
        # Verify aggregates
        self.assertEqual(packet["vehicle"]["plate_number"], "KA-01-AB-1234")
        self.assertEqual(packet["violation_type"], "multiple")
        self.assertIn("Stop-Line Violation", packet["violation_name"])
        self.assertIn("Red-Light Violation", packet["violation_name"])
        # Fine sum: 500 (stop line) + 1000 (red light) = 1500
        self.assertEqual(packet["fine_amount"], 1500)
        self.assertEqual(len(packet["violations_list"]), 2)
        
        # Verify PDF compiles successfully with the table layout
        pdf_path = generator.generate_challan_pdf(packet)
        self.assertTrue(os.path.exists(pdf_path))
        print(f"Aggregated PDF Challan generated at {pdf_path}")
        
        # Cleanup generated test evidence files
        try:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            # Remove directory if empty
            os.rmdir(os.path.dirname(pdf_path))
        except Exception:
            pass

    def test_driver_only_helmet_compliance(self) -> None:
        """Test driver-only helmet compliance toggle."""
        # Save original setting
        orig_setting = SETTINGS.pillion_helmet_required
        
        try:
            # Construct scene graph
            sg = SceneGraph(SETTINGS)
            
            # Detections representing:
            # - Motorcycle (moving EAST, facing right)
            # - Rider 1 (Driver, in front/right, center x = 270)
            # - Rider 2 (Pillion, in back/left, center x = 170)
            # - Helmet on Rider 1 (Driver)
            # - No Helmet on Rider 2 (Pillion)
            dets = [
                Detection(bbox=(100, 300, 350, 450), class_id=3, class_name="motorcycle", entity_class=EntityClass.MOTORCYCLE, confidence=0.95, track_id=5),
                Detection(bbox=(230, 250, 310, 380), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.91, track_id=6), # Driver
                Detection(bbox=(130, 250, 210, 380), class_id=0, class_name="person", entity_class=EntityClass.PERSON, confidence=0.91, track_id=7), # Pillion
                Detection(bbox=(255, 250, 285, 280), class_id=9, class_name="helmet", entity_class=EntityClass.HELMET, confidence=0.88), # Driver wears helmet
                Detection(bbox=(155, 250, 185, 280), class_id=10, class_name="no_helmet", entity_class=EntityClass.NO_HELMET, confidence=0.89), # Pillion has no helmet
            ]
            
            # Build scene graph
            sg.build(dets)
            
            # Explicitly set direction to EAST so Rider 1 is identified as driver (max x)
            for node in sg.nodes.values():
                if node.entity_class == EntityClass.MOTORCYCLE:
                    node.attributes["direction"] = "EAST"
            
            # 1. Pillion helmet REQUIRED: Should trigger 1 violation (pillion lacks helmet)
            SETTINGS.pillion_helmet_required = True
            engine = ViolationEngine(SETTINGS)
            violations_all = engine.analyze(sg, self.test_img)
            
            # Check we got a helmet violation for the pillion
            helmet_vios_all = [v for v in violations_all if v.violation_type == ViolationType.HELMET_NON_COMPLIANCE]
            self.assertEqual(len(helmet_vios_all), 1)
            self.assertIn("Pillion", helmet_vios_all[0].description)
            
            # 2. Pillion helmet NOT REQUIRED: Should trigger 0 violations (driver wears helmet)
            SETTINGS.pillion_helmet_required = False
            violations_driver_only = engine.analyze(sg, self.test_img)
            
            helmet_vios_do = [v for v in violations_driver_only if v.violation_type == ViolationType.HELMET_NON_COMPLIANCE]
            self.assertEqual(len(helmet_vios_do), 0)
            
            print("Driver-Only Helmet Compliance Test: Passed successfully.")
            
        finally:
            # Restore setting
            SETTINGS.pillion_helmet_required = orig_setting

    def setUp_annotated_frame(self, violations: List[ViolationRecord]) -> np.ndarray:
        """Helper to draw basic annotations for testing."""
        annotated = self.test_img.copy()
        for v in violations:
            cv2.rectangle(annotated, v.bbox[:2], v.bbox[2:], (0, 0, 255), 2)
            cv2.putText(annotated, v.description, (v.bbox[0], v.bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return annotated


if __name__ == "__main__":
    unittest.main()
