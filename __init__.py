# TrafficSentinel AI
# Automated Traffic Violation Detection & Classification System
# Gridlock Hackathon 2.0 — Flipkart × BTP

"""
TrafficSentinel AI uses a novel Hierarchical Scene Understanding Pipeline (HSUP)
to detect, classify, and document traffic violations from surveillance imagery.

Architecture:
    Layer 1: Adaptive Scene Conditioning (degradation-aware preprocessing)
    Layer 2: Multi-Scale Entity Detection + Tracking (YOLOv11 + SAHI + BoT-SORT)
    Layer 3: Scene Graph Construction (spatial relationship reasoning)
    Layer 4: Violation Reasoning Engine (rule + ML hybrid)
    Layer 5: Evidence Generation + e-Challan (legal-grade output)
"""

__version__ = "1.0.0"
__author__ = "TrafficSentinel Team"
