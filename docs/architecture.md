# 🚦 TrafficSentinel AI — System Architecture

This document details the engineering architecture, data flows, and layer-by-layer design of **TrafficSentinel AI**'s **Hierarchical Scene Understanding Pipeline (HSUP)**.

---

## 1. Pipeline Overview

Traditional traffic systems run isolated classifiers directly on detections, which fails in crowded, low-light, or complex environments (such as Indian cities). TrafficSentinel AI solves this by constructing an explicit **Scene Graph** that models spatial and temporal relationships.

```
       Input Frame (BGR)
              │
              ▼
   ┌──────────────────────┐
   │ Layer 1: Conditioner │ ◄── Degradation Classification & CLAHE/Gamma Correction
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Layer 2: Detection   │ ◄── YOLOv11 + SAHI Sliced Inference
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Layer 3: Scene Graph │ ◄── Relational Spatial Modeling (RIDES, HAS_PLATE)
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Layer 4: Reasoning   │ ◄── Violation Heuristics & Registry Toggles
   └──────────────────────┘
              │
              ▼
   ┌──────────────────────┐
   │ Layer 5: Evidence    │ ◄── EasyOCR plate extraction & PDF Challan Generation
   └──────────────────────┘
```

---

## 2. Directory Layout & Modules

The codebase is organized into modular directories ensuring a clean separation of concerns:

```
flipkart2r/
├── app.py                              # Streamlit dashboard shell & configuration
├── requirements.txt                    # Python dependencies
├── run_dashboard.bat                   # Quick-start script for Streamlit
│
├── config/
│   ├── settings.py                     # Global configurations & thresholds
│   └── indian_vehicle_taxonomy.py      # State codes & license plate structures
│
├── core/
│   ├── scene_conditioner.py            # Layer 1: Preprocessing & enhancement
│   ├── entity_detector.py              # Layer 2: YOLOv11 + SAHI detection
│   ├── multi_tracker.py                # Layer 2: BoT-SORT tracker
│   ├── scene_graph.py                  # Layer 3: Relation-graph construction
│   ├── violation_engine.py             # Layer 4: Rules coordinator
│   ├── plate_recognizer.py             # Layer 5: EasyOCR plate text reading
│   └── evidence_generator.py           # Layer 5: PDF challan compilation
│
├── violations/
│   ├── __init__.py                     # Package init (autosubclass registry)
│   ├── base.py                         # AbstractBaseClass detector contract
│   ├── helmet.py                       # Helmet non-compliance check
│   ├── seatbelt.py                     # Front-row seatbelt checks
│   ├── triple_riding.py                # Triple passenger check
│   ├── wrong_side.py                   # Driving flow direction check
│   ├── stop_line.py                    # Stop line traversal checks
│   ├── red_light.py                    # Traffic light state violation checks
│   └── illegal_parking.py              # Restrictive parking checks
│
├── utils/
│   ├── visualization.py                # Drawing overlay utilities
│   ├── indian_plate_validator.py       # Regex format matching & text correction
│   └── metrics.py                      # IoU calculations & evaluation benchmarks
│
└── tests/
    ├── __init__.py
    └── test_pipeline.py                # End-to-end integration unittests
```

---

## 3. Data Flow & Layer Details

### Layer 1: Adaptive Scene Conditioner
- **Purpose**: Maximizes downstream accuracy in adverse conditions (rain, low light, headlights glare).
- **Logic**: Classifies degradation category and applies targeted enhancement:
  - `LOW_LIGHT`: Gamma correction ($\gamma = 0.40$) + CLAHE (clip limit = 3.0).
  - `RAIN`: Guided filter dehazing.
  - `BLUR`: Laplacian sharpening.
  - `GLARE`: Flare attenuation and local contrast stretching.

### Layer 2: Multi-Scale Entity Detection & Tracking
- **Primary Model**: YOLOv11m for vehicle/person detection.
- **Distant/Small Objects**: Integrates **SAHI (Slicing Aided Hyper Inference)**, slicing the frame into overlapping $512\times512$ patches before merging using Non-Maximum Suppression (NMS).
- **Secondary Heads**: Lazy-loaded YOLOv11s models fine-tuned on:
  - Helmet vs. No-Helmet classification on person crops.
  - License Plate localization.
- **Tracker**: BoT-SORT tracks object identities across frames, providing motion trajectory coordinates used to estimate speed and direction vectors.

### Layer 3: Relational Scene Graph
- **Data Structure**: Constructed on each frame as a directed graph $G = (V, E)$, where:
  - $V$ represents `SceneNode` entities (Vehicles, Persons, Plates, Helmets).
  - $E$ represents `SceneEdge` relationships (`RIDES`, `DRIVES`, `WEARS`, `NOT_WEARS`, `HAS_PLATE`).
- **Association Heuristics**:
  - `RIDES` edge: Calculated using a bounding box vertical stack overlap threshold (Intersection-over-Union $\ge 0.15$ with the person center above the vehicle).
  - `WEARS` / `NOT_WEARS` edge: Focuses on the top 30% of the rider's bounding box (head region) and matches with helmet/no-helmet detections.
  - `HAS_PLATE` edge: Bounding box containment matching plate detections inside or near the lower bounding region of vehicles.

### Layer 4: Violation Reasoning Engine
- **Contract**: Individual violation classes subclass `BaseViolationDetector` and implement `.detect(scene_graph, frame)`.
- **Registry Pattern**: Concrete detectors are imported on boot and automatically register themselves in a global mapping. The `ViolationEngine` loops through this registry, executes the checks, and normalizes violation metadata into a standard list of `ViolationRecord` instances.

### Layer 5: Evidence & e-Challan Generator
- **Plate Text Extraction**: Crops license plates, scales them by $2.5\times$ using bicubic interpolation, grayscales, and runs EasyOCR character extraction. The raw output is corrected by the position-aware validator (e.g. converting `O` to `0` in numeric columns).
- **Notice Creation**: Compiles all metadata into a JSON packet, generates a secure Parivahan payment URL inside a QR code, draws crops of the vehicle/plate on the PDF, and outputs a formatted PDF e-Challan slip ready for ASTraM.
