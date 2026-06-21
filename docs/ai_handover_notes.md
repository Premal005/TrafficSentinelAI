# 🤖 AI-to-AI Handover & Architecture Briefing

Use this document as a context prompt or system briefing when handing over this codebase to another LLM or AI coding agent. It summarizes the architecture, core modules, structural modifications, and specific mathematical heuristics implemented in the project.

---

## 📋 Context & Project Hook
* **Project Name**: TrafficSentinel AI
* **Competition Context**: Gridlock Hackathon 2.0 (Flipkart × Bengaluru Traffic Police)
* **Goal**: Real-time traffic violation detection in high-density, heterogeneous mixed traffic environments.
* **Core Concept**: **Hierarchical Scene Understanding Pipeline (HSUP)**. Instead of flat classification over isolated bounding boxes (which causes false-positives on pedestrians standing near vehicles), this system builds an explicit directed spatial scene graph $G = (V, E)$ to model relationships.

---

## 🛠️ The 5-Layer HSUP Architecture

### Layer 1: Adaptive Scene Conditioning (`core/scene_conditioner.py`)
* **Purpose**: Enhance degraded CCTV frames dynamically prior to downstream detection.
* **Logic**: Classifies degradation category and applies targeted OpenCV preprocessing:
  - `LOW_LIGHT`: Gamma correction ($\gamma = 0.40$) + CLAHE (clip limit = 3.0).
  - `RAIN`: Guided filter dehazing.
  - `BLUR`: Laplacian sharpening.
  - `GLARE`: local contrast stretching.
  - Memory-intensive steps like `cv2.detailEnhance` are wrapped in exception handlers with unsharp mask fallbacks to prevent OOM crashes on resource-constrained devices.

### Layer 2: Multi-Scale Entity Detection & Tracking (`core/entity_detector.py` & `core/multi_tracker.py`)
* **Models**: YOLOv11m for vehicle/person detection, optional SAHI (Sliced Aided Hyper Inference) for small-object recall, and BoT-SORT for persistent identity tracking.
* **Specialized Heads**: Lazy-loaded YOLOv11s models for License Plate detection and Helmet/No-Helmet classification.
* **Foreground Focus filter**: Inside the `detect` method, all vehicle detections are filtered by area ratio ($A_{\text{vehicle}} / A_{\text{image}} \ge 0.03$). Pedestrians are filtered out unless they overlap with kept vehicles (overlap/IoU $\ge 0.08$) or are large foreground pedestrians themselves (height $\ge 120$px, area $\ge 12,000$px). This eliminates background noise and speeds up processing.

### Layer 3: Directed Scene Graph Builder (`core/scene_graph.py`)
* **Nodes ($V$)**: `SceneNode` dataclasses representing vehicles, persons, helmets, and plates.
* **Edges ($E$)**: `SceneEdge` relationships:
  - `RIDES`: Two-wheeler rider association.
  - `DRIVES`: Four-wheeler occupant/driver association.
  - `WEARS` / `NOT_WEARS`: Safety gear association.
  - `HAS_PLATE`: Plate-to-vehicle association.
* **Spatial Heuristics**:
  - **RIDES (Alignment check)**: A person rides a two-wheeler if 2D overlap (IoU) $\ge 0.15$, OR if they satisfy a vertical-horizontal stack: horizontal overlap $\ge 40\%$ of person width and vertical distance ratio (from bottom of person to top of vehicle, relative to person height) is within $[-0.50, +0.25]$. This links riders sitting high on the seat with near-zero 2D overlap.
  - **DRIVES (Windshield constraint)**: Driver must overlap with the four-wheeler box by $\ge 50\%$, preventing false-flagging of pedestrians standing outside the car.
  - **WEARS (Helmet region)**: Target top 30% of the rider's box (head region) and calculate overlap with helmet/no-helmet boxes.

### Layer 4: Violation Reasoning Engine (`core/violation_engine.py` & `violations/`)
* **Logic**: Uses a **Registry Pattern** where individual violation experts subclass `BaseViolationDetector` and implement `.detect()`.
* **Registry experts**:
  - `helmet.py`: Scan two-wheelers for riders linked via `NOT_WEARS` to `no_helmet` (runs pillion check if toggled). Includes a fallback to label unassociated `no_helmet` nodes directly.
  - `seatbelt.py`: Scan drivers of four-wheelers linked to `no_seatbelt`. Torso crops are segmented (height 30%-85%), equalized (`cv2.equalizeHist`), and analyzed using Hough transform (`cv2.HoughLinesP`) with dynamic length limits (`max(15, 0.3 * height)`) to find diagonal strap lines.
  - `triple_riding.py`: Flag two-wheelers with $\ge 3$ `RIDES` edges.
  - `wrong_side.py`: Compare tracker trajectory motion vectors (minimum displacement magnitude $\ge 2.0$px to filter jitter) against designated lane flow angles.
  - `stop_line.py`: Bounding box contact point overlap with stop-line polygons during red light. If stop line markings are faded/absent, falls back to a virtual horizontal line at 70% of frame height.
  - `red_light.py`: Cross stop line during red signal.
  - `illegal_parking.py`: Temporal tracker stationary analysis ($\ge 30.0$ seconds).
  - `missing_plate.py`: Vehicles operating without a detected plate.

### Layer 5: Evidence & e-Challan Generator (`core/plate_recognizer.py` & `core/evidence_generator.py`)
* **Multi-Stage OCR Fallback**: Crops the plate, upscales it $2.5\times$ via bicubic interpolation, applies CLAHE, and runs bilateral filtering. If EasyOCR returns no text, falls back to the raw crop, and then to adaptive thresholding + morphological closing.
* **Confidence Pruning**: Individual text segments with confidence $< 0.35$ are discarded to prevent reading background noise (e.g. `'TC'`).
* **Validator (`utils/indian_plate_validator.py`)**: Checks parsed text against RTO formats (state code registry, BH-series, temporary registration) and corrects confusions (e.g. converting `'O'` to `'0'` in numeric zones, and `'A'` to `'4'`).
* **Challan generation**: Creates a JSON evidence metadata file, saves vehicle and plate crops, generates a scan-to-pay QR code (mocked Parivahan payment URL), and outputs a secure PDF challan slip with double golden borders.

---

## 📂 Directories Mapping
* `app.py`: Streamlit Dashboard main entry.
* `config/settings.py`: Central repository of enums, model configurations, and thresholds.
* `core/`: Layer 1-5 core pipeline classes.
* `violations/`: Registry subclass experts for Layer 4.
* `utils/`: Visualizers, metrics, and plate validator.
* `tests/test_pipeline.py`: Integration smoke-tests checking E2E functionality.

---

## 🎯 Explaining Outcomes & Recent Fixes
When explaining outcomes to another AI, highlight these key resolution metrics:
1. **Background Clutter resolved**: Moving background filtering up to Layer 2 (detector level) instead of Layer 4 reduces the scene graph complexity from 29+ nodes to 6-9 nodes in busy scenes, speeding up execution and preventing false seatbelt/helmet challans on distant pedestrians.
2. **Triple Riding resolved**: The vertical-horizontal stack heuristic in Layer 3 correctly associates all three riders on a two-wheeler, allowing the pipeline to correctly count 3+ riders and trigger the `triple_riding` violation (verified on `triple_riding.png`).
3. **OCR False Positives resolved**: Filtering OCR text blocks with a confidence threshold $< 0.35$ prevents EasyOCR from stitching background noise characters into the plate string.
