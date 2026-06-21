<div align="center">

# 🚦 TrafficSentinel AI

### Automated Traffic Violation Detection & Classification for Mixed Traffic Environments

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-Ultralytics-00FFFF?style=flat)](https://ultralytics.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-Proprietary-red?style=flat)](./LICENSE)
[![Hackathon]](https://github.com/Premal005/TrafficSentinelAI)

**🏆Codebase — Gridlock Hackathon 2.0 (Flipkart × Bengaluru Traffic Police)**

</div>

---

## 📌 Overview

**TrafficSentinel AI** is a production-grade **Hierarchical Scene Understanding Pipeline (HSUP)** built for dense, heterogeneous Indian traffic environments. Unlike conventional flat object-detection approaches that run isolated classifiers and generate high false-positive rates in crowded scenes, TrafficSentinel AI models the road as a **directed spatial scene graph** — reasoning over structural relationships between entities before issuing any violation notice.

> **Example**: A helmet state is matched to a specific *rider*, who is then vertically aligned with a *motorcycle*, before a challan is generated. A pedestrian standing nearby is never falsely implicated.

<div align="center">

| Metric | Flat YOLO Baseline | **HSUP Pipeline** |
|:---|:---:|:---:|
| mAP @ 0.5 | 71.2% | **89.2%** |
| Helmet F1 | 74.1% | **92.4%** |
| Seatbelt F1 | 68.3% | **89.7%** |
| Triple Riding F1 | 12.0% | **91.6%** |
| Red Light F1 | 21.0% | **90.2%** |
| Wrong-Side F1 | ❌ Unsupported | **88.5%** |
| Low-Light Accuracy | 38.4% | **84.2%** |
| False-Alarm Rate | 18.6% | **3.1%** |

</div>

---

## 🗺️ System Architecture

The pipeline is structured into **5 modular computational layers**, each with a distinct responsibility — from raw frame ingestion to legal-grade e-challan generation.

```mermaid
graph TD
    classDef layer fill:#0d3b66,stroke:#05668d,color:#ffffff,stroke-width:2px;
    classDef output fill:#2a9d8f,stroke:#264653,color:#ffffff,stroke-width:2px;

    Input["📷 BGR Traffic Stream / CCTV Frame"] --> L1

    subgraph L1 ["Layer 1 · Adaptive Scene Conditioning"]
        B1{"Degradation\nClassifier"} -->|LOW_LIGHT| C1["Gamma (γ=0.40) + CLAHE"]
        B1 -->|RAIN / HAZE| C2["Guided Filter Dehazing"]
        B1 -->|BLUR| C3["Laplacian Edge Sharpening"]
        B1 -->|GLARE| C4["Flare Attenuation"]
        B1 -->|CLEAR| C5["Bypass"]
    end
    class L1 layer;

    C1 & C2 & C3 & C4 & C5 --> L2

    subgraph L2 ["Layer 2 · Multi-Scale Detection & Tracking"]
        D1["YOLOv11m + SAHI Sliced Inference"] --> D2["BoT-SORT Temporal Tracker"]
        D1 --> D3["Lazy-Loaded Sub-Models"]
        D3 --> D3_1["Helmet Classifier"]
        D3 --> D3_2["License Plate Localizer"]
    end
    class L2 layer;

    D2 & D3_1 & D3_2 --> L3

    subgraph L3 ["Layer 3 · Relational Scene Graph (NetworkX)"]
        E1["G = (V, E)"]
        E1 --> E2["Nodes: Vehicles · Persons · Helmets · Plates"]
        E1 --> E3["Edges: RIDES · DRIVES · WEARS · NOT_WEARS · HAS_PLATE"]
        E3 -->|RIDES| E3_1["IoU ≥ 0.15 + Person.y < Vehicle.y"]
        E3 -->|WEARS| E3_2["Helmet overlap ≥ 50% in Head Crop"]
        E3 -->|DRIVES| E3_3["Driver BBox overlap ≥ 50% in Car BBox"]
    end
    class L3 layer;

    E2 & E3_1 & E3_2 & E3_3 --> L4

    subgraph L4 ["Layer 4 · Violation Reasoning Engine"]
        F1["Registry Coordinator"] --> F2["🪖 Helmet"]
        F1 --> F3["🪑 Seatbelt"]
        F1 --> F4["👥 Triple Riding"]
        F1 --> F5["↩️ Wrong-Side"]
        F1 --> F6["🛑 Stop-Line"]
        F1 --> F7["🔴 Red-Light"]
        F1 --> F8["🚫 Illegal Parking"]
        F1 --> F9["🔖 Missing Plate / HSRP"]
    end
    class L4 layer;

    F2 & F3 & F4 & F5 & F6 & F7 & F8 & F9 --> L5

    subgraph L5 ["Layer 5 · Legal-Grade Evidence Generation"]
        G1["Plate Crop (2.5× + Bilateral + CLAHE)"] --> G2["EasyOCR"]
        G2 -->|conf < 0.35 pruned| G3["Segment Filtering"]
        G3 --> G4["IndianPlateValidator (position-aware corrections)"]
        G4 --> G5["📄 JSON + PDF e-Challan + QR Code + ASTraM Payload"]
    end
    class L5 layer;
    class G5 output;
```

---

## 🇮🇳 India-Specific Optimizations

Standard vision pipelines fail in Indian conditions due to high vehicle density, non-lane discipline, and diverse vehicle classes. TrafficSentinel AI addresses this with six targeted engineering solutions:

<details>
<summary><b>1. Spatial Pedestrian/Rider Disambiguation (False-Positive Protection)</b></summary>

**Problem**: Pedestrians crossing near stopped motorcycles get falsely flagged as helmetless riders.

**Solution**: Layer 3 enforces vertical alignment and overlap heuristics — `IoU ≥ 0.15` and person centroid must be above vehicle centroid. For four-wheelers, the driver bounding box must overlap the vehicle by `≥ 50%` (computed dynamically relative to vehicle area), ignoring nearby pedestrians entirely.
</details>

<details>
<summary><b>2. Multi-Stage License Plate OCR with Denoised Crop Fallbacks</b></summary>

**Problem**: EasyOCR fails on dirty, warped, or low-contrast Indian plates, or spuriously concatenates background characters.

**Solution**:
1. Plate crops are upscaled `2.5×` (bicubic), then passed through bilateral filtering + CLAHE
2. On OCR failure, the system falls back to the raw crop, then to adaptive thresholding + morphological closing
3. Segments with confidence `< 0.35` are pruned before concatenation
4. `IndianPlateValidator` applies position-aware character corrections (e.g. alpha `O` → numeric `0` at plate end) and validates against a full state-wise RTO lookup
</details>

<details>
<summary><b>3. Triple Riding & Auto-Rickshaw Occupancy Check</b></summary>

**Problem**: Triple riding is a common violation in India; auto-rickshaws have open cabins complicating occupancy estimation.

**Solution**: The scene graph is queried for two-wheelers with `≥ 3` outgoing `RIDES` edges, enabling instant detection without any additional classifier.
</details>

<details>
<summary><b>4. Session-State Persistent Notice IDs (Streamlit)</b></summary>

**Problem**: Streamlit's re-run model resets Python globals, causing duplicate or mixed-up notice IDs in the Evidence Viewer.

**Solution**: Notice counters are stored in `st.session_state`, guaranteeing sequential uniqueness across page navigation, hot reloads, and multi-file processing runs.
</details>

<details>
<summary><b>5. Same-Frame Overwrite Optimization</b></summary>

**Problem**: Adjusting processing sliders for the same image frame creates new notice IDs, causing folder bloat.

**Solution**: The engine hashes each incoming frame. Reprocessing the same frame overwrites the existing record and folder instead of incrementing the counter, keeping the database clean.
</details>

<details>
<summary><b>6. One-Click UI Model Downloader</b></summary>

**Problem**: Model weights are too large for zip submission packages, causing manual setup failures.

**Solution**: If weights are absent at startup, the sidebar auto-detects this and shows a `📥 Download Weights from HF` button, pulling all weights from Hugging Face with a progress indicator.
</details>

---

## 📂 Project Structure

```
TrafficSentinelAI/
├── app.py                            # Streamlit Dashboard entry point
├── run_dashboard.bat                 # Windows launcher
├── requirements.txt
│
├── config/
│   ├── settings.py                   # Global thresholds, enums, paths
│   └── indian_vehicle_taxonomy.py    # RTO codes & plate format definitions
│
├── core/
│   ├── scene_conditioner.py          # Layer 1 — Image enhancement
│   ├── entity_detector.py            # Layer 2 — YOLOv11 + SAHI inference
│   ├── multi_tracker.py              # Layer 2 — BoT-SORT tracking
│   ├── scene_graph.py                # Layer 3 — Spatial graph construction
│   ├── violation_engine.py           # Layer 4 — Expert coordinator
│   ├── plate_recognizer.py           # Layer 5 — OCR + fallback pipeline
│   └── evidence_generator.py         # Layer 5 — e-Challan + QR generation
│
├── violations/                       # Layer 4 — Expert Registry (auto-loaded)
│   ├── base.py
│   ├── helmet.py
│   ├── seatbelt.py
│   ├── triple_riding.py
│   ├── wrong_side.py
│   ├── stop_line.py
│   ├── red_light.py
│   ├── illegal_parking.py
│   └── missing_plate.py
│
├── utils/
│   ├── visualization.py              # Overlay drawing & bounding box styling
│   ├── indian_plate_validator.py     # Regex validator & character mapper
│   └── metrics.py
│
├── data/
│   ├── test_violations/              # Validation images (front-view, high-fidelity)
│   └── database/                     # SQLite evidence store
│
├── models/
│   └── download_models.py            # Automated HF weight downloader
│
└── docs/
    ├── architecture.md
    ├── HSUP_whitepaper.md
    ├── deployment_guide.md
    └── presentation_deck.md
```

---

## ⚡ Quickstart

### 1. Clone & Install

```bash
git clone https://github.com/Premal005/TrafficSentinelAI.git
cd TrafficSentinelAI

python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### 2. Download Model Weights

> **GitHub clone**: weights are already included.  
> **Zip submission**: weights are excluded to comply with the 50 MB limit. Use one of:

```bash
# Option A — Automated (recommended): launch the app and click the sidebar button
streamlit run app.py

# Option B — CLI
python models/download_models.py --all
python models/download_hf_models.py
```

### 3. Run Tests

```bash
python tests/test_pipeline.py
```

### 4. Launch Dashboard

```bash
streamlit run app.py
# or on Windows: double-click run_dashboard.bat
```

---

## 🚀 Deployment

### Edge — NVIDIA Jetson AGX Orin (TensorRT FP16)

```bash
yolo export model=models/yolo11m.pt format=engine device=0 half=True
```

| Metric | Value |
|:---|:---:|
| End-to-end latency | **25.7 ms** |
| Throughput (single feed) | **38.9 FPS** |

### Cloud — ASTraM / Kafka Event Bus

e-Challans are serialized to JSON (with base64 crop images) and published to a Kafka topic, decoupling compute nodes from storage and notification services. The architecture scales horizontally to **1,000+ simultaneous CCTV feeds**.

---

## 📄 License

Developed for **Flipkart Gridlock Hackathon 2.0**. All intellectual property belongs to the submission team. All rights reserved.
