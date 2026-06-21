# 🚀 Edge & Cloud Deployment Guide — TrafficSentinel AI

This document provides guidelines for deploying **TrafficSentinel AI**'s **Hierarchical Scene Understanding Pipeline (HSUP)** in real-world production systems, including edge devices (NVIDIA Jetson) and high-throughput cloud environments.

---

## 1. Edge Deployment (NVIDIA Jetson AGX Orin / Xavier)

For direct junction deployment (CCTV edge analytics), we compile models to **NVIDIA TensorRT** to maximize throughput and minimize latency.

### 1.1 Model Conversion to TensorRT
Export the PyTorch YOLO weights (`.pt`) to TensorRT (`.engine`) format directly using Ultralytics:

```bash
# Convert primary vehicle detection model
yolo export model=models/weights/yolo11m.pt format=engine device=0 half=True imgsz=640

# Convert specialized helmet detection model
yolo export model=models/weights/yolov8s_helmet.pt format=engine device=0 half=True imgsz=640
```

*Note: The `--half` flag compiles to FP16 precision, cutting latency in half on Jetson Tensor Cores with negligible accuracy loss ($< 0.1\%\text{ mAP}$).*

### 1.2 Pipeline Profiling on Jetson
Estimated performance comparison on Jetson AGX Orin (64GB, 275 TOPS):

| Pipeline Stage | Precision | Latency (FP32) | Latency (TRT FP16) |
|---|---|---|---|
| **Layer 1: Scene Conditioner** | OpenCV/Denoise | 8.5 ms | 8.5 ms |
| **Layer 2: Entity Detection** | YOLOv11m | 24.2 ms | 5.8 ms |
| **Layer 2: Helmet Classifier** | YOLOv11s | 12.1 ms | 3.1 ms |
| **Layer 3: Scene Graph** | NumPy/CPU | 1.8 ms | 1.8 ms |
| **Layer 5: EasyOCR** | ResNet+CTC | 18.0 ms | 6.5 ms |
| **Total Pipeline Latency** | — | **64.6 ms** | **25.7 ms** |
| **Throughput (FPS)** | — | **15.4 FPS** | **38.9 FPS** |

---

## 2. Docker Containerization

To deploy the dashboard and backend at scale in a Kubernetes cluster or a virtual server, run the containerized application.

### 2.1 Dockerfile
Create a `Dockerfile` at the root directory:

```dockerfile
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Install system dependencies
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose Streamlit dashboard port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2.2 Docker Build & Run
```bash
# Build image
docker build -t trafficsentinel-ai:latest .

# Run with GPU support enabled
docker run -d --gpus all -p 8501:8501 --name trafficsentinel trafficsentinel-ai:latest
```

---

## 3. Integration with BTP's ASTraM Unit

To feed results into the Bengaluru Traffic Police (BTP) ASTraM Command Center:
1. **RTSP Stream Capture**: Connect Layer 1 directly to junction IP cameras using OpenCV's `VideoCapture("rtsp://admin:password@IP_ADDR:554/stream1")`.
2. **REST API Endpoint**: Modify `evidence_generator.py` to POST the JSON evidence packet directly to BTP's violation logging API instead of saving locally:
   ```python
   import requests
   response = requests.post("https://astram.btp.gov.in/api/v1/violations", json=packet, headers=auth_headers)
   ```
3. **Kafka Event Bus**: In high-throughput city-wide deployments (1,000+ cameras), route evidence JSON packets to an Apache Kafka topic for asynchronous processing and load-balanced e-challan generation.
