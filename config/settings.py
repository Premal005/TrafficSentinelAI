"""
Global configuration for TrafficSentinel AI.

Central configuration hub defining all enums, thresholds, model paths,
and system-wide constants. Every module imports from here to ensure
consistency across the pipeline.
"""

from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os


# ══════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════

class ViolationType(Enum):
    """All supported traffic violation categories."""
    HELMET_NON_COMPLIANCE = "helmet_non_compliance"
    SEATBELT_NON_COMPLIANCE = "seatbelt_non_compliance"
    TRIPLE_RIDING = "triple_riding"
    WRONG_SIDE_DRIVING = "wrong_side_driving"
    STOP_LINE_VIOLATION = "stop_line_violation"
    RED_LIGHT_VIOLATION = "red_light_violation"
    ILLEGAL_PARKING = "illegal_parking"
    MISSING_LICENSE_PLATE = "missing_license_plate"


class DegradationType(Enum):
    """Image degradation categories for adaptive preprocessing."""
    CLEAR = "clear"
    LOW_LIGHT = "low_light"
    RAIN = "rain"
    SHADOW = "shadow"
    BLUR = "blur"
    GLARE = "glare"


class EntityClass(Enum):
    """Detectable entity classes in the traffic scene."""
    # Vehicles
    CAR = "car"
    MOTORCYCLE = "motorcycle"
    AUTO_RICKSHAW = "auto_rickshaw"
    BUS = "bus"
    TRUCK = "truck"
    BICYCLE = "bicycle"
    TEMPO = "tempo"
    
    # Road Users
    PERSON = "person"
    RIDER = "rider"
    DRIVER = "driver"
    PEDESTRIAN = "pedestrian"
    
    # Safety Gear
    HELMET = "helmet"
    NO_HELMET = "no_helmet"
    SEATBELT = "seatbelt"
    NO_SEATBELT = "no_seatbelt"
    
    # Infrastructure
    TRAFFIC_LIGHT_RED = "traffic_light_red"
    TRAFFIC_LIGHT_GREEN = "traffic_light_green"
    TRAFFIC_LIGHT_YELLOW = "traffic_light_yellow"
    STOP_LINE = "stop_line"
    LANE_MARKING = "lane_marking"
    
    # License Plate
    LICENSE_PLATE = "license_plate"


class TrafficLightState(Enum):
    """Traffic light states."""
    RED = "red"
    GREEN = "green"
    YELLOW = "yellow"
    UNKNOWN = "unknown"


# ══════════════════════════════════════════════════════════════
# COCO CLASS MAPPING (YOLOv11 pretrained on COCO)
# ══════════════════════════════════════════════════════════════

# COCO classes relevant to traffic scenes
COCO_TRAFFIC_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    9: "traffic light",
    11: "stop sign",
}

# Map COCO class IDs to our EntityClass
COCO_TO_ENTITY = {
    0: EntityClass.PERSON,
    1: EntityClass.BICYCLE,
    2: EntityClass.CAR,
    3: EntityClass.MOTORCYCLE,
    5: EntityClass.BUS,
    7: EntityClass.TRUCK,
}

# Two-wheeler classes (for helmet/triple-riding checks)
TWO_WHEELER_CLASSES = {EntityClass.MOTORCYCLE, EntityClass.BICYCLE}

# Four-wheeler classes (for seatbelt checks)
FOUR_WHEELER_CLASSES = {EntityClass.CAR, EntityClass.BUS, EntityClass.TRUCK, EntityClass.TEMPO}


# ══════════════════════════════════════════════════════════════
# VIOLATION FINE AMOUNTS (Indian Motor Vehicles Act, 2019)
# ══════════════════════════════════════════════════════════════

VIOLATION_FINES = {
    ViolationType.HELMET_NON_COMPLIANCE: 1000,
    ViolationType.SEATBELT_NON_COMPLIANCE: 1000,
    ViolationType.TRIPLE_RIDING: 1000,
    ViolationType.WRONG_SIDE_DRIVING: 5000,
    ViolationType.STOP_LINE_VIOLATION: 500,
    ViolationType.RED_LIGHT_VIOLATION: 1000,
    ViolationType.ILLEGAL_PARKING: 500,
    ViolationType.MISSING_LICENSE_PLATE: 2000,
}

VIOLATION_SECTIONS = {
    ViolationType.HELMET_NON_COMPLIANCE: "Section 194D - MVA 1988 (Amended 2019)",
    ViolationType.SEATBELT_NON_COMPLIANCE: "Section 194B(1) - MVA 1988 (Amended 2019)",
    ViolationType.TRIPLE_RIDING: "Section 194C - MVA 1988 (Amended 2019)",
    ViolationType.WRONG_SIDE_DRIVING: "Section 184 - MVA 1988 (Amended 2019)",
    ViolationType.STOP_LINE_VIOLATION: "Section 177A - MVA 1988 (Amended 2019)",
    ViolationType.RED_LIGHT_VIOLATION: "Section 119 read with Section 177A - MVA 1988 (Amended 2019)",
    ViolationType.ILLEGAL_PARKING: "Section 122 read with Section 177A - MVA 1988 (Amended 2019)",
    ViolationType.MISSING_LICENSE_PLATE: "Rule 50 & 51 - CMVR read with Section 177/192 - MVA",
}

VIOLATION_DISPLAY_NAMES = {
    ViolationType.HELMET_NON_COMPLIANCE: "Helmet Non-Compliance",
    ViolationType.SEATBELT_NON_COMPLIANCE: "Seatbelt Non-Compliance",
    ViolationType.TRIPLE_RIDING: "Triple Riding",
    ViolationType.WRONG_SIDE_DRIVING: "Wrong-Side Driving",
    ViolationType.STOP_LINE_VIOLATION: "Stop-Line Violation",
    ViolationType.RED_LIGHT_VIOLATION: "Red-Light Violation",
    ViolationType.ILLEGAL_PARKING: "Illegal Parking",
    ViolationType.MISSING_LICENSE_PLATE: "Missing License Plate / HSRP Violation",
}


# ══════════════════════════════════════════════════════════════
# SETTINGS DATACLASS
# ══════════════════════════════════════════════════════════════

@dataclass
class ModelConfig:
    """Configuration for a single YOLO model."""
    weights_path: str
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    image_size: int = 640
    device: str = "auto"  # auto, cpu, cuda:0


@dataclass
class SAHIConfig:
    """SAHI (Slicing Aided Hyper Inference) configuration."""
    enabled: bool = False
    slice_height: int = 512
    slice_width: int = 512
    overlap_ratio: float = 0.2
    postprocess_type: str = "NMS"
    postprocess_match_threshold: float = 0.5


@dataclass
class TrackerConfig:
    """BoT-SORT tracker configuration."""
    tracker_type: str = "botsort"  # botsort or bytetrack
    track_high_thresh: float = 0.5
    track_low_thresh: float = 0.1
    new_track_thresh: float = 0.6
    track_buffer: int = 30
    match_thresh: float = 0.8


@dataclass
class SceneGraphConfig:
    """Scene graph construction parameters."""
    # IoU thresholds for associating entities
    rider_vehicle_iou_thresh: float = 0.15
    person_vehicle_iou_thresh: float = 0.2
    plate_vehicle_iou_thresh: float = 0.1
    helmet_person_proximity: float = 0.3  # relative to person bbox height
    
    # Direction estimation
    min_trajectory_length: int = 5  # minimum frames for direction estimation
    direction_smoothing_window: int = 3


@dataclass  
class PreprocessorConfig:
    """Adaptive preprocessing configuration."""
    # CLAHE parameters
    clahe_clip_limit: float = 3.0
    clahe_grid_size: Tuple[int, int] = (8, 8)
    
    # Low-light enhancement
    gamma_low: float = 0.4
    gamma_high: float = 2.5
    
    # Brightness thresholds for degradation detection
    low_light_threshold: float = 60.0
    glare_threshold: float = 220.0
    blur_threshold: float = 100.0  # Laplacian variance
    
    # Shadow detection
    shadow_ratio_threshold: float = 0.3


@dataclass
class EvidenceConfig:
    """Evidence generation configuration."""
    output_dir: str = "evidence"
    save_original: bool = True
    save_annotated: bool = True
    save_vehicle_crop: bool = True
    save_plate_crop: bool = True
    save_attention_map: bool = True
    jpeg_quality: int = 95


@dataclass
class Settings:
    """
    Central settings for TrafficSentinel AI.
    
    All modules import from this class to ensure consistency.
    Modify thresholds here to tune the entire system.
    """
    # Project paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    models_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "models" / "weights")
    evidence_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "evidence")
    data_dir: Path = field(default_factory=lambda: Path(__file__).parent.parent / "data")
    
    # Model configurations
    vehicle_detector: ModelConfig = field(default_factory=lambda: ModelConfig(
        weights_path="yolo11m.pt",
        confidence_threshold=0.20,  # Lowered to 0.20 to capture small/occluded objects (e.g. children)
        iou_threshold=0.45,
        image_size=640,
        device="cpu",
    ))
    
    helmet_detector: ModelConfig = field(default_factory=lambda: ModelConfig(
        weights_path="yolo11s_helmet.pt",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        image_size=640,
        device="cpu",
    ))
    
    plate_detector: ModelConfig = field(default_factory=lambda: ModelConfig(
        weights_path="yolo11s_plate.pt",
        confidence_threshold=0.25,
        iou_threshold=0.45,
        image_size=640,
        device="cpu",
    ))
    
    # Sub-configs
    sahi: SAHIConfig = field(default_factory=SAHIConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    scene_graph: SceneGraphConfig = field(default_factory=SceneGraphConfig)
    preprocessor: PreprocessorConfig = field(default_factory=PreprocessorConfig)
    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)
    
    # Violation thresholds
    min_violation_confidence: float = 0.4
    triple_riding_min_persons: int = 3
    illegal_parking_duration_sec: float = 30.0
    pillion_helmet_required: bool = True  # Mandatory in India/BTP, but can be toggled off for driver-only check
    min_vehicle_area_ratio: float = 0.03  # Focus on near/foreground vehicles, filter distant background traffic
    min_vehicle_sharpness: float = 80.0  # Minimum Laplacian variance for vehicle ROI to be considered in-focus
    enable_focal_pruning: bool = True    # Filter out out-of-focus background vehicles/pedestrians
    
    # Processing
    max_image_dimension: int = 1920
    batch_size: int = 1
    
    # Dashboard
    dashboard_title: str = "🚦 TrafficSentinel AI — Traffic Command Center"
    dashboard_icon: str = "🚦"
    
    def __post_init__(self):
        """Ensure directories exist."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def get_model_path(self, model_config: ModelConfig) -> Path:
        """Get full path for a model, checking models_dir first with yolov8s fallback."""
        local_path = self.models_dir / model_config.weights_path
        if local_path.exists():
            return local_path
            
        # Fallback to yolov8s if yolo11s is missing
        if "yolo11s_" in model_config.weights_path:
            fallback_name = model_config.weights_path.replace("yolo11s_", "yolov8s_")
            fallback_path = self.models_dir / fallback_name
            if fallback_path.exists():
                import logging
                logging.getLogger(__name__).info(f"Fallback: Using {fallback_name} instead of {model_config.weights_path}")
                return fallback_path
                
        # Ultralytics will auto-download standard models
        return Path(model_config.weights_path)


# Global settings instance
SETTINGS = Settings()
