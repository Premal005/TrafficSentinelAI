"""
Live Detection Page — TrafficSentinel AI.
Allows uploading images or videos, running the 5-layer HSUP pipeline,
and visualizing results, including the spatial Scene Graph.
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import plotly.graph_objects as go
import networkx as nx
from typing import Dict, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import render_sidebar, inject_custom_css, configure_page

# Configure page, inject styles and render sidebar
configure_page()
inject_custom_css()
render_sidebar()

from config.settings import SETTINGS, EntityClass, ViolationType, VIOLATION_DISPLAY_NAMES

from core.scene_conditioner import SceneConditioner
from core.entity_detector import EntityDetector, Detection
from core.multi_tracker import MultiTracker
from core.scene_graph import SceneGraph, RIDES, DRIVES, WEARS, NOT_WEARS, HAS_PLATE
from core.violation_engine import ViolationEngine
from core.plate_recognizer import PlateRecognizer
from core.evidence_generator import EvidenceGenerator
from utils.visualization import draw_detections, draw_violations, resize_for_display

# Initialize Session State variables if not present
if "total_violations" not in st.session_state:
    st.session_state["total_violations"] = 0
if "total_processed" not in st.session_state:
    st.session_state["total_processed"] = 0
if "plates_recognized" not in st.session_state:
    st.session_state["plates_recognized"] = 0
if "avg_confidence" not in st.session_state:
    st.session_state["avg_confidence"] = 0.85
if "avg_fps" not in st.session_state:
    st.session_state["avg_fps"] = 15.4
if "violation_history" not in st.session_state:
    st.session_state["violation_history"] = []


def get_hierarchical_layout(scene_graph: SceneGraph) -> Dict[str, Tuple[float, float]]:
    """
    Computes a clean, layered hierarchical layout for the scene graph.
    Tiers:
      Y = 3.0: Vehicles
      Y = 2.0: License Plates (centered under their vehicle)
      Y = 1.0: Persons/Riders (spaced horizontally under their vehicle)
      Y = 0.0: Safety Gear / Helmets (aligned under the person)
    """
    pos = {}
    
    vehicles = []
    plates = []
    persons = []
    gear = []
    others = []
    
    vehicle_classes = {"car", "motorcycle", "bus", "truck", "auto_rickshaw", "tempo", "bicycle"}
    person_classes = {"person", "rider", "driver", "pedestrian"}
    gear_classes = {"helmet", "no_helmet", "seatbelt", "no_seatbelt"}
    
    for node_id, node in scene_graph.nodes.items():
        cls = node.entity_class.value if node.entity_class else ""
        if cls in vehicle_classes:
            vehicles.append(node)
        elif cls == "license_plate":
            plates.append(node)
        elif cls in person_classes:
            persons.append(node)
        elif cls in gear_classes:
            gear.append(node)
        else:
            others.append(node)
            
    # Deterministic sorting
    vehicles.sort(key=lambda n: n.node_id)
    plates.sort(key=lambda n: n.node_id)
    persons.sort(key=lambda n: n.node_id)
    gear.sort(key=lambda n: n.node_id)
    others.sort(key=lambda n: n.node_id)
    
    # 1. Place vehicles at Y = 3.0
    num_vehicles = len(vehicles)
    vehicle_x = {}
    if num_vehicles > 0:
        spacing = 2.0 / max(1, num_vehicles - 1) if num_vehicles > 1 else 0.0
        start_x = -1.0 if num_vehicles > 1 else 0.0
        for idx, node in enumerate(vehicles):
            x = start_x + idx * spacing
            pos[node.node_id] = (x, 3.0)
            vehicle_x[node.node_id] = x
            
    # Map connections from edges
    rides_or_drives = {}
    has_plate = {}
    wears_gear = {}
    
    for edge in scene_graph.edges:
        if edge.relation in ("RIDES", "DRIVES"):
            rides_or_drives[edge.source_id] = edge.target_id
        elif edge.relation == "HAS_PLATE":
            has_plate[edge.target_id] = edge.source_id
        elif edge.relation in ("WEARS", "NOT_WEARS"):
            wears_gear[edge.target_id] = edge.source_id

    # 2. Place license plates at Y = 2.0
    for node in plates:
        veh_id = has_plate.get(node.node_id)
        if veh_id in vehicle_x:
            pos[node.node_id] = (vehicle_x[veh_id], 2.0)
        else:
            pos[node.node_id] = (0.0, 2.0)
            
    # 3. Place persons at Y = 1.0
    vehicle_to_riders = {}
    independent_persons = []
    for node in persons:
        veh_id = rides_or_drives.get(node.node_id)
        if veh_id:
            if veh_id not in vehicle_to_riders:
                vehicle_to_riders[veh_id] = []
            vehicle_to_riders[veh_id].append(node)
        else:
            independent_persons.append(node)
            
    for veh_id, riders in vehicle_to_riders.items():
        veh_x = vehicle_x.get(veh_id, 0.0)
        num_riders = len(riders)
        if num_riders == 1:
            pos[riders[0].node_id] = (veh_x, 1.0)
        else:
            span = 0.5
            rider_spacing = span / (num_riders - 1)
            rider_start_x = veh_x - span / 2.0
            for idx, r_node in enumerate(riders):
                pos[r_node.node_id] = (rider_start_x + idx * rider_spacing, 1.0)
                
    num_ind = len(independent_persons)
    if num_ind > 0:
        spacing = 2.0 / max(1, num_ind + 1)
        for idx, node in enumerate(independent_persons):
            pos[node.node_id] = (-1.0 + (idx + 1) * spacing, 1.0)
            
    # 4. Place safety gear at Y = 0.0
    person_x = {p_id: xy[0] for p_id, xy in pos.items() if p_id in [p.node_id for p in persons]}
    for node in gear:
        p_id = wears_gear.get(node.node_id)
        if p_id in person_x:
            pos[node.node_id] = (person_x[p_id], 0.0)
        else:
            pos[node.node_id] = (0.0, 0.0)
            
    # 5. Place other/uncategorized nodes at Y = 1.5
    num_others = len(others)
    if num_others > 0:
        spacing = 2.0 / max(1, num_others + 1)
        for idx, node in enumerate(others):
            pos[node.node_id] = (-1.0 + (idx + 1) * spacing, 1.5)
            
    # Fallback safety assignment
    for node_id in scene_graph.nodes:
        if node_id not in pos:
            pos[node_id] = (0.0, 1.5)
            
    return pos


def render_scene_graph_plotly(scene_graph: SceneGraph):
    """Render the scene graph using Plotly and NetworkX with a premium hierarchical layout."""
    if not scene_graph.nodes:
        st.info("No nodes in scene graph to display.")
        return

    G = nx.DiGraph()
    
    # Add nodes
    for node_id, node in scene_graph.nodes.items():
        G.add_node(
            node_id, 
            label=f"{node.entity_class.value.upper()}\n({node.confidence:.2f})",
            cls=node.entity_class.value
        )
        
    # Add edges
    for edge in scene_graph.edges:
        G.add_edge(edge.source_id, edge.target_id, label=edge.relation)

    # Calculate layout positions using our clean hierarchical logic
    pos = get_hierarchical_layout(scene_graph)

    # Edge traces
    edge_x = []
    edge_y = []
    annotations = []

    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        # Center of edge for text label
        xc = (x0 + x1) / 2.0
        yc = (y0 + y1) / 2.0
        relation = edge[2].get("label", "")
        
        annotations.append(
            dict(
                x=xc, y=yc,
                text=relation,
                showarrow=False,
                font=dict(size=8, color="#00d2ff"),
                bgcolor="#0e1117",
                bordercolor="rgba(0,210,255,0.15)",
                borderwidth=1,
                borderpad=1.5
            )
        )

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1.5, color="rgba(255,255,255,0.15)"),
        hoverinfo='none',
        mode='lines'
    )

    # Node traces
    node_x = []
    node_y = []
    node_text = []
    node_color = []
    
    color_map = {
        "car": "#3a7bd5", "motorcycle": "#00d2ff", "auto_rickshaw": "#ffc107", 
        "person": "#28a745", "rider": "#20c997", "driver": "#198754",
        "helmet": "#28a745", "no_helmet": "#dc3545",
        "license_plate": "#fd7e14"
    }

    for node in G.nodes(data=True):
        x, y = pos[node[0]]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node[1].get("label", ""))
        
        cls = node[1].get("cls", "")
        node_color.append(color_map.get(cls, "#6c757d"))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[node[0].split('_')[0].upper() for node in G.nodes()], 
        textposition="bottom center",
        marker=dict(
            showscale=False,
            color=node_color,
            size=26,
            line=dict(width=2, color='#ffffff')
        ),
        textfont=dict(color="#ffffff", size=9)
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20, l=10, r=10, t=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            annotations=annotations
        )
    )
    
    st.plotly_chart(fig, width="stretch")


def group_violations_by_vehicle(violations, scene_graph):
    grouped = {}
    for v in violations:
        vehicle_id = "unknown_vehicle"
        for node_id in v.involved_nodes:
            node = scene_graph.nodes.get(node_id)
            if node and node.entity_class.value in ["car", "motorcycle", "auto_rickshaw", "bus", "truck", "tempo"]:
                vehicle_id = node_id
                break
        if vehicle_id == "unknown_vehicle":
            for node_id in v.involved_nodes:
                node = scene_graph.nodes.get(node_id)
                if node and node.entity_class.value == "license_plate":
                    vehicle_id = f"plate_{node_id}"
                    break
        if vehicle_id not in grouped:
            grouped[vehicle_id] = []
        grouped[vehicle_id].append(v)
    return grouped


def process_image(image_bytes, sample_option="-- None --"):
    """Execute full 5-layer HSUP pipeline on the uploaded image."""
    # Convert uploaded image to BGR numpy array
    file_bytes = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if frame is None:
        st.error("Failed to decode image.")
        return

    # Load thresholds from session state
    conf_thresh = st.session_state.get("confidence_threshold", 0.35)
    min_area_ratio = st.session_state.get("min_vehicle_area_ratio", 0.03)
    use_sahi = st.session_state.get("use_sahi", False)
    use_preprocessing = st.session_state.get("use_preprocessing", True)
    active_violations = st.session_state.get("active_violations", {})
    pillion_helmet_required = st.session_state.get("pillion_helmet_required", True)
    SETTINGS.pillion_helmet_required = pillion_helmet_required
    SETTINGS.vehicle_detector.confidence_threshold = conf_thresh
    SETTINGS.min_vehicle_area_ratio = min_area_ratio

    st.info("Pipeline Status: Initialising core modules...")
    
    # Instantiate modules
    conditioner = SceneConditioner(SETTINGS)
    detector = EntityDetector(SETTINGS)
    scene_graph_builder = SceneGraph(SETTINGS)
    violation_engine = ViolationEngine(SETTINGS)
    plate_recognizer = PlateRecognizer(SETTINGS)
    evidence_generator = EvidenceGenerator(SETTINGS)

    # Layer 1: Adaptive Preprocessing
    start_time = time.time()
    if use_preprocessing:
        st.info("Executing Layer 1: Adaptive Scene Conditioning...")
        l1_start = time.time()
        cond_res = conditioner.condition(frame)
        enhanced_frame = cond_res.enhanced_image
        deg_type = cond_res.degradation_type.value
        quality_score = cond_res.quality_score
        l1_time = time.time() - l1_start
    else:
        enhanced_frame = frame.copy()
        deg_type = "disabled"
        quality_score = 1.0
        l1_time = 0.0


    # Layer 2: Entity Detection & Tracking
    st.info("Executing Layer 2: Multi-Scale Entity Detection...")
    l2_start = time.time()
    # Standard detection using vehicle/person model
    detections = detector.detect(enhanced_frame, use_sahi=use_sahi)
    
    # Real models handle detection — no mock injections needed

    # Crop riders and detect helmets/seatbelts based on vehicle association
    rider_nodes = [d for d in detections if d.entity_class == EntityClass.PERSON]
    if rider_nodes:
        from core.scene_graph import compute_iou, compute_overlap_ratio, is_four_wheeler_occupant
        from config.settings import TWO_WHEELER_CLASSES, FOUR_WHEELER_CLASSES
        
        helmet_crops = []
        seatbelt_crops = []
        
        # Gather all vehicle detections in the frame
        vehicles = [d for d in detections if d.entity_class is not None and d.entity_class in (TWO_WHEELER_CLASSES | FOUR_WHEELER_CLASSES)]
        
        for r in rider_nodes:
            is_two_wheeler_rider = False
            is_occupant = False
            
            for v in vehicles:
                iou = compute_iou(r.bbox, v.bbox)
                overlap = compute_overlap_ratio(r.bbox, v.bbox)
                score = max(iou, overlap)
                
                if v.entity_class in TWO_WHEELER_CLASSES:
                    if score >= 0.15:
                        is_two_wheeler_rider = True
                elif v.entity_class in FOUR_WHEELER_CLASSES:
                    if is_four_wheeler_occupant(r.bbox, v.bbox):
                        is_occupant = True
            
            if is_two_wheeler_rider:
                helmet_crops.append(r.bbox)
            if is_occupant:
                seatbelt_crops.append(r.bbox)
                
        if helmet_crops:
            helmet_detections = detector.detect_helmets(enhanced_frame, helmet_crops)
            detections.extend(helmet_detections)
            
        if seatbelt_crops:
            seatbelt_detections = detector.detect_seatbelts(enhanced_frame, seatbelt_crops)
            detections.extend(seatbelt_detections)

    # Crop license plates and recognize
    plate_detections = detector.detect_plates(enhanced_frame, detections)
    detections.extend(plate_detections)
    l2_time = time.time() - l2_start

    # Layer 3: Scene Graph construction
    st.info("Executing Layer 3: Scene Graph Construction...")
    l3_start = time.time()
    scene_graph_builder.build(detections, frame_shape=enhanced_frame.shape[:2])

    # Real models handle detection — no mock injections needed


    # Layer 5a: OCR on detected plates & validation
    st.info("Executing Layer 5a: Character Recognition on License Plates...")
    for node_id, node in list(scene_graph_builder.nodes.items()):
        if node.entity_class == EntityClass.LICENSE_PLATE:
            # Recognize plate text
            plate_text, ocr_conf, val_res = plate_recognizer.recognize(enhanced_frame, node.bbox)
            
            # Filter out false-positive license plates (grills, fairings, logos, noise)
            import re
            cleaned_text = re.sub(r'[\s\-\.\,]', '', plate_text).upper()
            has_letter = any(c.isalpha() for c in cleaned_text)
            has_digit = any(c.isdigit() for c in cleaned_text)
            
            if len(cleaned_text) >= 4 and has_letter and has_digit:
                node.attributes["plate_text"] = plate_text
                node.attributes["plate_state"] = val_res.state or "UNKNOWN"
                node.attributes["plate_type"] = val_res.format_name or "private"
                node.attributes["ocr_confidence"] = ocr_conf
                st.session_state["plates_recognized"] += 1
            else:
                import logging
                logging.getLogger(__name__).info(
                    f"Discarding false-positive plate detection {node_id}: text='{plate_text}'"
                )
                if node_id in scene_graph_builder.nodes:
                    del scene_graph_builder.nodes[node_id]
                scene_graph_builder.edges = [
                    e for e in scene_graph_builder.edges 
                    if e.source_id != node_id and e.target_id != node_id
                ]
    l3_time = time.time() - l3_start

    # Layer 4: Violation reasoning
    st.info("Executing Layer 4: Violation Reasoning...")
    l4_start = time.time()
    violations = violation_engine.analyze(scene_graph_builder, enhanced_frame)
    
    # Filter violations based on sidebar active toggles
    filtered_violations = []
    for v in violations:
        enabled = True
        if v.violation_type == ViolationType.HELMET_NON_COMPLIANCE and not active_violations.get("Helmet", True):
            enabled = False
        elif v.violation_type == ViolationType.SEATBELT_NON_COMPLIANCE and not active_violations.get("Seatbelt", True):
            enabled = False
        elif v.violation_type == ViolationType.TRIPLE_RIDING and not active_violations.get("Triple Riding", True):
            enabled = False
        elif v.violation_type == ViolationType.WRONG_SIDE_DRIVING and not active_violations.get("Wrong Side", True):
            enabled = False
        elif v.violation_type == ViolationType.STOP_LINE_VIOLATION and not active_violations.get("Stop Line", True):
            enabled = False
        elif v.violation_type == ViolationType.RED_LIGHT_VIOLATION and not active_violations.get("Red Light", True):
            enabled = False
        elif v.violation_type == ViolationType.ILLEGAL_PARKING and not active_violations.get("Illegal Parking", True):
            enabled = False
        elif v.violation_type == ViolationType.MISSING_LICENSE_PLATE and not active_violations.get("Missing Plate", True):
            enabled = False
            
        if enabled:
            filtered_violations.append(v)
            
    violations = filtered_violations
    l4_time = time.time() - l4_start

    # Layer 5b: Evidence & e-Challan Generation
    st.info("Executing Layer 5b: Evidence and Challan Compilation...")
    l5_start = time.time()
    
    # Filter detections to only keep foreground/active scene graph nodes or infrastructure
    sg_bboxes = {node.bbox for node in scene_graph_builder.nodes.values()}
    filtered_detections = [
        d for d in detections 
        if d.bbox in sg_bboxes or d.entity_class in (
            EntityClass.TRAFFIC_LIGHT_RED, EntityClass.TRAFFIC_LIGHT_GREEN, 
            EntityClass.TRAFFIC_LIGHT_YELLOW, EntityClass.STOP_LINE, EntityClass.LANE_MARKING
        )
    ]
    
    annotated_frame = draw_detections(enhanced_frame, filtered_detections)
    annotated_frame = draw_violations(annotated_frame, violations)
    
    import hashlib
    frame_bytes = cv2.imencode('.jpg', enhanced_frame)[1].tobytes()
    current_frame_hash = hashlib.sha256(frame_bytes).hexdigest()
    
    # Check if there are any old packets in history for this exact image frame
    old_packets = [p for p in st.session_state["violation_history"]
                   if p.get("evidence_integrity", {}).get("frame_hash") == current_frame_hash]
                   
    # Match new violations to old violation IDs to avoid incrementing the sequence number on rerun
    if old_packets:
        for v in violations:
            v_veh_node = None
            for node_id in v.involved_nodes:
                node = scene_graph_builder.nodes.get(node_id)
                if node and node.entity_class.value in ["car", "motorcycle", "auto_rickshaw", "bus", "truck", "tempo"]:
                    v_veh_node = node
                    break
            
            if v_veh_node is not None:
                plate_node = scene_graph_builder.get_plate_of(v_veh_node.node_id)
                plate_text = plate_node.attributes.get("plate_text", "") if plate_node else ""
                
                matched_op = None
                if plate_text and plate_text != "UNKNOWN":
                    for op in old_packets:
                        if op.get("vehicle", {}).get("plate_number") == plate_text:
                            matched_op = op
                            break
                if matched_op is None:
                    for op in old_packets:
                        if op.get("vehicle", {}).get("type") == v_veh_node.entity_class.value:
                            matched_op = op
                            break
                if matched_op is not None:
                    v.violation_id = matched_op["violation_id"]
                    
        # Remove old packets for this same image frame from history to prevent duplication/mixups
        st.session_state["violation_history"] = [p for p in st.session_state["violation_history"]
                                                 if p.get("evidence_integrity", {}).get("frame_hash") != current_frame_hash]

    evidence_packets = []
    grouped_vios = group_violations_by_vehicle(violations, scene_graph_builder)
    for vehicle_id, vios in grouped_vios.items():
        packet = evidence_generator.generate_evidence_packet(
            enhanced_frame, annotated_frame, vios, scene_graph_builder
        )
        pdf_path = evidence_generator.generate_challan_pdf(packet)
        packet["pdf_path"] = pdf_path
        evidence_packets.append(packet)
        st.session_state["violation_history"].append(packet)
    l5_time = time.time() - l5_start
        
    elapsed = time.time() - start_time
    fps = 1.0 / elapsed if elapsed > 0 else 30.0

    # Update global metrics
    st.session_state["total_violations"] += len(violations)
    st.session_state["total_processed"] += 1
    st.session_state["avg_fps"] = (st.session_state["avg_fps"] * 0.9) + (fps * 0.1)
    if violations:
        mean_conf = sum(v.confidence for v in violations) / len(violations)
        st.session_state["avg_confidence"] = (st.session_state["avg_confidence"] * 0.9) + (mean_conf * 0.1)

    st.success(f"Pipeline executed successfully in {elapsed:.2f}s ({fps:.1f} FPS)")

    # Store layer timings for display
    layer_timings = {
        "L1: Scene Conditioning": l1_time if use_preprocessing else 0,
        "L2: Entity Detection": l2_time,
        "L3: Scene Graph": l3_time,
        "L4: Violation Reasoning": l4_time,
        "L5: Evidence Generation": l5_time,
    }
    
    return frame, annotated_frame, scene_graph_builder, violations, evidence_packets, deg_type, quality_score, layer_timings


# Page Layout
st.title("🔍 Live Violation Detection")
st.write("Upload an image or choose a preloaded sample to process through the complete 5-layer HSUP pipeline.")

sample_option = st.selectbox(
    "Select a preloaded sample case:",
    options=["-- None --", "Sample 1: Daylight Bengaluru Traffic", "Sample 2: Rainy Night Bengaluru Traffic"]
)

uploaded_file = st.file_uploader("Upload Traffic Frame", type=["jpg", "jpeg", "png"])

image_bytes = None

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
elif sample_option == "Sample 1: Daylight Bengaluru Traffic":
    sample_path = PROJECT_ROOT / "data" / "sample_images" / "traffic_scene_1.png"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
    else:
        st.error(f"Sample file not found at {sample_path}")
elif sample_option == "Sample 2: Rainy Night Bengaluru Traffic":
    sample_path = PROJECT_ROOT / "data" / "sample_images" / "traffic_scene_night.png"
    if sample_path.exists():
        with open(sample_path, "rb") as f:
            image_bytes = f.read()
    else:
        st.error(f"Sample file not found at {sample_path}")

col_main, col_sidebar = st.columns([2, 1])

if image_bytes is not None:
    # Process
    orig, anno, sg, violations, packets, deg_type, quality, layer_timings = process_image(image_bytes, sample_option)


    with col_main:
        st.subheader("📷 Pipeline Outputs")
        
        tab_original, tab_annotated = st.tabs(["Original Frame", "Annotated Pipeline Result"])
        
        with tab_original:
            st.image(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB), width="stretch")
            
        with tab_annotated:
            st.image(cv2.cvtColor(anno, cv2.COLOR_BGR2RGB), width="stretch")

        st.subheader("🧠 Scene Graph Visualisation")
        render_scene_graph_plotly(sg)

    with col_sidebar:
        st.subheader("🛡️ Scene Diagnostics")
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span>Degradation Type:</span>
                <strong style="color: #00d2ff;">{deg_type.upper()}</strong>
            </div>
            <div style="display: flex; justify-content: space-between;">
                <span>Scene Quality Score:</span>
                <strong style="color: #00ff88;">{quality:.2f} / 1.00</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader("⏱️ Layer Timing Breakdown")
        for layer_name, layer_t in layer_timings.items():
            st.markdown(f"**{layer_name}**: `{layer_t:.3f}s`")

        st.subheader("🚨 Detected Violations")
        if not violations:
            st.success("No violations detected in this frame.")
        else:
            for idx, p in enumerate(packets):
                color_map = {
                    "helmet_non_compliance": "red",
                    "seatbelt_non_compliance": "orange",
                    "triple_riding": "red",
                    "wrong_side_driving": "purple",
                    "stop_line_violation": "orange",
                    "red_light_violation": "red",
                    "illegal_parking": "blue",
                    "multiple": "red"
                }
                
                v_color = color_map.get(p["violation_type"], "blue")
                
                dispatch_html = ""
                if "police_dispatch" in p:
                    disp = p["police_dispatch"]
                    dispatch_html = f"""<style>
@keyframes pulse {{
    0% {{ opacity: 0.85; }}
    50% {{ opacity: 1; border-color: #ff0000; box-shadow: 0 0 8px rgba(255, 0, 0, 0.4); }}
    100% {{ opacity: 0.85; }}
}}
</style>
<div style="background-color: rgba(217, 4, 41, 0.15); border: 1px solid #d90429; border-radius: 4px; padding: 8px; margin-top: 8px; font-size: 12px; color: #ff4d4d; animation: pulse 2s infinite;">
    <strong>🚨 Patrol Intercept Dispatched:</strong><br>
    Unit {disp['patrol_unit']} (Officer {disp['officer_id']}) notified!<br>
    (Dist: {disp['distance']} | ETA: {disp['eta']})
</div>"""
                
                st.markdown(f"""<div class="glass-card" style="border-left: 4px solid {v_color};">
<h5 style="margin: 0; color: #ffffff;">{p['violation_name']}</h5>
<p style="color: rgba(255,255,255,0.6); font-size: 13px; margin: 4px 0;">
    Vehicle: <strong>{p['vehicle']['plate_number']}</strong> ({p['vehicle']['type'].upper()})
</p>
<div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 8px; border-bottom: { '1px solid rgba(255,255,255,0.1)' if dispatch_html else 'none' }; padding-bottom: { '6px' if dispatch_html else '0px' };">
    <span>Confidence: <strong>{p['confidence']:.2f}</strong></span>
    <span style="color: #ffaa00;">Fine: <strong>INR {p['fine_amount']}</strong></span>
</div>
{dispatch_html}
</div>""", unsafe_allow_html=True)

                # Show XAI Reasoning Chain if available from packet
                if "reasoning_chain" in p and p["reasoning_chain"]:
                    with st.expander("🔬 AI Reasoning Chain", expanded=False):
                        for step_idx, step in enumerate(p["reasoning_chain"]):
                            icon = "✅" if step_idx == len(p["reasoning_chain"]) - 1 else "➡️"
                            st.markdown(f"{icon} {step}")
                
                # Download link for challan PDF
                if p.get("pdf_path") and Path(p["pdf_path"]).exists():
                    with open(p["pdf_path"], "rb") as pdf_file:
                        st.download_button(
                            label=f"📄 Download e-Challan #{p['violation_id']}",
                            data=pdf_file,
                            file_name=Path(p["pdf_path"]).name,
                            mime="application/pdf",
                            key=f"dl_{p['violation_id']}_{idx}"
                        )
else:
    st.info("Please upload an image file using the uploader above to begin detection.")
