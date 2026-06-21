"""
TrafficSentinel AI — Traffic Command Center
============================================

Main Streamlit application entry point.
Provides a multi-page dashboard for traffic violation detection,
evidence review, analytics, and reporting.

Run with: streamlit run app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def configure_page():
    """Configure Streamlit page settings."""
    st.set_page_config(
        page_title="TrafficSentinel AI — Traffic Command Center",
        page_icon="🚦",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": None,
            "Report a bug": None,
            "About": (
                "## 🚦 TrafficSentinel AI\n"
                "### Hierarchical Scene Understanding Pipeline (HSUP)\n\n"
                "Automated traffic violation detection system using "
                "scene graph-based spatial reasoning.\n\n"
                "**Gridlock Hackathon 2.0** — Flipkart × BTP"
            ),
        },
    )


def inject_custom_css():
    """Inject custom CSS for premium dark-themed styling."""
    st.markdown("""
    <style>
        /* ═══════ GLOBAL STYLES ═══════ */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        .stApp {
            background: linear-gradient(135deg, #0a0a0f 0%, #0d1117 50%, #0a0f1a 100%);
        }
        
        /* ═══════ HEADER ═══════ */
        .main-header {
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 24px 32px;
            margin-bottom: 24px;
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5, #00d2ff);
            background-size: 200% 100%;
            animation: shimmer 3s ease-in-out infinite;
        }
        
        @keyframes shimmer {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        .main-header h1 {
            color: #ffffff;
            font-size: 28px;
            font-weight: 700;
            margin: 0;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            color: rgba(255,255,255,0.6);
            font-size: 14px;
            margin: 4px 0 0 0;
        }
        
        /* ═══════ METRIC CARDS ═══════ */
        .metric-card {
            background: linear-gradient(135deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        
        .metric-card:hover {
            border-color: rgba(0, 210, 255, 0.3);
            transform: translateY(-2px);
            box-shadow: 0 8px 25px rgba(0, 210, 255, 0.1);
        }
        
        .metric-value {
            font-size: 36px;
            font-weight: 800;
            background: linear-gradient(135deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            line-height: 1.2;
        }
        
        .metric-label {
            color: rgba(255,255,255,0.5);
            font-size: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }
        
        /* ═══════ STATUS BADGES ═══════ */
        .status-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
        }
        
        .status-online {
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.2);
        }
        
        .status-processing {
            background: rgba(255, 170, 0, 0.1);
            color: #ffaa00;
            border: 1px solid rgba(255, 170, 0, 0.2);
        }
        
        /* ═══════ VIOLATION BADGES ═══════ */
        .violation-badge {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            margin: 2px;
        }
        
        .violation-helmet { background: rgba(255, 59, 48, 0.15); color: #ff3b30; border: 1px solid rgba(255, 59, 48, 0.3); }
        .violation-seatbelt { background: rgba(255, 149, 0, 0.15); color: #ff9500; border: 1px solid rgba(255, 149, 0, 0.3); }
        .violation-triple { background: rgba(255, 45, 85, 0.15); color: #ff2d55; border: 1px solid rgba(255, 45, 85, 0.3); }
        .violation-wrongside { background: rgba(175, 82, 222, 0.15); color: #af52de; border: 1px solid rgba(175, 82, 222, 0.3); }
        .violation-stopline { background: rgba(0, 122, 255, 0.15); color: #007aff; border: 1px solid rgba(0, 122, 255, 0.3); }
        .violation-redlight { background: rgba(255, 59, 48, 0.15); color: #ff3b30; border: 1px solid rgba(255, 59, 48, 0.3); }
        .violation-parking { background: rgba(90, 200, 250, 0.15); color: #5ac8fa; border: 1px solid rgba(90, 200, 250, 0.3); }
        .violation-plate { background: rgba(255, 69, 58, 0.15); color: #ff453a; border: 1px solid rgba(255, 69, 58, 0.3); }
        
        /* ═══════ SIDEBAR ═══════ */
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0d1117 0%, #0a0f1a 100%);
            border-right: 1px solid rgba(255,255,255,0.06);
        }
        
        [data-testid="stSidebar"] .stMarkdown h1 {
            font-size: 20px;
            color: #ffffff;
        }
        
        /* ═══════ CARDS / CONTAINERS ═══════ */
        .glass-card {
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 20px;
            margin: 8px 0;
        }
        
        /* ═══════ STREAMLIT OVERRIDES ═══════ */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: transparent;
        }
        
        .stTabs [data-baseweb="tab"] {
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            color: rgba(255,255,255,0.6);
            border: 1px solid rgba(255,255,255,0.08);
            padding: 8px 16px;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, rgba(0,210,255,0.15), rgba(58,123,213,0.15));
            color: #00d2ff;
            border-color: rgba(0,210,255,0.3);
        }
        
        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.02);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 10px;
        }
        
        /* ═══════ BUTTONS ═══════ */
        .stButton > button {
            background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3);
        }
        
        /* ═══════ FILE UPLOADER ═══════ */
        [data-testid="stFileUploader"] {
            background: rgba(255,255,255,0.02);
            border: 2px dashed rgba(0, 210, 255, 0.2);
            border-radius: 12px;
            padding: 16px;
        }
        
        /* ═══════ PIPELINE INDICATOR ═══════ */
        .pipeline-step {
            display: inline-flex;
            align-items: center;
            padding: 6px 14px;
            margin: 3px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 500;
        }
        
        .pipeline-active {
            background: rgba(0, 210, 255, 0.15);
            color: #00d2ff;
            border: 1px solid rgba(0, 210, 255, 0.3);
        }
        
        .pipeline-done {
            background: rgba(0, 255, 136, 0.1);
            color: #00ff88;
            border: 1px solid rgba(0, 255, 136, 0.2);
        }
        
        .pipeline-pending {
            background: rgba(255, 255, 255, 0.05);
            color: rgba(255,255,255,0.3);
            border: 1px solid rgba(255,255,255,0.08);
        }
    </style>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render the sidebar with system info and navigation."""
    with st.sidebar:
        # Logo and title
        st.markdown("""
        <div style="text-align: center; padding: 16px 0;">
            <div style="font-size: 48px; margin-bottom: 8px;">🚦</div>
            <h1 style="margin: 0; font-size: 22px; font-weight: 700; 
                        background: linear-gradient(135deg, #00d2ff, #3a7bd5);
                        -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                TrafficSentinel AI
            </h1>
            <p style="color: rgba(255,255,255,0.4); font-size: 11px; margin-top: 4px; letter-spacing: 1px;">
                HIERARCHICAL SCENE UNDERSTANDING
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # System Status — Real-time model file checks
        st.markdown("### ⚙️ System Status")
        
        _vehicle_ok = (PROJECT_ROOT / "yolo11m.pt").exists() or (PROJECT_ROOT / "models" / "weights" / "yolo11m.pt").exists()
        _helmet_11 = (PROJECT_ROOT / "models" / "weights" / "yolo11s_helmet.pt").exists()
        _helmet_8 = (PROJECT_ROOT / "models" / "weights" / "yolov8s_helmet.pt").exists()
        _helmet_ok = _helmet_11 or _helmet_8
        _helmet_label = "YOLOv11s" if _helmet_11 else ("YOLOv8s" if _helmet_8 else "YOLOv11s")
        
        _plate_11 = (PROJECT_ROOT / "models" / "weights" / "yolo11s_plate.pt").exists()
        _plate_8 = (PROJECT_ROOT / "models" / "weights" / "yolov8s_plate.pt").exists()
        _plate_ok = _plate_11 or _plate_8
        _plate_label = "YOLOv11s" if _plate_11 else ("YOLOv8s" if _plate_8 else "YOLOv11s")
        
        def _status(ok, name):
            if ok:
                return f'<span style="color: #00ff88; font-size: 13px;">✅ {name}</span>'
            return f'<span style="color: #ff3b30; font-size: 13px;">❌ Missing</span>'
        
        st.markdown(f"""
        <div class="glass-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Pipeline</span>
                <span class="status-badge status-online">● Online</span>
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Detection</span>
                {_status(_vehicle_ok, "YOLOv11m")}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Helmet</span>
                {_status(_helmet_ok, _helmet_label)}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="color: rgba(255,255,255,0.6); font-size: 13px;">Plate</span>
                {_status(_plate_ok, _plate_label)}
            </div>
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <span style="color: rgba(255,255,255,0.6); font-size: 13px;">OCR Engine</span>
                <span style="color: #00d2ff; font-size: 13px;">EasyOCR</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Pipeline Architecture
        st.markdown("### 🧠 HSUP Pipeline")
        st.markdown("""
        <div class="glass-card" style="font-size: 12px;">
            <div class="pipeline-step pipeline-done">✓ L1: Scene Conditioning</div>
            <div class="pipeline-step pipeline-done">✓ L2: Entity Detection</div>
            <div class="pipeline-step pipeline-done">✓ L3: Scene Graph</div>
            <div class="pipeline-step pipeline-done">✓ L4: Violation Reasoning</div>
            <div class="pipeline-step pipeline-done">✓ L5: Evidence Generation</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # Detection Settings
        st.markdown("### 🎛️ Detection Settings")
        
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.35,
            step=0.05,
            help="Minimum confidence for detections"
        )

        min_vehicle_area_ratio = st.slider(
            "Minimum Vehicle Size Filter",
            min_value=0.0,
            max_value=0.15,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="Filters out small/distant background vehicles to focus on near violations (area ratio threshold)"
        )
        
        use_sahi = st.toggle(
            "Enable SAHI (Small Object Detection)",
            value=False,
            help="Slicing Aided Hyper Inference for distant/small objects"
        )
        
        use_preprocessing = st.toggle(
            "Adaptive Preprocessing",
            value=True,
            help="Auto-detect and correct image degradation"
        )
        
        pillion_helmet_required = st.toggle(
            "Require Pillion Helmet",
            value=True,
            help="Enforce helmet compliance for pillion passengers (BTP regulations)"
        )
        
        # Store in session state
        st.session_state["confidence_threshold"] = confidence_threshold
        st.session_state["min_vehicle_area_ratio"] = min_vehicle_area_ratio
        st.session_state["use_sahi"] = use_sahi
        st.session_state["use_preprocessing"] = use_preprocessing
        st.session_state["pillion_helmet_required"] = pillion_helmet_required
        
        # Update global settings
        from config.settings import SETTINGS
        SETTINGS.pillion_helmet_required = pillion_helmet_required
        SETTINGS.vehicle_detector.confidence_threshold = confidence_threshold
        SETTINGS.min_vehicle_area_ratio = min_vehicle_area_ratio
        
        st.divider()
        
        # Violation Types Toggle
        st.markdown("### 🚨 Active Violations")
        violations = {
            "Helmet": st.toggle("Helmet Non-Compliance", value=True),
            "Seatbelt": st.toggle("Seatbelt Non-Compliance", value=True),
            "Triple Riding": st.toggle("Triple Riding", value=True),
            "Wrong Side": st.toggle("Wrong-Side Driving", value=True),
            "Stop Line": st.toggle("Stop-Line Violation", value=True),
            "Red Light": st.toggle("Red-Light Violation", value=True),
            "Illegal Parking": st.toggle("Illegal Parking", value=True),
            "Missing Plate": st.toggle("Missing License Plate / HSRP", value=True),
        }
        st.session_state["active_violations"] = violations
        
        st.divider()
        
        # Credits
        st.markdown("""
        <div style="text-align: center; padding: 8px 0;">
            <p style="color: rgba(255,255,255,0.3); font-size: 11px;">
                Gridlock Hackathon 2.0<br>
                Flipkart × BTP<br>
                v1.0.0
            </p>
        </div>
        """, unsafe_allow_html=True)


def render_home():
    """Render the home page with overview and quick start."""
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🚦 Traffic Command Center</h1>
        <p>Hierarchical Scene Understanding Pipeline (HSUP) — Automated Traffic Violation Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Hero metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    total_violations = st.session_state.get("total_violations", 0)
    total_processed = st.session_state.get("total_processed", 0)
    avg_confidence = st.session_state.get("avg_confidence", 0.0)
    plates_recognized = st.session_state.get("plates_recognized", 0)
    avg_fps = st.session_state.get("avg_fps", 0.0)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_violations}</div>
            <div class="metric-label">Violations Detected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_processed}</div>
            <div class="metric-label">Images Processed</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_confidence:.0%}</div>
            <div class="metric-label">Avg Confidence</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{plates_recognized}</div>
            <div class="metric-label">Plates Recognized</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_fps:.1f}</div>
            <div class="metric-label">Avg FPS</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture Overview
    st.markdown("### 🧠 HSUP Architecture Overview")
    
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00d2ff; margin-top: 0;">5-Layer Hierarchical Scene Understanding Pipeline</h4>
            <p style="color: rgba(255,255,255,0.7); font-size: 14px; line-height: 1.8;">
                Unlike traditional detect-and-classify approaches, TrafficSentinel AI builds a 
                <strong style="color: #ffffff;">spatial scene graph</strong> that models relationships 
                between detected entities — understanding <em>who</em> is riding <em>what</em>, 
                <em>where</em> they are relative to lane markings, and <em>whether</em> they comply 
                with traffic rules. This enables contextual violation reasoning that goes beyond 
                simple object detection.
            </p>
            <br>
            <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <div class="pipeline-step pipeline-done" style="flex: 1; min-width: 140px; text-align: center;">
                    <strong>Layer 1</strong><br>Scene Conditioning
                </div>
                <div class="pipeline-step pipeline-done" style="flex: 1; min-width: 140px; text-align: center;">
                    <strong>Layer 2</strong><br>Entity Detection
                </div>
                <div class="pipeline-step pipeline-done" style="flex: 1; min-width: 140px; text-align: center;">
                    <strong>Layer 3</strong><br>Scene Graph
                </div>
                <div class="pipeline-step pipeline-done" style="flex: 1; min-width: 140px; text-align: center;">
                    <strong>Layer 4</strong><br>Violation Reasoning
                </div>
                <div class="pipeline-step pipeline-done" style="flex: 1; min-width: 140px; text-align: center;">
                    <strong>Layer 5</strong><br>Evidence + e-Challan
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        st.markdown("""
        <div class="glass-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Supported Violations</h4>
            <div class="violation-badge violation-helmet">🪖 Helmet Non-Compliance</div>
            <div class="violation-badge violation-seatbelt">🔗 Seatbelt Non-Compliance</div>
            <div class="violation-badge violation-triple">👥 Triple Riding</div>
            <div class="violation-badge violation-wrongside">↩️ Wrong-Side Driving</div>
            <div class="violation-badge violation-stopline">🛑 Stop-Line Violation</div>
            <div class="violation-badge violation-redlight">🔴 Red-Light Violation</div>
            <div class="violation-badge violation-parking">🅿️ Illegal Parking</div>
            <div class="violation-badge violation-plate">🚫 Missing Number Plate / HSRP</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Start Guide
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🚀 Quick Start")
    
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 36px; margin-bottom: 8px;">📸</div>
            <h4 style="color: #ffffff; margin: 0;">1. Upload Image</h4>
            <p style="color: rgba(255,255,255,0.5); font-size: 13px;">
                Go to <strong>Live Detection</strong> and upload a traffic image or video
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_b:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 36px; margin-bottom: 8px;">🔍</div>
            <h4 style="color: #ffffff; margin: 0;">2. Detect Violations</h4>
            <p style="color: rgba(255,255,255,0.5); font-size: 13px;">
                The HSUP pipeline analyzes the scene, builds a graph, and identifies violations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_c:
        st.markdown("""
        <div class="glass-card" style="text-align: center; min-height: 160px;">
            <div style="font-size: 36px; margin-bottom: 8px;">📋</div>
            <h4 style="color: #ffffff; margin: 0;">3. Review Evidence</h4>
            <p style="color: rgba(255,255,255,0.5); font-size: 13px;">
                Browse the <strong>Evidence Viewer</strong> for annotated proof and e-challans
            </p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    configure_page()
    inject_custom_css()
    render_sidebar()
    render_home()


if __name__ == "__main__":
    main()
