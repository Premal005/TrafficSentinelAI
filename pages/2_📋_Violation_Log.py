"""
Violation Log Page — TrafficSentinel AI.
Displays a searchable, filterable database table of all infractions.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import render_sidebar, inject_custom_css, configure_page

# Configure page, inject styles and render sidebar
configure_page()
inject_custom_css()
render_sidebar()

from config.settings import ViolationType, VIOLATION_DISPLAY_NAMES, VIOLATION_FINES


# Ensure state lists are initialized
if "violation_history" not in st.session_state:
    st.session_state["violation_history"] = []
if "selected_violation_id" not in st.session_state:
    st.session_state["selected_violation_id"] = None

# Populate with realistic mock records if history is empty, to ensure the demo looks premium
if not st.session_state["violation_history"]:
    mock_records = [
        {
            "violation_id": "VIO-20260616-001",
            "timestamp": "2026-06-16T07:15:30+05:30",
            "violation_type": "helmet_non_compliance",
            "violation_name": "Helmet Non-Compliance",
            "confidence": 0.94,
            "location": "CAM-JCN-01 (MG Road)",
            "law_section": "Section 194D - MVA 2019",
            "fine_amount": 1000,
            "vehicle": {"type": "motorcycle", "plate_number": "KA-03-HA-8821", "plate_state": "Karnataka", "plate_type": "private"},
            "evidence_paths": {},
            "scene_context": {"weather": "clear", "traffic_density": "high"}
        },
        {
            "violation_id": "VIO-20260616-002",
            "timestamp": "2026-06-16T07:22:45+05:30",
            "violation_type": "triple_riding",
            "violation_name": "Triple Riding",
            "confidence": 0.88,
            "location": "CAM-JCN-03 (Indiranagar)",
            "law_section": "Section 194C - MVA 2019",
            "fine_amount": 2000,
            "vehicle": {"type": "motorcycle", "plate_number": "KA-05-EQ-4491", "plate_state": "Karnataka", "plate_type": "private"},
            "evidence_paths": {},
            "scene_context": {"weather": "clear", "traffic_density": "medium"}
        },
        {
            "violation_id": "VIO-20260616-003",
            "timestamp": "2026-06-16T07:34:12+05:30",
            "violation_type": "seatbelt_non_compliance",
            "violation_name": "Seatbelt Non-Compliance",
            "confidence": 0.81,
            "location": "CAM-JCN-01 (MG Road)",
            "law_section": "Section 194B(1) - MVA 2019",
            "fine_amount": 1000,
            "vehicle": {"type": "car", "plate_number": "KA-51-MB-1002", "plate_state": "Karnataka", "plate_type": "private"},
            "evidence_paths": {},
            "scene_context": {"weather": "rainy", "traffic_density": "high"}
        },
        {
            "violation_id": "VIO-20260616-004",
            "timestamp": "2026-06-16T07:48:05+05:30",
            "violation_type": "red_light_violation",
            "violation_name": "Red-Light Violation",
            "confidence": 0.96,
            "location": "CAM-JCN-02 (Koramangala)",
            "law_section": "Section 177A - MVA 2019",
            "fine_amount": 5000,
            "vehicle": {"type": "auto_rickshaw", "plate_number": "KA-01-F-9902", "plate_state": "Karnataka", "plate_type": "commercial"},
            "evidence_paths": {},
            "scene_context": {"weather": "clear", "traffic_density": "low"}
        },
        {
            "violation_id": "VIO-20260616-005",
            "timestamp": "2026-06-16T08:05:19+05:30",
            "violation_type": "wrong_side_driving",
            "violation_name": "Wrong-Side Driving",
            "confidence": 0.90,
            "location": "CAM-JCN-04 (Hebbal Flyover)",
            "law_section": "Section 194D - MVA 2019",
            "fine_amount": 5000,
            "vehicle": {"type": "truck", "plate_number": "MH-12-Q-7781", "plate_state": "Maharashtra", "plate_type": "commercial"},
            "evidence_paths": {},
            "scene_context": {"weather": "cloudy", "traffic_density": "medium"}
        }
    ]
    st.session_state["violation_history"] = mock_records
    st.session_state["total_violations"] = len(mock_records)
    st.session_state["total_processed"] = len(mock_records) + 4

st.title("📋 Violation Database Log")
st.write("Browse, search, and manage all traffic violations registered across junctions.")

# Sidebar Filters
st.sidebar.subheader("🔍 Log Filters")

search_query = st.sidebar.text_input("Search License Plate", value="", placeholder="e.g. KA-05")

violation_filter = st.sidebar.multiselect(
    "Violation Type",
    options=list(VIOLATION_DISPLAY_NAMES.values()),
    default=[]
)

vehicle_filter = st.sidebar.multiselect(
    "Vehicle Category",
    options=["car", "motorcycle", "auto_rickshaw", "bus", "truck", "tempo"],
    default=[]
)

min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.4, 0.05)

# Process Data into DataFrame
data = []
for v in st.session_state["violation_history"]:
    status = f"Dispatched ({v['police_dispatch']['patrol_unit']})" if "police_dispatch" in v else "Challan Issued"
    data.append({
        "ID": v["violation_id"],
        "Timestamp": v["timestamp"].split('+')[0].replace('T', ' '),
        "Violation": v["violation_name"],
        "Plate Number": v["vehicle"]["plate_number"],
        "Vehicle Type": v["vehicle"]["type"].upper(),
        "Confidence": v["confidence"],
        "Fine (INR)": v["fine_amount"],
        "Enforcement Status": status,
        "Location": v["location"]
    })

df = pd.DataFrame(data)

# Apply Filters
if not df.empty:
    if search_query:
        df = df[df["Plate Number"].str.contains(search_query, case=False)]
        
    if violation_filter:
        df = df[df["Violation"].isin(violation_filter)]
        
    if vehicle_filter:
        df = df[df["Vehicle Type"].str.lower().isin(vehicle_filter)]
        
    df = df[df["Confidence"] >= min_conf]

# Show Results
st.subheader(f"📊 Query Results ({len(df)} records found)")

if df.empty:
    st.warning("No records match the selected filters.")
else:
    # Format confidence as percentage
    df_display = df.copy()
    df_display["Confidence"] = df_display["Confidence"].apply(lambda x: f"{x:.1%}")
    df_display["Fine (INR)"] = df_display["Fine (INR)"].apply(lambda x: f"Rs. {x:,}")

    # Display table with streamlit selection dataframe
    # Allow clicking/selecting a row
    selected_rows = st.dataframe(
        df_display,
        use_container_width=True,
        hide_index=True,
        column_order=["ID", "Timestamp", "Violation", "Plate Number", "Vehicle Type", "Confidence", "Fine (INR)", "Enforcement Status", "Location"]
    )
    
    # Layout for quick actions
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("⚡ Quick Actions")
    
    col_sel, col_action = st.columns([1, 1])
    
    with col_sel:
        selected_id = st.selectbox(
            "Select Violation Notice to Review:",
            options=df["ID"].tolist()
        )
        
    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        if st.button("🔍 Load in Evidence Viewer", use_container_width=True):
            st.session_state["selected_violation_id"] = selected_id
            st.success(f"Loaded {selected_id}. Please navigate to 📄 Evidence Viewer.")
            
            # Auto navigate page (or instruct user to click)
            st.info("Notice selected! Click '4 📄 Evidence Viewer' on the sidebar to review full details.")
