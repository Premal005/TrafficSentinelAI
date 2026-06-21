"""
Evidence Viewer Page — TrafficSentinel AI.
Renders the detailed evidence packet and lets users inspect and download the official PDF e-Challan.
"""

import sys
from pathlib import Path
import streamlit as st
import cv2
import json

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app import render_sidebar, inject_custom_css, configure_page

# Configure page, inject styles and render sidebar
configure_page()
inject_custom_css()
render_sidebar()

st.title("📄 Evidence File Viewer & e-Challan")
st.write("Review legally binding evidence packets and generate ASTraM enforcement challans.")


# Retrieve history and selected notice
history = st.session_state.get("violation_history", [])
selected_id = st.session_state.get("selected_violation_id")

# If none selected, default to the most recent one
if not selected_id and history:
    selected_id = history[-1]["violation_id"]

# Find the packet
packet = None
for p in reversed(history):
    if p["violation_id"] == selected_id:
        packet = p
        break

if not packet:
    if not history:
        st.info("No violations registered yet. Please upload a traffic frame in Live Detection.")
    else:
        st.warning("Selected notice not found. Choose one from the Violation Log database.")
else:
    # ══════════════════════════════════════════════════════════
    # NOTICE HEADER
    # ══════════════════════════════════════════════════════════
    st.subheader(f"🔍 Notice: {packet['violation_id']}")
    
    col_meta, col_actions = st.columns([2, 1])
    
    with col_meta:
        # Load details
        v_type = packet["violation_name"]
        ts = packet["timestamp"].split('.')[0].replace('T', ' ')
        loc = packet["location"]
        section = packet["law_section"]
        fine = packet["fine_amount"]
        
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Infraction Summary</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Offence Type:</td>
                    <td style="padding: 8px 0; font-weight: 600; color: #ff3b30;">{v_type}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Registered Law Section:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{section}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Junction / Location:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{loc}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Time of Detection:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{ts}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Aggregated Fine:</td>
                    <td style="padding: 8px 0; font-weight: 700; color: #ffaa00; font-size: 16px;">INR {fine}/-</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
    with col_actions:
        veh = packet["vehicle"]
        st.markdown(f"""
        <div class="glass-card">
            <h4 style="color: #00d2ff; margin-top: 0;">Vehicle Diagnostics</h4>
            <table style="width: 100%; border-collapse: collapse; font-size: 14px;">
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Plate Number:</td>
                    <td style="padding: 8px 0; font-weight: 700; color: #00ff88; font-size: 15px;">{veh['plate_number']}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Vehicle Type:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{veh['type'].upper()}</td>
                </tr>
                <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">State of Issue:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{veh['plate_state']}</td>
                </tr>
                <tr>
                    <td style="padding: 8px 0; color: rgba(255,255,255,0.6);">Usage Category:</td>
                    <td style="padding: 8px 0; font-weight: 500;">{veh['plate_type'].upper()}</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)
        
        # Download PDF Challan Button
        if packet.get("pdf_path") and Path(packet["pdf_path"]).exists():
            with open(packet["pdf_path"], "rb") as pdf_file:
                st.download_button(
                    label="📥 Download Legal PDF Challan",
                    data=pdf_file,
                    file_name=Path(packet["pdf_path"]).name,
                    mime="application/pdf",
                    use_container_width=True
                )
        else:
            st.warning("PDF Challan is not compiled for this record.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # EVIDENCE IMAGES
    # ══════════════════════════════════════════════════════════
    st.subheader("🖼️ Photographic Evidence")
    
    # Check what images exist
    paths = packet.get("evidence_paths", {})
    
    col_photo1, col_photo2 = st.columns(2)
    
    with col_photo1:
        if paths.get("annotated_frame") and Path(PROJECT_ROOT / paths["annotated_frame"]).exists():
            img_path = PROJECT_ROOT / paths["annotated_frame"]
            st.image(str(img_path), caption="Annotated Overlays (BBox and Relations)", use_container_width=True)
        else:
            # Placeholder for mock record
            st.info("Annotated overview not available for this mock case.")
            
    with col_photo2:
        col_crop1, col_crop2 = st.columns(2)
        
        with col_crop1:
            if paths.get("vehicle_crop") and Path(PROJECT_ROOT / paths["vehicle_crop"]).exists():
                img_path = PROJECT_ROOT / paths["vehicle_crop"]
                st.image(str(img_path), caption="Cropped Vehicle", use_container_width=True)
            else:
                st.info("Vehicle crop crop not available.")
                
        with col_crop2:
            if paths.get("plate_crop") and Path(PROJECT_ROOT / paths["plate_crop"]).exists():
                img_path = PROJECT_ROOT / paths["plate_crop"]
                st.image(str(img_path), caption="Cropped License Plate", use_container_width=True)
            else:
                st.info("Plate crop not available.")

    # ══════════════════════════════════════════════════════════
    # LIVE CHALLAN RECEIPT PREVIEW (HTML STYLING)
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🧾 E-Challan Invoice Preview")
    
    is_dispatch = "police_dispatch" in packet
    dispatch_banner_html = ""
    payment_note_html = '<p style="margin: 0;">Please pay the fine within 60 days to avoid prosecution.</p>'
    
    if is_dispatch:
        disp = packet["police_dispatch"]
        dispatch_banner_html = f"""<div style="background-color: #ffebeb; border: 2px solid #ff3b30; border-radius: 6px; padding: 12px; margin-bottom: 16px; color: #d90429; font-weight: bold; font-size: 12px; text-align: left; line-height: 1.5;">
<div style="font-size: 14px; margin-bottom: 4px;">🚨 POLICE INTERCEPT DISPATCHED</div>
<div style="color: #1a1a2e; font-weight: normal;">
<strong>Unit:</strong> {disp['patrol_unit']} | <strong>Officer ID:</strong> {disp['officer_id']}<br>
<strong>Instructions:</strong> {disp['instructions']}
</div>
</div>"""
        payment_note_html = '<p style="margin: 0; color: #ff3b30; font-weight: bold;">OFFLINE ENFORCEMENT: PATROL DISPATCHED FOR INTERCEPTION. NO ONLINE PAYMENT.</p>'
    
    v_rows_html = ""
    v_list = packet.get("violations_list", [])
    if not v_list:
        v_list = [{
            "violation_name": packet["violation_name"],
            "law_section": packet["law_section"],
            "fine_amount": packet["fine_amount"]
        }]
        
    for item in v_list:
        v_rows_html += f"""<div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
<span style="font-weight: bold;">{item['violation_name']}</span>
<span>INR {item['fine_amount']}.00</span>
</div>
<div style="font-size: 11px; color: #555555; margin-bottom: 8px; border-bottom: 1px dashed #cccccc; padding-bottom: 4px;">
Law: {item['law_section']}
</div>"""

    st.markdown(f"""<div style="background-color: #ffffff; color: #1a1a2e; border: 1px solid #cccccc; border-radius: 8px; padding: 24px; font-family: monospace; max-width: 650px; margin: 0 auto; box-shadow: 0 4px 15px rgba(0,0,0,0.15);">
{dispatch_banner_html}
<div style="text-align: center; border-bottom: 2px dashed #1a1a2e; padding-bottom: 12px; margin-bottom: 16px;">
<h3 style="margin: 0; font-weight: 800;">BENGALURU TRAFFIC POLICE</h3>
<p style="margin: 4px 0 0 0; font-size: 12px;">AUTOMATED ENFORCEMENT ASTraM UNIT</p>
<p style="margin: 2px 0 0 0; font-size: 11px;">GOVERNMENT OF KARNATAKA</p>
</div>
<div style="font-size: 12px; line-height: 1.6; margin-bottom: 16px;">
<div style="display: flex; justify-content: space-between;">
<span>Notice Number:</span>
<strong>{packet['violation_id']}</strong>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Date & Time:</span>
<strong>{ts}</strong>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Location:</span>
<strong>{loc}</strong>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Vehicle Reg:</span>
<strong>{veh['plate_number']} ({veh['plate_state']})</strong>
</div>
<div style="display: flex; justify-content: space-between;">
<span>Vehicle Class:</span>
<strong>{veh['type'].upper()}</strong>
</div>
</div>
<div style="border-top: 1px dashed #1a1a2e; border-bottom: 1px dashed #1a1a2e; padding: 12px 0 4px 0; margin-bottom: 16px; font-size: 13px;">
{v_rows_html}
</div>
<div style="display: flex; justify-content: space-between; font-size: 16px; font-weight: 800; margin-bottom: 24px;">
<span>TOTAL FINE DUE:</span>
<span>INR {fine}.00</span>
</div>
<div style="text-align: center; border-top: 2px dashed #1a1a2e; padding-top: 16px; font-size: 10px;">
{payment_note_html}
<p style="margin: 4px 0 0 0; font-weight: bold;">*** THANK YOU FOR DRIVING SAFELY ***</p>
</div>
</div>""", unsafe_allow_html=True)
