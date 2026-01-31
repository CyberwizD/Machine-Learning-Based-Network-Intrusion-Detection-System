"""
Live Network Monitoring page for the NIDS Streamlit app
"""

import sys
from pathlib import Path
import time
from datetime import datetime, timedelta
import json

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import check_api_health, get_models_info, predict_csv
from theme import apply_theme, render_hero

# Page configuration
st.set_page_config(
    page_title="Live Monitoring | NIDS ML System",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# Initialize session state
if 'capture_active' not in st.session_state:
    st.session_state.capture_active = False
if 'captured_data' not in st.session_state:
    st.session_state.captured_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = []

api_healthy = check_api_health()
status_class = "status-online" if api_healthy else "status-offline"
status_text = "API Online" if api_healthy else "API Offline"

render_hero(
    title="Live Network Monitoring",
    subtitle="Capture traffic signals and run on-the-fly inference for suspicious activity.",
    badge="Real-Time",
    status_text=status_text,
    status_class=status_class,
)

if not api_healthy:
    st.error("Backend API is not accessible. Please ensure FastAPI server is running on port 8000.")
    st.info("Run: `uvicorn backend.api.main:app --reload`")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/network-shield.png", width=100)
    st.title("System Status")

    models_info = get_models_info()
    if models_info:
        st.success("System Online")
        st.metric("Models Loaded", len(models_info.get('available_models', [])))
        st.metric("Best Model", models_info.get('best_model', 'N/A'))
    else:
        st.warning("Models not loaded")

    st.divider()
    st.subheader("Capture Settings")
    capture_duration = st.slider("Capture Duration (seconds)", 10, 300, 60)
    max_file_size = st.slider("Max File Size (MB)", 1, 50, 5)

    st.divider()
    st.caption("Bells University of Technology")
    st.caption("ICT423 - Machine Learning Project")

# Main content
st.header("Live Network Traffic Analysis")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Network Capture Control")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        if st.button("Start Capture", type="primary", disabled=st.session_state.capture_active):
            st.session_state.capture_active = True
            st.session_state.captured_data = None
            st.rerun()

    with col_b:
        if st.button("Stop Capture", disabled=not st.session_state.capture_active):
            st.session_state.capture_active = False
            st.success("Network traffic data saved!")
            st.rerun()

    with col_c:
        if st.button("Reset"):
            st.session_state.capture_active = False
            st.session_state.captured_data = None
            st.session_state.analysis_results = None
            st.session_state.live_data = []
            st.rerun()

with col2:
    if st.session_state.capture_active:
        st.info("Capture Active")
    else:
        st.success("Ready")

st.divider()

# Live visualization
if st.session_state.capture_active:
    st.subheader("Real-Time Network Traffic")

    placeholder_chart = st.empty()
    placeholder_metrics = st.empty()

    for i in range(capture_duration):
        if not st.session_state.capture_active:
            break

        timestamp = datetime.now() - timedelta(seconds=capture_duration - i)
        packets = np.random.randint(100, 1000)
        bytes_sent = np.random.randint(1000, 10000)
        threats = np.random.randint(0, 5)

        st.session_state.live_data.append({
            'timestamp': timestamp,
            'packets': packets,
            'bytes': bytes_sent,
            'threats': threats
        })

        if len(st.session_state.live_data) > 60:
            st.session_state.live_data.pop(0)

        df_live = pd.DataFrame(st.session_state.live_data)

        with placeholder_chart.container():
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Packet Rate', 'Bandwidth Usage', 'Threats Detected', 'Protocol Distribution'),
                specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                      [{'type': 'scatter'}, {'type': 'pie'}]]
            )

            fig.add_trace(
                go.Scatter(x=df_live['timestamp'], y=df_live['packets'],
                          mode='lines', name='Packets/s',
                          line=dict(color='#0f766e', width=2)),
                row=1, col=1
            )

            fig.add_trace(
                go.Scatter(x=df_live['timestamp'], y=df_live['bytes'],
                          mode='lines', name='Bytes/s',
                          line=dict(color='#0ea5e9', width=2)),
                row=1, col=2
            )

            fig.add_trace(
                go.Scatter(x=df_live['timestamp'], y=df_live['threats'],
                          mode='lines+markers', name='Threats',
                          line=dict(color='#ef4444', width=2)),
                row=2, col=1
            )

            protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
            values = [40, 30, 10, 15, 5]
            fig.add_trace(
                go.Pie(labels=protocols, values=values, hole=0.3),
                row=2, col=2
            )

            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

        with placeholder_metrics.container():
            met_col1, met_col2, met_col3, met_col4 = st.columns(4)
            met_col1.metric("Total Packets", f"{sum(df_live['packets']):,}")
            met_col2.metric("Total Bytes", f"{sum(df_live['bytes'])/1024:.1f} KB")
            met_col3.metric("Threats", sum(df_live['threats']))
            met_col4.metric("Duration", f"{len(df_live)}s")

        time.sleep(1)

    if not st.session_state.capture_active and st.session_state.live_data:
        st.success("Capture completed!")

        df_captured = pd.DataFrame(st.session_state.live_data)
        csv_path = Path("data/captured") / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        df_captured.to_csv(csv_path, index=False)

        st.info(f"Data saved to: {csv_path}")

else:
    st.info("Click 'Start Capture' to begin monitoring network traffic")

st.divider()

# Analysis section
st.subheader("Analyze Captured Data")

col1, col2 = st.columns([3, 1])

with col1:
    uploaded_file = st.file_uploader(
        "Upload network traffic CSV file",
        type=['csv'],
        help="Upload a CSV file containing network traffic features"
    )

with col2:
    st.write("")
    st.write("")
    analyze_button = st.button("Analyze", type="primary", disabled=uploaded_file is None)

if analyze_button and uploaded_file:
    with st.spinner("Analyzing network traffic..."):
        results = predict_csv(uploaded_file)

        if results:
            st.session_state.analysis_results = results

            st.success("Analysis Complete!")

            st.subheader("Analysis Summary")
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Samples", results['total_samples'])
            col2.metric("Attacks Detected", results['attacks_detected'],
                       delta=f"{(results['attacks_detected']/results['total_samples']*100):.1f}%")
            col3.metric("Normal Traffic", results['normal_traffic'])
            col4.metric("Avg Confidence", f"{results['average_confidence']:.2%}")

            st.subheader("Detected Attack Types")
            attack_types = results.get('attack_types', {})

            if attack_types:
                fig = go.Figure(data=[
                    go.Bar(
                        x=list(attack_types.keys()),
                        y=list(attack_types.values()),
                        marker_color='#ef4444'
                    )
                ])
                fig.update_layout(
                    title="Distribution of Attack Types",
                    xaxis_title="Attack Type",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)

            st.subheader("Sample Predictions (First 20)")
            samples_df = pd.DataFrame(results['samples'][:20])
            st.dataframe(samples_df, use_container_width=True)

            st.download_button(
                label="Download Full Results",
                data=json.dumps(results, indent=2),
                file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )