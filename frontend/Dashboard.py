"""
Streamlit Frontend for Network Intrusion Detection System
Dashboard overview page
"""

import streamlit as st

from api_client import check_api_health, get_models_info
from theme import apply_theme, render_hero

# Page configuration
st.set_page_config(
    page_title="NIDS ML System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

# API health
api_healthy = check_api_health()
status_class = "status-online" if api_healthy else "status-offline"
status_text = "API Online" if api_healthy else "API Offline"

render_hero(
    title="Network Intrusion Detection System",
    subtitle="Operational overview, system health, and quick access to analysis tools.",
    badge="ML-Based NIDS",
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
        st.metric("Features", models_info.get('n_features', 'N/A'))
        st.metric("Classes", models_info.get('n_classes', 'N/A'))
    else:
        st.warning("Models not loaded")

    st.divider()
    st.caption("Bells University of Technology")
    st.caption("ICT423 - Machine Learning Project")

# Main content
st.header("System Overview")

models_info = get_models_info()
if models_info:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Models", len(models_info.get('available_models', [])))
    col2.metric("Best Model", models_info.get('best_model', 'N/A'))
    col3.metric("Features", models_info.get('n_features', 'N/A'))
    col4.metric("Classes", models_info.get('n_classes', 'N/A'))

st.divider()

st.subheader("What You Can Do")
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("""
    <div class="section-card">
        <h4>Live Monitoring</h4>
        <p>Start a live capture session and watch traffic trends in real time. Export captured CSV data for model inference.</p>
        <p><strong>Go to:</strong> Live Monitoring</p>
    </div>
    """, unsafe_allow_html=True)

with col_b:
    st.markdown("""
    <div class="section-card">
        <h4>ML Analysis</h4>
        <p>Compare model performance, inspect confusion matrices, ROC curves, and feature importance for explainability.</p>
        <p><strong>Go to:</strong> ML Analysis</p>
    </div>
    """, unsafe_allow_html=True)

st.divider()

st.subheader("Workflow")
st.markdown("""
1. Train models and generate metadata using the training pipeline.
2. Start the FastAPI backend to serve predictions and metrics.
3. Use Live Monitoring for capture and quick inference.
4. Review model performance and artifacts in ML Analysis.
""")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #4b5563; padding: 20px;'>
    <p><strong>Bells University of Technology</strong></p>
    <p>ICT423 - Machine Learning Research Project</p>
    <p>Network Intrusion Detection System with Machine Learning</p>
    <p>Developed using Streamlit and FastAPI</p>
</div>
""", unsafe_allow_html=True)