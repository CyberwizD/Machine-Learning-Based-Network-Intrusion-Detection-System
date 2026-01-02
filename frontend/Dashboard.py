"""
Streamlit Frontend for Network Intrusion Detection System
Complete UI with Live Monitoring and ML Analysis
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import os

# Page configuration
st.set_page_config(
    page_title="NIDS ML System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")

# Initialize session state
if 'capture_active' not in st.session_state:
    st.session_state.capture_active = False
if 'captured_data' not in st.session_state:
    st.session_state.captured_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'live_data' not in st.session_state:
    st.session_state.live_data = []


def check_api_health():
    """Check if API is accessible"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False


def get_models_info():
    """Get information about available models"""
    try:
        response = requests.get(f"{API_BASE_URL}/models")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def get_all_metrics():
    """Get metrics for all models"""
    try:
        response = requests.get(f"{API_BASE_URL}/metrics")
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def predict_csv(file_content):
    """Send CSV for prediction"""
    try:
        files = {'file': file_content}
        response = requests.post(f"{API_BASE_URL}/predict/csv", files=files)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error: {e}")
        return None


# Header
st.markdown('<h1 class="main-header">🛡️ Network Intrusion Detection System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Machine Learning-Powered Cyber Threat Detection</p>', unsafe_allow_html=True)

# Check API health
api_healthy = check_api_health()

if not api_healthy:
    st.error("⚠️ Backend API is not accessible. Please ensure FastAPI server is running on port 8000.")
    st.info("Run: `uvicorn backend.api.main:app --reload`")
    st.stop()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/network-shield.png", width=100)
    st.title("System Status")
    
    # System metrics
    models_info = get_models_info()
    if models_info:
        st.success("✅ System Online")
        st.metric("Models Loaded", len(models_info.get('available_models', [])))
        st.metric("Best Model", models_info.get('best_model', 'N/A'))
        st.metric("Features", models_info.get('n_features', 'N/A'))
        st.metric("Classes", models_info.get('n_classes', 'N/A'))
    else:
        st.warning("⚠️ Models not loaded")
    
    st.divider()
    
    # Settings
    st.subheader("⚙️ Settings")
    capture_duration = st.slider("Capture Duration (seconds)", 10, 300, 60)
    max_file_size = st.slider("Max File Size (MB)", 1, 50, 5)
    
    st.divider()
    
    st.caption("Bells University of Technology")
    st.caption("ICT423 - Machine Learning Project")

# Main tabs
tab1, tab2 = st.tabs(["🌐 Live Network Monitoring", "📊 ML Analysis & Training Results"])

# ==========================
# TAB 1: Live Network Monitoring
# ==========================
with tab1:
    st.header("Live Network Traffic Analysis")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Network Capture Control")
        
        col_a, col_b, col_c = st.columns(3)
        
        with col_a:
            if st.button("🎬 Start Capture", type="primary", disabled=st.session_state.capture_active):
                st.session_state.capture_active = True
                st.session_state.captured_data = None
                st.rerun()
        
        with col_b:
            if st.button("⏹️ Stop Capture", disabled=not st.session_state.capture_active):
                st.session_state.capture_active = False
                st.success("Network traffic data saved!")
                st.rerun()
        
        with col_c:
            if st.button("🔄 Reset"):
                st.session_state.capture_active = False
                st.session_state.captured_data = None
                st.session_state.analysis_results = None
                st.session_state.live_data = []
                st.rerun()
    
    with col2:
        if st.session_state.capture_active:
            st.info("🔴 Capture Active")
        else:
            st.success("⚪ Ready")
    
    st.divider()
    
    # Live visualization
    if st.session_state.capture_active:
        st.subheader("📡 Real-Time Network Traffic")
        
        # Simulate live data (in production, this would be real packet capture)
        placeholder_chart = st.empty()
        placeholder_metrics = st.empty()
        
        for i in range(capture_duration):
            if not st.session_state.capture_active:
                break
            
            # Simulate network data
            timestamp = datetime.now() - timedelta(seconds=capture_duration-i)
            packets = np.random.randint(100, 1000)
            bytes_sent = np.random.randint(1000, 10000)
            threats = np.random.randint(0, 5)
            
            st.session_state.live_data.append({
                'timestamp': timestamp,
                'packets': packets,
                'bytes': bytes_sent,
                'threats': threats
            })
            
            # Keep only last 60 seconds
            if len(st.session_state.live_data) > 60:
                st.session_state.live_data.pop(0)
            
            # Create dataframe
            df_live = pd.DataFrame(st.session_state.live_data)
            
            # Plot
            with placeholder_chart.container():
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=('Packet Rate', 'Bandwidth Usage', 'Threats Detected', 'Protocol Distribution'),
                    specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                          [{'type': 'scatter'}, {'type': 'pie'}]]
                )
                
                # Packet rate
                fig.add_trace(
                    go.Scatter(x=df_live['timestamp'], y=df_live['packets'],
                              mode='lines', name='Packets/s',
                              line=dict(color='#667eea', width=2)),
                    row=1, col=1
                )
                
                # Bandwidth
                fig.add_trace(
                    go.Scatter(x=df_live['timestamp'], y=df_live['bytes'],
                              mode='lines', name='Bytes/s',
                              line=dict(color='#764ba2', width=2)),
                    row=1, col=2
                )
                
                # Threats
                fig.add_trace(
                    go.Scatter(x=df_live['timestamp'], y=df_live['threats'],
                              mode='lines+markers', name='Threats',
                              line=dict(color='#f56565', width=2)),
                    row=2, col=1
                )
                
                # Protocol distribution (simulated)
                protocols = ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS']
                values = [40, 30, 10, 15, 5]
                fig.add_trace(
                    go.Pie(labels=protocols, values=values, hole=0.3),
                    row=2, col=2
                )
                
                fig.update_layout(height=600, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            # Metrics
            with placeholder_metrics.container():
                met_col1, met_col2, met_col3, met_col4 = st.columns(4)
                met_col1.metric("Total Packets", f"{sum(df_live['packets']):,}")
                met_col2.metric("Total Bytes", f"{sum(df_live['bytes'])/1024:.1f} KB")
                met_col3.metric("Threats", sum(df_live['threats']))
                met_col4.metric("Duration", f"{len(df_live)}s")
            
            time.sleep(1)
        
        if not st.session_state.capture_active and st.session_state.live_data:
            st.success("✅ Capture completed!")
            
            # Save to CSV (simulated)
            df_captured = pd.DataFrame(st.session_state.live_data)
            csv_path = Path("data/captured") / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df_captured.to_csv(csv_path, index=False)
            
            st.info(f"📁 Data saved to: {csv_path}")
    
    else:
        st.info("👆 Click 'Start Capture' to begin monitoring network traffic")
    
    st.divider()
    
    # Analysis section
    st.subheader("🔍 Analyze Captured Data")
    
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
        analyze_button = st.button("🚀 Analyze", type="primary", disabled=uploaded_file is None)
    
    if analyze_button and uploaded_file:
        with st.spinner("Analyzing network traffic..."):
            results = predict_csv(uploaded_file)
            
            if results:
                st.session_state.analysis_results = results
                
                # Display results
                st.success("✅ Analysis Complete!")
                
                # Summary metrics
                st.subheader("Analysis Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                col1.metric("Total Samples", results['total_samples'])
                col2.metric("Attacks Detected", results['attacks_detected'], 
                           delta=f"{(results['attacks_detected']/results['total_samples']*100):.1f}%")
                col3.metric("Normal Traffic", results['normal_traffic'])
                col4.metric("Avg Confidence", f"{results['average_confidence']:.2%}")
                
                # Attack types
                st.subheader("Detected Attack Types")
                attack_types = results.get('attack_types', {})
                
                if attack_types:
                    fig = go.Figure(data=[
                        go.Bar(
                            x=list(attack_types.keys()),
                            y=list(attack_types.values()),
                            marker_color='indianred'
                        )
                    ])
                    fig.update_layout(
                        title="Distribution of Attack Types",
                        xaxis_title="Attack Type",
                        yaxis_title="Count",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Sample predictions
                st.subheader("Sample Predictions (First 20)")
                samples_df = pd.DataFrame(results['samples'][:20])
                st.dataframe(samples_df, use_container_width=True)
                
                # Download results
                st.download_button(
                    label="📥 Download Full Results",
                    data=json.dumps(results, indent=2),
                    file_name=f"analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )


# ==========================
# TAB 2: ML Analysis & Training Results
# ==========================
with tab2:
    st.header("Machine Learning Analysis Dashboard")
    
    # Get metrics
    metrics_data = get_all_metrics()
    
    if metrics_data:
        st.success(f"✅ Best Performing Model: **{metrics_data['best_model']}**")
        
        # Model comparison
        st.subheader("📈 Model Performance Comparison")
        
        metrics_df = pd.DataFrame(metrics_data['metrics'])
        
        # Create comparison chart
        fig = go.Figure()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        
        for i, metric in enumerate(metrics_to_plot):
            fig.add_trace(go.Bar(
                name=metric.replace('_', ' ').title(),
                x=metrics_df['model_name'],
                y=metrics_df[metric],
                marker_color=colors[i]
            ))
        
        fig.update_layout(
            barmode='group',
            title="Model Performance Metrics Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.divider()
        
        # Detailed metrics table
        st.subheader("📊 Detailed Performance Metrics")
        st.dataframe(
            metrics_df.style.highlight_max(axis=0, subset=['accuracy', 'precision', 'recall', 'f1_score']),
            use_container_width=True
        )
        
        st.divider()
        
        # Model selection for detailed analysis
        st.subheader("🔬 Detailed Model Analysis")
        
        selected_model = st.selectbox(
            "Select a model for detailed analysis:",
            metrics_df['model_name'].tolist()
        )
        
        if selected_model:
            col1, col2 = st.columns(2)
            
            with col1:
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                try:
                    response = requests.get(f"{API_BASE_URL}/confusion-matrix/{selected_model}")
                    if response.status_code == 200:
                        cm_data = response.json()
                        cm = np.array(cm_data['confusion_matrix'])
                        class_names = cm_data['class_names']
                        
                        fig = px.imshow(
                            cm,
                            labels=dict(x="Predicted", y="True", color="Count"),
                            x=class_names,
                            y=class_names,
                            color_continuous_scale='Blues',
                            text_auto=True
                        )
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                except:
                    st.warning("Confusion matrix not available")
            
            with col2:
                # ROC Curve
                st.subheader("ROC Curve")
                try:
                    response = requests.get(f"{API_BASE_URL}/roc-curve/{selected_model}")
                    if response.status_code == 200:
                        roc_data = response.json()
                        
                        if not roc_data.get('multiclass'):
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=roc_data['fpr'],
                                y=roc_data['tpr'],
                                mode='lines',
                                name=f"AUC = {roc_data['auc']:.3f}",
                                line=dict(color='#667eea', width=2)
                            ))
                            fig.add_trace(go.Scatter(
                                x=[0, 1],
                                y=[0, 1],
                                mode='lines',
                                name='Random Classifier',
                                line=dict(color='gray', dash='dash')
                            ))
                            fig.update_layout(
                                title="ROC Curve",
                                xaxis_title="False Positive Rate",
                                yaxis_title="True Positive Rate",
                                height=400
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Multiclass ROC curves available - showing macro average")
                except:
                    st.warning("ROC curve not available")
            
            # Classification Report
            st.subheader("Classification Report")
            try:
                response = requests.get(f"{API_BASE_URL}/classification-report/{selected_model}")
                if response.status_code == 200:
                    report_data = response.json()
                    report = report_data['report']
                    
                    # Convert to DataFrame for better display
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)
            except:
                st.warning("Classification report not available")
        
        st.divider()
        
        # Feature Importance
        st.subheader("🎯 Feature Importance")
        try:
            response = requests.get(f"{API_BASE_URL}/feature-importance")
            if response.status_code == 200:
                fi_data = response.json()
                
                features = [f['feature'] for f in fi_data['top_features']]
                importances = [f['importance'] for f in fi_data['top_features']]
                
                fig = go.Figure(go.Bar(
                    x=importances,
                    y=features,
                    orientation='h',
                    marker_color='#667eea'
                ))
                fig.update_layout(
                    title=f"Top 20 Most Important Features - {fi_data['model_name']}",
                    xaxis_title="Importance",
                    yaxis_title="Feature",
                    height=600
                )
                st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Feature importance available only for tree-based models")
        
        st.divider()
        
        # Training visualizations from saved images
        st.subheader("📸 Training Visualizations")
        
        viz_dir = Path("models/visualizations")
        if viz_dir.exists():
            viz_files = list(viz_dir.glob("*.png"))
            
            if viz_files:
                selected_viz = st.selectbox(
                    "Select visualization:",
                    [f.stem.replace('_', ' ').title() for f in viz_files]
                )
                
                if selected_viz:
                    viz_file = viz_dir / f"{selected_viz.lower().replace(' ', '_')}.png"
                    if viz_file.exists():
                        st.image(str(viz_file), use_column_width=True)
            else:
                st.info("No training visualizations found. Run training pipeline first.")
        else:
            st.info("Visualizations directory not found. Run training pipeline first.")
    
    else:
        st.warning("⚠️ No model metrics available. Please train models first.")
        
        st.info("""
        ### How to Train Models:
        
        1. Prepare your dataset in CSV format
        2. Run the training pipeline:
        ```python
        python backend/ml/train.py
        ```
        3. Refresh this page to see results
        """)

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>Bells University of Technology</strong></p>
    <p>ICT423 - Machine Learning Research Project</p>
    <p>Network Intrusion Detection System with Machine Learning</p>
    <p>Developed with ❤️ using Streamlit & FastAPI</p>
</div>
""", unsafe_allow_html=True)
