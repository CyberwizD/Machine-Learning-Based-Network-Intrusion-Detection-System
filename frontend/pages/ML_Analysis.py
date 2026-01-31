"""
ML Analysis page for the NIDS Streamlit app
"""

import sys
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

sys.path.append(str(Path(__file__).resolve().parents[1]))

from api_client import (
    check_api_health,
    get_models_info,
    get_all_metrics,
    get_confusion_matrix,
    get_roc_curve,
    get_classification_report,
    get_feature_importance,
)
from theme import apply_theme, render_hero

# Page configuration
st.set_page_config(
    page_title="ML Analysis | NIDS ML System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

apply_theme()

api_healthy = check_api_health()
status_class = "status-online" if api_healthy else "status-offline"
status_text = "API Online" if api_healthy else "API Offline"

render_hero(
    title="ML Analysis & Training Results",
    subtitle="Compare models, inspect errors, and review explainability artifacts.",
    badge="Model Lab",
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
    st.caption("Bells University of Technology")
    st.caption("ICT423 - Machine Learning Project")

# Main content
st.header("Machine Learning Analysis Dashboard")

metrics_data = get_all_metrics()

if metrics_data:
    st.success(f"Best Performing Model: {metrics_data['best_model']}")

    st.subheader("Model Performance Comparison")

    metrics_df = pd.DataFrame(metrics_data['metrics'])

    fig = go.Figure()

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
    colors = ['#0f766e', '#0ea5e9', '#f59e0b', '#ef4444']

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

    st.subheader("Detailed Performance Metrics")
    st.dataframe(
        metrics_df.style.highlight_max(axis=0, subset=['accuracy', 'precision', 'recall', 'f1_score']),
        use_container_width=True
    )

    st.divider()

    st.subheader("Detailed Model Analysis")

    selected_model = st.selectbox(
        "Select a model for detailed analysis:",
        metrics_df['model_name'].tolist()
    )

    if selected_model:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Confusion Matrix")
            cm_data = get_confusion_matrix(selected_model)
            if cm_data:
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
            else:
                st.warning("Confusion matrix not available")

        with col2:
            st.subheader("ROC Curve")
            roc_data = get_roc_curve(selected_model)
            if roc_data:
                if not roc_data.get('multiclass'):
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=roc_data['fpr'],
                        y=roc_data['tpr'],
                        mode='lines',
                        name=f"AUC = {roc_data['auc']:.3f}",
                        line=dict(color='#0f766e', width=2)
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
            else:
                st.warning("ROC curve not available")

        st.subheader("Classification Report")
        report_data = get_classification_report(selected_model)
        if report_data:
            report = report_data['report']
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)
        else:
            st.warning("Classification report not available")

    st.divider()

    st.subheader("Feature Importance")
    fi_data = get_feature_importance()
    if fi_data:
        features = [f['feature'] for f in fi_data['top_features']]
        importances = [f['importance'] for f in fi_data['top_features']]

        fig = go.Figure(go.Bar(
            x=importances,
            y=features,
            orientation='h',
            marker_color='#0f766e'
        ))
        fig.update_layout(
            title=f"Top 20 Most Important Features - {fi_data['model_name']}",
            xaxis_title="Importance",
            yaxis_title="Feature",
            height=600
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance available only for tree-based models")

    st.divider()

    st.subheader("Training Visualizations")

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
    st.warning("No model metrics available. Please train models first.")

    st.info("""
    How to Train Models:

    1. Prepare your dataset in CSV format
    2. Run the training pipeline:
    ```python
    python backend/ml/train.py
    ```
    3. Refresh this page to see results
    """)