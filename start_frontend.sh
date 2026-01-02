#!/bin/bash
echo "Starting Streamlit frontend..."
source venv/bin/activate
cd frontend
streamlit run app.py
