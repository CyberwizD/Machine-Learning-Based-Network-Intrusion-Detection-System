#!/bin/bash
echo "Starting FastAPI backend..."
source venv/bin/activate
cd backend/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
