#!/bin/bash

# NIDS ML System Setup Script
# Bells University of Technology - ICT423

echo "=========================================="
echo "NIDS ML System Setup"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.8.0"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo -e "${GREEN}✓ Python $python_version detected${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required. Found: $python_version${NC}"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠ Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip setuptools wheel
echo -e "${GREEN}✓ Pip upgraded${NC}"

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install dependencies${NC}"
    exit 1
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/captured
mkdir -p models
mkdir -p models/visualizations
mkdir -p notebooks
mkdir -p logs
echo -e "${GREEN}✓ Directory structure created${NC}"

# Create .env file
echo ""
echo "Creating environment configuration..."
if [ ! -f ".env" ]; then
    cat > .env << EOF
# API Configuration
API_URL=http://localhost:8000
API_HOST=0.0.0.0
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501

# Model Configuration
MODEL_DIR=models
DATA_DIR=data

# Logging
LOG_LEVEL=INFO
EOF
    echo -e "${GREEN}✓ .env file created${NC}"
else
    echo -e "${YELLOW}⚠ .env file already exists${NC}"
fi

# Download sample dataset (NSL-KDD)
echo ""
echo "Downloading sample dataset (NSL-KDD)..."
if [ ! -f "data/raw/KDDTrain+.txt" ]; then
    echo "Please download NSL-KDD dataset manually from:"
    echo "https://www.unb.ca/cic/datasets/nsl.html"
    echo ""
    echo "Place the files in data/raw/ directory:"
    echo "  - KDDTrain+.txt"
    echo "  - KDDTest+.txt"
    echo -e "${YELLOW}⚠ Manual download required${NC}"
else
    echo -e "${GREEN}✓ Dataset files found${NC}"
fi

# Create sample configuration files
echo ""
echo "Creating sample configuration files..."

# Create config.py
cat > backend/utils/config.py << 'EOF'
"""
Configuration management for NIDS ML System
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Directories
    BASE_DIR = Path(__file__).parent.parent.parent
    MODEL_DIR = BASE_DIR / os.getenv('MODEL_DIR', 'models')
    DATA_DIR = BASE_DIR / os.getenv('DATA_DIR', 'data')
    LOG_DIR = BASE_DIR / 'logs'
    
    # API Settings
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 8000))
    
    # Model Settings
    BEST_MODEL_PATH = MODEL_DIR / 'best_model.pkl'
    SCALER_PATH = MODEL_DIR / 'scaler.pkl'
    LABEL_ENCODER_PATH = MODEL_DIR / 'label_encoder.pkl'
    METADATA_PATH = MODEL_DIR / 'metadata.json'
    
    # Training Settings
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    N_JOBS = -1
    
    # Logging
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

config = Config()
EOF

echo -e "${GREEN}✓ Configuration files created${NC}"

# Create logger utility
cat > backend/utils/logger.py << 'EOF'
"""
Logging utility for NIDS ML System
"""
import logging
import sys
from pathlib import Path
from datetime import datetime

def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """Setup logger with console and file handlers"""
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"{log_file}_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setLevel(level)
        file_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger
EOF

echo -e "${GREEN}✓ Logger utility created${NC}"

# Create quick start scripts
echo ""
echo "Creating quick start scripts..."

# Start API script
cat > start_api.sh << 'EOF'
#!/bin/bash
echo "Starting FastAPI backend..."
source venv/bin/activate
cd backend/api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
EOF
chmod +x start_api.sh

# Start Frontend script
cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "Starting Streamlit frontend..."
source venv/bin/activate
cd frontend
streamlit run app.py
EOF
chmod +x start_frontend.sh

# Train models script
cat > train_models.sh << 'EOF'
#!/bin/bash
echo "Training ML models..."
source venv/bin/activate
cd backend/ml
python train.py
EOF
chmod +x train_models.sh

echo -e "${GREEN}✓ Quick start scripts created${NC}"

# Create data preprocessing script
cat > backend/ml/preprocessing.py << 'EOF'
"""
Data preprocessing utilities for NIDS
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

def load_nsl_kdd(train_path: str, test_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load NSL-KDD dataset
    """
    # Column names for NSL-KDD
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]
    
    # Load training data
    train_df = pd.read_csv(train_path, names=columns)
    
    # Load test data if provided
    test_df = None
    if test_path:
        test_df = pd.read_csv(test_path, names=columns)
    
    return train_df, test_df

def preprocess_data(df: pd.DataFrame, remove_difficulty: bool = True) -> pd.DataFrame:
    """
    Preprocess network traffic data
    """
    df = df.copy()
    
    # Remove difficulty column if present
    if remove_difficulty and 'difficulty' in df.columns:
        df = df.drop(columns=['difficulty'])
    
    # Handle missing values
    df = df.dropna()
    
    # Simplify labels (binary: normal vs attack)
    if 'label' in df.columns:
        df['label'] = df['label'].apply(lambda x: 'normal' if x == 'normal' else 'attack')
    
    return df

def encode_categorical_features(df: pd.DataFrame, encoders: dict = None) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features
    """
    df = df.copy()
    
    if encoders is None:
        encoders = {}
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    categorical_cols = [col for col in categorical_cols if col != 'label']
    
    for col in categorical_cols:
        if col in encoders:
            df[col] = encoders[col].transform(df[col].astype(str))
        else:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, encoders
EOF

echo -e "${GREEN}✓ Preprocessing utilities created${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next Steps:"
echo ""
echo "1. Download NSL-KDD dataset:"
echo "   https://www.unb.ca/cic/datasets/nsl.html"
echo "   Place files in: data/raw/"
echo ""
echo "2. Train models:"
echo "   ./train_models.sh"
echo ""
echo "3. Start backend API:"
echo "   ./start_api.sh"
echo ""
echo "4. Start frontend (in new terminal):"
echo "   ./start_frontend.sh"
echo ""
echo "5. Access application:"
echo "   Frontend: http://localhost:8501"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo -e "${GREEN}Happy researching! 🚀${NC}"
