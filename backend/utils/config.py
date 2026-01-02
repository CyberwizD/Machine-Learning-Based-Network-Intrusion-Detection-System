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
