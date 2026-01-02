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
