"""
FastAPI Backend for Network Intrusion Detection System
Provides REST API for ML predictions and analysis
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from datetime import datetime
import asyncio

app = FastAPI(
    title="NIDS ML API",
    description="Network Intrusion Detection System with Machine Learning",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global variables for models
models = {}
scaler = None
label_encoder = None
metadata = None
feature_names = []


# Pydantic models
class PredictionRequest(BaseModel):
    features: List[float]


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]
    is_attack: bool
    threat_level: str


class BatchPredictionRequest(BaseModel):
    data: List[List[float]]


class ModelMetrics(BaseModel):
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    training_time: float


# Startup event to load models
@app.on_event("startup")
async def load_models():
    """Load trained models and artifacts on startup"""
    global models, scaler, label_encoder, metadata, feature_names
    
    # Resolve models directory relative to this file (robust to current working dir)
    models_dir = Path(__file__).resolve().parents[1] / "ml" / "models"

    # Fallback: support running from repo root or other working directories
    if not models_dir.exists():
        alt = Path.cwd() / "backend" / "ml" / "models"
        if alt.exists():
            models_dir = alt
        else:
            raise FileNotFoundError(
                f"Models directory not found. Checked: {models_dir} and {alt}"
            )

    try:
        # Load best model
        best_path = models_dir / "best_model.pkl"
        if not best_path.exists():
            raise FileNotFoundError(f"Missing model file: {best_path}")
        with open(best_path, "rb") as f:
            models['best'] = pickle.load(f)
        
        # Load all models
        for model_file in models_dir.glob("model_*.pkl"):
            model_name = model_file.stem.replace("model_", "").replace("_", " ").title()
            with open(model_file, "rb") as f:
                models[model_name] = pickle.load(f)
        
        # Load preprocessing artifacts
        scaler_path = models_dir / "scaler.pkl"
        label_enc_path = models_dir / "label_encoder.pkl"
        metadata_path = models_dir / "metadata.json"

        if not scaler_path.exists():
            raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
        if not label_enc_path.exists():
            raise FileNotFoundError(f"Missing label encoder file: {label_enc_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        with open(label_enc_path, "rb") as f:
            label_encoder = pickle.load(f)

        # Load metadata
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
            feature_names = metadata['feature_names']
        
        print(f"✓ Loaded {len(models)} models successfully")
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "NIDS ML API",
        "version": "1.0.0",
        "status": "active",
        "models_loaded": len(models)
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_available": list(models.keys()),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/models")
async def get_models_info():
    """Get information about available models"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    return {
        "best_model": metadata['best_model'],
        "available_models": list(models.keys()),
        "n_features": metadata['n_features'],
        "n_classes": metadata['n_classes'],
        "class_names": metadata['class_names'],
        "feature_names": feature_names[:10]  # First 10 features
    }


@app.get("/metrics")
async def get_all_metrics():
    """Get performance metrics for all models"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    results = metadata.get('results', {})
    
    metrics_list = []
    for model_name, metrics in results.items():
        metrics_list.append({
            "model_name": model_name,
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision_weighted'],
            "recall": metrics['recall_weighted'],
            "f1_score": metrics['f1_weighted'],
            "training_time": metrics.get('training_time', 0)
        })
    
    return {
        "metrics": metrics_list,
        "best_model": metadata['best_model']
    }


@app.get("/metrics/{model_name}")
async def get_model_metrics(model_name: str):
    """Get detailed metrics for a specific model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    results = metadata.get('results', {})
    
    if model_name not in results:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return {
        "model_name": model_name,
        "metrics": results[model_name]
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make prediction on single sample"""
    if models.get('best') is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.features).reshape(1, -1)
        
        # Validate feature count
        if features.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make prediction
        prediction = models['best'].predict(features_scaled)[0]
        probabilities = models['best'].predict_proba(features_scaled)[0]
        
        # Get class name
        predicted_class = label_encoder.inverse_transform([prediction])[0]
        
        # Determine if it's an attack
        is_attack = predicted_class.lower() != 'normal'
        
        # Calculate confidence and threat level
        confidence = float(np.max(probabilities))
        
        if confidence >= 0.9:
            threat_level = "High"
        elif confidence >= 0.7:
            threat_level = "Medium"
        else:
            threat_level = "Low"
        
        # Create probability dictionary
        prob_dict = {
            label_encoder.inverse_transform([i])[0]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        return PredictionResponse(
            prediction=predicted_class,
            confidence=confidence,
            probabilities=prob_dict,
            is_attack=is_attack,
            threat_level=threat_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Make predictions on multiple samples"""
    if models.get('best') is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare features
        features = np.array(request.data)
        
        # Validate
        if features.shape[1] != len(feature_names):
            raise HTTPException(
                status_code=400,
                detail=f"Expected {len(feature_names)} features, got {features.shape[1]}"
            )
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Make predictions
        predictions = models['best'].predict(features_scaled)
        probabilities = models['best'].predict_proba(features_scaled)
        
        # Process results
        results = []
        for i, (pred, probs) in enumerate(zip(predictions, probabilities)):
            predicted_class = label_encoder.inverse_transform([pred])[0]
            confidence = float(np.max(probs))
            is_attack = predicted_class.lower() != 'normal'
            
            results.append({
                "sample_id": i,
                "prediction": predicted_class,
                "confidence": confidence,
                "is_attack": is_attack
            })
        
        # Calculate summary statistics
        attack_count = sum(1 for r in results if r['is_attack'])
        normal_count = len(results) - attack_count
        
        return {
            "predictions": results,
            "summary": {
                "total_samples": len(results),
                "attacks_detected": attack_count,
                "normal_traffic": normal_count,
                "attack_percentage": (attack_count / len(results)) * 100
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/csv")
async def predict_from_csv(file: UploadFile = File(...)):
    """Make predictions from uploaded CSV file"""
    if models.get('best') is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Read CSV
        contents = await file.read()
        df = pd.read_csv(pd.io.common.BytesIO(contents))
        
        # Remove target column if present
        if 'label' in df.columns:
            df = df.drop(columns=['label'])
        
        # Encode categorical columns
        from sklearn.preprocessing import LabelEncoder
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        
        # Get features
        features = df.values
        
        # Scale
        features_scaled = scaler.transform(features)
        
        # Predict
        predictions = models['best'].predict(features_scaled)
        probabilities = models['best'].predict_proba(features_scaled)
        
        # Add predictions to dataframe
        df['prediction'] = label_encoder.inverse_transform(predictions)
        df['confidence'] = np.max(probabilities, axis=1)
        df['is_attack'] = df['prediction'].apply(lambda x: x.lower() != 'normal')
        
        # Summary statistics
        attack_count = df['is_attack'].sum()
        
        return {
            "total_samples": len(df),
            "attacks_detected": int(attack_count),
            "normal_traffic": int(len(df) - attack_count),
            "attack_types": df[df['is_attack']]['prediction'].value_counts().to_dict(),
            "average_confidence": float(df['confidence'].mean()),
            "samples": df.head(100).to_dict(orient='records')
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/confusion-matrix/{model_name}")
async def get_confusion_matrix(model_name: str):
    """Get confusion matrix for specific model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    results = metadata.get('results', {})
    
    if model_name not in results:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    cm = results[model_name].get('confusion_matrix')
    
    if cm is None:
        raise HTTPException(status_code=404, detail="Confusion matrix not available")
    
    return {
        "model_name": model_name,
        "confusion_matrix": cm,
        "class_names": metadata['class_names']
    }


@app.get("/roc-curve/{model_name}")
async def get_roc_curve(model_name: str):
    """Get ROC curve data for specific model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    results = metadata.get('results', {})
    
    if model_name not in results:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    metrics = results[model_name]
    
    if 'fpr' in metrics and 'tpr' in metrics:
        return {
            "model_name": model_name,
            "fpr": metrics['fpr'],
            "tpr": metrics['tpr'],
            "auc": metrics.get('roc_auc', None)
        }
    elif 'fpr_multiclass' in metrics:
        return {
            "model_name": model_name,
            "multiclass": True,
            "curves": {
                str(i): {
                    "fpr": metrics['fpr_multiclass'][str(i)],
                    "tpr": metrics['tpr_multiclass'][str(i)],
                    "auc": metrics['roc_auc_multiclass'][str(i)]
                }
                for i in range(len(metadata['class_names']))
            },
            "class_names": metadata['class_names']
        }
    else:
        raise HTTPException(status_code=404, detail="ROC curve data not available")


@app.get("/classification-report/{model_name}")
async def get_classification_report(model_name: str):
    """Get detailed classification report for specific model"""
    if metadata is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    results = metadata.get('results', {})
    
    if model_name not in results:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    report = results[model_name].get('classification_report')
    
    if report is None:
        raise HTTPException(status_code=404, detail="Classification report not available")
    
    return {
        "model_name": model_name,
        "report": report,
        "class_names": metadata['class_names']
    }


@app.get("/feature-importance")
async def get_feature_importance():
    """Get feature importance from best model"""
    if models.get('best') is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model = models['best']
    
    if not hasattr(model, 'feature_importances_'):
        raise HTTPException(
            status_code=400,
            detail="Model does not support feature importance"
        )
    
    importances = model.feature_importances_
    
    # Sort by importance
    indices = np.argsort(importances)[::-1]
    
    # Top 20 features
    top_features = []
    for i in indices[:20]:
        top_features.append({
            "feature": feature_names[i],
            "importance": float(importances[i])
        })
    
    return {
        "model_name": metadata['best_model'],
        "top_features": top_features,
        "total_features": len(feature_names)
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    