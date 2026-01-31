import os
import requests

API_BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def check_api_health():
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def get_models_info():
    try:
        response = requests.get(f"{API_BASE_URL}/models", timeout=8)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_all_metrics():
    try:
        response = requests.get(f"{API_BASE_URL}/metrics", timeout=12)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def predict_csv(file_content):
    try:
        files = {'file': file_content}
        response = requests.post(f"{API_BASE_URL}/predict/csv", files=files, timeout=30)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_confusion_matrix(model_name):
    try:
        response = requests.get(f"{API_BASE_URL}/confusion-matrix/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_roc_curve(model_name):
    try:
        response = requests.get(f"{API_BASE_URL}/roc-curve/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_classification_report(model_name):
    try:
        response = requests.get(f"{API_BASE_URL}/classification-report/{model_name}", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None


def get_feature_importance():
    try:
        response = requests.get(f"{API_BASE_URL}/feature-importance", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None