"""
Machine Learning Training Pipeline for Network Intrusion Detection System
Implements multiple ML algorithms with comprehensive evaluation
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

import xgboost as xgb
import lightgbm as lgb

import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


class NIDSTrainer:
    """
    Comprehensive ML training pipeline for Network Intrusion Detection
    """
    
    def __init__(self, data_path: str, output_dir: str = "models"):
        self.data_path = Path(data_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Store training artifacts
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Load and preprocess network traffic data
        Supports NSL-KDD, CIC-IDS2017, or custom datasets
        """
        print("Loading dataset...")
        
        # Load data (adjust based on your dataset format)
        if self.data_path.suffix == '.csv':
            df = pd.read_csv(self.data_path)
        else:
            df = pd.read_csv(
                self.data_path,
                header=None,
                sep=','
            )
        
        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        # Adjust 'label' to your actual target column name
        target_col = 'label' if 'label' in df.columns else df.columns[-1]
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Encode categorical features
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
        
        # Encode target labels
        y = self.label_encoder.fit_transform(y)
        
        self.feature_names = X.columns.tolist()
        
        print(f"Features: {len(self.feature_names)}")
        print(f"Classes: {np.unique(y)}")
        
        return X, y
    
    def split_and_scale_data(self, X: pd.DataFrame, y: pd.Series, 
                            test_size: float = 0.2, random_state: int = 42):
        """
        Split data and apply scaling
        """
        print("\nSplitting and scaling data...")
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        
    def initialize_models(self) -> Dict:
        """
        Initialize multiple ML models for comparison
        """
        print("\nInitializing models...")
        
        models = {
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1
            ),
            'LightGBM': lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                max_depth=10,
                learning_rate=0.1,
                random_state=42
            ),
            'Decision Tree': DecisionTreeClassifier(
                max_depth=20,
                random_state=42
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )
        }
        
        return models
    
    def train_models(self):
        """
        Train all models and store results
        """
        models = self.initialize_models()
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            start_time = datetime.now()
            
            # Train model
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = self.calculate_metrics(self.y_test, y_pred, y_pred_proba)
            
            training_time = (datetime.now() - start_time).total_seconds()
            metrics['training_time'] = training_time
            
            # Store model and results
            self.models[name] = model
            self.results[name] = metrics
            
            print(f"✓ {name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_weighted']:.4f}")
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None) -> Dict:
        """
        Calculate comprehensive evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist(),
        }
        
        # Per-class metrics
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        # ROC-AUC for binary or multiclass
        if y_pred_proba is not None:
            n_classes = len(np.unique(y_true))
            if n_classes == 2:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                metrics['roc_auc'] = auc(fpr, tpr)
                metrics['fpr'] = fpr.tolist()
                metrics['tpr'] = tpr.tolist()
            else:
                # Multiclass ROC-AUC (one-vs-rest)
                from sklearn.preprocessing import label_binarize
                y_true_bin = label_binarize(y_true, classes=range(n_classes))
                
                fpr_dict = {}
                tpr_dict = {}
                roc_auc_dict = {}
                
                for i in range(n_classes):
                    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
                    fpr_dict[i] = fpr.tolist()
                    tpr_dict[i] = tpr.tolist()
                    roc_auc_dict[i] = auc(fpr, tpr)
                
                metrics['fpr_multiclass'] = fpr_dict
                metrics['tpr_multiclass'] = tpr_dict
                metrics['roc_auc_multiclass'] = roc_auc_dict
        
        return metrics
    
    def generate_visualizations(self):
        """
        Generate comprehensive visualizations for paper
        """
        print("\nGenerating visualizations...")
        viz_dir = self.output_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)
        
        # 1. Model Comparison
        self._plot_model_comparison(viz_dir)
        
        # 2. Confusion Matrices
        self._plot_confusion_matrices(viz_dir)
        
        # 3. ROC Curves
        self._plot_roc_curves(viz_dir)
        
        # 4. Feature Importance (for tree-based models)
        self._plot_feature_importance(viz_dir)
        
        # 5. Training Metrics Comparison
        self._plot_metrics_comparison(viz_dir)
        
        print(f"✓ Visualizations saved to {viz_dir}")
    
    def _plot_model_comparison(self, viz_dir: Path):
        """Plot comparison of all models"""
        metrics = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for idx, metric in enumerate(metrics):
            values = [self.results[model][metric] for model in self.results.keys()]
            
            axes[idx].bar(self.results.keys(), values, color='steelblue', alpha=0.8)
            axes[idx].set_title(f'{metric.replace("_", " ").title()}', fontsize=12, fontweight='bold')
            axes[idx].set_ylim([0, 1.1])
            axes[idx].set_ylabel('Score')
            axes[idx].tick_params(axis='x', rotation=45)
            axes[idx].grid(axis='y', alpha=0.3)
            
            # Add value labels
            for i, v in enumerate(values):
                axes[idx].text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confusion_matrices(self, viz_dir: Path):
        """Plot confusion matrices for all models"""
        n_models = len(self.models)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        axes = axes.ravel() if n_models > 1 else [axes]
        
        for idx, (name, metrics) in enumerate(self.results.items()):
            cm = np.array(metrics['confusion_matrix'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       cbar_kws={'label': 'Count'})
            axes[idx].set_title(f'{name}', fontweight='bold')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        # Hide extra subplots
        for idx in range(n_models, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(viz_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_roc_curves(self, viz_dir: Path):
        """Plot ROC curves"""
        plt.figure(figsize=(10, 8))
        
        for name, metrics in self.results.items():
            if 'fpr' in metrics and 'tpr' in metrics:
                plt.plot(metrics['fpr'], metrics['tpr'], 
                        label=f"{name} (AUC = {metrics['roc_auc']:.3f})",
                        linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.savefig(viz_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_importance(self, viz_dir: Path):
        """Plot feature importance for tree-based models"""
        for name, model in self.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                indices = np.argsort(importances)[-20:]  # Top 20 features
                
                plt.figure(figsize=(10, 8))
                plt.barh(range(len(indices)), importances[indices], color='steelblue', alpha=0.8)
                plt.yticks(range(len(indices)), [self.feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importances - {name}', fontweight='bold')
                plt.tight_layout()
                
                filename = f"feature_importance_{name.lower().replace(' ', '_')}.png"
                plt.savefig(viz_dir / filename, dpi=300, bbox_inches='tight')
                plt.close()
    
    def _plot_metrics_comparison(self, viz_dir: Path):
        """Plot detailed metrics comparison"""
        models = list(self.results.keys())
        metrics_to_plot = ['accuracy', 'precision_weighted', 'recall_weighted', 
                          'f1_weighted', 'precision_macro', 'recall_macro', 'f1_macro']
        
        data = []
        for model in models:
            for metric in metrics_to_plot:
                data.append({
                    'Model': model,
                    'Metric': metric.replace('_', ' ').title(),
                    'Score': self.results[model][metric]
                })
        
        df = pd.DataFrame(data)
        
        plt.figure(figsize=(14, 8))
        sns.barplot(data=df, x='Metric', y='Score', hue='Model')
        plt.xticks(rotation=45, ha='right')
        plt.title('Comprehensive Metrics Comparison', fontsize=14, fontweight='bold')
        plt.ylabel('Score')
        plt.ylim([0, 1.1])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(viz_dir / 'metrics_comparison_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_models_and_artifacts(self):
        """
        Save trained models, scalers, and results
        """
        print("\nSaving models and artifacts...")
        
        # Save best model (highest F1 score)
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['f1_weighted'])
        best_model = self.models[best_model_name]
        
        # Save best model
        with open(self.output_dir / 'best_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Save all models
        for name, model in self.models.items():
            filename = f"model_{name.lower().replace(' ', '_')}.pkl"
            with open(self.output_dir / filename, 'wb') as f:
                pickle.dump(model, f)
        
        # Save preprocessing artifacts
        with open(self.output_dir / 'scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(self.output_dir / 'label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save results and metadata
        metadata = {
            'best_model': best_model_name,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'n_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'training_date': datetime.now().isoformat(),
            'results': self.results
        }
        
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Best model: {best_model_name}")
        print(f"✓ Models saved to {self.output_dir}")
    
    def run_complete_pipeline(self):
        """
        Execute the complete training pipeline
        """
        print("="*80)
        print("NIDS ML Training Pipeline")
        print("="*80)
        
        # Load and preprocess
        X, y = self.load_and_preprocess_data()
        
        # Split and scale
        self.split_and_scale_data(X, y)
        
        # Train models
        self.train_models()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Save everything
        self.save_models_and_artifacts()
        
        print("\n" + "="*80)
        print("Training Complete!")
        print("="*80)
        
        # Print summary
        print("\nModel Performance Summary:")
        for name, metrics in self.results.items():
            print(f"\n{name}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Precision: {metrics['precision_weighted']:.4f}")
            print(f"  Recall:    {metrics['recall_weighted']:.4f}")
            print(f"  F1-Score:  {metrics['f1_weighted']:.4f}")
            print(f"  Time:      {metrics['training_time']:.2f}s")


if __name__ == "__main__":
    # Example usage
    trainer = NIDSTrainer(
        data_path=r".\data\raw\KDDTrain+.txt",
        output_dir="models"
    )
    
    trainer.run_complete_pipeline()
