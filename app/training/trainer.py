"""
Model Trainer
==============

Handles model training for Phase 3 predictive models.

Features:
- LightGBM model training
- Feature extraction from historical data
- Cross-validation
- Model artifact generation

Design Principles:
- Lightweight training (no heavy deep learning)
- Reproducible training runs
- Clear metrics reporting
- S3-ready artifact output
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import json
import uuid
import os

import numpy as np

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""
    model_name: str
    model_version: str
    training_id: str
    
    # Dataset info
    train_samples: int
    validation_samples: int
    feature_count: int
    
    # Performance metrics
    train_loss: float
    validation_loss: float
    train_mae: float
    validation_mae: float
    
    # Cross-validation
    cv_scores: List[float] = field(default_factory=list)
    cv_mean: float = 0.0
    cv_std: float = 0.0
    
    # Timing
    training_duration_seconds: float = 0.0
    
    # Metadata
    trained_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    hyperparameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainingResult:
    """Result from model training."""
    success: bool
    model_path: Optional[str]
    onnx_path: Optional[str]
    metrics: Optional[TrainingMetrics]
    error: Optional[str] = None


class ModelTrainer:
    """
    Trains Phase 3 predictive models.
    
    Supports:
    - Delay prediction model (regression)
    - Threat classification model (binary classification)
    - Drift detection model (multi-output regression)
    """
    
    def __init__(
        self,
        output_dir: str = "models",
        random_state: int = 42,
    ):
        """
        Initialize trainer.
        
        Args:
            output_dir: Directory for model artifacts
            random_state: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state
    
    def _get_lightgbm_params(
        self,
        model_type: str,
    ) -> Dict[str, Any]:
        """Get LightGBM parameters for model type."""
        base_params = {
            "random_state": self.random_state,
            "n_jobs": -1,
            "verbose": -1,
        }
        
        if model_type == "delay_predictor":
            return {
                **base_params,
                "objective": "regression",
                "metric": "mae",
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
            }
        elif model_type == "threat_classifier":
            return {
                **base_params,
                "objective": "binary",
                "metric": "binary_logloss",
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "num_leaves": 31,
                "min_child_samples": 20,
            }
        elif model_type == "drift_detector":
            return {
                **base_params,
                "objective": "regression",
                "metric": "mae",
                "n_estimators": 80,
                "max_depth": 4,
                "learning_rate": 0.1,
                "num_leaves": 15,
            }
        else:
            return base_params
    
    async def train_delay_predictor(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        version: Optional[str] = None,
    ) -> TrainingResult:
        """
        Train delay prediction model.
        
        Args:
            X_train: Training features
            y_train: Training targets (delay in minutes)
            X_val: Validation features
            y_val: Validation targets
            version: Model version string
            
        Returns:
            TrainingResult
        """
        import time
        start_time = time.time()
        
        training_id = f"train_{uuid.uuid4().hex[:8]}"
        model_name = "delay_predictor"
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            import lightgbm as lgb
            from sklearn.model_selection import cross_val_score
            
            params = self._get_lightgbm_params(model_name)
            
            # Train model
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
            
            # Evaluate
            train_pred = model.predict(X_train)
            val_pred = model.predict(X_val)
            
            train_mae = float(np.mean(np.abs(train_pred - y_train)))
            val_mae = float(np.mean(np.abs(val_pred - y_val)))
            
            # Cross-validation
            cv_scores = cross_val_score(
                lgb.LGBMRegressor(**params),
                X_train, y_train,
                cv=5,
                scoring="neg_mean_absolute_error",
            )
            cv_scores = -cv_scores  # Convert to positive MAE
            
            # Save model
            model_path = self.output_dir / f"{model_name}_{version}.txt"
            model.booster_.save_model(str(model_path))
            
            # Export to ONNX
            onnx_path = await self._export_to_onnx(
                model, model_name, version, X_train.shape[1]
            )
            
            training_duration = time.time() - start_time
            
            metrics = TrainingMetrics(
                model_name=model_name,
                model_version=version,
                training_id=training_id,
                train_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=X_train.shape[1],
                train_loss=train_mae,
                validation_loss=val_mae,
                train_mae=train_mae,
                validation_mae=val_mae,
                cv_scores=cv_scores.tolist(),
                cv_mean=float(np.mean(cv_scores)),
                cv_std=float(np.std(cv_scores)),
                training_duration_seconds=training_duration,
                hyperparameters=params,
            )
            
            # Save metrics
            metrics_path = self.output_dir / f"{model_name}_{version}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump({
                    "model_name": metrics.model_name,
                    "version": metrics.model_version,
                    "training_id": metrics.training_id,
                    "train_samples": metrics.train_samples,
                    "validation_mae": metrics.validation_mae,
                    "cv_mean": metrics.cv_mean,
                    "cv_std": metrics.cv_std,
                    "trained_at": metrics.trained_at.isoformat(),
                }, f, indent=2)
            
            logger.info(
                "model_trained",
                model=model_name,
                version=version,
                val_mae=val_mae,
                duration=training_duration,
            )
            
            return TrainingResult(
                success=True,
                model_path=str(model_path),
                onnx_path=onnx_path,
                metrics=metrics,
            )
            
        except Exception as e:
            logger.error("training_failed", model=model_name, error=str(e))
            return TrainingResult(
                success=False,
                model_path=None,
                onnx_path=None,
                metrics=None,
                error=str(e),
            )
    
    async def train_threat_classifier(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        version: Optional[str] = None,
    ) -> TrainingResult:
        """Train threat classification model."""
        import time
        start_time = time.time()
        
        training_id = f"train_{uuid.uuid4().hex[:8]}"
        model_name = "threat_classifier"
        version = version or datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            import lightgbm as lgb
            from sklearn.metrics import log_loss
            
            params = self._get_lightgbm_params(model_name)
            
            model = lgb.LGBMClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
            )
            
            # Evaluate
            train_proba = model.predict_proba(X_train)[:, 1]
            val_proba = model.predict_proba(X_val)[:, 1]
            
            train_loss = log_loss(y_train, train_proba)
            val_loss = log_loss(y_val, val_proba)
            
            # MAE for probability predictions
            train_mae = float(np.mean(np.abs(train_proba - y_train)))
            val_mae = float(np.mean(np.abs(val_proba - y_val)))
            
            # Save model
            model_path = self.output_dir / f"{model_name}_{version}.txt"
            model.booster_.save_model(str(model_path))
            
            # Export to ONNX
            onnx_path = await self._export_to_onnx(
                model, model_name, version, X_train.shape[1]
            )
            
            training_duration = time.time() - start_time
            
            metrics = TrainingMetrics(
                model_name=model_name,
                model_version=version,
                training_id=training_id,
                train_samples=len(X_train),
                validation_samples=len(X_val),
                feature_count=X_train.shape[1],
                train_loss=train_loss,
                validation_loss=val_loss,
                train_mae=train_mae,
                validation_mae=val_mae,
                training_duration_seconds=training_duration,
                hyperparameters=params,
            )
            
            logger.info(
                "model_trained",
                model=model_name,
                version=version,
                val_loss=val_loss,
            )
            
            return TrainingResult(
                success=True,
                model_path=str(model_path),
                onnx_path=onnx_path,
                metrics=metrics,
            )
            
        except Exception as e:
            logger.error("training_failed", model=model_name, error=str(e))
            return TrainingResult(
                success=False,
                model_path=None,
                onnx_path=None,
                metrics=None,
                error=str(e),
            )
    
    async def _export_to_onnx(
        self,
        model: Any,
        model_name: str,
        version: str,
        input_dim: int,
    ) -> Optional[str]:
        """Export model to ONNX format."""
        try:
            from onnxmltools import convert_lightgbm
            from onnxmltools.convert.common.data_types import FloatTensorType
            import onnx
            
            initial_types = [
                ("input", FloatTensorType([None, input_dim]))
            ]
            
            onnx_model = convert_lightgbm(
                model.booster_,
                name=model_name,
                initial_types=initial_types,
                target_opset=13,
            )
            
            onnx_path = self.output_dir / f"{model_name}_{version}.onnx"
            onnx.save(onnx_model, str(onnx_path))
            
            logger.info("onnx_exported", model=model_name, path=str(onnx_path))
            return str(onnx_path)
            
        except Exception as e:
            logger.warning("onnx_export_failed", error=str(e))
            return None


# Singleton instance
_trainer: Optional[ModelTrainer] = None


def get_trainer() -> ModelTrainer:
    """Get or create singleton trainer."""
    global _trainer
    if _trainer is None:
        _trainer = ModelTrainer()
    return _trainer
