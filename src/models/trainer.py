"""
Dynamic Machine Learning Model Trainer with Gemini AI Integration.
"""
import asyncio
import json
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import SGDRegressor, SGDClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import google.generativeai as genai
import mlflow
import mlflow.sklearn
from dataclasses import dataclass
import threading
import time

from src.config.settings import settings


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ModelMetrics:
    """Model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class ModelTrainer:
    """Dynamic model trainer with online learning capabilities."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.is_training = False
        self.last_training_time: Optional[datetime] = None
        self.gemini_model = None
        self.auto_retrain_thread: Optional[threading.Thread] = None
        self.stop_auto_retrain = threading.Event()

        # Create models directory
        os.makedirs("models", exist_ok=True)

    async def initialize(self):
        """Initialize the model trainer."""
        try:
            # Configure Gemini AI
            genai.configure(api_key=settings.google_api_key)
            self.gemini_model = genai.GenerativeModel(settings.gemini_model)
            logger.info("✅ Gemini AI model configured")

            # Set up MLflow
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("real-time-ml-experiment")
            logger.info("✅ MLflow tracking configured")

            # Load existing models if available
            await self._load_existing_models()

            # Start auto-retraining if enabled
            if settings.auto_retraining:
                self._start_auto_retraining()

            logger.info("🚀 Model trainer initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize model trainer: {e}")
            raise

    async def _load_existing_models(self):
        """Load previously trained models from disk."""
        try:
            models_dir = "models"
            if not os.path.exists(models_dir):
                return

            for filename in os.listdir(models_dir):
                if filename.endswith('.pkl'):
                    model_name = filename[:-4]  # Remove .pkl extension
                    model_path = os.path.join(models_dir, filename)

                    try:
                        model_data = joblib.load(model_path)
                        self.models[model_name] = model_data['model']
                        self.scalers[model_name] = model_data.get('scaler')
                        self.label_encoders[model_name] = model_data.get('label_encoder')

                        # Load metrics if available
                        metrics_file = os.path.join(models_dir, f"{model_name}_metrics.json")
                        if os.path.exists(metrics_file):
                            with open(metrics_file, 'r') as f:
                                metrics_data = json.load(f)
                                self.model_metrics[model_name] = ModelMetrics(**metrics_data)

                        logger.info(f"📥 Loaded model: {model_name}")

                    except Exception as e:
                        logger.warning(f"⚠️ Error loading model {model_name}: {e}")

        except Exception as e:
            logger.error(f"❌ Error loading existing models: {e}")

    def _start_auto_retraining(self):
        """Start automatic model retraining thread."""
        if self.auto_retrain_thread and self.auto_retrain_thread.is_alive():
            return

        self.auto_retrain_thread = threading.Thread(
            target=self._auto_retrain_loop,
            daemon=True
        )
        self.auto_retrain_thread.start()
        logger.info("🔄 Auto-retraining thread started")

    def _auto_retrain_loop(self):
        """Background loop for automatic model retraining."""
        while not self.stop_auto_retrain.is_set():
            try:
                # Check if it's time to retrain
                if self._should_retrain():
                    asyncio.run(self.retrain_models())

                # Wait for next check
                self.stop_auto_retrain.wait(settings.model_update_interval)

            except Exception as e:
                logger.error(f"❌ Error in auto-retrain loop: {e}")
                self.stop_auto_retrain.wait(60)  # Wait 1 minute before retrying

    def _should_retrain(self) -> bool:
        """Check if models should be retrained."""
        if not self.last_training_time:
            return True

        time_since_last_training = datetime.now() - self.last_training_time
        return time_since_last_training.total_seconds() >= settings.model_update_interval

    async def train_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "auto",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Train a machine learning model.

        Args:
            model_name: Name identifier for the model
            X: Feature matrix
            y: Target variable
            model_type: Type of model ("classification", "regression", "auto")
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility

        Returns:
            Dict containing training results and metrics
        """
        try:
            self.is_training = True
            logger.info(f"🏋️ Starting training for model: {model_name}")

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            # Determine model type if auto
            if model_type == "auto":
                model_type = self._detect_model_type(y)

            # Preprocess features
            X_train_scaled, X_test_scaled, scaler = await self._preprocess_features(
                X_train, X_test, model_name
            )

            # Encode labels if classification
            if model_type == "classification":
                y_train_encoded, y_test_encoded, label_encoder = await self._encode_labels(
                    y_train, y_test, model_name
                )
            else:
                y_train_encoded, y_test_encoded, label_encoder = y_train, y_test, None

            # Train model
            model, training_params = await self._train_model(
                X_train_scaled, y_train_encoded, model_type, model_name
            )

            # Make predictions
            y_pred = model.predict(X_test_scaled)

            # Calculate metrics
            metrics = await self._calculate_metrics(
                y_test_encoded, y_pred, model_type
            )

            # Save model
            await self._save_model(model_name, model, scaler, label_encoder, metrics)

            # Log to MLflow
            await self._log_to_mlflow(model_name, model, metrics, training_params)

            # Update tracking
            self.models[model_name] = model
            self.scalers[model_name] = scaler
            self.label_encoders[model_name] = label_encoder
            self.model_metrics[model_name] = metrics
            self.last_training_time = datetime.now()

            logger.info(f"✅ Model {model_name} trained successfully")

            return {
                "model_name": model_name,
                "model_type": model_type,
                "metrics": metrics,
                "training_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X.columns),
                "training_time": datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"❌ Error training model {model_name}: {e}")
            raise
        finally:
            self.is_training = False

    def _detect_model_type(self, y: pd.Series) -> str:
        """Detect whether the problem is classification or regression."""
        if y.dtype in ['object', 'string', 'category']:
            return "classification"
        elif len(y.unique()) < 20:  # Arbitrary threshold for classification
            return "classification"
        else:
            return "regression"

    async def _preprocess_features(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        model_name: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
        """Preprocess and scale features."""
        # Handle missing values
        X_train = X_train.fillna(X_train.mean())
        X_test = X_test.fillna(X_train.mean())

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        return X_train_scaled, X_test_scaled, scaler

    async def _encode_labels(
        self,
        y_train: pd.Series,
        y_test: pd.Series,
        model_name: str
    ) -> Tuple[pd.Series, pd.Series, LabelEncoder]:
        """Encode categorical labels."""
        label_encoder = LabelEncoder()
        y_train_encoded = pd.Series(
            label_encoder.fit_transform(y_train),
            index=y_train.index,
            name=y_train.name
        )
        y_test_encoded = pd.Series(
            label_encoder.transform(y_test),
            index=y_test.index,
            name=y_test.name
        )

        return y_train_encoded, y_test_encoded, label_encoder

    async def _train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_type: str,
        model_name: str
    ) -> Tuple[Any, Dict[str, Any]]:
        """Train the actual model."""
        if model_type == "classification":
            # Use Random Forest for classification
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            training_params = {
                "model_class": "RandomForestClassifier",
                "n_estimators": 100,
                "random_state": 42
            }
        else:
            # Use Random Forest for regression
            model = RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
            training_params = {
                "model_class": "RandomForestRegressor",
                "n_estimators": 100,
                "random_state": 42
            }

        # Train model
        model.fit(X_train, y_train)

        return model, training_params

    async def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        model_type: str
    ) -> ModelMetrics:
        """Calculate model performance metrics."""
        if model_type == "classification":
            accuracy = accuracy_score(y_true, y_pred)
            # For multi-class, we'll use a general approach
            metrics = ModelMetrics(
                accuracy=accuracy,
                created_at=datetime.now()
            )
        else:
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)
            metrics = ModelMetrics(
                mse=mse,
                rmse=rmse,
                created_at=datetime.now()
            )

        return metrics

    async def _save_model(
        self,
        model_name: str,
        model: Any,
        scaler: StandardScaler,
        label_encoder: Optional[LabelEncoder],
        metrics: ModelMetrics
    ):
        """Save model and associated components."""
        try:
            # Save model data
            model_data = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'metrics': metrics,
                'saved_at': datetime.now().isoformat()
            }

            model_path = f"models/{model_name}.pkl"
            joblib.dump(model_data, model_path)

            # Save metrics separately
            metrics_path = f"models/{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics.__dict__, f, default=str, indent=2)

            logger.info(f"💾 Model {model_name} saved to {model_path}")

        except Exception as e:
            logger.error(f"❌ Error saving model {model_name}: {e}")

    async def _log_to_mlflow(
        self,
        model_name: str,
        model: Any,
        metrics: ModelMetrics,
        params: Dict[str, Any]
    ):
        """Log model and metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                if metrics.accuracy > 0:
                    mlflow.log_metric("accuracy", metrics.accuracy)
                if metrics.mse > 0:
                    mlflow.log_metric("mse", metrics.mse)
                    mlflow.log_metric("rmse", metrics.rmse)

                # Log model
                mlflow.sklearn.log_model(model, f"model_{model_name}")

                logger.info(f"📊 Model {model_name} logged to MLflow")

        except Exception as e:
            logger.error(f"❌ Error logging to MLflow: {e}")

    async def predict(
        self,
        model_name: str,
        X: pd.DataFrame,
        return_probabilities: bool = False
    ) -> np.ndarray:
        """
        Make predictions using a trained model.

        Args:
            model_name: Name of the model to use
            X: Feature matrix for prediction
            return_probabilities: Whether to return prediction probabilities

        Returns:
            Array of predictions
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found")

            model = self.models[model_name]
            scaler = self.scalers.get(model_name)

            # Preprocess features
            if scaler:
                X_scaled = pd.DataFrame(
                    scaler.transform(X),
                    columns=X.columns,
                    index=X.index
                )
            else:
                X_scaled = X

            # Make predictions
            predictions = model.predict(X_scaled)

            # Get probabilities if requested and available
            if return_probabilities and hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X_scaled)
                return predictions, probabilities

            return predictions

        except Exception as e:
            logger.error(f"❌ Error making predictions with {model_name}: {e}")
            raise

    async def retrain_models(self):
        """Retrain all existing models with new data."""
        try:
            logger.info("🔄 Starting model retraining...")

            # This would typically fetch new data from the data pipeline
            # For now, we'll simulate retraining existing models

            for model_name in list(self.models.keys()):
                try:
                    # Get recent data for retraining
                    recent_data = await self._get_recent_training_data(model_name)

                    if recent_data and len(recent_data) > 10:  # Minimum data threshold
                        X, y = self._prepare_training_data(recent_data)
                        await self.train_model(model_name, X, y, model_type="auto")

                        logger.info(f"✅ Model {model_name} retrained")

                except Exception as e:
                    logger.error(f"❌ Error retraining model {model_name}: {e}")

            logger.info("🔄 Model retraining completed")

        except Exception as e:
            logger.error(f"❌ Error during model retraining: {e}")

    async def _get_recent_training_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """Get recent data for model retraining."""
        # This would typically fetch from the data pipeline
        # For now, return None to indicate no new data
        return None

    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare data for training."""
        # This is a placeholder - in practice, you'd extract features and target
        # from your specific data structure
        return data.drop('target', axis=1), data['target']

    async def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get information about a trained model."""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")

        model = self.models[model_name]
        metrics = self.model_metrics.get(model_name)

        return {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "features": getattr(model, 'n_features_in_', 'Unknown'),
            "training_date": self.last_training_time.isoformat() if self.last_training_time else None,
            "metrics": metrics.__dict__ if metrics else None,
            "parameters": model.get_params() if hasattr(model, 'get_params') else {}
        }

    async def list_models(self) -> List[str]:
        """List all available trained models."""
        return list(self.models.keys())

    async def delete_model(self, model_name: str) -> bool:
        """Delete a trained model."""
        try:
            if model_name in self.models:
                del self.models[model_name]
                if model_name in self.scalers:
                    del self.scalers[model_name]
                if model_name in self.label_encoders:
                    del self.label_encoders[model_name]
                if model_name in self.model_metrics:
                    del self.model_metrics[model_name]

                # Delete files
                model_path = f"models/{model_name}.pkl"
                metrics_path = f"models/{model_name}_metrics.json"

                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)

                logger.info(f"🗑️ Model {model_name} deleted")
                return True

        except Exception as e:
            logger.error(f"❌ Error deleting model {model_name}: {e}")

        return False

    async def analyze_data_with_gemini(
        self,
        data_sample: pd.DataFrame,
        analysis_type: str = "general"
    ) -> str:
        """
        Use Gemini AI to analyze data and provide insights.

        Args:
            data_sample: Sample of the data to analyze
            analysis_type: Type of analysis to perform

        Returns:
            AI-generated analysis and insights
        """
        try:
            if not self.gemini_model:
                raise ValueError("Gemini AI model not initialized")

            # Prepare data for analysis
            data_info = {
                "shape": data_sample.shape,
                "columns": list(data_sample.columns),
                "dtypes": data_sample.dtypes.to_dict(),
                "sample_data": data_sample.head(5).to_dict(),
                "statistics": data_sample.describe().to_dict()
            }

            # Create analysis prompt
            prompt = f"""
            Analyze this dataset and provide insights for {analysis_type}:

            Dataset Information:
            - Shape: {data_info['shape']}
            - Columns: {data_info['columns']}
            - Data Types: {data_info['dtypes']}

            Sample Data:
            {data_info['sample_data']}

            Statistics:
            {data_info['statistics']}

            Please provide:
            1. Key patterns and trends in the data
            2. Potential features for machine learning
            3. Data quality issues or anomalies
            4. Recommendations for model selection
            5. Feature engineering suggestions

            Format your response as a structured analysis.
            """

            # Get Gemini AI response
            response = self.gemini_model.generate_content(prompt)

            logger.info("🤖 Gemini AI analysis completed")
            return response.text

        except Exception as e:
            logger.error(f"❌ Error in Gemini AI analysis: {e}")
            return f"Error in AI analysis: {str(e)}"

    async def shutdown(self):
        """Shutdown the model trainer."""
        try:
            # Stop auto-retraining
            self.stop_auto_retrain.set()

            if self.auto_retrain_thread:
                self.auto_retrain_thread.join(timeout=5)

            # Save all models
            for model_name in self.models.keys():
                model = self.models[model_name]
                scaler = self.scalers.get(model_name)
                label_encoder = self.label_encoders.get(model_name)
                metrics = self.model_metrics.get(model_name)

                if metrics:
                    await self._save_model(model_name, model, scaler, label_encoder, metrics)

            logger.info("🛑 Model trainer shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during model trainer shutdown: {e}")
