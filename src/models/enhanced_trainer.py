"""
Enhanced ML Trainer with Advanced Algorithms and Ensemble Methods
Provides superior performance for immediate user value while enabling autonomous learning.
"""
import asyncio
import json
import logging
import pickle
import os
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier,
    VotingRegressor, VotingClassifier,
    StackingRegressor, StackingClassifier
)
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    LogisticRegression, SGDRegressor, SGDClassifier
)
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (
    mean_squared_error, accuracy_score, classification_report,
    r2_score, mean_absolute_error, precision_recall_fscore_support
)
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, RFE
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import mlflow
import mlflow.sklearn
from dataclasses import dataclass, asdict
import threading
import time

from src.config.settings import settings
from src.database.service import db_service

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
class EnhancedModelMetrics:
    """Enhanced model performance metrics."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    rmse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    cross_val_score: float = 0.0
    training_time: float = 0.0
    inference_time: float = 0.0
    model_size_mb: float = 0.0
    feature_importance: Dict[str, float] = None
    created_at: datetime = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.feature_importance is None:
            self.feature_importance = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return asdict(self)

class EnhancedModelTrainer:
    """Enhanced model trainer with superior performance and autonomous learning capabilities."""

    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.model_metrics: Dict[str, EnhancedModelMetrics] = {}
        self.is_training = False
        self.last_training_time: Optional[datetime] = None
        self.auto_retrain_thread: Optional[threading.Thread] = None
        self.stop_auto_retrain = threading.Event()

        # Enhanced model configurations
        self.model_configs = {
            'classification': {
                'simple': [
                    ('logistic', LogisticRegression(random_state=42, max_iter=1000)),
                    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingClassifier(random_state=42)),
                    ('et', ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1))
                ],
                'ensemble': None,  # Will be created dynamically
                'advanced': [
                    ('svm', SVC(probability=True, random_state=42)),
                    ('xgb', None)  # XGBoost if available
                ]
            },
            'regression': {
                'simple': [
                    ('linear', LinearRegression()),
                    ('ridge', Ridge(random_state=42)),
                    ('lasso', Lasso(random_state=42)),
                    ('rf', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
                    ('gb', GradientBoostingRegressor(random_state=42)),
                    ('et', ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1))
                ],
                'ensemble': None,  # Will be created dynamically
                'advanced': [
                    ('svr', SVR()),
                    ('elastic', ElasticNet(random_state=42))
                ]
            }
        }

        # Create models directory
        os.makedirs("models/enhanced", exist_ok=True)

        # Initialize MLflow
        self._setup_mlflow()

    def _setup_mlflow(self):
        """Setup MLflow for experiment tracking."""
        try:
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("enhanced-autonomous-ml-experiment")
            logger.info("✅ MLflow tracking configured for enhanced models")
        except Exception as e:
            logger.error(f"❌ Error setting up MLflow: {e}")

    async def initialize(self):
        """Initialize the enhanced model trainer."""
        try:
            # Create ensemble models
            self._create_ensemble_models()

            # Load existing models if available
            await self._load_existing_models()

            # Start auto-retraining if enabled
            if settings.auto_retraining:
                self._start_auto_retraining()

            logger.info("🚀 Enhanced model trainer initialized successfully")

        except Exception as e:
            logger.error(f"❌ Failed to initialize enhanced model trainer: {e}")
            raise

    def _create_ensemble_models(self):
        """Create ensemble models for better performance."""
        try:
            # Classification ensemble
            base_classifiers = [
                ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingClassifier(n_estimators=50, random_state=42)),
                ('et', ExtraTreesClassifier(n_estimators=50, random_state=42))
            ]

            self.model_configs['classification']['ensemble'] = [
                ('voting_soft', VotingClassifier(estimators=base_classifiers, voting='soft')),
                ('voting_hard', VotingClassifier(estimators=base_classifiers, voting='hard')),
                ('stacking', StackingClassifier(
                    estimators=base_classifiers,
                    final_estimator=LogisticRegression(random_state=42)
                ))
            ]

            # Regression ensemble
            base_regressors = [
                ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
                ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
                ('et', ExtraTreesRegressor(n_estimators=50, random_state=42))
            ]

            self.model_configs['regression']['ensemble'] = [
                ('voting', VotingRegressor(estimators=base_regressors)),
                ('stacking', StackingRegressor(
                    estimators=base_regressors,
                    final_estimator=LinearRegression()
                ))
            ]

            logger.info("✅ Ensemble models created")

        except Exception as e:
            logger.error(f"❌ Error creating ensemble models: {e}")

    async def _load_existing_models(self):
        """Load previously trained enhanced models from disk."""
        try:
            models_dir = "models/enhanced"
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
                                self.model_metrics[model_name] = EnhancedModelMetrics(**metrics_data)

                        logger.info(f"📥 Loaded enhanced model: {model_name}")

                    except Exception as e:
                        logger.warning(f"⚠️ Error loading enhanced model {model_name}: {e}")

        except Exception as e:
            logger.error(f"❌ Error loading existing enhanced models: {e}")

    def _start_auto_retraining(self):
        """Start automatic model retraining thread."""
        if self.auto_retrain_thread and self.auto_retrain_thread.is_alive():
            return

        self.auto_retrain_thread = threading.Thread(
            target=self._auto_retrain_loop,
            daemon=True
        )
        self.auto_retrain_thread.start()
        logger.info("🔄 Auto-retraining thread started for enhanced models")

    def _auto_retrain_loop(self):
        """Background loop for automatic model retraining."""
        while not self.stop_auto_retrain.is_set():
            try:
                # Check if it's time to retrain
                if self._should_retrain():
                    asyncio.run(self.retrain_enhanced_models())

                # Wait for next check
                self.stop_auto_retrain.wait(settings.model_update_interval)

            except Exception as e:
                logger.error(f"❌ Error in enhanced auto-retrain loop: {e}")
                self.stop_auto_retrain.wait(60)  # Wait 1 minute before retrying

    def _should_retrain(self) -> bool:
        """Check if enhanced models should be retrained."""
        if not self.last_training_time:
            return True

        time_since_last_training = datetime.now() - self.last_training_time
        return time_since_last_training.total_seconds() >= settings.model_update_interval

    async def train_enhanced_model(
        self,
        model_name: str,
        X: pd.DataFrame,
        y: pd.Series,
        model_type: str = "auto",
        test_size: float = 0.2,
        random_state: int = 42,
        use_ensemble: bool = True,
        feature_selection: bool = True
    ) -> Dict[str, Any]:
        """
        Train an enhanced machine learning model with superior performance.

        Args:
            model_name: Name identifier for the model
            X: Feature matrix
            y: Target variable
            model_type: Type of model ("classification", "regression", "auto")
            test_size: Proportion of data for testing
            random_state: Random seed for reproducibility
            use_ensemble: Whether to use ensemble methods
            feature_selection: Whether to perform feature selection

        Returns:
            Dict containing training results and enhanced metrics
        """
        try:
            self.is_training = True
            start_time = time.time()

            logger.info(f"🏋️ Starting enhanced training for model: {model_name}")

            # Determine model type if auto
            if model_type == "auto":
                model_type = self._detect_model_type(y)

            # Preprocessing pipeline
            X_train, X_test, y_train, y_test, scaler, label_encoder = await self._enhanced_preprocessing(
                X, y, model_type, model_name, test_size, random_state
            )

            # Feature selection
            if feature_selection and len(X_train.columns) > 5:
                X_train, X_test = await self._perform_feature_selection(
                    X_train, X_test, y_train, model_type
                )

            # Train multiple models and select best
            best_model, training_params, model_metrics = await self._train_multiple_models(
                X_train, X_test, y_train, y_test, model_type, model_name, use_ensemble
            )

            # Calculate comprehensive metrics
            enhanced_metrics = await self._calculate_enhanced_metrics(
                best_model, X_test, y_test, model_type, (time.time() - start_time)
            )

            # Save model with enhanced metadata
            await self._save_enhanced_model(
                model_name, best_model, scaler, label_encoder, enhanced_metrics, training_params
            )

            # Log to MLflow
            await self._log_to_mlflow(model_name, best_model, enhanced_metrics, training_params)

            # Update tracking
            self.models[model_name] = best_model
            self.scalers[model_name] = scaler
            self.label_encoders[model_name] = label_encoder
            self.model_metrics[model_name] = enhanced_metrics
            self.last_training_time = datetime.now()

            # Save to database
            db_service.save_model_performance(
                model_name=model_name,
                model_version="enhanced_v1",
                accuracy=enhanced_metrics.accuracy,
                precision=enhanced_metrics.precision,
                recall=enhanced_metrics.recall,
                f1_score=enhanced_metrics.f1_score,
                mse=enhanced_metrics.mse,
                rmse=enhanced_metrics.rmse,
                training_time=enhanced_metrics.training_time,
                dataset_size=len(X),
                metadata={
                    'algorithm': training_params.get('algorithm', 'unknown'),
                    'features': len(X_train.columns),
                    'model_type': model_type,
                    'ensemble_used': use_ensemble
                }
            )

            logger.info(f"✅ Enhanced model {model_name} trained successfully")

            return {
                "model_name": model_name,
                "model_type": model_type,
                "algorithm": training_params.get('algorithm', 'unknown'),
                "metrics": enhanced_metrics.to_dict(),
                "training_size": len(X_train),
                "test_size": len(X_test),
                "features": list(X.columns),
                "training_time": enhanced_metrics.training_time,
                "performance_score": self._calculate_performance_score(enhanced_metrics, model_type)
            }

        except Exception as e:
            logger.error(f"❌ Error training enhanced model {model_name}: {e}")
            raise
        finally:
            self.is_training = False

    def _detect_model_type(self, y: pd.Series) -> str:
        """Enhanced model type detection."""
        if y.dtype in ['object', 'string', 'category']:
            return "classification"
        elif len(y.unique()) < 20 and y.dtype in ['int64', 'int32']:
            return "classification"
        else:
            return "regression"

    async def _enhanced_preprocessing(
        self, X: pd.DataFrame, y: pd.Series, model_type: str, model_name: str,
        test_size: float, random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler, Optional[LabelEncoder]]:
        """Enhanced preprocessing with advanced techniques."""

        # Handle missing values with intelligent imputation
        X = X.copy()

        # Numeric columns: use median for outliers, mean for normal
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if X[col].isnull().sum() > 0:
                # Use median for columns with outliers
                if X[col].skew() > 1 or X[col].skew() < -1:
                    X[col] = X[col].fillna(X[col].median())
                else:
                    X[col] = X[col].fillna(X[col].mean())

        # Categorical columns: use mode
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns
        for col in categorical_cols:
            if X[col].isnull().sum() > 0:
                X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y if model_type == "classification" else None
        )

        # Scale features using RobustScaler for better outlier handling
        scaler = RobustScaler()
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

        # Encode labels for classification
        label_encoder = None
        if model_type == "classification":
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
        else:
            y_train_encoded, y_test_encoded = y_train, y_test

        return X_train_scaled, X_test_scaled, y_train_encoded, y_test_encoded, scaler, label_encoder

    async def _perform_feature_selection(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, model_type: str
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Perform intelligent feature selection."""

        try:
            # Select top K features based on statistical tests
            n_features = min(len(X_train.columns), max(10, len(X_train.columns) // 2))

            if model_type == "classification":
                selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                selector = SelectKBest(score_func=f_regression, k=n_features)

            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)

            # Get selected feature names
            selected_features = X_train.columns[selector.get_support()].tolist()

            # Create new DataFrames with selected features
            X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features, index=X_train.index)
            X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features, index=X_test.index)

            logger.info(f"🎯 Selected {len(selected_features)} features from {len(X_train.columns)}")
            return X_train_selected, X_test_selected

        except Exception as e:
            logger.warning(f"⚠️ Feature selection failed, using all features: {e}")
            return X_train, X_test

    async def _train_multiple_models(
        self, X_train: pd.DataFrame, X_test: pd.DataFrame,
        y_train: pd.Series, y_test: pd.Series, model_type: str,
        model_name: str, use_ensemble: bool
    ) -> Tuple[Any, Dict[str, Any], EnhancedModelMetrics]:
        """Train multiple models and select the best performer."""

        best_model = None
        best_score = -1
        best_params = {}
        best_metrics = None

        # Test simple models first
        models_to_test = self.model_configs[model_type]['simple'].copy()

        # Add ensemble models if requested
        if use_ensemble and self.model_configs[model_type]['ensemble']:
            models_to_test.extend(self.model_configs[model_type]['ensemble'])

        for model_info in models_to_test:
            try:
                model_key, model = model_info

                # Skip if model is None (e.g., XGBoost not installed)
                if model is None:
                    continue

                # Train model
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time

                # Make predictions
                y_pred = model.predict(X_test)

                # Calculate metrics
                if model_type == "classification":
                    accuracy = accuracy_score(y_test, y_pred)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

                    # Use accuracy as primary score for model selection
                    current_score = accuracy
                    current_metrics = EnhancedModelMetrics(
                        accuracy=accuracy,
                        precision=precision,
                        recall=recall,
                        f1_score=f1,
                        training_time=training_time
                    )
                else:
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    # Use negative MSE for model selection (higher is better)
                    current_score = -mse
                    current_metrics = EnhancedModelMetrics(
                        mse=mse,
                        rmse=rmse,
                        mae=mae,
                        r2_score=r2,
                        training_time=training_time
                    )

                # Update best model if this one is better
                if current_score > best_score:
                    best_score = current_score
                    best_model = model
                    best_params = {
                        'algorithm': model_key,
                        'model_type': model_type,
                        'training_time': training_time,
                        'feature_count': len(X_train.columns)
                    }
                    best_metrics = current_metrics

                logger.debug(f"📊 {model_key} - Score: {current_score:.4f}, Time: {training_time:.2f}s")

            except Exception as e:
                logger.warning(f"⚠️ Error training {model_info[0]}: {e}")
                continue

        if best_model is None:
            raise ValueError("No models could be trained successfully")

        logger.info(f"🏆 Best model: {best_params['algorithm']} with score: {best_score".4f"}")
        return best_model, best_params, best_metrics

    async def _calculate_enhanced_metrics(
        self, model: Any, X_test: pd.DataFrame, y_test: pd.Series,
        model_type: str, training_time: float
    ) -> EnhancedModelMetrics:
        """Calculate comprehensive enhanced metrics."""

        # Measure inference time
        start_time = time.time()
        y_pred = model.predict(X_test)
        inference_time = (time.time() - start_time) / len(X_test)  # Per sample

        # Calculate model size
        model_size_mb = len(pickle.dumps(model)) / (1024 * 1024)

        if model_type == "classification":
            accuracy = accuracy_score(y_test, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')

            # Cross-validation score
            try:
                cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='accuracy')
                cross_val_score_mean = cv_scores.mean()
            except:
                cross_val_score_mean = accuracy  # Fallback

            metrics = EnhancedModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cross_val_score=cross_val_score_mean,
                training_time=training_time,
                inference_time=inference_time,
                model_size_mb=model_size_mb
            )

        else:
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Cross-validation score (negative MSE)
            try:
                cv_scores = cross_val_score(model, X_test, y_test, cv=3, scoring='neg_mean_squared_error')
                cross_val_score_mean = -cv_scores.mean()  # Convert back to positive
            except:
                cross_val_score_mean = mse  # Fallback

            metrics = EnhancedModelMetrics(
                mse=mse,
                rmse=rmse,
                mae=mae,
                r2_score=r2,
                cross_val_score=cross_val_score_mean,
                training_time=training_time,
                inference_time=inference_time,
                model_size_mb=model_size_mb
            )

        # Get feature importance if available
        if hasattr(model, 'feature_importances_'):
            feature_names = X_test.columns
            importance_dict = dict(zip(feature_names, model.feature_importances_))
            metrics.feature_importance = importance_dict

        return metrics

    def _calculate_performance_score(self, metrics: EnhancedModelMetrics, model_type: str) -> float:
        """Calculate overall performance score for model comparison."""
        if model_type == "classification":
            # Weighted score for classification
            return (
                metrics.accuracy * 0.3 +
                metrics.f1_score * 0.3 +
                metrics.cross_val_score * 0.2 +
                (1 - metrics.training_time / 300) * 0.1 +  # Prefer faster models
                (1 - metrics.model_size_mb / 100) * 0.1     # Prefer smaller models
            )
        else:
            # Weighted score for regression (higher R2 is better, lower MSE is better)
            r2_normalized = max(0, min(1, metrics.r2_score))
            mse_normalized = max(0, min(1, 1 - metrics.mse / (metrics.mse + 100)))  # Normalize MSE

            return (
                r2_normalized * 0.3 +
                mse_normalized * 0.3 +
                metrics.cross_val_score * 0.2 +
                (1 - metrics.training_time / 300) * 0.1 +
                (1 - metrics.model_size_mb / 100) * 0.1
            )

    async def _save_enhanced_model(
        self, model_name: str, model: Any, scaler: StandardScaler,
        label_encoder: Optional[LabelEncoder], metrics: EnhancedModelMetrics,
        training_params: Dict[str, Any]
    ):
        """Save enhanced model with comprehensive metadata."""
        try:
            # Save model data
            model_data = {
                'model': model,
                'scaler': scaler,
                'label_encoder': label_encoder,
                'metrics': metrics.to_dict(),
                'training_params': training_params,
                'saved_at': datetime.now().isoformat(),
                'model_type': 'enhanced',
                'version': '1.0'
            }

            model_path = f"models/enhanced/{model_name}.pkl"
            joblib.dump(model_data, model_path)

            # Save metrics separately for easy access
            metrics_path = f"models/enhanced/{model_name}_metrics.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics.to_dict(), f, default=str, indent=2)

            logger.info(f"💾 Enhanced model {model_name} saved to {model_path}")

        except Exception as e:
            logger.error(f"❌ Error saving enhanced model {model_name}: {e}")

    async def _log_to_mlflow(
        self, model_name: str, model: Any, metrics: EnhancedModelMetrics, params: Dict[str, Any]
    ):
        """Log enhanced model and metrics to MLflow."""
        try:
            with mlflow.start_run(run_name=f"enhanced_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                # Log parameters
                mlflow.log_params(params)

                # Log metrics
                if metrics.accuracy > 0:
                    mlflow.log_metric("accuracy", metrics.accuracy)
                if metrics.f1_score > 0:
                    mlflow.log_metric("f1_score", metrics.f1_score)
                if metrics.mse > 0:
                    mlflow.log_metric("mse", metrics.mse)
                    mlflow.log_metric("rmse", metrics.rmse)
                if metrics.r2_score != 0:
                    mlflow.log_metric("r2_score", metrics.r2_score)

                # Log training time and model size
                mlflow.log_metric("training_time", metrics.training_time)
                mlflow.log_metric("model_size_mb", metrics.model_size_mb)

                # Log model
                mlflow.sklearn.log_model(model, f"enhanced_model_{model_name}")

                logger.info(f"📊 Enhanced model {model_name} logged to MLflow")

        except Exception as e:
            logger.error(f"❌ Error logging enhanced model to MLflow: {e}")

    async def predict_enhanced(
        self,
        model_name: str,
        X: pd.DataFrame,
        return_probabilities: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Make predictions using an enhanced trained model.

        Args:
            model_name: Name of the model to use
            X: Feature matrix for prediction
            return_probabilities: Whether to return prediction probabilities

        Returns:
            Array of predictions or tuple of (predictions, probabilities)
        """
        try:
            if model_name not in self.models:
                raise ValueError(f"Enhanced model {model_name} not found")

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
            logger.error(f"❌ Error making predictions with enhanced model {model_name}: {e}")
            raise

    async def get_model_comparison(self) -> Dict[str, Any]:
        """Get comparison of all trained enhanced models."""
        comparison = {
            'models': [],
            'best_classification': None,
            'best_regression': None,
            'summary': {}
        }

        for model_name, metrics in self.model_metrics.items():
            model_info = {
                'name': model_name,
                'metrics': metrics.to_dict(),
                'performance_score': self._calculate_performance_score(metrics, 'classification' if metrics.accuracy > 0 else 'regression')
            }
            comparison['models'].append(model_info)

        # Find best models by type
        classification_models = [m for m in comparison['models'] if m['metrics']['accuracy'] > 0]
        regression_models = [m for m in comparison['models'] if m['metrics']['mse'] > 0]

        if classification_models:
            comparison['best_classification'] = max(
                classification_models,
                key=lambda x: x['performance_score']
            )

        if regression_models:
            comparison['best_regression'] = max(
                regression_models,
                key=lambda x: x['performance_score']
            )

        # Summary statistics
        if comparison['models']:
            scores = [m['performance_score'] for m in comparison['models']]
            comparison['summary'] = {
                'total_models': len(comparison['models']),
                'average_performance': sum(scores) / len(scores),
                'best_performance': max(scores),
                'worst_performance': min(scores)
            }

        return comparison

    async def retrain_enhanced_models(self):
        """Retrain all existing enhanced models with new data."""
        try:
            logger.info("🔄 Starting enhanced model retraining...")

            for model_name in list(self.models.keys()):
                try:
                    # Get recent data for retraining
                    recent_data = await self._get_recent_training_data(model_name)

                    if recent_data and len(recent_data) > 10:  # Minimum data threshold
                        X, y = self._prepare_training_data(recent_data)
                        await self.train_enhanced_model(
                            model_name, X, y, model_type="auto",
                            use_ensemble=True, feature_selection=True
                        )

                        logger.info(f"✅ Enhanced model {model_name} retrained")

                except Exception as e:
                    logger.error(f"❌ Error retraining enhanced model {model_name}: {e}")

            logger.info("🔄 Enhanced model retraining completed")

        except Exception as e:
            logger.error(f"❌ Error during enhanced model retraining: {e}")

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

    async def get_enhanced_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about an enhanced trained model."""
        if model_name not in self.models:
            raise ValueError(f"Enhanced model {model_name} not found")

        model = self.models[model_name]
        metrics = self.model_metrics.get(model_name)

        return {
            "model_name": model_name,
            "model_type": type(model).__name__,
            "features": getattr(model, 'n_features_in_', 'Unknown'),
            "training_date": self.last_training_time.isoformat() if self.last_training_time else None,
            "metrics": metrics.to_dict() if metrics else None,
            "parameters": model.get_params() if hasattr(model, 'get_params') else {},
            "performance_score": self._calculate_performance_score(metrics, 'classification' if metrics.accuracy > 0 else 'regression') if metrics else 0.0
        }

    async def list_enhanced_models(self) -> List[str]:
        """List all available enhanced trained models."""
        return list(self.models.keys())

    async def delete_enhanced_model(self, model_name: str) -> bool:
        """Delete an enhanced trained model."""
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
                model_path = f"models/enhanced/{model_name}.pkl"
                metrics_path = f"models/enhanced/{model_name}_metrics.json"

                if os.path.exists(model_path):
                    os.remove(model_path)
                if os.path.exists(metrics_path):
                    os.remove(metrics_path)

                logger.info(f"🗑️ Enhanced model {model_name} deleted")
                return True

        except Exception as e:
            logger.error(f"❌ Error deleting enhanced model {model_name}: {e}")

        return False

    async def shutdown(self):
        """Shutdown the enhanced model trainer."""
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
                    await self._save_enhanced_model(model_name, model, scaler, label_encoder, metrics, {})

            logger.info("🛑 Enhanced model trainer shutdown complete")

        except Exception as e:
            logger.error(f"❌ Error during enhanced model trainer shutdown: {e}")

# Global enhanced model trainer instance
enhanced_trainer = EnhancedModelTrainer()
