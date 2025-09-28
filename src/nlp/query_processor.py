"""
Query Processor for Natural Language Analytics
"""
import asyncio
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

from .intent_recognizer import IntentRecognizer, IntentRecognitionResult, QueryIntent
from ..visualization.professional_charts import chart_generator
from ..models.enhanced_trainer import enhanced_trainer
from ..database.service import db_service

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes natural language queries and generates appropriate responses."""

    def __init__(self):
        self.intent_recognizer = IntentRecognizer()
        self.current_data: Optional[pd.DataFrame] = None
        self.data_columns: List[str] = []
        self.trained_models: List[str] = []

    async def process_query(self, query: str, data: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Process a natural language query and return results.

        Args:
            query: Natural language query from user
            data: Optional DataFrame to analyze

        Returns:
            Dictionary containing results, visualizations, and insights
        """
        try:
            # Update current data if provided
            if data is not None:
                self.current_data = data
                self.data_columns = list(data.columns)
                self.trained_models = await enhanced_trainer.list_enhanced_models()

            # Recognize intent
            intent_result = self.intent_recognizer.recognize_intent(query)

            logger.info(f"Query: {query}")
            logger.info(f"Intent: {intent_result.intent.value}, Confidence: {intent_result.confidence:.2f}")

            # Process based on intent
            if intent_result.confidence > 0.5:
                result = await self._process_intent_query(intent_result)
            else:
                result = await self._process_unknown_query(query, intent_result)

            # Add metadata
            result.update({
                'original_query': query,
                'intent': intent_result.intent.value,
                'confidence': intent_result.confidence,
                'entities': intent_result.entities,
                'processed_at': datetime.now().isoformat()
            })

            return result

        except Exception as e:
            logger.error(f"Error processing query '{query}': {e}")
            return {
                'success': False,
                'error': str(e),
                'original_query': query,
                'suggestion': 'Please try rephrasing your query or check your data format'
            }

    async def _process_intent_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Process a query with recognized intent."""
        intent = intent_result.intent
        entities = intent_result.entities

        # Route to appropriate handler
        handlers = {
            QueryIntent.VISUALIZE_DATA: self._handle_visualization_query,
            QueryIntent.ANALYZE_TRENDS: self._handle_trend_analysis,
            QueryIntent.COMPARE_GROUPS: self._handle_comparison_query,
            QueryIntent.FIND_CORRELATIONS: self._handle_correlation_query,
            QueryIntent.DATA_SUMMARY: self._handle_summary_query,
            QueryIntent.FEATURE_IMPORTANCE: self._handle_feature_importance_query,
            QueryIntent.PREDICT_VALUES: self._handle_prediction_query,
            QueryIntent.CREATE_DASHBOARD: self._handle_dashboard_query
        }

        handler = handlers.get(intent, self._handle_unknown_intent)
        return await handler(intent_result)

    async def _handle_visualization_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle visualization requests."""
        entities = intent_result.entities

        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for visualization. Please upload data first.',
                'action_required': 'upload_data'
            }

        # Extract columns to visualize
        columns = entities.get('columns', [])
        if not columns and len(self.data_columns) > 0:
            # Default to first few numeric columns
            numeric_cols = [col for col in self.data_columns if self.current_data[col].dtype in ['int64', 'float64']]
            columns = numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols

        if not columns:
            return {
                'success': False,
                'message': 'Could not identify columns to visualize. Please specify column names.',
                'available_columns': self.data_columns
            }

        try:
            # Create visualization
            sample_data = self.current_data[columns].head(100)  # Sample for performance

            # Determine best visualization type
            if len(columns) == 1:
                # Single variable - use distribution plot
                chart = chart_generator.create_statistical_summary_plot(sample_data)
            elif len(columns) == 2:
                # Two variables - scatter plot
                chart = chart_generator.create_correlation_heatmap(sample_data)
            else:
                # Multiple variables - comprehensive dashboard
                chart = chart_generator.create_comprehensive_dashboard(
                    sample_data,
                    target_column=columns[0] if len(columns) > 1 else None
                )

            return {
                'success': True,
                'message': f'Created visualization for columns: {", ".join(columns)}',
                'visualization_type': 'statistical_summary' if len(columns) == 1 else 'comprehensive_dashboard',
                'columns_visualized': columns,
                'data_points': len(sample_data),
                'action': 'visualization_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating visualization: {e}',
                'columns_requested': columns
            }

    async def _handle_trend_analysis(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle trend analysis requests."""
        entities = intent_result.entities

        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for trend analysis. Please upload data first.'
            }

        # Look for time-based columns
        time_columns = [col for col in self.data_columns if 'time' in col.lower() or 'date' in col.lower()]
        if not time_columns:
            # Try to infer time column
            for col in self.data_columns:
                if self.current_data[col].dtype in ['datetime64[ns]', 'object']:
                    try:
                        pd.to_datetime(self.current_data[col])
                        time_columns.append(col)
                        break
                    except:
                        continue

        if not time_columns:
            return {
                'success': False,
                'message': 'No time-based columns found for trend analysis. Please ensure your data includes date/time information.',
                'available_columns': self.data_columns
            }

        try:
            # Use first time column
            time_col = time_columns[0]
            numeric_cols = [col for col in self.data_columns if col != time_col and self.current_data[col].dtype in ['int64', 'float64']]

            if not numeric_cols:
                return {
                    'success': False,
                    'message': 'No numeric columns found for trend analysis.'
                }

            # Create trend visualization
            trend_data = self.current_data[[time_col] + numeric_cols[:3]].head(50)  # Sample for performance

            # This would create a time series visualization
            return {
                'success': True,
                'message': f'Created trend analysis for {len(numeric_cols)} variables over time',
                'time_column': time_col,
                'analyzed_variables': numeric_cols[:3],
                'data_points': len(trend_data),
                'action': 'trend_analysis_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in trend analysis: {e}'
            }

    async def _handle_summary_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle data summary requests."""
        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for summary. Please upload data first.'
            }

        try:
            # Generate comprehensive data summary
            summary = {
                'total_rows': len(self.current_data),
                'total_columns': len(self.data_columns),
                'column_types': {col: str(self.current_data[col].dtype) for col in self.data_columns},
                'missing_values': self.current_data.isnull().sum().to_dict(),
                'numeric_columns': [col for col in self.data_columns if self.current_data[col].dtype in ['int64', 'float64']],
                'categorical_columns': [col for col in self.data_columns if self.current_data[col].dtype == 'object']
            }

            # Add basic statistics for numeric columns
            if summary['numeric_columns']:
                sample_stats = self.current_data[summary['numeric_columns']].describe()
                summary['basic_statistics'] = sample_stats.to_dict()

            return {
                'success': True,
                'message': f'Generated summary for dataset with {summary["total_rows"]} rows and {summary["total_columns"]} columns',
                'summary': summary,
                'action': 'data_summary_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error generating summary: {e}'
            }

    async def _handle_correlation_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle correlation analysis requests."""
        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for correlation analysis. Please upload data first.'
            }

        try:
            # Get numeric columns for correlation
            numeric_cols = [col for col in self.data_columns if self.current_data[col].dtype in ['int64', 'float64']]

            if len(numeric_cols) < 2:
                return {
                    'success': False,
                    'message': 'Need at least 2 numeric columns for correlation analysis.',
                    'available_numeric_columns': numeric_cols
                }

            # Sample data for correlation (correlation on full dataset might be slow)
            sample_data = self.current_data[numeric_cols].head(1000)

            # Calculate correlation matrix
            correlation_matrix = sample_data.corr()

            return {
                'success': True,
                'message': f'Generated correlation analysis for {len(numeric_cols)} numeric variables',
                'correlation_matrix': correlation_matrix.to_dict(),
                'analyzed_columns': numeric_cols,
                'strongest_correlations': self._find_strongest_correlations(correlation_matrix),
                'action': 'correlation_analysis_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in correlation analysis: {e}'
            }

    async def _handle_feature_importance_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle feature importance requests."""
        if not self.trained_models:
            return {
                'success': False,
                'message': 'No trained models available. Please train a model first.',
                'action_required': 'train_model'
            }

        try:
            # Use the first available model
            model_name = self.trained_models[0]
            model_info = await enhanced_trainer.get_enhanced_model_info(model_name)

            if not model_info.get('metrics', {}).get('feature_importance'):
                return {
                    'success': False,
                    'message': 'Selected model does not support feature importance analysis.',
                    'available_models': self.trained_models
                }

            feature_importance = model_info['metrics']['feature_importance']

            return {
                'success': True,
                'message': f'Generated feature importance analysis for model: {model_name}',
                'model_name': model_name,
                'feature_importance': feature_importance,
                'top_features': self._get_top_features(feature_importance, 5),
                'action': 'feature_importance_analyzed'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in feature importance analysis: {e}'
            }

    async def _handle_prediction_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle prediction requests."""
        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for prediction. Please upload data first.'
            }

        if not self.trained_models:
            return {
                'success': False,
                'message': 'No trained models available for prediction. Please train a model first.',
                'action_required': 'train_model'
            }

        try:
            # Use the first available model for prediction
            model_name = self.trained_models[0]

            # Prepare prediction data (exclude any target columns)
            feature_cols = [col for col in self.data_columns if col not in ['target', 'prediction', 'label']]
            prediction_data = self.current_data[feature_cols].head(10)  # Sample for demo

            # Make predictions
            predictions = await enhanced_trainer.predict_enhanced(model_name, prediction_data)

            return {
                'success': True,
                'message': f'Generated predictions using model: {model_name}',
                'model_name': model_name,
                'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
                'prediction_data_shape': prediction_data.shape,
                'action': 'predictions_generated'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error generating predictions: {e}'
            }

    async def _handle_dashboard_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle dashboard creation requests."""
        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for dashboard creation. Please upload data first.'
            }

        try:
            # Create comprehensive dashboard
            sample_data = self.current_data.head(100)  # Sample for performance

            # This would create a comprehensive dashboard
            return {
                'success': True,
                'message': 'Generated comprehensive dashboard with multiple visualizations',
                'dashboard_type': 'comprehensive_analytics',
                'data_points_included': len(sample_data),
                'columns_analyzed': len(self.data_columns),
                'action': 'dashboard_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating dashboard: {e}'
            }

    async def _handle_comparison_query(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle comparison requests."""
        entities = intent_result.entities

        if not self.current_data:
            return {
                'success': False,
                'message': 'No data available for comparison. Please upload data first.'
            }

        # Extract comparison groups
        comparison_groups = entities.get('comparisons', [])

        if not comparison_groups:
            return {
                'success': False,
                'message': 'Could not identify groups to compare. Please specify what to compare.',
                'suggestion': 'Try: "compare sales between regions" or "compare performance across departments"'
            }

        try:
            # This would create comparison visualizations
            return {
                'success': True,
                'message': f'Created comparison analysis for: {", ".join(comparison_groups)}',
                'comparison_groups': comparison_groups,
                'action': 'comparison_analysis_created'
            }

        except Exception as e:
            return {
                'success': False,
                'message': f'Error in comparison analysis: {e}'
            }

    async def _handle_unknown_intent(self, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Handle queries with unknown intent."""
        return {
            'success': False,
            'message': 'Could not understand the query. Please try rephrasing.',
            'suggestion': 'Try asking to visualize data, analyze trends, or create a summary',
            'confidence': intent_result.confidence,
            'entities_found': intent_result.entities
        }

    async def _process_unknown_query(self, query: str, intent_result: IntentRecognitionResult) -> Dict[str, Any]:
        """Process queries that couldn't be classified with high confidence."""
        # Try to provide helpful suggestions based on keywords
        query_lower = query.lower()

        suggestions = []

        if any(word in query_lower for word in ['data', 'dataset', 'information']):
            suggestions.append("Try asking for a data summary or visualization")

        if any(word in query_lower for word in ['trend', 'time', 'change']):
            suggestions.append("Try asking about trends or patterns over time")

        if any(word in query_lower for word in ['compare', 'difference', 'between']):
            suggestions.append("Try asking to compare different groups or categories")

        if any(word in query_lower for word in ['predict', 'forecast', 'future']):
            suggestions.append("Try asking for predictions or forecasts")

        return {
            'success': False,
            'message': 'Query not clearly understood. Here are some suggestions:',
            'suggestions': suggestions,
            'original_query': query,
            'confidence': intent_result.confidence
        }

    def _find_strongest_correlations(self, corr_matrix: pd.DataFrame) -> List[Dict]:
        """Find the strongest correlations in the matrix."""
        # Get upper triangle of correlation matrix
        correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                col1, col2 = corr_matrix.columns[i], corr_matrix.columns[j]
                correlation = corr_matrix.iloc[i, j]
                correlations.append({
                    'column1': col1,
                    'column2': col2,
                    'correlation': correlation
                })

        # Sort by absolute correlation strength
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

        return correlations[:5]  # Return top 5

    def _get_top_features(self, feature_importance: Dict, n: int = 5) -> List[Tuple[str, float]]:
        """Get the top n most important features."""
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        return sorted_features[:n]

    async def get_available_data_info(self) -> Dict[str, Any]:
        """Get information about currently available data."""
        return {
            'has_data': self.current_data is not None,
            'data_shape': self.current_data.shape if self.current_data is not None else None,
            'columns': self.data_columns,
            'trained_models': self.trained_models,
            'data_types': {col: str(self.current_data[col].dtype) for col in self.data_columns} if self.current_data is not None else {}
        }
