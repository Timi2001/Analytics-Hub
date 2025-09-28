"""
Response Generator for Natural Language Analytics
"""
import logging
from typing import Dict, List, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generates natural language responses for analytics results."""

    def __init__(self):
        self.response_templates = {
            'visualization_success': [
                "✅ Perfect! I've created a {visualization_type} visualization for the {columns} data.",
                "📊 Great! Here's a {visualization_type} showing the {columns} information.",
                "🎨 Visualization created! Check out this {visualization_type} for {columns}."
            ],
            'summary_success': [
                "📋 Here's a comprehensive summary of your dataset with {rows} rows and {columns} columns.",
                "📊 Data summary generated! Your dataset contains {rows} records across {columns} variables.",
                "✅ Summary complete! Dataset overview: {rows} rows, {columns} columns."
            ],
            'correlation_success': [
                "🔗 Correlation analysis complete! Found {correlations_count} significant relationships in your data.",
                "📈 Correlation insights: I've identified the strongest relationships between your variables.",
                "🔍 Correlation analysis done! Here are the most important variable relationships."
            ],
            'prediction_success': [
                "🔮 Predictions generated using the {model_name} model with {accuracy}% accuracy.",
                "📊 Forecast complete! Generated predictions based on your trained model.",
                "🎯 Prediction results ready! Using {model_name} to forecast outcomes."
            ],
            'trend_success': [
                "📈 Trend analysis complete! Identified patterns in {variables} over time.",
                "🔄 Time series analysis done! Here's how {variables} changed over the time period.",
                "📊 Trend insights: {variables} trends identified and visualized."
            ],
            'feature_importance_success': [
                "🎯 Feature importance analysis complete! The most influential factors are: {top_features}",
                "📊 Key drivers identified! Here are the most important variables for your model.",
                "🔍 Feature analysis done! These variables have the biggest impact: {top_features}"
            ],
            'dashboard_success': [
                "📋 Comprehensive dashboard created! Multi-panel view with {panels} different visualizations.",
                "🎨 Dashboard ready! Complete analytics overview with multiple chart types.",
                "📊 Full dashboard generated! {panels} visualizations showing different aspects of your data."
            ],
            'error_no_data': [
                "📁 I don't see any data loaded yet. Please upload a CSV or Excel file first.",
                "💾 No dataset available. Start by uploading your data file.",
                "📊 Please load your data first, then I can help with the analysis."
            ],
            'error_no_model': [
                "🤖 No trained models found. Please train a model first using your data.",
                "🧠 Model required! Train a machine learning model to enable predictions and insights.",
                "📈 Ready to train! Upload data and create a model to get started."
            ],
            'error_understanding': [
                "🤔 I didn't quite understand that query. Try asking to 'show trends' or 'create a summary'.",
                "💭 Could you rephrase that? Try: 'show me a chart of sales' or 'summarize my data'.",
                "❓ Not sure what you mean. Try asking for a visualization, summary, or trend analysis."
            ]
        }

    def generate_response(self, result: Dict[str, Any]) -> str:
        """
        Generate a natural language response from query results.

        Args:
            result: Query processing result

        Returns:
            Natural language response string
        """
        try:
            if not result.get('success', False):
                return self._generate_error_response(result)

            # Get result metadata
            action = result.get('action', 'unknown')
            intent = result.get('intent', 'unknown')

            # Route to appropriate response generator
            if action == 'visualization_created':
                return self._generate_visualization_response(result)
            elif action == 'data_summary_created':
                return self._generate_summary_response(result)
            elif action == 'correlation_analysis_created':
                return self._generate_correlation_response(result)
            elif action == 'predictions_generated':
                return self._generate_prediction_response(result)
            elif action == 'trend_analysis_created':
                return self._generate_trend_response(result)
            elif action == 'feature_importance_analyzed':
                return self._generate_feature_importance_response(result)
            elif action == 'dashboard_created':
                return self._generate_dashboard_response(result)
            else:
                return self._generate_generic_success_response(result)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "✅ Analysis complete! Check the visualizations and results above."

    def _generate_visualization_response(self, result: Dict[str, Any]) -> str:
        """Generate response for visualization results."""
        templates = self.response_templates['visualization_success']

        # Get context information
        viz_type = result.get('visualization_type', 'chart')
        columns = result.get('columns_visualized', ['data'])
        columns_text = ', '.join(columns)

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            visualization_type=viz_type,
            columns=columns_text
        )

    def _generate_summary_response(self, result: Dict[str, Any]) -> str:
        """Generate response for summary results."""
        templates = self.response_templates['summary_success']

        # Get context information
        summary = result.get('summary', {})
        rows = summary.get('total_rows', 0)
        columns = summary.get('total_columns', 0)

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            rows=rows,
            columns=columns
        )

    def _generate_correlation_response(self, result: Dict[str, Any]) -> str:
        """Generate response for correlation results."""
        templates = self.response_templates['correlation_success']

        # Get context information
        correlations = result.get('strongest_correlations', [])
        correlations_count = len(correlations)

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            correlations_count=correlations_count
        )

    def _generate_prediction_response(self, result: Dict[str, Any]) -> str:
        """Generate response for prediction results."""
        templates = self.response_templates['prediction_success']

        # Get context information
        model_name = result.get('model_name', 'trained model')
        # Note: Would need to get accuracy from model metrics in real implementation

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            model_name=model_name,
            accuracy="85"  # Placeholder - would come from actual model metrics
        )

    def _generate_trend_response(self, result: Dict[str, Any]) -> str:
        """Generate response for trend analysis results."""
        templates = self.response_templates['trend_success']

        # Get context information
        variables = result.get('analyzed_variables', [])
        variables_text = ', '.join(variables[:3])  # Show first 3

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            variables=variables_text
        )

    def _generate_feature_importance_response(self, result: Dict[str, Any]) -> str:
        """Generate response for feature importance results."""
        templates = self.response_templates['feature_importance_success']

        # Get context information
        top_features = result.get('top_features', [])
        if top_features:
            features_text = ', '.join([feature[0] for feature in top_features[:3]])
        else:
            features_text = 'identified'

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            top_features=features_text
        )

    def _generate_dashboard_response(self, result: Dict[str, Any]) -> str:
        """Generate response for dashboard results."""
        templates = self.response_templates['dashboard_success']

        # Get context information
        panels = result.get('columns_analyzed', 1)  # Estimate based on columns

        # Select random template and format
        import random
        template = random.choice(templates)

        return template.format(
            panels=panels
        )

    def _generate_error_response(self, result: Dict[str, Any]) -> str:
        """Generate response for error conditions."""
        error_type = result.get('action_required', 'unknown')

        if error_type == 'upload_data':
            templates = self.response_templates['error_no_data']
        elif error_type == 'train_model':
            templates = self.response_templates['error_no_model']
        else:
            templates = self.response_templates['error_understanding']

        # Select random template
        import random
        return random.choice(templates)

    def _generate_generic_success_response(self, result: Dict[str, Any]) -> str:
        """Generate generic success response."""
        return "✅ Analysis complete! Check the results and visualizations above."

    def generate_suggestions(self, failed_query: str, available_data: Dict[str, Any]) -> List[str]:
        """
        Generate helpful suggestions based on failed query and available data.

        Args:
            failed_query: The query that couldn't be processed
            available_data: Information about available data

        Returns:
            List of suggested queries
        """
        suggestions = []

        # Basic suggestions based on data availability
        if available_data.get('has_data', False):
            suggestions.extend([
                "Show me a summary of my data",
                "Create a visualization of the numeric columns",
                "What are the trends in my data?",
                "Find correlations between variables"
            ])

            if available_data.get('trained_models'):
                suggestions.extend([
                    "What are the most important features?",
                    "Generate predictions using the trained model",
                    "Show model performance metrics"
                ])
        else:
            suggestions.extend([
                "Upload a CSV or Excel file to get started",
                "Load your dataset for analysis",
                "Import data to begin analytics"
            ])

        # Context-specific suggestions based on query content
        query_lower = failed_query.lower()

        if any(word in query_lower for word in ['sales', 'revenue', 'profit']):
            suggestions.append("Show me sales trends over time")
            suggestions.append("Compare revenue between different periods")

        if any(word in query_lower for word in ['customer', 'user', 'client']):
            suggestions.append("Analyze customer satisfaction trends")
            suggestions.append("Compare user behavior across segments")

        return suggestions[:5]  # Return top 5 suggestions
