"""
Intent Recognition for Natural Language Analytics Queries
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    """Types of user intents in analytics queries."""
    VISUALIZE_DATA = "visualize_data"
    ANALYZE_TRENDS = "analyze_trends"
    COMPARE_GROUPS = "compare_groups"
    FIND_CORRELATIONS = "find_correlations"
    PREDICT_VALUES = "predict_values"
    IDENTIFY_ANOMALIES = "identify_anomalies"
    CREATE_DASHBOARD = "create_dashboard"
    TRAIN_MODEL = "train_model"
    FEATURE_IMPORTANCE = "feature_importance"
    DATA_SUMMARY = "data_summary"
    EXPORT_DATA = "export_data"
    UNKNOWN = "unknown"

@dataclass
class IntentRecognitionResult:
    """Result of intent recognition."""
    intent: QueryIntent
    confidence: float
    entities: Dict[str, List[str]]
    original_query: str
    suggested_action: str

class IntentRecognizer:
    """Recognizes user intent from natural language queries."""

    def __init__(self):
        # Intent patterns with regex and keywords
        self.intent_patterns = {
            QueryIntent.VISUALIZE_DATA: {
                'patterns': [
                    r'(?:show|display|plot|chart|graph|visualize)\s+(?:me\s+)?(.+?)(?:\s+over\s+time|\s+by\s+\w+|\s+for\s+\w+)?$',
                    r'(?:create|make)\s+(?:a\s+)?(?:chart|plot|graph|visualization)\s+(?:of|for)\s+(.+)$',
                    r'(?:i\s+)?(?:want\s+to\s+)?(?:see|view)\s+(.+?)\s+(?:in\s+)?(?:a\s+)?(?:chart|plot|graph)$'
                ],
                'keywords': ['show', 'display', 'plot', 'chart', 'graph', 'visualize', 'see', 'view'],
                'confidence_weight': 0.9
            },
            QueryIntent.ANALYZE_TRENDS: {
                'patterns': [
                    r'(?:analyze|examine|study|look\s+at)\s+(?:the\s+)?trends?\s+(?:in|of|for)\s+(.+)$',
                    r'(?:what\s+are\s+the\s+)?trends?\s+(?:in|of|for)\s+(.+?)(?:\s+over\s+time)?$',
                    r'(?:how\s+(?:has|have|did)\s+(.+?)\s+changed?\s+over\s+time)$'
                ],
                'keywords': ['trend', 'trends', 'over time', 'change', 'pattern', 'analyze', 'examine'],
                'confidence_weight': 0.85
            },
            QueryIntent.COMPARE_GROUPS: {
                'patterns': [
                    r'(?:compare|contrast)\s+(.+?)\s+(?:between|among|across)\s+(.+)$',
                    r'(?:what\s+(?:is\s+the\s+)?difference)\s+(?:between|among)\s+(.+)$',
                    r'(?:how\s+does?|do)\s+(.+?)\s+(?:differ|vary)\s+(?:between|among|across)\s+(.+)$'
                ],
                'keywords': ['compare', 'contrast', 'difference', 'between', 'among', 'across', 'versus', 'vs'],
                'confidence_weight': 0.8
            },
            QueryIntent.FIND_CORRELATIONS: {
                'patterns': [
                    r'(?:find|identify|show)\s+(?:the\s+)?correlations?\s+(?:between|among)\s+(.+)$',
                    r'(?:what\s+(?:is\s+the\s+)?relationship)\s+(?:between|among)\s+(.+)$',
                    r'(?:how\s+(?:are|is)\s+(.+?)\s+related?\s+to\s+(.+))$'
                ],
                'keywords': ['correlation', 'relationship', 'related', 'correlated', 'association'],
                'confidence_weight': 0.8
            },
            QueryIntent.PREDICT_VALUES: {
                'patterns': [
                    r'(?:predict|forecast|estimate)\s+(.+?)\s+(?:for|based\s+on)\s+(.+)$',
                    r'(?:what\s+will\s+(.+?)\s+be\s+in)\s+(.+)$',
                    r'(?:forecast|project)\s+(.+?)\s+(?:into\s+the\s+)?future$'
                ],
                'keywords': ['predict', 'forecast', 'estimate', 'projection', 'future', 'will be'],
                'confidence_weight': 0.75
            },
            QueryIntent.DATA_SUMMARY: {
                'patterns': [
                    r'(?:give\s+me\s+)?(?:a\s+)?summary\s+(?:of\s+)?(.+?)(?:\s+data)?$',
                    r'(?:what\s+(?:are\s+the\s+)?statistics)\s+(?:for|of)\s+(.+)$',
                    r'(?:describe|summarize)\s+(.+?)(?:\s+data)?$'
                ],
                'keywords': ['summary', 'summarize', 'statistics', 'describe', 'overview', 'stats'],
                'confidence_weight': 0.9
            },
            QueryIntent.FEATURE_IMPORTANCE: {
                'patterns': [
                    r'(?:what\s+(?:are\s+the\s+)?most\s+important)\s+features?\s+(?:for|in)\s+(.+)$',
                    r'(?:which\s+factors?\s+(?:most\s+)?influence)\s+(.+)$',
                    r'(?:feature\s+importance|important\s+variables)\s+(?:for|in)\s+(.+)$'
                ],
                'keywords': ['important', 'feature importance', 'factors', 'variables', 'influence', 'impact'],
                'confidence_weight': 0.85
            },
            QueryIntent.CREATE_DASHBOARD: {
                'patterns': [
                    r'(?:create|build|make)\s+(?:a\s+)?dashboard\s+(?:for|with|showing)\s+(.+)$',
                    r'(?:design|generate)\s+(?:a\s+)?(?:comprehensive\s+)?dashboard\s+(?:for|with)\s+(.+)$'
                ],
                'keywords': ['dashboard', 'create dashboard', 'build dashboard', 'comprehensive view'],
                'confidence_weight': 0.8
            }
        }

        # Entity extraction patterns
        self.entity_patterns = {
            'columns': r'(?:column|field|variable|feature)s?\s+(?:called\s+)?["\']?([^"\']+)["\']?',
            'time_periods': r'(?:last|past|previous)\s+(\d+)\s+(?:days?|weeks?|months?|years?)',
            'comparisons': r'(?:between|among|versus|vs|compared?\s+to)\s+([^,\s]+)',
            'numbers': r'\b\d+(?:\.\d+)?\b',
            'percentages': r'\b\d+(?:\.\d+)?%',
            'dates': r'\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}/\d{1,2}/\d{4}\b'
        }

    def recognize_intent(self, query: str) -> IntentRecognitionResult:
        """
        Recognize the intent from a natural language query.

        Args:
            query: The user's natural language query

        Returns:
            IntentRecognitionResult with recognized intent and entities
        """
        query = query.lower().strip()
        original_query = query

        # Initialize with unknown intent
        best_intent = QueryIntent.UNKNOWN
        best_confidence = 0.0
        entities = {}
        suggested_action = "Process query with AI assistance"

        # Check each intent pattern
        for intent, config in self.intent_patterns.items():
            confidence = self._calculate_intent_confidence(query, config)

            if confidence > best_confidence and confidence > 0.3:  # Minimum threshold
                best_intent = intent
                best_confidence = confidence

                # Extract entities for this intent
                entities = self._extract_entities(query, intent)

                # Generate suggested action
                suggested_action = self._generate_suggested_action(intent, entities)

        # If confidence is too low, try to use keyword matching
        if best_confidence < 0.5:
            best_intent, best_confidence = self._keyword_fallback(query)

        return IntentRecognitionResult(
            intent=best_intent,
            confidence=best_confidence,
            entities=entities,
            original_query=original_query,
            suggested_action=suggested_action
        )

    def _calculate_intent_confidence(self, query: str, config: Dict) -> float:
        """Calculate confidence score for an intent."""
        confidence = 0.0

        # Pattern matching
        for pattern in config['patterns']:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                confidence += config['confidence_weight'] * 0.7

        # Keyword matching
        query_words = set(query.split())
        keywords = config['keywords']
        keyword_matches = sum(1 for keyword in keywords if keyword in query)
        if keyword_matches > 0:
            confidence += (keyword_matches / len(keywords)) * config['confidence_weight'] * 0.3

        return min(confidence, 1.0)

    def _extract_entities(self, query: str, intent: QueryIntent) -> Dict[str, List[str]]:
        """Extract relevant entities from the query."""
        entities = {}

        # Extract columns/variables
        column_matches = re.findall(self.entity_patterns['columns'], query, re.IGNORECASE)
        if column_matches:
            entities['columns'] = [match.strip(' "\'') for match in column_matches]

        # Extract time periods
        time_matches = re.findall(self.entity_patterns['time_periods'], query, re.IGNORECASE)
        if time_matches:
            entities['time_periods'] = time_matches

        # Extract comparison groups
        comparison_matches = re.findall(self.entity_patterns['comparisons'], query, re.IGNORECASE)
        if comparison_matches:
            entities['comparisons'] = [match.strip() for match in comparison_matches]

        # Extract numbers
        number_matches = re.findall(self.entity_patterns['numbers'], query)
        if number_matches:
            entities['numbers'] = number_matches

        return entities

    def _generate_suggested_action(self, intent: QueryIntent, entities: Dict) -> str:
        """Generate a suggested action based on intent and entities."""
        actions = {
            QueryIntent.VISUALIZE_DATA: "Create appropriate visualization for the specified data",
            QueryIntent.ANALYZE_TRENDS: "Perform trend analysis and time series visualization",
            QueryIntent.COMPARE_GROUPS: "Create comparison charts between specified groups",
            QueryIntent.FIND_CORRELATIONS: "Generate correlation heatmap and statistical analysis",
            QueryIntent.PREDICT_VALUES: "Train prediction model and generate forecasts",
            QueryIntent.DATA_SUMMARY: "Create statistical summary and data overview",
            QueryIntent.FEATURE_IMPORTANCE: "Analyze and visualize feature importance",
            QueryIntent.CREATE_DASHBOARD: "Generate comprehensive dashboard with relevant visualizations"
        }

        return actions.get(intent, "Process query with AI assistance")

    def _keyword_fallback(self, query: str) -> Tuple[QueryIntent, float]:
        """Fallback method using keyword matching."""
        # Simple keyword-based intent detection
        keywords_map = {
            QueryIntent.VISUALIZE_DATA: ['show', 'plot', 'chart', 'graph', 'visualize'],
            QueryIntent.ANALYZE_TRENDS: ['trend', 'over time', 'pattern', 'change'],
            QueryIntent.COMPARE_GROUPS: ['compare', 'difference', 'between', 'versus'],
            QueryIntent.FIND_CORRELATIONS: ['correlation', 'relationship', 'related'],
            QueryIntent.DATA_SUMMARY: ['summary', 'statistics', 'describe', 'overview']
        }

        best_intent = QueryIntent.UNKNOWN
        best_score = 0.0

        for intent, keywords in keywords_map.items():
            score = sum(1 for keyword in keywords if keyword in query)
            if score > best_score:
                best_score = score
                best_intent = intent

        confidence = min(best_score * 0.2, 0.6)  # Lower confidence for keyword fallback
        return best_intent, confidence
