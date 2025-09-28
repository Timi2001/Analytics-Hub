"""
User Interaction Tracking Agent for Autonomous Learning
Tracks user behavior to enable reinforcement learning and system improvement.
"""
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
import uuid

from src.config.settings import settings

logger = logging.getLogger(__name__)

class InteractionType(Enum):
    """Types of user interactions to track."""
    DASHBOARD_VIEW = "dashboard_view"
    CHART_INTERACTION = "chart_interaction"
    DATA_UPLOAD = "data_upload"
    MODEL_TRAINING = "model_training"
    ANALYSIS_REQUEST = "analysis_request"
    EXPORT_DATA = "export_data"
    SETTINGS_CHANGE = "settings_change"
    PAGE_NAVIGATION = "page_navigation"
    FEATURE_USAGE = "feature_usage"

class UserSentiment(Enum):
    """User satisfaction indicators."""
    VERY_SATISFIED = "very_satisfied"
    SATISFIED = "satisfied"
    NEUTRAL = "neutral"
    DISSATISFIED = "dissatisfied"
    VERY_DISSATISFIED = "very_dissatisfied"

@dataclass
class UserInteraction:
    """Represents a single user interaction event."""
    interaction_id: str
    session_id: str
    user_id: str
    interaction_type: InteractionType
    timestamp: datetime
    duration: Optional[float]  # How long they spent on this interaction
    metadata: Dict[str, Any]  # Additional context
    sentiment: Optional[UserSentiment]  # User satisfaction indicator
    success: bool  # Whether the interaction completed successfully
    error_message: Optional[str]  # If there was an error

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            **asdict(self),
            'interaction_type': self.interaction_type.value,
            'timestamp': self.timestamp.isoformat(),
            'sentiment': self.sentiment.value if self.sentiment else None
        }

@dataclass
class SessionMetrics:
    """Aggregated metrics for a user session."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime]
    total_interactions: int
    successful_interactions: int
    total_time_spent: float
    average_sentiment: Optional[UserSentiment]
    features_used: List[str]
    errors_encountered: int

class UserInteractionTracker:
    """Tracks user interactions for autonomous learning."""

    def __init__(self):
        self.active_sessions: Dict[str, SessionMetrics] = {}
        self.interaction_history: List[UserInteraction] = []
        self.max_history_size = 10000  # Keep last 10k interactions in memory
        self.session_timeout = 3600  # 1 hour session timeout

        # Create data directory for storing interaction logs
        import os
        os.makedirs("data/interactions", exist_ok=True)

    def start_session(self, user_id: str) -> str:
        """Start tracking a new user session."""
        session_id = str(uuid.uuid4())

        session_metrics = SessionMetrics(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            end_time=None,
            total_interactions=0,
            successful_interactions=0,
            total_time_spent=0.0,
            average_sentiment=None,
            features_used=[],
            errors_encountered=0
        )

        self.active_sessions[session_id] = session_metrics
        logger.info(f"🧠 Started tracking session {session_id} for user {user_id}")

        return session_id

    def end_session(self, session_id: str) -> Optional[SessionMetrics]:
        """End a user session and return final metrics."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        session.end_time = datetime.now()

        # Calculate final metrics
        session.total_time_spent = (
            session.end_time - session.start_time
        ).total_seconds()

        # Calculate average sentiment from interactions
        sentiments = [
            interaction.sentiment for interaction in self.interaction_history
            if (interaction.session_id == session_id and
                interaction.sentiment is not None)
        ]

        if sentiments:
            sentiment_counts = {}
            for sentiment in sentiments:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

            # Use the most common sentiment
            session.average_sentiment = max(
                sentiment_counts.keys(),
                key=lambda x: sentiment_counts[x]
            )

        # Save session data
        self._save_session_data(session)

        # Remove from active sessions
        del self.active_sessions[session_id]

        logger.info(f"🧠 Ended session {session_id}. Total interactions: {session.total_interactions}")
        return session

    def track_interaction(
        self,
        session_id: str,
        interaction_type: InteractionType,
        metadata: Dict[str, Any] = None,
        duration: float = None,
        success: bool = True,
        error_message: str = None,
        sentiment: UserSentiment = None
    ) -> str:
        """Track a user interaction event."""

        if metadata is None:
            metadata = {}

        # Generate unique interaction ID
        interaction_id = str(uuid.uuid4())

        # Get user ID from session
        user_id = "anonymous"
        if session_id in self.active_sessions:
            user_id = self.active_sessions[session_id].user_id

        # Create interaction record
        interaction = UserInteraction(
            interaction_id=interaction_id,
            session_id=session_id,
            user_id=user_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            duration=duration,
            metadata=metadata,
            sentiment=sentiment,
            success=success,
            error_message=error_message
        )

        # Add to history
        self.interaction_history.append(interaction)

        # Update session metrics
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.total_interactions += 1

            if success:
                session.successful_interactions += 1

            if duration:
                session.total_time_spent += duration

            # Track features used
            feature_name = interaction_type.value
            if feature_name not in session.features_used:
                session.features_used.append(feature_name)

            if not success:
                session.errors_encountered += 1

        # Maintain history size limit
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[-self.max_history_size:]

        # Save to persistent storage periodically
        if len(self.interaction_history) % 100 == 0:
            self._save_interaction_data()

        logger.debug(f"📊 Tracked interaction: {interaction_type.value}")
        return interaction_id

    def get_user_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze user behavior patterns for autonomous learning."""

        # Filter interactions for this user in the last N days
        cutoff_date = datetime.now().timestamp() - (days * 24 * 3600)

        user_interactions = [
            interaction for interaction in self.interaction_history
            if (interaction.user_id == user_id and
                interaction.timestamp.timestamp() >= cutoff_date)
        ]

        if not user_interactions:
            return {}

        # Analyze patterns
        patterns = {
            'total_interactions': len(user_interactions),
            'interaction_types': {},
            'time_patterns': {},
            'success_rate': 0.0,
            'average_session_duration': 0.0,
            'preferred_features': [],
            'sentiment_trend': []
        }

        # Count interaction types
        for interaction in user_interactions:
            interaction_type = interaction.interaction_type.value
            patterns['interaction_types'][interaction_type] = (
                patterns['interaction_types'].get(interaction_type, 0) + 1
            )

            if interaction.sentiment:
                patterns['sentiment_trend'].append(interaction.sentiment.value)

        # Calculate success rate
        successful = sum(1 for i in user_interactions if i.success)
        patterns['success_rate'] = successful / len(user_interactions)

        # Find preferred features (most used)
        sorted_features = sorted(
            patterns['interaction_types'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        patterns['preferred_features'] = [feature[0] for feature in sorted_features[:5]]

        return patterns

    def get_system_learning_data(self) -> Dict[str, Any]:
        """Generate learning data for the autonomous system."""

        # Aggregate data for RL training
        learning_data = {
            'total_users': len(set(i.user_id for i in self.interaction_history)),
            'total_sessions': len(self.active_sessions),
            'total_interactions': len(self.interaction_history),
            'success_rate': 0.0,
            'feature_usage': {},
            'sentiment_distribution': {},
            'error_patterns': {}
        }

        if not self.interaction_history:
            return learning_data

        # Calculate overall success rate
        successful = sum(1 for i in self.interaction_history if i.success)
        learning_data['success_rate'] = successful / len(self.interaction_history)

        # Analyze feature usage
        for interaction in self.interaction_history:
            feature = interaction.interaction_type.value
            learning_data['feature_usage'][feature] = (
                learning_data['feature_usage'].get(feature, 0) + 1
            )

            # Track sentiment
            if interaction.sentiment:
                sentiment = interaction.sentiment.value
                learning_data['sentiment_distribution'][sentiment] = (
                    learning_data['sentiment_distribution'].get(sentiment, 0) + 1
                )

            # Track errors
            if not interaction.success and interaction.error_message:
                error_key = interaction.error_message[:50]  # First 50 chars
                learning_data['error_patterns'][error_key] = (
                    learning_data['error_patterns'].get(error_key, 0) + 1
                )

        return learning_data

    def _save_interaction_data(self):
        """Save interaction history to persistent storage."""
        try:
            filename = f"data/interactions/interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # Convert recent interactions to serializable format
            recent_interactions = self.interaction_history[-1000:]  # Last 1000 interactions
            serializable_data = [interaction.to_dict() for interaction in recent_interactions]

            with open(filename, 'w') as f:
                json.dump(serializable_data, f, indent=2)

            logger.debug(f"💾 Saved {len(serializable_data)} interactions to {filename}")

        except Exception as e:
            logger.error(f"❌ Error saving interaction data: {e}")

    def _save_session_data(self, session: SessionMetrics):
        """Save session metrics to persistent storage."""
        try:
            filename = f"data/interactions/session_{session.session_id}.json"

            with open(filename, 'w') as f:
                json.dump(asdict(session), f, indent=2, default=str)

            logger.debug(f"💾 Saved session metrics to {filename}")

        except Exception as e:
            logger.error(f"❌ Error saving session data: {e}")

    def cleanup_old_sessions(self):
        """Clean up expired sessions."""
        current_time = time.time()
        expired_sessions = []

        for session_id, session in self.active_sessions.items():
            session_age = current_time - session.start_time.timestamp()
            if session_age > self.session_timeout:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            logger.info(f"🧹 Cleaning up expired session: {session_id}")
            self.end_session(session_id)

    def get_dashboard_engagement_metrics(self) -> Dict[str, Any]:
        """Get metrics specifically for dashboard optimization learning."""

        dashboard_interactions = [
            i for i in self.interaction_history
            if i.interaction_type in [
                InteractionType.DASHBOARD_VIEW,
                InteractionType.CHART_INTERACTION,
                InteractionType.PAGE_NAVIGATION
            ]
        ]

        if not dashboard_interactions:
            return {}

        metrics = {
            'total_dashboard_interactions': len(dashboard_interactions),
            'average_session_duration': 0.0,
            'most_used_features': [],
            'user_satisfaction_trend': [],
            'navigation_patterns': {},
            'feature_success_rates': {}
        }

        # Calculate average session duration for dashboard interactions
        durations = [i.duration for i in dashboard_interactions if i.duration]
        if durations:
            metrics['average_session_duration'] = sum(durations) / len(durations)

        # Analyze feature usage
        feature_counts = {}
        feature_success = {}

        for interaction in dashboard_interactions:
            feature = interaction.interaction_type.value

            feature_counts[feature] = feature_counts.get(feature, 0) + 1

            if feature not in feature_success:
                feature_success[feature] = {'total': 0, 'successful': 0}

            feature_success[feature]['total'] += 1
            if interaction.success:
                feature_success[feature]['successful'] += 1

        # Most used features
        sorted_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)
        metrics['most_used_features'] = [feature[0] for feature in sorted_features[:10]]

        # Success rates by feature
        for feature, counts in feature_success.items():
            success_rate = counts['successful'] / counts['total']
            metrics['feature_success_rates'][feature] = success_rate

        # User satisfaction trend
        satisfaction_scores = []
        for interaction in dashboard_interactions:
            if interaction.sentiment:
                # Convert sentiment to numeric score
                sentiment_scores = {
                    UserSentiment.VERY_DISSATISFIED.value: 1,
                    UserSentiment.DISSATISFIED.value: 2,
                    UserSentiment.NEUTRAL.value: 3,
                    UserSentiment.SATISFIED.value: 4,
                    UserSentiment.VERY_SATISFIED.value: 5
                }
                satisfaction_scores.append(sentiment_scores.get(interaction.sentiment.value, 3))

        if satisfaction_scores:
            metrics['user_satisfaction_trend'] = satisfaction_scores

        return metrics

# Global user tracker instance
user_tracker = UserInteractionTracker()
