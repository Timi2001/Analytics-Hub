"""
Database service for autonomous learning system.
Handles persistence of user interactions, RL learning data, and system metrics.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, desc

from src.database.connection import get_db_session
from src.database.models import (
    UserInteraction, UserSession, RLLearningData, SystemMetrics,
    ModelPerformance, DataIngestionLog, AutonomousInsights,
    LearningProgress, UserFeedback
)
from src.agents.user_tracker import UserInteraction as TrackerInteraction, SessionMetrics as TrackerSession

logger = logging.getLogger(__name__)

class DatabaseService:
    """Service for database operations in autonomous learning system."""

    def __init__(self):
        self.db_session = None

    def save_user_interaction(self, interaction: TrackerInteraction) -> bool:
        """Save user interaction to database."""
        try:
            with get_db_session() as db:
                # Convert tracker interaction to database model
                db_interaction = UserInteraction(
                    session_id=interaction.session_id,
                    user_id=interaction.user_id,
                    interaction_type=interaction.interaction_type.value,
                    timestamp=interaction.timestamp,
                    duration=interaction.duration,
                    metadata=interaction.metadata,
                    sentiment=interaction.sentiment.value if interaction.sentiment else None,
                    success=interaction.success,
                    error_message=interaction.error_message
                )

                db.add(db_interaction)
                db.commit()
                db.refresh(db_interaction)

                logger.debug(f"💾 Saved interaction {db_interaction.id} to database")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving user interaction: {e}")
            return False

    def save_user_session(self, session: TrackerSession) -> bool:
        """Save user session to database."""
        try:
            with get_db_session() as db:
                # Check if session already exists
                existing_session = db.query(UserSession).filter(
                    UserSession.session_id == session.session_id
                ).first()

                if existing_session:
                    # Update existing session
                    existing_session.end_time = session.end_time
                    existing_session.total_interactions = session.total_interactions
                    existing_session.successful_interactions = session.successful_interactions
                    existing_session.total_time_spent = session.total_time_spent
                    existing_session.average_sentiment = session.average_sentiment.value if session.average_sentiment else None
                    existing_session.features_used = session.features_used
                    existing_session.errors_encountered = session.errors_encountered
                else:
                    # Create new session
                    db_session = UserSession(
                        session_id=session.session_id,
                        user_id=session.user_id,
                        start_time=session.start_time,
                        end_time=session.end_time,
                        total_interactions=session.total_interactions,
                        successful_interactions=session.successful_interactions,
                        total_time_spent=session.total_time_spent,
                        average_sentiment=session.average_sentiment.value if session.average_sentiment else None,
                        features_used=session.features_used,
                        errors_encountered=session.errors_encountered
                    )
                    db.add(db_session)

                db.commit()
                logger.debug(f"💾 Saved session {session.session_id} to database")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving user session: {e}")
            return False

    def save_rl_learning_data(self, agent_type: str, state: Dict, action: Dict,
                            reward: float, next_state: Dict = None,
                            episode_id: str = None) -> bool:
        """Save reinforcement learning training data."""
        try:
            with get_db_session() as db:
                rl_data = RLLearningData(
                    agent_type=agent_type,
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    episode_id=episode_id,
                    learning_iteration=1  # Would be incremented in practice
                )

                db.add(rl_data)
                db.commit()

                logger.debug(f"💾 Saved RL learning data for {agent_type}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving RL learning data: {e}")
            return False

    def save_system_metric(self, metric_name: str, metric_value: float,
                          category: str = "general", metadata: Dict = None) -> bool:
        """Save system performance metric."""
        try:
            with get_db_session() as db:
                metric = SystemMetrics(
                    metric_name=metric_name,
                    metric_value=metric_value,
                    category=category,
                    metadata=metadata or {}
                )

                db.add(metric)
                db.commit()

                logger.debug(f"💾 Saved system metric: {metric_name} = {metric_value}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving system metric: {e}")
            return False

    def save_model_performance(self, model_name: str, model_version: str,
                             accuracy: float = None, precision: float = None,
                             recall: float = None, f1_score: float = None,
                             mse: float = None, rmse: float = None,
                             training_time: float = None, dataset_size: int = None,
                             metadata: Dict = None) -> bool:
        """Save ML model performance metrics."""
        try:
            with get_db_session() as db:
                performance = ModelPerformance(
                    model_name=model_name,
                    model_version=model_version,
                    accuracy=accuracy,
                    precision=precision,
                    recall=recall,
                    f1_score=f1_score,
                    mse=mse,
                    rmse=rmse,
                    training_time=training_time,
                    dataset_size=dataset_size,
                    metadata=metadata or {}
                )

                db.add(performance)
                db.commit()

                logger.debug(f"💾 Saved model performance for {model_name}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving model performance: {e}")
            return False

    def get_user_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Get user behavior patterns from database."""
        try:
            with get_db_session() as db:
                # Calculate cutoff date
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                # Get user interactions
                interactions = db.query(UserInteraction).filter(
                    and_(
                        UserInteraction.user_id == user_id,
                        UserInteraction.timestamp >= cutoff_date
                    )
                ).all()

                if not interactions:
                    return {}

                # Analyze patterns
                patterns = {
                    'total_interactions': len(interactions),
                    'interaction_types': {},
                    'success_rate': 0.0,
                    'average_session_duration': 0.0,
                    'sentiment_trend': []
                }

                # Count interaction types and sentiments
                for interaction in interactions:
                    # Interaction types
                    patterns['interaction_types'][interaction.interaction_type] = (
                        patterns['interaction_types'].get(interaction.interaction_type, 0) + 1
                    )

                    # Sentiments
                    if interaction.sentiment:
                        patterns['sentiment_trend'].append(interaction.sentiment)

                # Calculate success rate
                successful = sum(1 for i in interactions if i.success)
                patterns['success_rate'] = successful / len(interactions)

                return patterns

        except Exception as e:
            logger.error(f"❌ Error getting user patterns: {e}")
            return {}

    def get_system_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive learning analytics for autonomous system."""
        try:
            with get_db_session() as db:
                analytics = {
                    'total_users': 0,
                    'total_sessions': 0,
                    'total_interactions': 0,
                    'success_rate': 0.0,
                    'feature_usage': {},
                    'sentiment_distribution': {},
                    'learning_progress': {},
                    'recent_activity': []
                }

                # Get basic counts
                analytics['total_users'] = db.query(UserSession.user_id).distinct().count()
                analytics['total_sessions'] = db.query(UserSession).count()
                analytics['total_interactions'] = db.query(UserInteraction).count()

                # Get success rate
                total_interactions = db.query(UserInteraction).count()
                if total_interactions > 0:
                    successful_interactions = db.query(UserInteraction).filter(
                        UserInteraction.success == True
                    ).count()
                    analytics['success_rate'] = successful_interactions / total_interactions

                # Get feature usage
                interactions = db.query(UserInteraction.interaction_type).all()
                for (interaction_type,) in interactions:
                    analytics['feature_usage'][interaction_type] = (
                        analytics['feature_usage'].get(interaction_type, 0) + 1
                    )

                # Get sentiment distribution
                sentiments = db.query(UserInteraction.sentiment).filter(
                    UserInteraction.sentiment.isnot(None)
                ).all()
                for (sentiment,) in sentiments:
                    analytics['sentiment_distribution'][sentiment] = (
                        analytics['sentiment_distribution'].get(sentiment, 0) + 1
                    )

                # Get recent activity (last 24 hours)
                recent_cutoff = datetime.utcnow() - timedelta(hours=24)
                recent_interactions = db.query(UserInteraction).filter(
                    UserInteraction.timestamp >= recent_cutoff
                ).limit(10).all()

                analytics['recent_activity'] = [
                    {
                        'user_id': interaction.user_id,
                        'interaction_type': interaction.interaction_type,
                        'timestamp': interaction.timestamp.isoformat(),
                        'success': interaction.success
                    }
                    for interaction in recent_interactions
                ]

                return analytics

        except Exception as e:
            logger.error(f"❌ Error getting system learning analytics: {e}")
            return {}

    def get_learning_progress_trends(self, agent_type: str = None,
                                   days: int = 30) -> Dict[str, Any]:
        """Get learning progress trends over time."""
        try:
            with get_db_session() as db:
                cutoff_date = datetime.utcnow() - timedelta(days=days)

                # Build query
                query = db.query(LearningProgress).filter(
                    LearningProgress.timestamp >= cutoff_date
                )

                if agent_type:
                    query = query.filter(LearningProgress.agent_type == agent_type)

                progress_records = query.order_by(LearningProgress.timestamp).all()

                if not progress_records:
                    return {}

                # Analyze trends
                trends = {
                    'agent_types': list(set(p.agent_type for p in progress_records)),
                    'performance_trend': [],
                    'improvement_rates': [],
                    'exploration_rates': [],
                    'timestamps': []
                }

                for record in progress_records:
                    trends['performance_trend'].append(record.performance_score)
                    trends['improvement_rates'].append(record.improvement_rate or 0.0)
                    trends['exploration_rates'].append(record.exploration_rate or 0.0)
                    trends['timestamps'].append(record.timestamp.isoformat())

                return trends

        except Exception as e:
            logger.error(f"❌ Error getting learning progress trends: {e}")
            return {}

    def save_autonomous_insight(self, insight_type: str, agent_type: str,
                              title: str, description: str, confidence_score: float,
                              supporting_data: Dict = None,
                              implementation_suggestion: str = None,
                              user_id: str = None) -> bool:
        """Save AI-generated insight to database."""
        try:
            with get_db_session() as db:
                insight = AutonomousInsights(
                    insight_type=insight_type,
                    agent_type=agent_type,
                    title=title,
                    description=description,
                    confidence_score=confidence_score,
                    supporting_data=supporting_data or {},
                    implementation_suggestion=implementation_suggestion,
                    user_id=user_id
                )

                db.add(insight)
                db.commit()

                logger.debug(f"💾 Saved autonomous insight: {title}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving autonomous insight: {e}")
            return False

    def get_recent_insights(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent autonomous insights."""
        try:
            with get_db_session() as db:
                insights = db.query(AutonomousInsights).filter(
                    AutonomousInsights.status == "active"
                ).order_by(desc(AutonomousInsights.timestamp)).limit(limit).all()

                return [
                    {
                        'id': str(insight.id),
                        'insight_type': insight.insight_type,
                        'agent_type': insight.agent_type,
                        'title': insight.title,
                        'description': insight.description,
                        'confidence_score': insight.confidence_score,
                        'timestamp': insight.timestamp.isoformat(),
                        'status': insight.status
                    }
                    for insight in insights
                ]

        except Exception as e:
            logger.error(f"❌ Error getting recent insights: {e}")
            return []

    def save_data_ingestion_log(self, source_type: str, source_name: str,
                              records_ingested: int = 0, records_failed: int = 0,
                              processing_time: float = 0.0, success: bool = True,
                              error_message: str = None, metadata: Dict = None) -> bool:
        """Save data ingestion log."""
        try:
            with get_db_session() as db:
                log_entry = DataIngestionLog(
                    source_type=source_type,
                    source_name=source_name,
                    records_ingested=records_ingested,
                    records_failed=records_failed,
                    processing_time=processing_time,
                    success=success,
                    error_message=error_message,
                    metadata=metadata or {}
                )

                db.add(log_entry)
                db.commit()

                logger.debug(f"💾 Saved data ingestion log: {source_name}")
                return True

        except Exception as e:
            logger.error(f"❌ Error saving data ingestion log: {e}")
            return False

    def get_database_stats(self) -> Dict[str, Any]:
        """Get comprehensive database statistics."""
        try:
            with get_db_session() as db:
                stats = {}

                # Table counts
                stats['total_interactions'] = db.query(UserInteraction).count()
                stats['total_sessions'] = db.query(UserSession).count()
                stats['total_rl_data_points'] = db.query(RLLearningData).count()
                stats['total_system_metrics'] = db.query(SystemMetrics).count()
                stats['total_model_performances'] = db.query(ModelPerformance).count()
                stats['total_insights'] = db.query(AutonomousInsights).count()
                stats['total_learning_records'] = db.query(LearningProgress).count()

                # Recent activity (last 7 days)
                week_ago = datetime.utcnow() - timedelta(days=7)
                stats['interactions_last_week'] = db.query(UserInteraction).filter(
                    UserInteraction.timestamp >= week_ago
                ).count()

                # User engagement metrics
                if stats['total_sessions'] > 0:
                    avg_interactions_per_session = stats['total_interactions'] / stats['total_sessions']
                    stats['avg_interactions_per_session'] = round(avg_interactions_per_session, 2)

                # Success rates
                if stats['total_interactions'] > 0:
                    successful_interactions = db.query(UserInteraction).filter(
                        UserInteraction.success == True
                    ).count()
                    stats['overall_success_rate'] = successful_interactions / stats['total_interactions']

                return stats

        except Exception as e:
            logger.error(f"❌ Error getting database stats: {e}")
            return {}

# Global database service instance
db_service = DatabaseService()
