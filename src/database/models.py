"""
Database models for autonomous learning system.
Defines tables for user interactions, RL learning data, and system metrics.
"""
import uuid
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Boolean, DateTime, Text, JSON, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
from src.database.connection import Base

class UserInteraction(Base):
    """Model for tracking user interactions for RL learning."""
    __tablename__ = "user_interactions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    interaction_type = Column(String, nullable=False)  # dashboard_view, chart_interaction, etc.
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    duration = Column(Float)  # Time spent on interaction in seconds
    metadata = Column(JSON)  # Additional context data
    sentiment = Column(String)  # very_satisfied, satisfied, neutral, dissatisfied, very_dissatisfied
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    # Relationships
    session = relationship("UserSession", back_populates="interactions")

    def __repr__(self):
        return f"<UserInteraction(id={self.id}, type={self.interaction_type}, user={self.user_id})>"

class UserSession(Base):
    """Model for tracking user sessions."""
    __tablename__ = "user_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    start_time = Column(DateTime, default=datetime.utcnow, nullable=False)
    end_time = Column(DateTime)
    total_interactions = Column(Integer, default=0)
    successful_interactions = Column(Integer, default=0)
    total_time_spent = Column(Float, default=0.0)
    average_sentiment = Column(String)
    features_used = Column(JSON)  # List of features used in session
    errors_encountered = Column(Integer, default=0)

    # Relationships
    interactions = relationship("UserInteraction", back_populates="session")

    def __repr__(self):
        return f"<UserSession(id={self.id}, user={self.user_id}, interactions={self.total_interactions})>"

class RLLearningData(Base):
    """Model for storing reinforcement learning training data."""
    __tablename__ = "rl_learning_data"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_type = Column(String, nullable=False)  # dashboard_designer, analysis_strategist, etc.
    state = Column(JSON, nullable=False)  # Environment state before action
    action = Column(JSON, nullable=False)  # Action taken by agent
    reward = Column(Float, nullable=False)  # Reward received
    next_state = Column(JSON)  # Environment state after action
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    episode_id = Column(String, index=True)  # Group related learning steps
    learning_iteration = Column(Integer, default=1)

    def __repr__(self):
        return f"<RLLearningData(id={self.id}, agent={self.agent_type}, reward={self.reward})>"

class SystemMetrics(Base):
    """Model for tracking system performance metrics."""
    __tablename__ = "system_metrics"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    metric_name = Column(String, nullable=False)
    metric_value = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)  # Additional context
    category = Column(String, default="general")  # performance, learning, user_experience, etc.

    def __repr__(self):
        return f"<SystemMetrics(id={self.id}, name={self.metric_name}, value={self.metric_value})>"

class ModelPerformance(Base):
    """Model for tracking ML model performance over time."""
    __tablename__ = "model_performance"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_name = Column(String, nullable=False)
    model_version = Column(String, nullable=False)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    mse = Column(Float)  # Mean squared error for regression
    rmse = Column(Float)  # Root mean squared error
    training_time = Column(Float)  # Training duration in seconds
    dataset_size = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)  # Additional model information

    def __repr__(self):
        return f"<ModelPerformance(id={self.id}, model={self.model_name}, accuracy={self.accuracy})>"

class DataIngestionLog(Base):
    """Model for tracking data ingestion activities."""
    __tablename__ = "data_ingestion_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_type = Column(String, nullable=False)  # file, api, stream, etc.
    source_name = Column(String, nullable=False)
    records_ingested = Column(Integer, default=0)
    records_failed = Column(Integer, default=0)
    processing_time = Column(Float)  # Processing duration in seconds
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)  # File info, data quality metrics, etc.
    success = Column(Boolean, default=True)
    error_message = Column(Text)

    def __repr__(self):
        return f"<DataIngestionLog(id={self.id}, source={self.source_name}, records={self.records_ingested})>"

class AutonomousInsights(Base):
    """Model for storing AI-generated insights and recommendations."""
    __tablename__ = "autonomous_insights"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    insight_type = Column(String, nullable=False)  # dashboard_improvement, analysis_suggestion, etc.
    agent_type = Column(String, nullable=False)  # Which agent generated the insight
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    confidence_score = Column(Float)  # AI confidence in the insight (0-1)
    supporting_data = Column(JSON)  # Data that supports the insight
    implementation_suggestion = Column(Text)  # How to implement the insight
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    user_id = Column(String)  # User who triggered the insight (if applicable)
    status = Column(String, default="active")  # active, implemented, dismissed

    def __repr__(self):
        return f"<AutonomousInsights(id={self.id}, type={self.insight_type}, confidence={self.confidence_score})>"

class LearningProgress(Base):
    """Model for tracking autonomous learning progress over time."""
    __tablename__ = "learning_progress"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    agent_type = Column(String, nullable=False)
    learning_iteration = Column(Integer, nullable=False)
    performance_score = Column(Float, nullable=False)
    improvement_rate = Column(Float)  # Rate of improvement over time
    exploration_rate = Column(Float)  # Current exploration vs exploitation ratio
    total_actions = Column(Integer, default=0)
    successful_actions = Column(Integer, default=0)
    average_reward = Column(Float, default=0.0)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)  # Additional learning metrics

    def __repr__(self):
        return f"<LearningProgress(id={self.id}, agent={self.agent_type}, score={self.performance_score})>"

class UserFeedback(Base):
    """Model for storing explicit user feedback on system performance."""
    __tablename__ = "user_feedback"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(String, nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    feedback_type = Column(String, nullable=False)  # rating, comment, bug_report, feature_request
    rating = Column(Integer)  # 1-5 star rating (if applicable)
    comment = Column(Text)
    category = Column(String)  # usability, performance, features, bugs
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    metadata = Column(JSON)  # Additional context

    def __repr__(self):
        return f"<UserFeedback(id={self.id}, type={self.feedback_type}, rating={self.rating})>"

# Create indexes for better query performance
from sqlalchemy import Index

# User interaction indexes
Index('ix_user_interactions_session_timestamp', UserInteraction.session_id, UserInteraction.timestamp)
Index('ix_user_interactions_user_timestamp', UserInteraction.user_id, UserInteraction.timestamp)
Index('ix_user_interactions_type_timestamp', UserInteraction.interaction_type, UserInteraction.timestamp)

# Session indexes
Index('ix_user_sessions_user_start', UserSession.user_id, UserSession.start_time)

# RL learning indexes
Index('ix_rl_learning_agent_timestamp', RLLearningData.agent_type, RLLearningData.timestamp)
Index('ix_rl_learning_episode', RLLearningData.episode_id, RLLearningData.learning_iteration)

# Metrics indexes
Index('ix_system_metrics_name_timestamp', SystemMetrics.metric_name, SystemMetrics.timestamp)
Index('ix_system_metrics_category_timestamp', SystemMetrics.category, SystemMetrics.timestamp)

# Model performance indexes
Index('ix_model_performance_name_timestamp', ModelPerformance.model_name, ModelPerformance.timestamp)
Index('ix_model_performance_version', ModelPerformance.model_version)

# Learning progress indexes
Index('ix_learning_progress_agent_iteration', LearningProgress.agent_type, LearningProgress.learning_iteration)
Index('ix_learning_progress_timestamp', LearningProgress.timestamp)

# Feedback indexes
Index('ix_user_feedback_user_timestamp', UserFeedback.user_id, UserFeedback.timestamp)
Index('ix_user_feedback_type_timestamp', UserFeedback.feedback_type, UserFeedback.timestamp)
