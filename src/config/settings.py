"""
Configuration settings for the Real-Time ML Application.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # Google AI Studio Configuration
    google_api_key: str
    gemini_model: str = "gemini-pro"

    # Application Configuration
    app_env: str = "development"
    debug: bool = True
    host: str = "localhost"
    port: int = 8000

    # Streamlit Configuration
    streamlit_port: int = 8501
    streamlit_address: str = "localhost"

    # Redis Configuration
    redis_url: str = "redis://localhost:6379"
    redis_password: Optional[str] = None

    # Kafka Configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_client_id: str = "real-time-ml-app"

    # Database Configuration
    database_url: str = "sqlite:///app.db"

    # Logging Configuration
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Security Configuration
    secret_key: str
    cors_origins: List[str] = ["http://localhost:3000", "http://localhost:8501"]

    # Model Configuration
    model_update_interval: int = 300  # 5 minutes
    max_model_versions: int = 5
    auto_retraining: bool = True

    # Monitoring Configuration
    prometheus_gateway: str = "http://localhost:9091"
    metrics_port: int = 9090

    @validator("cors_origins", pre=True)
    def assemble_cors_origins(cls, v):
        """Parse CORS origins from string to list."""
        if isinstance(v, str):
            return [i.strip() for i in v.split(",")]
        return v

    @validator("debug", pre=True)
    def parse_debug(cls, v):
        """Parse debug as boolean."""
        if isinstance(v, str):
            return v.lower() in ("yes", "true", "t", "1")
        return bool(v)

    @validator("auto_retraining", pre=True)
    def parse_auto_retraining(cls, v):
        """Parse auto_retraining as boolean."""
        if isinstance(v, str):
            return v.lower() in ("yes", "true", "t", "1")
        return bool(v)

    class Config:
        """Pydantic configuration."""
        env_file = ".env"
        case_sensitive = False


# Global settings instance
settings = Settings()

# Ensure logs directory exists
os.makedirs("logs", exist_ok=True)
