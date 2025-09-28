"""
Database connection and session management for Railway PostgreSQL.
Handles persistent storage for autonomous learning data.
"""
import logging
from typing import Generator
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
import os

logger = logging.getLogger(__name__)

# Database configuration
DATABASE_URL = os.getenv(
    'DATABASE_URL',
    'postgresql://postgres:gHEaSsDtVTtFCWaaaYthCuokXbEEFLjU@crossover.proxy.rlwy.net:24171/railway'
)

# Create engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=False  # Set to True for SQL debugging
)

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create base class for models
Base = declarative_base()

def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    Yields database session and ensures proper cleanup.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def get_db_session():
    """Context manager for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def test_database_connection() -> bool:
    """Test database connection and return status."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            row = result.fetchone()
            if row and row[0] == 1:
                logger.info("✅ Database connection successful")
                return True
            else:
                logger.error("❌ Database connection test failed")
                return False
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        return False

def get_database_info() -> dict:
    """Get database connection information."""
    try:
        with engine.connect() as conn:
            # Get database version
            result = conn.execute(text("SELECT version()"))
            version = result.fetchone()[0]

            # Get database name
            result = conn.execute(text("SELECT current_database()"))
            db_name = result.fetchone()[0]

            # Get user name
            result = conn.execute(text("SELECT current_user"))
            user = result.fetchone()[0]

            return {
                'connected': True,
                'database_name': db_name,
                'user': user,
                'version': version,
                'engine': 'PostgreSQL'
            }
    except Exception as e:
        logger.error(f"Error getting database info: {e}")
        return {
            'connected': False,
            'error': str(e)
        }

def create_tables():
    """Create all database tables."""
    try:
        logger.info("🗄️ Creating database tables...")
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
    except Exception as e:
        logger.error(f"❌ Error creating database tables: {e}")
        raise

def drop_tables():
    """Drop all database tables (use with caution)."""
    try:
        logger.warning("⚠️ Dropping all database tables...")
        Base.metadata.drop_all(bind=engine)
        logger.info("✅ Database tables dropped successfully")
    except Exception as e:
        logger.error(f"❌ Error dropping database tables: {e}")
        raise

# Database health check
def check_database_health() -> dict:
    """Comprehensive database health check."""
    health_info = {
        'connection_status': 'unknown',
        'tables_exist': False,
        'can_write': False,
        'performance_ms': 0,
        'error': None
    }

    try:
        import time

        # Test connection
        start_time = time.time()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        health_info['performance_ms'] = (time.time() - start_time) * 1000

        if test_database_connection():
            health_info['connection_status'] = 'healthy'

            # Check if tables exist
            try:
                with get_db_session() as db:
                    # Try to query one of our tables
                    from src.database.models import UserInteraction
                    db.query(UserInteraction).first()
                    health_info['tables_exist'] = True
            except:
                health_info['tables_exist'] = False

            # Test write capability
            try:
                with get_db_session() as db:
                    # Try a simple insert
                    test_interaction = UserInteraction(
                        session_id="test",
                        user_id="test",
                        interaction_type="test",
                        success=True
                    )
                    db.add(test_interaction)
                    db.commit()
                    health_info['can_write'] = True

                    # Clean up test data
                    db.delete(test_interaction)
                    db.commit()
            except Exception as e:
                health_info['can_write'] = False
                health_info['error'] = str(e)

        else:
            health_info['connection_status'] = 'unhealthy'

    except Exception as e:
        health_info['connection_status'] = 'error'
        health_info['error'] = str(e)

    return health_info
