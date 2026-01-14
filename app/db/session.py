"""
Database Session Management
============================

Database connection and session handling.

Phase 1: Placeholder for future database integration.
Phase 2+: MySQL with SQLAlchemy async.

Design Decisions:
- Use async SQLAlchemy for non-blocking I/O
- Connection pooling for production
- MySQL for reliable RDBMS storage
"""

from typing import AsyncGenerator, Optional

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Base(DeclarativeBase):
    """SQLAlchemy declarative base."""
    pass


# Async engine (will be initialized on first use)
_engine = None
_session_factory = None


def get_engine():
    """
    Get or create the async database engine.
    
    Uses connection pooling for production performance.
    """
    global _engine
    
    if _engine is None:
        _engine = create_async_engine(
            settings.database_url,
            pool_size=settings.database_pool_size,
            max_overflow=settings.database_max_overflow,
            pool_pre_ping=True,  # Verify connections before use
            echo=settings.debug,  # SQL logging in debug mode
        )
        logger.info("database_engine_created")
    
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _session_factory
    
    if _session_factory is None:
        engine = get_engine()
        _session_factory = async_sessionmaker(
            engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        logger.info("session_factory_created")
    
    return _session_factory


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting a database session.
    
    Usage:
        @app.get("/items")
        async def get_items(session: AsyncSession = Depends(get_session)):
            ...
    
    Yields:
        AsyncSession for database operations
    """
    factory = get_session_factory()
    
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def init_db() -> None:
    """
    Initialize database tables.
    
    Should be called on application startup.
    """
    engine = get_engine()
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
    
    logger.info("database_initialized")


async def close_db() -> None:
    """
    Close database connections.
    
    Should be called on application shutdown.
    """
    global _engine
    
    if _engine is not None:
        await _engine.dispose()
        _engine = None
        logger.info("database_connections_closed")


# Placeholder for Phase 2 models
# class TrackingEvent(Base):
#     """Tracking event database model."""
#     __tablename__ = "tracking_events"
#     
#     id = Column(String, primary_key=True)
#     user_id = Column(Integer, index=True)
#     trip_id = Column(Integer, index=True)
#     timestamp = Column(DateTime(timezone=True), index=True)
#     lat = Column(Float)
#     lon = Column(Float)
#     confidence = Column(Float)
#     # ... more fields
