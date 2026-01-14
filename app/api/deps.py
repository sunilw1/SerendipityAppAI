"""
API Dependencies
=================

Dependency injection for FastAPI routes.

Provides:
- Service instances
- Database sessions (Phase 2+)
- Authentication (Phase 2+)
"""

from typing import Generator, Optional

from fastapi import Depends, HTTPException, Header, status

from app.core.config import settings
from app.services.intelligence_service import (
    IntelligenceService,
    get_intelligence_service,
)
from app.services.baseline_service import (
    BaselineService,
    get_baseline_service,
)


def get_intelligence_svc() -> IntelligenceService:
    """
    Dependency for intelligence service.
    
    Returns:
        IntelligenceService instance
    """
    return get_intelligence_service()


def get_baseline_svc() -> BaselineService:
    """
    Dependency for baseline service.
    
    Returns:
        BaselineService instance
    """
    return get_baseline_service()


def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias="X-API-Key")
) -> bool:
    """
    Verify API key for protected endpoints.
    
    In development mode, API key is optional.
    In production, API key is required.
    
    Args:
        x_api_key: API key from header
        
    Returns:
        True if valid
        
    Raises:
        HTTPException: If invalid in production
    """
    # Development mode: skip verification
    if settings.is_development:
        return True
    
    # Production mode: require valid key
    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    if x_api_key != settings.api_secret_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )
    
    return True


# Placeholder for database session (Phase 2+)
# def get_db() -> Generator:
#     """Get database session."""
#     from app.db.session import SessionLocal
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()
