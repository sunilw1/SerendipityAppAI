"""
Health Check Routes
====================

Provides health check endpoints for:
- Liveness probes
- Readiness probes
- Service status

Used by Kubernetes, load balancers, and monitoring systems.
"""

from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends

from app.core.config import settings
from app.models.schemas import HealthResponse
from app.api.deps import get_intelligence_svc
from app.services.intelligence_service import IntelligenceService

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check service health and dependencies",
)
async def health_check(
    service: IntelligenceService = Depends(get_intelligence_svc),
) -> HealthResponse:
    """
    Health check endpoint.
    
    Returns:
        - Service status
        - API version
        - Database connectivity
        - Dataset availability
        - Basic statistics
    """
    # Check dataset
    dataset_path = settings.dataset_file_path
    dataset_available = dataset_path.exists() and dataset_path.is_file()
    
    # Get stats if available
    stats = {}
    if dataset_available:
        try:
            file_stats = service.get_dataset_stats()
            stats = {
                "total_rows": file_stats.get("total_rows", 0),
                "unique_users": file_stats.get("unique_users", 0),
                "unique_trips": file_stats.get("unique_trips", 0),
                "file_size_mb": file_stats.get("file_size_mb", 0),
            }
        except Exception:
            pass
    
    # Database status (placeholder for Phase 2)
    database_connected = True  # Will be actual check in Phase 2
    
    # Determine overall status
    if dataset_available:
        status = "healthy"
    else:
        status = "degraded"
    
    return HealthResponse(
        status=status,
        version=settings.api_version,
        timestamp=datetime.now(timezone.utc),
        database_connected=database_connected,
        dataset_available=dataset_available,
        stats=stats,
    )


@router.get(
    "/health/live",
    summary="Liveness probe",
    description="Simple liveness check for Kubernetes",
)
async def liveness() -> dict:
    """
    Liveness probe.
    
    Returns 200 if the service is running.
    Used by Kubernetes to detect crashed containers.
    """
    return {"status": "alive"}


@router.get(
    "/health/ready",
    summary="Readiness probe",
    description="Readiness check for Kubernetes",
)
async def readiness(
    service: IntelligenceService = Depends(get_intelligence_svc),
) -> dict:
    """
    Readiness probe.
    
    Returns 200 if the service is ready to accept traffic.
    Used by Kubernetes to control traffic routing.
    """
    dataset_available = settings.dataset_file_path.exists()
    
    if not dataset_available:
        return {"status": "not_ready", "reason": "dataset_unavailable"}
    
    return {"status": "ready"}
