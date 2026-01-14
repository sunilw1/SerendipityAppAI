"""
Serendipity AI Backend - FastAPI Application
==============================================

Production-grade AI intelligence layer for family safety geolocation tracking.

Phase 1 Capabilities:
- Real-time data ingestion simulation
- GPS data cleaning and validation
- Confidence scoring for location accuracy
- Baseline behavior learning (observe-only)
- Clean intelligence APIs

Entry Points:
- Development: uvicorn app.main:app --reload
- Production: gunicorn app.main:app -k uvicorn.workers.UvicornWorker
"""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.v1.router import api_router

# Initialize logging
setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """
    Application lifespan manager.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        environment=settings.app_env,
        debug=settings.debug,
    )
    
    # Verify dataset exists
    if settings.dataset_file_path.exists():
        logger.info(
            "dataset_found",
            path=str(settings.dataset_file_path),
            size_mb=round(settings.dataset_file_path.stat().st_size / (1024 * 1024), 2),
        )
    else:
        logger.warning(
            "dataset_not_found",
            expected_path=str(settings.dataset_file_path),
        )
    
    yield
    
    # Shutdown
    logger.info("application_shutting_down")


# Create FastAPI application
app = FastAPI(
    title="Serendipity AI Backend",
    description="""
    ## Family Safety Geolocation Tracking - AI Intelligence Layer
    
    This API provides:
    - **Data Ingestion**: Process tracking data from mobile apps
    - **Data Quality**: Validate and clean GPS coordinates
    - **Confidence Scoring**: Score reliability of each location point
    - **Baseline Learning**: Learn normal user behavior patterns
    - **Location Intelligence**: Return standardized, enriched location data
    
    ### Phase 1 Focus
    - Observe raw signals
    - Clean and normalize data  
    - Learn baseline behavior
    - Produce confidence-scored intelligence
    
    ### API Versioning
    All endpoints are prefixed with `/api/v1`
    """,
    version="0.1.0",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = datetime.now(timezone.utc)
    
    response = await call_next(request)
    
    duration_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
    
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration_ms, 2),
    )
    
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions."""
    logger.error(
        "unhandled_exception",
        path=request.url.path,
        error=str(exc),
        error_type=type(exc).__name__,
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "detail": str(exc) if settings.debug else "An unexpected error occurred",
        },
    )


# Include API router
app.include_router(api_router, prefix="/api/v1")


# Root endpoint
@app.get("/", tags=["Root"])
async def root():
    """
    Root endpoint.
    
    Returns basic API information.
    """
    return {
        "name": "Serendipity AI Backend",
        "version": "0.1.0",
        "phase": 1,
        "description": "AI Intelligence Layer for Family Safety Geolocation Tracking",
        "docs": "/docs" if settings.debug else "Disabled in production",
        "api_prefix": "/api/v1",
        "health": "/api/v1/health",
    }


# Direct health endpoint (for load balancers that check root)
@app.get("/ping", tags=["Health"])
async def ping():
    """Simple ping endpoint."""
    return {"status": "pong"}


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        workers=1 if settings.debug else settings.workers,
    )
