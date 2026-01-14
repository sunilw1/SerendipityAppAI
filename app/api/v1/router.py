"""
API v1 Router
==============

Aggregates all v1 API routes.
"""

from fastapi import APIRouter

from app.api.v1.routes import health, ingest, intelligence

# Create main v1 router
api_router = APIRouter()

# Include route modules
api_router.include_router(
    health.router,
    tags=["Health"],
)

api_router.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["Data Ingestion"],
)

api_router.include_router(
    intelligence.router,
    prefix="/intelligence",
    tags=["Location Intelligence"],
)
