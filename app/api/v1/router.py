"""
API v1 Router
==============

Aggregates all v1 API routes.

Phase 1: Data ingestion, confidence scoring, baselines
Phase 2: Behavioral intelligence, anomaly detection, risk scoring
Phase 2 Viz: Visualization and analytics endpoints
Phase 3: Predictive intelligence, GPU inference, retraining
"""

from fastapi import APIRouter

from app.api.v1.routes import health, ingest, intelligence, phase2, phase2_visualization, phase3, safety

# Create main v1 router
api_router = APIRouter()

# Include route modules

# Health check
api_router.include_router(
    health.router,
    tags=["Health"],
)

# Phase 1: Data Ingestion
api_router.include_router(
    ingest.router,
    prefix="/ingest",
    tags=["Data Ingestion"],
)

# Phase 1: Location Intelligence (confidence-scored data)
api_router.include_router(
    intelligence.router,
    prefix="/intelligence",
    tags=["Location Intelligence"],
)

# Phase 2: Behavioral Intelligence & Threat Detection
api_router.include_router(
    phase2.router,
    prefix="/phase2",
    tags=["Phase 2 - Behavioral Intelligence"],
)

# Phase 2: Visualization & Analytics (demo-ready, dashboard-friendly)
api_router.include_router(
    phase2_visualization.router,
    prefix="/phase2/viz",
    tags=["Phase 2 - Visualization"],
)

# Phase 3: Predictive Intelligence & GPU Inference
api_router.include_router(
    phase3.router,
    tags=["Phase 3 - Predictive Intelligence"],
)

# Phase 3: Safety Detection (Crash, Fall, Battery)
api_router.include_router(
    safety.router,
    prefix="/phase3",
    tags=["Phase 3 - Safety Detection"],
)
