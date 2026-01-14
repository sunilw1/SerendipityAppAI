"""Data models and Pydantic schemas for the Serendipity AI Backend."""

from app.models.schemas import (
    # Base models
    LocationPoint,
    RawTrackingEvent,
    NormalizedTrackingEvent,
    ProcessedTrackingEvent,
    
    # Trip models
    Trip,
    TripSummary,
    
    # Intelligence models
    ConfidenceScore,
    DataQualityReport,
    LocationIntelligence,
    
    # Baseline models
    UserBaseline,
    BaselineMetrics,
    
    # API models
    IngestRequest,
    IngestResponse,
    IntelligenceRequest,
    IntelligenceResponse,
    HealthResponse,
)

__all__ = [
    # Base models
    "LocationPoint",
    "RawTrackingEvent",
    "NormalizedTrackingEvent",
    "ProcessedTrackingEvent",
    
    # Trip models
    "Trip",
    "TripSummary",
    
    # Intelligence models
    "ConfidenceScore",
    "DataQualityReport",
    "LocationIntelligence",
    
    # Baseline models
    "UserBaseline",
    "BaselineMetrics",
    
    # API models
    "IngestRequest",
    "IngestResponse",
    "IntelligenceRequest",
    "IntelligenceResponse",
    "HealthResponse",
]
