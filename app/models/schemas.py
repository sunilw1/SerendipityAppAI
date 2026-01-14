"""
Pydantic Schemas for Serendipity AI Backend
=============================================

Type-safe data models for all data flowing through the system.

Schema Hierarchy:
1. RawTrackingEvent - Direct from mobile app / CSV
2. NormalizedTrackingEvent - After cleaning and normalization
3. ProcessedTrackingEvent - With features and confidence scores

Design Principles:
- Immutable models where possible (frozen=True)
- Strict validation at boundaries
- Clear separation between raw, normalized, and enriched data
- All fields documented with descriptions
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Set
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

from app.core.constants import ActivityType, DataQualityFlag


# =============================================================================
# BASE LOCATION MODELS
# =============================================================================

class LocationPoint(BaseModel):
    """
    A single geographic coordinate with accuracy.
    
    This is the atomic unit of location data.
    """
    model_config = ConfigDict(frozen=True)
    
    lat: float = Field(..., ge=-90, le=90, description="Latitude in degrees")
    lon: float = Field(..., ge=-180, le=180, description="Longitude in degrees")
    accuracy_meters: Optional[float] = Field(
        None, 
        ge=0, 
        description="GPS accuracy radius in meters"
    )
    
    @field_validator("lat", "lon")
    @classmethod
    def validate_not_null_island(cls, v: float, info) -> float:
        """Reject coordinates at (0, 0) which is often a GPS error."""
        # This is checked in combination with the other coordinate
        return v
    
    @model_validator(mode="after")
    def validate_not_null_island_combined(self) -> "LocationPoint":
        """Reject Null Island coordinates (0, 0)."""
        if abs(self.lat) < 0.0001 and abs(self.lon) < 0.0001:
            raise ValueError("Null Island coordinates (0, 0) are invalid")
        return self


# =============================================================================
# RAW TRACKING EVENT
# =============================================================================

class RawTrackingEvent(BaseModel):
    """
    Raw tracking event as received from mobile app or CSV.
    
    This model accepts data with minimal validation to preserve
    the original signal. Cleaning happens in subsequent stages.
    """
    model_config = ConfigDict(frozen=True)
    
    # Identifiers
    user_id: int = Field(..., description="Unique user identifier")
    trip_index_for_user: int = Field(..., ge=0, description="Trip index for this user")
    trip_global_index: int = Field(..., ge=0, description="Global trip index")
    point_index: int = Field(..., ge=0, description="Point index within trip")
    
    # Trip metadata
    trip_title: Optional[str] = Field(None, description="Human-readable trip title")
    trip_session: Optional[str] = Field(None, description="Session identifier")
    trip_start: Optional[datetime] = Field(None, description="Trip start time")
    trip_end: Optional[datetime] = Field(None, description="Trip end time")
    trip_duration_minutes: Optional[float] = Field(None, ge=0, description="Trip duration in minutes")
    trip_distance_km: Optional[float] = Field(None, ge=0, description="Trip distance in km")
    
    # Location data
    timestamp: datetime = Field(..., description="Event timestamp")
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    
    # Movement data
    speed: Optional[float] = Field(None, description="Speed in m/s (-1 means missing)")
    is_moving: int = Field(..., ge=0, le=1, description="Movement flag (0 or 1)")
    
    # Activity detection
    activity_type: str = Field(..., description="Detected activity type")
    activity_confidence: int = Field(..., ge=0, le=100, description="Activity confidence 0-100")
    
    # GPS quality
    coords_accuracy: Optional[float] = Field(None, ge=0, description="GPS accuracy in meters")
    
    # Additional context
    odometer: Optional[float] = Field(None, description="Odometer reading in meters")
    live_address: Optional[str] = Field(None, description="Reverse geocoded address")
    
    @field_validator("speed", mode="before")
    @classmethod
    def handle_missing_speed(cls, v: Any) -> Optional[float]:
        """Convert -1 sentinel to None for missing speed."""
        if v == -1 or v == "-1":
            return None
        return v
    
    @property
    def has_valid_coordinates(self) -> bool:
        """Check if coordinates are within valid ranges."""
        return -90 <= self.lat <= 90 and -180 <= self.lon <= 180
    
    @property
    def has_valid_speed(self) -> bool:
        """Check if speed is present and non-negative."""
        return self.speed is not None and self.speed >= 0


# =============================================================================
# NORMALIZED TRACKING EVENT
# =============================================================================

class NormalizedTrackingEvent(BaseModel):
    """
    Tracking event after normalization and initial cleaning.
    
    At this stage:
    - Timestamps are standardized to UTC
    - Coordinates are validated
    - Activity types are normalized to enum
    - Data quality flags are attached
    """
    model_config = ConfigDict(frozen=True)
    
    # Core identifiers
    event_id: str = Field(..., description="Unique event identifier")
    user_id: int = Field(..., description="User identifier")
    trip_id: int = Field(..., description="Global trip identifier")
    point_index: int = Field(..., ge=0, description="Point index within trip")
    
    # Timestamp (normalized to UTC)
    timestamp: datetime = Field(..., description="Event timestamp (UTC)")
    timestamp_unix: float = Field(..., description="Unix timestamp for calculations")
    
    # Location (validated)
    location: LocationPoint = Field(..., description="Validated location")
    
    # Movement data (normalized)
    speed_ms: Optional[float] = Field(None, ge=0, description="Speed in m/s")
    is_moving: bool = Field(..., description="Movement flag")
    
    # Activity (normalized to enum)
    activity_type: ActivityType = Field(..., description="Normalized activity type")
    activity_confidence: int = Field(..., ge=0, le=100, description="Confidence 0-100")
    
    # GPS quality
    gps_accuracy_meters: Optional[float] = Field(None, ge=0, description="GPS accuracy")
    
    # Data quality
    quality_flags: List[str] = Field(
        default_factory=list, 
        description="Data quality flags"
    )
    is_valid: bool = Field(default=True, description="Overall validity flag")
    
    # Original data reference
    raw_event_hash: str = Field(..., description="Hash of original event for traceability")


# =============================================================================
# COMPUTED FEATURES
# =============================================================================

class ComputedFeatures(BaseModel):
    """
    Features computed from sequential tracking events.
    
    These features support confidence scoring and baseline learning.
    """
    model_config = ConfigDict(frozen=True)
    
    # Temporal features
    time_delta_seconds: Optional[float] = Field(
        None, 
        ge=0, 
        description="Time since previous point"
    )
    update_rate_hz: Optional[float] = Field(
        None, 
        ge=0, 
        description="Update frequency"
    )
    
    # Spatial features
    distance_meters: Optional[float] = Field(
        None, 
        ge=0, 
        description="Distance from previous point (Haversine)"
    )
    bearing_degrees: Optional[float] = Field(
        None, 
        ge=0, 
        le=360, 
        description="Bearing to this point"
    )
    bearing_change_degrees: Optional[float] = Field(
        None, 
        description="Change in bearing"
    )
    
    # Derived movement
    calculated_speed_ms: Optional[float] = Field(
        None, 
        ge=0, 
        description="Speed calculated from distance/time"
    )
    speed_difference_ms: Optional[float] = Field(
        None, 
        description="Reported speed - calculated speed"
    )
    acceleration_ms2: Optional[float] = Field(
        None, 
        description="Acceleration from speed change"
    )
    
    # Stop detection
    is_stop: bool = Field(default=False, description="Detected as a stop")
    stop_duration_seconds: Optional[float] = Field(
        None, 
        ge=0, 
        description="Duration of stop if applicable"
    )
    
    # Gap detection
    has_time_gap: bool = Field(default=False, description="Significant time gap detected")
    gap_duration_seconds: Optional[float] = Field(
        None, 
        ge=0, 
        description="Gap duration if applicable"
    )


# =============================================================================
# CONFIDENCE SCORE
# =============================================================================

class ConfidenceScore(BaseModel):
    """
    Detailed confidence score breakdown for a tracking event.
    
    The overall score is a weighted combination of component scores.
    Each component is in [0, 1] where 1 is highest confidence.
    """
    model_config = ConfigDict(frozen=True)
    
    # Overall score
    overall: float = Field(..., ge=0, le=1, description="Overall confidence score")
    
    # Component scores
    gps_accuracy_score: float = Field(..., ge=0, le=1, description="GPS accuracy confidence")
    speed_validity_score: float = Field(..., ge=0, le=1, description="Speed validity confidence")
    acceleration_validity_score: float = Field(..., ge=0, le=1, description="Acceleration validity")
    temporal_consistency_score: float = Field(..., ge=0, le=1, description="Temporal consistency")
    activity_consistency_score: float = Field(..., ge=0, le=1, description="Activity consistency")
    signal_continuity_score: float = Field(..., ge=0, le=1, description="Signal continuity")
    
    # Explanation
    flags: List[str] = Field(default_factory=list, description="Factors affecting score")
    
    @property
    def confidence_level(self) -> str:
        """Human-readable confidence level."""
        if self.overall >= 0.9:
            return "excellent"
        elif self.overall >= 0.75:
            return "good"
        elif self.overall >= 0.5:
            return "moderate"
        elif self.overall >= 0.25:
            return "low"
        else:
            return "unreliable"


# =============================================================================
# PROCESSED TRACKING EVENT
# =============================================================================

class ProcessedTrackingEvent(BaseModel):
    """
    Fully processed tracking event with features and confidence.
    
    This is the primary output of the Phase 1 pipeline.
    """
    model_config = ConfigDict(frozen=True)
    
    # Normalized event data
    event: NormalizedTrackingEvent = Field(..., description="Normalized event")
    
    # Computed features
    features: ComputedFeatures = Field(..., description="Computed features")
    
    # Confidence scoring
    confidence: ConfidenceScore = Field(..., description="Confidence score")
    
    @property
    def is_high_confidence(self) -> bool:
        """Check if this is a high-confidence event."""
        return self.confidence.overall >= 0.75


# =============================================================================
# TRIP MODELS
# =============================================================================

class TripSummary(BaseModel):
    """Summary statistics for a trip."""
    model_config = ConfigDict(frozen=True)
    
    trip_id: int = Field(..., description="Trip identifier")
    user_id: int = Field(..., description="User identifier")
    
    # Temporal
    start_time: datetime = Field(..., description="Trip start time")
    end_time: datetime = Field(..., description="Trip end time")
    duration_seconds: float = Field(..., ge=0, description="Trip duration")
    
    # Spatial
    distance_meters: float = Field(..., ge=0, description="Total distance")
    start_location: LocationPoint = Field(..., description="Start location")
    end_location: LocationPoint = Field(..., description="End location")
    
    # Data quality
    total_points: int = Field(..., ge=1, description="Number of points")
    valid_points: int = Field(..., ge=0, description="Number of valid points")
    average_confidence: float = Field(..., ge=0, le=1, description="Average confidence")
    
    # Activity breakdown
    primary_activity: ActivityType = Field(..., description="Most common activity")
    activity_breakdown: Dict[str, float] = Field(
        default_factory=dict, 
        description="Percentage by activity type"
    )


class Trip(BaseModel):
    """Complete trip with all processed events."""
    model_config = ConfigDict(frozen=False)  # Mutable for building
    
    summary: TripSummary = Field(..., description="Trip summary")
    events: List[ProcessedTrackingEvent] = Field(
        default_factory=list, 
        description="Processed events"
    )


# =============================================================================
# DATA QUALITY REPORT
# =============================================================================

class DataQualityReport(BaseModel):
    """Aggregated data quality report for a dataset or batch."""
    model_config = ConfigDict(frozen=True)
    
    # Scope
    user_id: Optional[int] = Field(None, description="User ID if user-specific")
    trip_id: Optional[int] = Field(None, description="Trip ID if trip-specific")
    
    # Counts
    total_events: int = Field(..., ge=0, description="Total events analyzed")
    valid_events: int = Field(..., ge=0, description="Valid events")
    flagged_events: int = Field(..., ge=0, description="Events with quality flags")
    
    # Flag breakdown
    flag_counts: Dict[str, int] = Field(
        default_factory=dict, 
        description="Count of each flag type"
    )
    
    # Confidence distribution
    confidence_mean: float = Field(..., ge=0, le=1, description="Mean confidence")
    confidence_std: float = Field(..., ge=0, description="Confidence std dev")
    confidence_min: float = Field(..., ge=0, le=1, description="Min confidence")
    confidence_max: float = Field(..., ge=0, le=1, description="Max confidence")
    
    # Quality buckets
    high_confidence_count: int = Field(..., ge=0, description="Events with confidence >= 0.75")
    medium_confidence_count: int = Field(..., ge=0, description="Events with 0.5 <= confidence < 0.75")
    low_confidence_count: int = Field(..., ge=0, description="Events with confidence < 0.5")
    
    @property
    def validity_rate(self) -> float:
        """Percentage of valid events."""
        if self.total_events == 0:
            return 0.0
        return self.valid_events / self.total_events


# =============================================================================
# BASELINE MODELS
# =============================================================================

class BaselineMetrics(BaseModel):
    """Statistical baseline metrics for a behavior dimension."""
    model_config = ConfigDict(frozen=True)
    
    metric_name: str = Field(..., description="Name of the metric")
    
    # Central tendency
    mean: float = Field(..., description="Mean value")
    median: float = Field(..., description="Median value")
    
    # Dispersion
    std: float = Field(..., ge=0, description="Standard deviation")
    min_value: float = Field(..., description="Minimum observed")
    max_value: float = Field(..., description="Maximum observed")
    
    # Percentiles for anomaly detection (Phase 2+)
    p5: float = Field(..., description="5th percentile")
    p25: float = Field(..., description="25th percentile")
    p75: float = Field(..., description="75th percentile")
    p95: float = Field(..., description="95th percentile")
    
    # Sample info
    sample_size: int = Field(..., ge=1, description="Number of observations")


class UserBaseline(BaseModel):
    """Learned baseline behavior for a user."""
    model_config = ConfigDict(frozen=True)
    
    user_id: int = Field(..., description="User identifier")
    created_at: datetime = Field(..., description="Baseline creation time")
    updated_at: datetime = Field(..., description="Last update time")
    
    # Data coverage
    trips_analyzed: int = Field(..., ge=0, description="Trips used for baseline")
    points_analyzed: int = Field(..., ge=0, description="Points used for baseline")
    date_range_days: float = Field(..., ge=0, description="Data time span in days")
    
    # Behavior metrics
    speed_baseline: BaselineMetrics = Field(..., description="Speed patterns")
    update_frequency_baseline: BaselineMetrics = Field(..., description="Update frequency patterns")
    trip_duration_baseline: BaselineMetrics = Field(..., description="Trip duration patterns")
    stop_duration_baseline: BaselineMetrics = Field(..., description="Stop duration patterns")
    
    # Activity patterns
    activity_distribution: Dict[str, float] = Field(
        default_factory=dict, 
        description="Activity type distribution"
    )
    
    # Confidence threshold (learned)
    typical_confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Typical confidence for this user"
    )
    
    @property
    def is_mature(self) -> bool:
        """Check if baseline has enough data to be reliable."""
        return self.trips_analyzed >= 5 and self.points_analyzed >= 100


# =============================================================================
# LOCATION INTELLIGENCE (API OUTPUT)
# =============================================================================

class LocationIntelligence(BaseModel):
    """
    High-level location intelligence output.
    
    This is the primary output exposed via API - clean, confidence-scored
    location data without raw details.
    """
    model_config = ConfigDict(frozen=True)
    
    # Identity
    user_id: int = Field(..., description="User identifier")
    event_id: str = Field(..., description="Event identifier")
    
    # Time
    timestamp: datetime = Field(..., description="Event time")
    
    # Location (standardized)
    latitude: float = Field(..., ge=-90, le=90, description="Latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude")
    accuracy_meters: Optional[float] = Field(None, description="Location accuracy")
    
    # Movement
    speed_ms: Optional[float] = Field(None, ge=0, description="Speed in m/s")
    is_moving: bool = Field(..., description="Movement status")
    activity: str = Field(..., description="Activity type")
    
    # Intelligence
    confidence: float = Field(..., ge=0, le=1, description="Confidence score")
    confidence_level: str = Field(..., description="Confidence level label")
    
    # Quality indicators (high-level)
    has_quality_issues: bool = Field(..., description="Has data quality flags")
    quality_issue_count: int = Field(..., ge=0, description="Number of quality issues")


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class IngestRequest(BaseModel):
    """Request to ingest tracking events."""
    events: List[Dict[str, Any]] = Field(
        ..., 
        min_length=1, 
        max_length=10000,
        description="List of tracking events to ingest"
    )
    validate_only: bool = Field(
        default=False, 
        description="Only validate, don't persist"
    )


class IngestResponse(BaseModel):
    """Response from ingestion endpoint."""
    success: bool = Field(..., description="Overall success status")
    events_received: int = Field(..., ge=0, description="Events received")
    events_accepted: int = Field(..., ge=0, description="Events accepted")
    events_rejected: int = Field(..., ge=0, description="Events rejected")
    validation_errors: List[Dict[str, Any]] = Field(
        default_factory=list, 
        description="Validation error details"
    )
    processing_time_ms: float = Field(..., ge=0, description="Processing time")


class IntelligenceRequest(BaseModel):
    """Request for location intelligence data."""
    user_id: int = Field(..., description="User to query")
    trip_id: Optional[int] = Field(None, description="Specific trip (optional)")
    start_time: Optional[datetime] = Field(None, description="Start of time range")
    end_time: Optional[datetime] = Field(None, description="End of time range")
    min_confidence: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="Minimum confidence threshold"
    )
    include_features: bool = Field(
        default=False, 
        description="Include computed features"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Max results")
    offset: int = Field(default=0, ge=0, description="Pagination offset")


class IntelligenceResponse(BaseModel):
    """Response with location intelligence data."""
    success: bool = Field(..., description="Query success")
    user_id: int = Field(..., description="Queried user")
    total_count: int = Field(..., ge=0, description="Total matching events")
    returned_count: int = Field(..., ge=0, description="Events in response")
    data: List[LocationIntelligence] = Field(
        default_factory=list, 
        description="Intelligence data"
    )
    quality_report: Optional[DataQualityReport] = Field(
        None, 
        description="Quality report for returned data"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    timestamp: datetime = Field(..., description="Response time")
    database_connected: bool = Field(..., description="Database connectivity")
    dataset_available: bool = Field(..., description="Dataset file exists")
    stats: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Service statistics"
    )
