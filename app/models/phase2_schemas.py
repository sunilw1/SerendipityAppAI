"""
Phase 2 Schemas - Behavioral Intelligence & Threat Detection
=============================================================

Pydantic models for Phase 2 outputs:
- Behavioral profiles (expanded baselines)
- Anomaly detection results
- Spoofing detection results
- Risk scores with explainability
- Session and location-level intelligence

Design Principles:
- All scores are continuous (0-1), not binary
- Every score includes explanation/reasons
- Builds on Phase 1 schemas (confidence, features)
- Tunable without code changes via config
"""

from datetime import datetime, time as dt_time
from typing import Dict, List, Optional, Set, Tuple
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict

from app.models.schemas import (
    BaselineMetrics,
    UserBaseline,
    LocationPoint,
    ConfidenceScore,
)


# =============================================================================
# ENUMS FOR PHASE 2
# =============================================================================

class AnomalyType(str, Enum):
    """Types of anomalies detected by the system."""
    ROUTE_DEVIATION = "route_deviation"
    SPEED_ANOMALY = "speed_anomaly"
    IMPOSSIBLE_JUMP = "impossible_jump"
    STOP_PATTERN_ANOMALY = "stop_pattern_anomaly"
    TIME_OF_DAY_ANOMALY = "time_of_day_anomaly"
    UPDATE_FREQUENCY_ANOMALY = "update_frequency_anomaly"
    MOVEMENT_CONSISTENCY_ANOMALY = "movement_consistency_anomaly"
    LOW_CONFIDENCE_SEQUENCE = "low_confidence_sequence"
    ACCELERATION_ANOMALY = "acceleration_anomaly"
    BEARING_ANOMALY = "bearing_anomaly"


class SpoofingIndicator(str, Enum):
    """Indicators of potential GPS spoofing or tampering."""
    TELEPORTATION = "teleportation"
    REPEATED_TELEPORT = "repeated_teleport"
    IMPOSSIBLE_SPEED = "impossible_speed"
    TIMESTAMP_MANIPULATION = "timestamp_manipulation"
    ACCURACY_DEGRADATION = "accuracy_degradation"
    SYNTHETIC_PATTERN = "synthetic_pattern"
    SESSION_INTEGRITY_VIOLATION = "session_integrity_violation"
    CLOCK_DRIFT = "clock_drift"
    PERFECT_LINE_PATH = "perfect_line_path"
    IDENTICAL_COORDINATES = "identical_coordinates"


class RiskLevel(str, Enum):
    """Human-readable risk levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


# =============================================================================
# BEHAVIORAL PROFILE (EXPANDED BASELINE)
# =============================================================================

class RoutePattern(BaseModel):
    """Learned route pattern from historical data."""
    model_config = ConfigDict(frozen=True)
    
    route_id: str = Field(..., description="Unique route identifier")
    start_zone: Tuple[float, float, float] = Field(
        ..., 
        description="Start zone (lat, lon, radius_meters)"
    )
    end_zone: Tuple[float, float, float] = Field(
        ..., 
        description="End zone (lat, lon, radius_meters)"
    )
    waypoint_zones: List[Tuple[float, float, float]] = Field(
        default_factory=list,
        description="Intermediate waypoint zones"
    )
    frequency: int = Field(..., ge=1, description="Times this route observed")
    avg_duration_seconds: float = Field(..., ge=0, description="Average trip duration")
    avg_distance_meters: float = Field(..., ge=0, description="Average distance")
    typical_speeds_ms: BaselineMetrics = Field(..., description="Speed distribution")
    typical_time_of_day: Optional[Tuple[int, int]] = Field(
        None, 
        description="Typical hour range (start, end)"
    )


class TimeOfDayPattern(BaseModel):
    """Activity patterns by time of day."""
    model_config = ConfigDict(frozen=True)
    
    hour: int = Field(..., ge=0, le=23, description="Hour of day (0-23)")
    activity_probability: float = Field(..., ge=0, le=1, description="Probability of activity")
    typical_activity: Optional[str] = Field(None, description="Most common activity")
    avg_speed_ms: Optional[float] = Field(None, ge=0, description="Average speed")
    sample_count: int = Field(..., ge=0, description="Number of observations")


class StopPattern(BaseModel):
    """Learned stop pattern (frequent locations)."""
    model_config = ConfigDict(frozen=True)
    
    stop_id: str = Field(..., description="Stop pattern identifier")
    location: LocationPoint = Field(..., description="Stop center")
    radius_meters: float = Field(..., ge=0, description="Stop zone radius")
    frequency: int = Field(..., ge=1, description="Visit frequency")
    avg_duration_seconds: float = Field(..., ge=0, description="Average stop duration")
    typical_arrival_hours: List[int] = Field(
        default_factory=list, 
        description="Typical arrival hours"
    )
    typical_departure_hours: List[int] = Field(
        default_factory=list, 
        description="Typical departure hours"
    )


class BehavioralProfile(BaseModel):
    """
    Extended behavioral profile for a user.
    
    Builds on Phase 1 UserBaseline with:
    - Route patterns
    - Time-of-day patterns
    - Stop patterns
    - Movement consistency metrics
    """
    model_config = ConfigDict(frozen=True)
    
    user_id: int = Field(..., description="User identifier")
    profile_version: str = Field(default="2.0", description="Profile schema version")
    created_at: datetime = Field(..., description="Profile creation time")
    updated_at: datetime = Field(..., description="Last update time")
    
    # Phase 1 baseline (inherited)
    baseline: UserBaseline = Field(..., description="Phase 1 baseline metrics")
    
    # Route patterns
    known_routes: List[RoutePattern] = Field(
        default_factory=list,
        description="Learned route patterns"
    )
    route_adherence_score: float = Field(
        default=0.0, 
        ge=0, 
        le=1,
        description="How consistently user follows known routes"
    )
    
    # Time-of-day patterns
    hourly_patterns: List[TimeOfDayPattern] = Field(
        default_factory=list,
        description="Activity by hour"
    )
    
    # Stop patterns
    frequent_stops: List[StopPattern] = Field(
        default_factory=list,
        description="Frequent stop locations"
    )
    
    # Movement consistency
    speed_consistency: float = Field(
        default=0.0, 
        ge=0, 
        le=1,
        description="How consistent speeds are vs baseline"
    )
    update_regularity: float = Field(
        default=0.0, 
        ge=0, 
        le=1,
        description="How regular update intervals are"
    )
    
    # Profile maturity
    total_trips_analyzed: int = Field(default=0, ge=0, description="Trips in profile")
    total_points_analyzed: int = Field(default=0, ge=0, description="Points in profile")
    profile_confidence: float = Field(
        default=0.0, 
        ge=0, 
        le=1,
        description="Overall profile reliability"
    )
    
    @property
    def is_mature(self) -> bool:
        """Check if profile has enough data."""
        return (
            self.total_trips_analyzed >= 10 and 
            self.total_points_analyzed >= 500 and
            self.baseline.is_mature
        )


# =============================================================================
# ANOMALY DETECTION RESULTS
# =============================================================================

class AnomalySignal(BaseModel):
    """Individual anomaly signal with explanation."""
    model_config = ConfigDict(frozen=True)
    
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    severity: float = Field(..., ge=0, le=1, description="Severity score (0-1)")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Human-readable explanation")
    observed_value: Optional[float] = Field(None, description="Observed value")
    expected_range: Optional[Tuple[float, float]] = Field(
        None, 
        description="Expected range (min, max)"
    )
    z_score: Optional[float] = Field(None, description="Z-score if applicable")
    contribution_weight: float = Field(
        default=1.0, 
        ge=0, 
        le=1,
        description="Weight in overall anomaly score"
    )


class AnomalyResult(BaseModel):
    """Complete anomaly detection result for a location point."""
    model_config = ConfigDict(frozen=True)
    
    event_id: str = Field(..., description="Event identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    
    # Overall anomaly score
    anomaly_score: float = Field(..., ge=0, le=1, description="Combined anomaly score")
    is_anomalous: bool = Field(..., description="Whether point is anomalous")
    
    # Individual signals
    signals: List[AnomalySignal] = Field(
        default_factory=list, 
        description="Contributing anomaly signals"
    )
    
    # Model outputs (for transparency)
    isolation_forest_score: Optional[float] = Field(
        None, 
        ge=-1, 
        le=1,
        description="Isolation Forest anomaly score"
    )
    statistical_score: Optional[float] = Field(
        None, 
        ge=0, 
        le=1,
        description="Statistical z-score based anomaly"
    )
    
    @property
    def primary_anomaly(self) -> Optional[AnomalyType]:
        """Get the most severe anomaly type."""
        if not self.signals:
            return None
        return max(self.signals, key=lambda s: s.severity).anomaly_type


# =============================================================================
# SPOOFING DETECTION RESULTS
# =============================================================================

class SpoofingSignal(BaseModel):
    """Individual spoofing indicator with explanation."""
    model_config = ConfigDict(frozen=True)
    
    indicator: SpoofingIndicator = Field(..., description="Type of spoofing indicator")
    likelihood: float = Field(..., ge=0, le=1, description="Spoofing likelihood")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    description: str = Field(..., description="Explanation of detection")
    evidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Supporting evidence values"
    )


class SpoofingResult(BaseModel):
    """Spoofing detection result for a point or session."""
    model_config = ConfigDict(frozen=True)
    
    scope: str = Field(..., description="Detection scope (point/session)")
    scope_id: str = Field(..., description="Event or session ID")
    timestamp: datetime = Field(..., description="Detection timestamp")
    
    # Overall spoofing likelihood (NOT a binary decision)
    spoofing_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Overall spoofing probability"
    )
    tampering_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Overall tampering probability"
    )
    
    # Individual indicators
    indicators: List[SpoofingSignal] = Field(
        default_factory=list,
        description="Contributing indicators"
    )
    
    # Session integrity
    session_integrity_score: float = Field(
        default=1.0, 
        ge=0, 
        le=1,
        description="Session data integrity"
    )
    
    @property
    def requires_attention(self) -> bool:
        """Check if this warrants review."""
        return self.spoofing_likelihood >= 0.5 or self.tampering_likelihood >= 0.5


# =============================================================================
# RISK SCORING
# =============================================================================

class RiskContributor(BaseModel):
    """Individual factor contributing to risk score."""
    model_config = ConfigDict(frozen=True)
    
    factor: str = Field(..., description="Risk factor name")
    weight: float = Field(..., ge=0, le=1, description="Factor weight")
    score: float = Field(..., ge=0, le=1, description="Factor score")
    weighted_contribution: float = Field(..., ge=0, le=1, description="Weighted score")
    description: str = Field(..., description="Explanation")


class LocationRiskScore(BaseModel):
    """Risk score for a single location point."""
    model_config = ConfigDict(frozen=True)
    
    event_id: str = Field(..., description="Event identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    location: LocationPoint = Field(..., description="Location")
    
    # Risk score (continuous 0-1)
    risk_score: float = Field(..., ge=0, le=1, description="Overall risk score")
    risk_level: RiskLevel = Field(..., description="Human-readable risk level")
    
    # Contributing factors
    contributors: List[RiskContributor] = Field(
        default_factory=list,
        description="Factors contributing to risk"
    )
    
    # Component scores (for explainability)
    confidence_factor: float = Field(..., ge=0, le=1, description="Phase 1 confidence influence")
    anomaly_factor: float = Field(..., ge=0, le=1, description="Anomaly detection influence")
    spoofing_factor: float = Field(..., ge=0, le=1, description="Spoofing detection influence")
    baseline_factor: float = Field(..., ge=0, le=1, description="Baseline deviation influence")
    
    # Explainability (MANDATORY)
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable reasons for risk score"
    )


class SessionRiskScore(BaseModel):
    """Aggregated risk score for a session/trip."""
    model_config = ConfigDict(frozen=True)
    
    session_id: str = Field(..., description="Session/trip identifier")
    user_id: int = Field(..., description="User identifier")
    start_time: datetime = Field(..., description="Session start")
    end_time: datetime = Field(..., description="Session end")
    
    # Session-level risk
    risk_score: float = Field(..., ge=0, le=1, description="Overall session risk")
    risk_level: RiskLevel = Field(..., description="Human-readable risk level")
    
    # Statistics
    total_points: int = Field(..., ge=0, description="Total location points")
    high_risk_points: int = Field(..., ge=0, description="Points with risk > 0.7")
    anomalous_points: int = Field(..., ge=0, description="Anomalous points detected")
    
    # Component scores
    max_location_risk: float = Field(..., ge=0, le=1, description="Maximum point risk")
    avg_location_risk: float = Field(..., ge=0, le=1, description="Average point risk")
    spoofing_likelihood: float = Field(..., ge=0, le=1, description="Session spoofing likelihood")
    integrity_score: float = Field(..., ge=0, le=1, description="Session integrity")
    
    # Explainability
    contributors: List[RiskContributor] = Field(
        default_factory=list,
        description="Top risk contributors"
    )
    reasons: List[str] = Field(
        default_factory=list,
        description="Human-readable risk explanations"
    )
    
    # Flagged events (top concerns)
    flagged_event_ids: List[str] = Field(
        default_factory=list,
        description="IDs of highest-risk events"
    )


# =============================================================================
# INTELLIGENCE OUTPUT (PHASE 2 API)
# =============================================================================

class Phase2Intelligence(BaseModel):
    """
    Complete Phase 2 intelligence output for a location.
    
    Combines Phase 1 confidence with Phase 2 analysis.
    """
    model_config = ConfigDict(frozen=True)
    
    # Identity
    event_id: str = Field(..., description="Event identifier")
    user_id: int = Field(..., description="User identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(..., description="Event timestamp")
    
    # Location
    location: LocationPoint = Field(..., description="Location")
    
    # Phase 1 outputs (preserved)
    phase1_confidence: float = Field(..., ge=0, le=1, description="Phase 1 confidence")
    confidence_level: str = Field(..., description="Confidence level label")
    
    # Phase 2 outputs
    risk_score: float = Field(..., ge=0, le=1, description="Risk score")
    risk_level: RiskLevel = Field(..., description="Risk level")
    anomaly_score: float = Field(..., ge=0, le=1, description="Anomaly score")
    spoofing_likelihood: float = Field(..., ge=0, le=1, description="Spoofing likelihood")
    
    # Explainability (MANDATORY)
    risk_reasons: List[str] = Field(..., description="Risk score explanations")
    anomaly_types: List[str] = Field(default_factory=list, description="Detected anomaly types")
    
    # Clean movement signal
    is_trusted: bool = Field(..., description="Can be trusted for downstream use")
    
    @classmethod
    def risk_level_from_score(cls, score: float) -> RiskLevel:
        """Convert risk score to level."""
        if score < 0.1:
            return RiskLevel.MINIMAL
        elif score < 0.25:
            return RiskLevel.LOW
        elif score < 0.45:
            return RiskLevel.MODERATE
        elif score < 0.65:
            return RiskLevel.ELEVATED
        elif score < 0.85:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL


class SessionIntelligence(BaseModel):
    """Complete Phase 2 session-level intelligence."""
    model_config = ConfigDict(frozen=True)
    
    session_id: str = Field(..., description="Session identifier")
    user_id: int = Field(..., description="User identifier")
    start_time: datetime = Field(..., description="Session start")
    end_time: datetime = Field(..., description="Session end")
    duration_seconds: float = Field(..., ge=0, description="Duration")
    
    # Session risk
    risk_score: SessionRiskScore = Field(..., description="Session risk assessment")
    
    # Location intelligence summary
    total_points: int = Field(..., ge=0, description="Total points")
    trusted_points: int = Field(..., ge=0, description="Trusted points")
    trust_rate: float = Field(..., ge=0, le=1, description="Percentage trusted")
    
    # Anomaly summary
    anomaly_count: int = Field(..., ge=0, description="Anomalous points")
    anomaly_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Anomaly type counts"
    )
    
    # Spoofing summary
    spoofing_likelihood: float = Field(..., ge=0, le=1, description="Overall likelihood")
    spoofing_indicators: List[str] = Field(
        default_factory=list,
        description="Detected indicators"
    )
    
    # High-level verdict
    is_session_valid: bool = Field(..., description="Session appears valid")
    validation_confidence: float = Field(..., ge=0, le=1, description="Validation confidence")
    recommendations: List[str] = Field(
        default_factory=list,
        description="Recommended actions"
    )


# =============================================================================
# API REQUEST/RESPONSE MODELS (PHASE 2)
# =============================================================================

class RiskAnalysisRequest(BaseModel):
    """Request for risk analysis."""
    user_id: int = Field(..., description="User to analyze")
    session_id: Optional[str] = Field(None, description="Specific session")
    start_time: Optional[datetime] = Field(None, description="Start of range")
    end_time: Optional[datetime] = Field(None, description="End of range")
    include_location_risks: bool = Field(
        default=True, 
        description="Include per-location risks"
    )
    include_explanations: bool = Field(
        default=True, 
        description="Include detailed explanations"
    )
    limit: int = Field(default=100, ge=1, le=10000, description="Max results")


class RiskAnalysisResponse(BaseModel):
    """Response with risk analysis results."""
    success: bool = Field(..., description="Analysis success")
    user_id: int = Field(..., description="Analyzed user")
    analysis_timestamp: datetime = Field(..., description="Analysis time")
    
    # Session-level results
    session_risks: List[SessionRiskScore] = Field(
        default_factory=list,
        description="Session risk scores"
    )
    
    # Location-level results (if requested)
    location_risks: List[LocationRiskScore] = Field(
        default_factory=list,
        description="Location risk scores"
    )
    
    # Summary statistics
    total_sessions: int = Field(..., ge=0, description="Sessions analyzed")
    high_risk_sessions: int = Field(..., ge=0, description="High risk sessions")
    avg_risk_score: float = Field(..., ge=0, le=1, description="Average risk")


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    user_id: int = Field(..., description="User to analyze")
    session_id: Optional[str] = Field(None, description="Specific session")
    sensitivity: float = Field(
        default=0.5, 
        ge=0, 
        le=1,
        description="Detection sensitivity (higher = more detections)"
    )
    include_model_scores: bool = Field(
        default=False, 
        description="Include raw model scores"
    )


class AnomalyDetectionResponse(BaseModel):
    """Response with anomaly detection results."""
    success: bool = Field(..., description="Detection success")
    user_id: int = Field(..., description="Analyzed user")
    
    # Anomalies detected
    anomalies: List[AnomalyResult] = Field(
        default_factory=list,
        description="Detected anomalies"
    )
    
    # Summary
    total_points_analyzed: int = Field(..., ge=0, description="Points analyzed")
    anomalous_points: int = Field(..., ge=0, description="Anomalous points")
    anomaly_rate: float = Field(..., ge=0, le=1, description="Anomaly rate")
    anomaly_type_breakdown: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by anomaly type"
    )


class SpoofingDetectionResponse(BaseModel):
    """Response with spoofing detection results."""
    success: bool = Field(..., description="Detection success")
    user_id: int = Field(..., description="Analyzed user")
    session_id: Optional[str] = Field(None, description="Analyzed session")
    
    # Overall assessment
    overall_spoofing_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Overall spoofing probability"
    )
    overall_tampering_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1,
        description="Overall tampering probability"
    )
    
    # Detailed results
    results: List[SpoofingResult] = Field(
        default_factory=list,
        description="Detection results"
    )
    
    # Recommendations
    requires_review: bool = Field(..., description="Manual review recommended")
    confidence_in_assessment: float = Field(..., ge=0, le=1, description="Assessment confidence")
