"""
Phase 3 Schemas - Predictive Intelligence
==========================================

Pydantic models for Phase 3 outputs:
- Travel context (weather + traffic fusion)
- Delay predictions
- Threat likelihood predictions
- Behavioral drift detection
- GPU inference statistics

Design Principles:
- All predictions include confidence scores (0-1)
- Every output includes human-readable explanation
- Builds on Phase 2 schemas
- Supports GPU-accelerated inference
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict


# =============================================================================
# ENUMS FOR PHASE 3
# =============================================================================

class WeatherCondition(str, Enum):
    """Weather conditions affecting travel."""
    CLEAR = "clear"
    CLOUDS = "clouds"
    RAIN = "rain"
    DRIZZLE = "drizzle"
    THUNDERSTORM = "thunderstorm"
    SNOW = "snow"
    MIST = "mist"
    FOG = "fog"
    HAZE = "haze"
    UNKNOWN = "unknown"


class TrafficSeverity(str, Enum):
    """Traffic congestion levels."""
    FREE_FLOW = "free_flow"
    LIGHT = "light"
    MODERATE = "moderate"
    HEAVY = "heavy"
    SEVERE = "severe"
    UNKNOWN = "unknown"


class DriftType(str, Enum):
    """Types of behavioral drift detected."""
    SPEED_RANGE = "speed_range"
    ROUTE_ADHERENCE = "route_adherence"
    STOP_PATTERN = "stop_pattern"
    ACTIVITY_TIMING = "activity_timing"
    UPDATE_FREQUENCY = "update_frequency"
    LOCATION_PATTERN = "location_pattern"
    COMBINED = "combined"


class DriftConcern(str, Enum):
    """Classification of drift concern level."""
    NORMAL_ADAPTATION = "normal_adaptation"
    MINOR_CHANGE = "minor_change"
    SIGNIFICANT_CHANGE = "significant_change"
    CONCERNING = "concerning"
    CRITICAL = "critical"


class ThreatLevel(str, Enum):
    """Threat likelihood levels."""
    MINIMAL = "minimal"
    LOW = "low"
    MODERATE = "moderate"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class InferenceBackend(str, Enum):
    """Inference backend types."""
    CPU = "cpu"
    GPU_PYTORCH = "gpu_pytorch"
    GPU_TENSORRT = "gpu_tensorrt"
    TRITON = "triton"


class ImpactSeverity(str, Enum):
    """Crash/fall impact severity levels."""
    NONE = "none"
    MINOR = "minor"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class BatteryStatus(str, Enum):
    """Battery health status levels."""
    HEALTHY = "healthy"
    LOW = "low"
    CRITICAL = "critical"
    CHARGING = "charging"


# =============================================================================
# EXTERNAL DATA SCHEMAS
# =============================================================================

class WeatherData(BaseModel):
    """Weather data from external API."""
    model_config = ConfigDict(frozen=True)
    
    condition: WeatherCondition = Field(..., description="Weather condition")
    temperature_celsius: float = Field(..., description="Temperature in Celsius")
    humidity_percent: float = Field(..., ge=0, le=100, description="Humidity percentage")
    wind_speed_ms: float = Field(..., ge=0, description="Wind speed in m/s")
    visibility_meters: float = Field(..., ge=0, description="Visibility in meters")
    precipitation_mm: float = Field(default=0.0, ge=0, description="Precipitation in mm")
    
    # Impact scoring
    travel_impact_score: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Impact on travel (0=none, 1=severe)"
    )
    
    # Metadata
    timestamp: datetime = Field(..., description="Data timestamp")
    location_lat: float = Field(..., description="Latitude")
    location_lon: float = Field(..., description="Longitude")
    source: str = Field(default="openweathermap", description="Data source")


class TrafficData(BaseModel):
    """Traffic data from external API."""
    model_config = ConfigDict(frozen=True)
    
    severity: TrafficSeverity = Field(..., description="Traffic severity")
    current_speed_kmh: float = Field(..., ge=0, description="Current speed km/h")
    free_flow_speed_kmh: float = Field(..., ge=0, description="Free flow speed km/h")
    congestion_ratio: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Congestion ratio (current/free_flow)"
    )
    
    # Delay estimation
    delay_factor: float = Field(
        ..., 
        ge=1.0, 
        description="Delay multiplier (1.0 = no delay)"
    )
    estimated_delay_seconds: float = Field(
        default=0.0, 
        ge=0, 
        description="Estimated delay in seconds"
    )
    
    # Route segment info
    segment_start: Tuple[float, float] = Field(..., description="Segment start (lat, lon)")
    segment_end: Tuple[float, float] = Field(..., description="Segment end (lat, lon)")
    segment_length_meters: float = Field(..., ge=0, description="Segment length")
    
    # Metadata
    timestamp: datetime = Field(..., description="Data timestamp")
    source: str = Field(default="tomtom", description="Data source")


class TravelContext(BaseModel):
    """
    Fused travel context combining weather and traffic.
    
    Used as input for predictive routing models.
    """
    model_config = ConfigDict(frozen=True)
    
    # Weather context
    weather: Optional[WeatherData] = Field(None, description="Weather data")
    weather_available: bool = Field(default=False, description="Weather data available")
    
    # Traffic context
    traffic: Optional[TrafficData] = Field(None, description="Traffic data")
    traffic_available: bool = Field(default=False, description="Traffic data available")
    
    # Combined metrics
    combined_delay_factor: float = Field(
        default=1.0, 
        ge=1.0, 
        description="Combined delay multiplier"
    )
    travel_risk_score: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="Overall travel risk (0=safe, 1=high risk)"
    )
    
    # Confidence in the context
    context_confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Confidence in context data"
    )
    
    # Explanation
    factors: List[str] = Field(
        default_factory=list, 
        description="Contributing factors to delay"
    )
    
    # Metadata
    generated_at: datetime = Field(..., description="Context generation time")
    valid_until: datetime = Field(..., description="Context validity expiration")


# =============================================================================
# PREDICTION SCHEMAS
# =============================================================================

class DelayPrediction(BaseModel):
    """
    Prediction of travel delay for a route.
    
    Output from predictive routing service.
    """
    model_config = ConfigDict(frozen=True)
    
    # Prediction identity
    prediction_id: str = Field(..., description="Unique prediction ID")
    user_id: int = Field(..., description="User ID")
    route_id: Optional[str] = Field(None, description="Known route ID if matched")
    
    # Route info
    origin: Tuple[float, float] = Field(..., description="Origin (lat, lon)")
    destination: Tuple[float, float] = Field(..., description="Destination (lat, lon)")
    expected_distance_meters: float = Field(..., ge=0, description="Expected distance")
    
    # Delay prediction
    predicted_delay_minutes: float = Field(
        ..., 
        ge=0, 
        description="Predicted delay in minutes"
    )
    delay_confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Confidence in delay prediction"
    )
    
    # Route prediction
    expected_duration_minutes: float = Field(
        ..., 
        ge=0, 
        description="Expected total duration"
    )
    route_confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Confidence in route matching"
    )
    
    # Delay breakdown
    weather_delay_minutes: float = Field(
        default=0.0, 
        ge=0, 
        description="Weather-related delay"
    )
    traffic_delay_minutes: float = Field(
        default=0.0, 
        ge=0, 
        description="Traffic-related delay"
    )
    
    # Context used
    travel_context: Optional[TravelContext] = Field(
        None, 
        description="Travel context used"
    )
    
    # Explainability (MANDATORY)
    explanation: List[str] = Field(
        ..., 
        description="Human-readable explanation of prediction"
    )
    
    # Metadata
    predicted_at: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(default="1.0", description="Model version used")
    inference_time_ms: float = Field(default=0.0, ge=0, description="Inference time")


class ThreatPrediction(BaseModel):
    """
    Prediction of threat likelihood based on behavioral patterns.
    
    Uses Phase 2 risk scores, anomaly history, and spoofing signals.
    """
    model_config = ConfigDict(frozen=True)
    
    # Prediction identity
    prediction_id: str = Field(..., description="Unique prediction ID")
    user_id: int = Field(..., description="User ID")
    
    # Threat prediction
    threat_likelihood: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Probability of elevated threat"
    )
    threat_level: ThreatLevel = Field(..., description="Threat level classification")
    threat_confidence: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Confidence in threat prediction"
    )
    
    # Time horizon
    time_horizon_hours: int = Field(
        default=24, 
        ge=1, 
        description="Prediction time horizon"
    )
    
    # Contributing factors
    contributing_factors: List[str] = Field(
        default_factory=list, 
        description="Factors contributing to threat"
    )
    factor_weights: Dict[str, float] = Field(
        default_factory=dict, 
        description="Weight of each factor"
    )
    
    # Input signals used
    recent_risk_score_avg: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Average recent risk score"
    )
    anomaly_rate_7d: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Anomaly rate in last 7 days"
    )
    spoofing_signals_count: int = Field(
        default=0, 
        ge=0, 
        description="Spoofing signals in window"
    )
    drift_detected: bool = Field(
        default=False, 
        description="Behavioral drift detected"
    )
    
    # Trend analysis
    risk_trend: str = Field(
        default="stable", 
        description="Risk trend (increasing/stable/decreasing)"
    )
    trend_strength: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="Strength of the trend"
    )
    
    # Explainability (MANDATORY)
    explanation: str = Field(
        ..., 
        description="Human-readable explanation"
    )
    recommendations: List[str] = Field(
        default_factory=list, 
        description="Recommended actions"
    )
    
    # Metadata
    predicted_at: datetime = Field(..., description="Prediction timestamp")
    model_version: str = Field(default="1.0", description="Model version")
    inference_time_ms: float = Field(default=0.0, ge=0, description="Inference time")
    
    @classmethod
    def threat_level_from_score(cls, score: float) -> ThreatLevel:
        """Convert threat score to level."""
        if score < 0.1:
            return ThreatLevel.MINIMAL
        elif score < 0.25:
            return ThreatLevel.LOW
        elif score < 0.45:
            return ThreatLevel.MODERATE
        elif score < 0.65:
            return ThreatLevel.ELEVATED
        elif score < 0.85:
            return ThreatLevel.HIGH
        else:
            return ThreatLevel.CRITICAL


# =============================================================================
# DRIFT DETECTION SCHEMAS
# =============================================================================

class DriftSignal(BaseModel):
    """Individual drift signal detected."""
    model_config = ConfigDict(frozen=True)
    
    drift_type: DriftType = Field(..., description="Type of drift")
    severity: float = Field(..., ge=0, le=1, description="Drift severity")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    
    # Statistical measures
    divergence_score: float = Field(
        ..., 
        ge=0, 
        description="Statistical divergence measure"
    )
    baseline_value: Optional[float] = Field(None, description="Baseline value")
    current_value: Optional[float] = Field(None, description="Current value")
    change_percent: Optional[float] = Field(None, description="Percentage change")
    
    # Description
    description: str = Field(..., description="Human-readable description")


class DriftResult(BaseModel):
    """
    Complete behavioral drift detection result.
    
    Analyzes changes in user behavior over time windows.
    """
    model_config = ConfigDict(frozen=True)
    
    # Result identity
    result_id: str = Field(..., description="Unique result ID")
    user_id: int = Field(..., description="User ID")
    
    # Detection result
    drift_detected: bool = Field(..., description="Whether drift was detected")
    drift_severity: float = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Overall drift severity"
    )
    
    # Primary drift
    primary_drift_type: Optional[DriftType] = Field(
        None, 
        description="Primary type of drift"
    )
    
    # Concern classification
    concern_level: DriftConcern = Field(
        ..., 
        description="Level of concern"
    )
    is_concerning: bool = Field(
        ..., 
        description="Whether drift is concerning vs normal adaptation"
    )
    
    # Individual signals
    signals: List[DriftSignal] = Field(
        default_factory=list, 
        description="Individual drift signals"
    )
    
    # Time windows analyzed
    window_7d_drift: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="7-day window drift score"
    )
    window_30d_drift: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="30-day window drift score"
    )
    window_90d_drift: float = Field(
        default=0.0, 
        ge=0, 
        le=1, 
        description="90-day window drift score"
    )
    
    # Profile comparison
    profile_snapshots_compared: int = Field(
        default=0, 
        ge=0, 
        description="Number of profile snapshots compared"
    )
    
    # Explainability (MANDATORY)
    explanation: str = Field(..., description="Human-readable explanation")
    details: List[str] = Field(
        default_factory=list, 
        description="Detailed drift descriptions"
    )
    
    # Metadata
    analyzed_at: datetime = Field(..., description="Analysis timestamp")
    model_version: str = Field(default="1.0", description="Model version")
    inference_time_ms: float = Field(default=0.0, ge=0, description="Inference time")
    
    @classmethod
    def concern_from_severity(cls, severity: float, is_gradual: bool) -> DriftConcern:
        """Determine concern level from severity and pattern."""
        if severity < 0.1:
            return DriftConcern.NORMAL_ADAPTATION
        elif severity < 0.3:
            return DriftConcern.MINOR_CHANGE if is_gradual else DriftConcern.SIGNIFICANT_CHANGE
        elif severity < 0.5:
            return DriftConcern.SIGNIFICANT_CHANGE
        elif severity < 0.7:
            return DriftConcern.CONCERNING
        else:
            return DriftConcern.CRITICAL


# =============================================================================
# INFERENCE STATISTICS SCHEMAS
# =============================================================================

class ModelStats(BaseModel):
    """Statistics for a single model."""
    model_config = ConfigDict(frozen=True)
    
    model_name: str = Field(..., description="Model name")
    model_version: str = Field(..., description="Model version")
    backend: InferenceBackend = Field(..., description="Inference backend")
    
    # Performance metrics
    total_inferences: int = Field(default=0, ge=0, description="Total inferences")
    avg_inference_time_ms: float = Field(default=0.0, ge=0, description="Average time")
    p50_inference_time_ms: float = Field(default=0.0, ge=0, description="P50 latency")
    p95_inference_time_ms: float = Field(default=0.0, ge=0, description="P95 latency")
    p99_inference_time_ms: float = Field(default=0.0, ge=0, description="P99 latency")
    
    # Throughput
    inferences_per_second: float = Field(default=0.0, ge=0, description="Throughput")
    
    # Errors
    error_count: int = Field(default=0, ge=0, description="Error count")
    error_rate: float = Field(default=0.0, ge=0, le=1, description="Error rate")


class GPUStats(BaseModel):
    """GPU utilization statistics."""
    model_config = ConfigDict(frozen=True)
    
    gpu_available: bool = Field(..., description="GPU available")
    gpu_name: Optional[str] = Field(None, description="GPU name")
    gpu_memory_total_mb: float = Field(default=0.0, ge=0, description="Total memory")
    gpu_memory_used_mb: float = Field(default=0.0, ge=0, description="Used memory")
    gpu_memory_free_mb: float = Field(default=0.0, ge=0, description="Free memory")
    gpu_utilization_percent: float = Field(
        default=0.0, 
        ge=0, 
        le=100, 
        description="GPU utilization"
    )
    tensorrt_available: bool = Field(default=False, description="TensorRT available")
    triton_connected: bool = Field(default=False, description="Triton connected")


class InferenceStats(BaseModel):
    """
    Complete inference statistics for Phase 3.
    
    Used to monitor GPU utilization and model performance.
    """
    model_config = ConfigDict(frozen=True)
    
    # GPU info
    gpu: GPUStats = Field(..., description="GPU statistics")
    
    # Model statistics
    models: Dict[str, ModelStats] = Field(
        default_factory=dict, 
        description="Per-model statistics"
    )
    
    # Aggregate metrics
    total_inferences: int = Field(default=0, ge=0, description="Total across all models")
    avg_inference_time_ms: float = Field(default=0.0, ge=0, description="Overall average")
    
    # Backend distribution
    cpu_inferences: int = Field(default=0, ge=0, description="CPU inferences")
    gpu_inferences: int = Field(default=0, ge=0, description="GPU inferences")
    triton_inferences: int = Field(default=0, ge=0, description="Triton inferences")
    
    # Health
    is_healthy: bool = Field(default=True, description="Overall health status")
    health_issues: List[str] = Field(
        default_factory=list, 
        description="Health issues if any"
    )
    
    # Metadata
    collected_at: datetime = Field(..., description="Stats collection time")
    uptime_seconds: float = Field(default=0.0, ge=0, description="Service uptime")


# =============================================================================
# API REQUEST/RESPONSE SCHEMAS
# =============================================================================

class DelayPredictionRequest(BaseModel):
    """Request for delay prediction."""
    user_id: int = Field(..., description="User ID")
    origin_lat: float = Field(..., description="Origin latitude")
    origin_lon: float = Field(..., description="Origin longitude")
    destination_lat: float = Field(..., description="Destination latitude")
    destination_lon: float = Field(..., description="Destination longitude")
    departure_time: Optional[datetime] = Field(
        None, 
        description="Planned departure time"
    )
    include_context: bool = Field(
        default=True, 
        description="Include travel context in response"
    )


class DelayPredictionResponse(BaseModel):
    """Response with delay prediction."""
    success: bool = Field(..., description="Prediction success")
    prediction: Optional[DelayPrediction] = Field(None, description="Prediction result")
    error: Optional[str] = Field(None, description="Error message if failed")


class ThreatPredictionRequest(BaseModel):
    """Request for threat prediction."""
    user_id: int = Field(..., description="User ID")
    time_horizon_hours: int = Field(
        default=24, 
        ge=1, 
        le=168, 
        description="Prediction horizon"
    )
    include_recommendations: bool = Field(
        default=True, 
        description="Include recommendations"
    )


class ThreatPredictionResponse(BaseModel):
    """Response with threat prediction."""
    success: bool = Field(..., description="Prediction success")
    prediction: Optional[ThreatPrediction] = Field(None, description="Prediction result")
    error: Optional[str] = Field(None, description="Error message if failed")


class DriftDetectionRequest(BaseModel):
    """Request for drift detection."""
    user_id: int = Field(..., description="User ID")
    time_windows: List[int] = Field(
        default=[7, 30, 90], 
        description="Time windows in days"
    )
    sensitivity: float = Field(
        default=0.5, 
        ge=0, 
        le=1, 
        description="Detection sensitivity"
    )


class DriftDetectionResponse(BaseModel):
    """Response with drift detection result."""
    success: bool = Field(..., description="Detection success")
    result: Optional[DriftResult] = Field(None, description="Detection result")
    error: Optional[str] = Field(None, description="Error message if failed")


class InferenceHealthResponse(BaseModel):
    """Response with inference health status."""
    healthy: bool = Field(..., description="Overall health")
    gpu_available: bool = Field(..., description="GPU availability")
    triton_connected: bool = Field(..., description="Triton connection")
    models_loaded: List[str] = Field(
        default_factory=list, 
        description="Loaded models"
    )
    issues: List[str] = Field(
        default_factory=list, 
        description="Health issues"
    )
    checked_at: datetime = Field(..., description="Check timestamp")


# =============================================================================
# CRASH / FALL DETECTION SCHEMAS
# =============================================================================

class AccelerometerReading(BaseModel):
    """Single accelerometer reading from device."""
    x: float = Field(..., description="X-axis acceleration (m/s²)")
    y: float = Field(..., description="Y-axis acceleration (m/s²)")
    z: float = Field(..., description="Z-axis acceleration (m/s²)")
    timestamp: datetime = Field(..., description="Reading timestamp")


class CrashDetectionRequest(BaseModel):
    """Request for crash detection analysis."""
    user_id: int = Field(..., description="User ID")
    readings: List[AccelerometerReading] = Field(
        ..., min_length=5, description="Accelerometer readings (minimum 5)"
    )
    speed_kmh: Optional[float] = Field(None, ge=0, description="Current speed in km/h")
    lat: Optional[float] = Field(None, description="Current latitude")
    lon: Optional[float] = Field(None, description="Current longitude")
    device_type: str = Field(default="mobile", description="Device type")


class CrashDetectionResult(BaseModel):
    """Result of crash detection analysis."""
    model_config = ConfigDict(frozen=True)

    detection_id: str = Field(..., description="Unique detection ID")
    user_id: int = Field(..., description="User ID")
    crash_detected: bool = Field(..., description="Whether a crash was detected")
    severity: ImpactSeverity = Field(..., description="Impact severity")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    peak_g_force: float = Field(..., ge=0, description="Peak G-force recorded")
    impact_duration_ms: float = Field(default=0, ge=0, description="Impact duration in ms")
    post_impact_motion: bool = Field(default=True, description="Motion detected after impact")
    speed_at_impact_kmh: Optional[float] = Field(None, description="Speed at time of impact")
    location: Optional[Tuple[float, float]] = Field(None, description="Impact location (lat, lon)")
    explanation: str = Field(..., description="Human-readable explanation")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    detected_at: datetime = Field(..., description="Detection timestamp")
    alert_triggered: bool = Field(default=False, description="Whether alert was triggered")


class CrashDetectionResponse(BaseModel):
    """Response for crash detection."""
    success: bool = Field(..., description="Detection success")
    result: Optional[CrashDetectionResult] = Field(None, description="Detection result")
    error: Optional[str] = Field(None, description="Error message if failed")


class FallDetectionRequest(BaseModel):
    """Request for fall detection analysis."""
    user_id: int = Field(..., description="User ID")
    readings: List[AccelerometerReading] = Field(
        ..., min_length=5, description="Accelerometer readings (minimum 5)"
    )
    gyroscope_data: Optional[List[Dict]] = Field(
        None, description="Optional gyroscope readings"
    )
    user_age: Optional[int] = Field(None, ge=0, le=120, description="User age for risk assessment")
    lat: Optional[float] = Field(None, description="Current latitude")
    lon: Optional[float] = Field(None, description="Current longitude")


class FallDetectionResult(BaseModel):
    """Result of fall detection analysis."""
    model_config = ConfigDict(frozen=True)

    detection_id: str = Field(..., description="Unique detection ID")
    user_id: int = Field(..., description="User ID")
    fall_detected: bool = Field(..., description="Whether a fall was detected")
    severity: ImpactSeverity = Field(..., description="Fall severity")
    confidence: float = Field(..., ge=0, le=1, description="Detection confidence")
    free_fall_duration_ms: float = Field(default=0, ge=0, description="Free-fall duration in ms")
    impact_g_force: float = Field(..., ge=0, description="Impact G-force")
    post_fall_stationary: bool = Field(default=False, description="User stationary after fall")
    stationary_duration_seconds: float = Field(default=0, ge=0, description="Time stationary after fall")
    location: Optional[Tuple[float, float]] = Field(None, description="Fall location (lat, lon)")
    explanation: str = Field(..., description="Human-readable explanation")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    detected_at: datetime = Field(..., description="Detection timestamp")
    alert_triggered: bool = Field(default=False, description="Whether alert was triggered")


class FallDetectionResponse(BaseModel):
    """Response for fall detection."""
    success: bool = Field(..., description="Detection success")
    result: Optional[FallDetectionResult] = Field(None, description="Detection result")
    error: Optional[str] = Field(None, description="Error message if failed")


# =============================================================================
# BATTERY MONITORING SCHEMAS
# =============================================================================

class BatteryReportRequest(BaseModel):
    """Battery status report from device."""
    user_id: int = Field(..., description="User ID")
    battery_level: float = Field(..., ge=0, le=100, description="Battery level percentage")
    is_charging: bool = Field(default=False, description="Whether device is charging")
    battery_temperature: Optional[float] = Field(None, description="Battery temperature °C")
    drain_rate_per_hour: Optional[float] = Field(None, ge=0, description="Drain rate %/hour")
    lat: Optional[float] = Field(None, description="Current latitude")
    lon: Optional[float] = Field(None, description="Current longitude")


class BatteryAnalysisResult(BaseModel):
    """Battery analysis result."""
    model_config = ConfigDict(frozen=True)

    user_id: int = Field(..., description="User ID")
    battery_level: float = Field(..., ge=0, le=100, description="Current battery level")
    status: BatteryStatus = Field(..., description="Battery status")
    estimated_hours_remaining: Optional[float] = Field(
        None, ge=0, description="Estimated hours until empty"
    )
    is_critical: bool = Field(default=False, description="Whether battery is critically low")
    drain_rate_per_hour: Optional[float] = Field(None, description="Current drain rate %/hour")
    explanation: str = Field(..., description="Human-readable explanation")
    alert_triggered: bool = Field(default=False, description="Whether alert was triggered")
    recommended_actions: List[str] = Field(default_factory=list, description="Recommended actions")
    analyzed_at: datetime = Field(..., description="Analysis timestamp")


class BatteryAnalysisResponse(BaseModel):
    """Response for battery analysis."""
    success: bool = Field(..., description="Analysis success")
    result: Optional[BatteryAnalysisResult] = Field(None, description="Analysis result")
    error: Optional[str] = Field(None, description="Error message if failed")
