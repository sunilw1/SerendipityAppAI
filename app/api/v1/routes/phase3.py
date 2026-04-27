"""
Phase 3 API Routes - Predictive Intelligence
==============================================

REST API endpoints for Phase 3 predictive services:
- Delay prediction
- Threat likelihood prediction
- Behavioral drift detection
- Inference health monitoring

Design Principles:
- All responses include confidence scores
- All responses include explanations
- Consistent error handling
- Async-first implementation
"""

from datetime import datetime, timezone
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.api.deps import verify_api_key
from app.models.phase3_schemas import (
    DelayPrediction,
    DelayPredictionRequest,
    DelayPredictionResponse,
    ThreatPrediction,
    ThreatPredictionRequest,
    ThreatPredictionResponse,
    DriftResult,
    DriftDetectionRequest,
    DriftDetectionResponse,
    InferenceStats,
    InferenceHealthResponse,
)
from app.services.predictive_routing import (
    get_predictive_routing_service,
    PredictiveRoutingService,
)
from app.services.threat_predictor import (
    get_threat_predictor,
    ThreatPredictor,
)
from app.detection.drift import (
    get_drift_detector,
    BehavioralDriftDetector,
)
from app.inference.triton_client import get_triton_client, TritonClient

logger = get_logger(__name__)

router = APIRouter(prefix="/phase3", tags=["Phase 3 - Predictive Intelligence"])


# =============================================================================
# DEPENDENCY INJECTION
# =============================================================================

def get_routing_service() -> PredictiveRoutingService:
    """Get predictive routing service."""
    return get_predictive_routing_service()


def get_threat_service() -> ThreatPredictor:
    """Get threat predictor service."""
    return get_threat_predictor()


def get_drift_service() -> BehavioralDriftDetector:
    """Get drift detector service."""
    return get_drift_detector()


def get_triton() -> TritonClient:
    """Get Triton client."""
    return get_triton_client()


# =============================================================================
# DELAY PREDICTION ENDPOINTS
# =============================================================================

@router.get(
    "/predict/delay/{user_id}",
    response_model=DelayPredictionResponse,
    summary="Predict travel delay",
    description="""
    Predict travel delay for a user's trip.
    
    Uses:
    - Historical route patterns from behavioral profile
    - Real-time weather data (OpenWeatherMap)
    - Real-time traffic data (TomTom)
    - ML model for delay prediction
    
    Returns prediction with confidence score and explanation.
    """,
)
async def predict_delay(
    user_id: int,
    origin_lat: float = Query(..., description="Origin latitude"),
    origin_lon: float = Query(..., description="Origin longitude"),
    destination_lat: float = Query(..., description="Destination latitude"),
    destination_lon: float = Query(..., description="Destination longitude"),
    departure_time: Optional[datetime] = Query(
        None, description="Planned departure time (ISO 8601)"
    ),
    include_context: bool = Query(
        True, description="Include travel context in response"
    ),
    routing_service: PredictiveRoutingService = Depends(get_routing_service),
    _api_key: str = Depends(verify_api_key),
) -> DelayPredictionResponse:
    """Predict travel delay for a trip."""
    try:
        logger.info(
            "delay_prediction_request",
            user_id=user_id,
            origin=(origin_lat, origin_lon),
            destination=(destination_lat, destination_lon),
        )
        
        prediction = await routing_service.predict_delay(
            user_id=user_id,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            destination_lat=destination_lat,
            destination_lon=destination_lon,
            departure_time=departure_time,
            include_context=include_context,
        )
        
        return DelayPredictionResponse(
            success=True,
            prediction=prediction,
            error=None,
        )
        
    except Exception as e:
        logger.error("delay_prediction_failed", user_id=user_id, error=str(e))
        return DelayPredictionResponse(
            success=False,
            prediction=None,
            error=str(e),
        )


@router.post(
    "/predict/delay",
    response_model=DelayPredictionResponse,
    summary="Predict travel delay (POST)",
    description="Same as GET but accepts request body.",
)
async def predict_delay_post(
    request: DelayPredictionRequest,
    routing_service: PredictiveRoutingService = Depends(get_routing_service),
    _api_key: str = Depends(verify_api_key),
) -> DelayPredictionResponse:
    """Predict travel delay (POST variant)."""
    try:
        prediction = await routing_service.predict_delay(
            user_id=request.user_id,
            origin_lat=request.origin_lat,
            origin_lon=request.origin_lon,
            destination_lat=request.destination_lat,
            destination_lon=request.destination_lon,
            departure_time=request.departure_time,
            include_context=request.include_context,
        )
        
        return DelayPredictionResponse(
            success=True,
            prediction=prediction,
            error=None,
        )
        
    except Exception as e:
        logger.error("delay_prediction_failed", error=str(e))
        return DelayPredictionResponse(
            success=False,
            prediction=None,
            error=str(e),
        )


# =============================================================================
# THREAT PREDICTION ENDPOINTS
# =============================================================================

@router.get(
    "/predict/threat/{user_id}",
    response_model=ThreatPredictionResponse,
    summary="Predict threat likelihood",
    description="""
    Predict threat likelihood for a user based on behavioral patterns.
    
    Uses:
    - Historical risk scores from Phase 2
    - Anomaly detection history
    - Spoofing signal history
    - Behavioral drift indicators
    
    Returns prediction with confidence score, explanation, and recommendations.
    """,
)
async def predict_threat(
    user_id: int,
    time_horizon_hours: int = Query(
        24, ge=1, le=168, description="Prediction time horizon in hours"
    ),
    include_recommendations: bool = Query(
        True, description="Include action recommendations"
    ),
    threat_service: ThreatPredictor = Depends(get_threat_service),
    _api_key: str = Depends(verify_api_key),
) -> ThreatPredictionResponse:
    """Predict threat likelihood for a user."""
    try:
        logger.info(
            "threat_prediction_request",
            user_id=user_id,
            horizon_hours=time_horizon_hours,
        )
        
        # In production, these would come from the database
        # For now, use empty lists (predictor handles this gracefully)
        prediction = await threat_service.predict(
            user_id=user_id,
            risk_scores=[],  # Would fetch from Phase 2 storage
            anomalies=[],
            spoofing_results=[],
            drift_detected=False,
            time_horizon_hours=time_horizon_hours,
        )
        
        return ThreatPredictionResponse(
            success=True,
            prediction=prediction,
            error=None,
        )
        
    except Exception as e:
        logger.error("threat_prediction_failed", user_id=user_id, error=str(e))
        return ThreatPredictionResponse(
            success=False,
            prediction=None,
            error=str(e),
        )


@router.post(
    "/predict/threat",
    response_model=ThreatPredictionResponse,
    summary="Predict threat likelihood (POST)",
)
async def predict_threat_post(
    request: ThreatPredictionRequest,
    threat_service: ThreatPredictor = Depends(get_threat_service),
    _api_key: str = Depends(verify_api_key),
) -> ThreatPredictionResponse:
    """Predict threat likelihood (POST variant)."""
    try:
        prediction = await threat_service.predict(
            user_id=request.user_id,
            risk_scores=[],
            anomalies=[],
            spoofing_results=[],
            drift_detected=False,
            time_horizon_hours=request.time_horizon_hours,
        )
        
        return ThreatPredictionResponse(
            success=True,
            prediction=prediction,
            error=None,
        )
        
    except Exception as e:
        return ThreatPredictionResponse(
            success=False,
            prediction=None,
            error=str(e),
        )


# =============================================================================
# DRIFT DETECTION ENDPOINTS
# =============================================================================

@router.get(
    "/detect/drift/{user_id}",
    response_model=DriftDetectionResponse,
    summary="Detect behavioral drift",
    description="""
    Detect behavioral drift for a user.
    
    Analyzes changes in:
    - Speed patterns
    - Route adherence
    - Stop patterns
    - Activity timing
    - Update frequency
    
    Returns detection result with severity, concern level, and explanation.
    """,
)
async def detect_drift(
    user_id: int,
    time_windows: str = Query(
        "7,30,90",
        description="Comma-separated time windows in days",
    ),
    sensitivity: float = Query(
        0.5, ge=0, le=1, description="Detection sensitivity"
    ),
    drift_service: BehavioralDriftDetector = Depends(get_drift_service),
    _api_key: str = Depends(verify_api_key),
) -> DriftDetectionResponse:
    """Detect behavioral drift for a user."""
    try:
        logger.info(
            "drift_detection_request",
            user_id=user_id,
            windows=time_windows,
        )
        
        # Parse time windows
        windows = [int(w.strip()) for w in time_windows.split(",")]
        
        # In production, fetch current profile from storage
        # For now, return a placeholder response
        from app.models.phase2_schemas import BehavioralProfile
        from app.models.schemas import UserBaseline, BaselineMetrics
        
        # Create minimal profile for demo
        now = datetime.now(timezone.utc)
        minimal_baseline = UserBaseline(
            user_id=user_id,
            created_at=now,
            updated_at=now,
            speed_baseline=BaselineMetrics(
                metric_name="speed_ms",
                mean=10.0,
                std=5.0,
                min_val=0.0,
                max_val=30.0,
                median=9.0,
                percentile_5=2.0,
                percentile_95=25.0,
                percentile_99=28.0,
                sample_size=100,
            ),
            update_frequency_baseline=BaselineMetrics(
                metric_name="update_freq",
                mean=5.0, std=2.0, min_val=1.0, max_val=30.0,
                median=4.0, percentile_5=1.5, percentile_95=15.0,
                percentile_99=25.0, sample_size=100,
            ),
            trip_duration_baseline=BaselineMetrics(
                metric_name="trip_duration",
                mean=1800.0, std=900.0, min_val=300.0, max_val=7200.0,
                median=1500.0, percentile_5=400.0, percentile_95=4000.0,
                percentile_99=6000.0, sample_size=20,
            ),
            stop_duration_baseline=BaselineMetrics(
                metric_name="stop_duration",
                mean=300.0, std=200.0, min_val=60.0, max_val=3600.0,
                median=180.0, percentile_5=60.0, percentile_95=900.0,
                percentile_99=2000.0, sample_size=50,
            ),
            total_trips=20,
            total_points=2000,
            activity_distribution={"walking": 0.3, "driving": 0.6, "stationary": 0.1},
            typical_confidence=0.85,
            is_mature=True,
        )
        
        profile = BehavioralProfile(
            user_id=user_id,
            profile_version="2.0",
            created_at=now,
            updated_at=now,
            baseline=minimal_baseline,
            known_routes=[],
            route_adherence_score=0.7,
            hourly_patterns=[],
            frequent_stops=[],
            speed_consistency=0.8,
            update_regularity=0.75,
            total_trips_analyzed=20,
            total_points_analyzed=2000,
            profile_confidence=0.8,
        )
        
        result = await drift_service.detect(
            user_id=user_id,
            current_profile=profile,
            time_windows=windows,
        )
        
        return DriftDetectionResponse(
            success=True,
            result=result,
            error=None,
        )
        
    except Exception as e:
        logger.error("drift_detection_failed", user_id=user_id, error=str(e))
        return DriftDetectionResponse(
            success=False,
            result=None,
            error=str(e),
        )


# =============================================================================
# INFERENCE HEALTH ENDPOINTS
# =============================================================================

@router.get(
    "/inference/health",
    response_model=InferenceHealthResponse,
    summary="Check inference health",
    description="Check health of GPU inference services (Triton).",
)
async def inference_health(
    triton: TritonClient = Depends(get_triton),
    _api_key: str = Depends(verify_api_key),
) -> InferenceHealthResponse:
    """Check inference service health."""
    try:
        is_healthy, issues = await triton.health_checker.check_health()
        
        return InferenceHealthResponse(
            healthy=is_healthy,
            gpu_available=True,  # Placeholder
            triton_connected=triton._connected,
            models_loaded=triton.health_checker.loaded_models,
            issues=issues,
            checked_at=datetime.now(timezone.utc),
        )
        
    except Exception as e:
        logger.error("inference_health_check_failed", error=str(e))
        return InferenceHealthResponse(
            healthy=False,
            gpu_available=False,
            triton_connected=False,
            models_loaded=[],
            issues=[str(e)],
            checked_at=datetime.now(timezone.utc),
        )


@router.get(
    "/inference/stats",
    response_model=InferenceStats,
    summary="Get inference statistics",
    description="Get detailed inference statistics including GPU utilization.",
)
async def inference_stats(
    triton: TritonClient = Depends(get_triton),
    _api_key: str = Depends(verify_api_key),
) -> InferenceStats:
    """Get inference statistics."""
    try:
        stats = await triton.get_stats()
        return stats
        
    except Exception as e:
        logger.error("inference_stats_failed", error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get inference stats: {str(e)}",
        )


# =============================================================================
# COMBINED INTELLIGENCE ENDPOINT
# =============================================================================

class Phase3IntelligenceResponse(BaseModel):
    """Combined Phase 3 intelligence response."""
    user_id: int
    timestamp: datetime
    
    # Delay prediction
    delay_prediction: Optional[DelayPrediction] = None
    delay_available: bool = False
    
    # Threat prediction
    threat_prediction: Optional[ThreatPrediction] = None
    threat_available: bool = False
    
    # Drift detection
    drift_result: Optional[DriftResult] = None
    drift_available: bool = False
    
    # Summary
    overall_risk_level: str = "unknown"
    recommendations: List[str] = Field(default_factory=list)


@router.get(
    "/intelligence/{user_id}",
    response_model=Phase3IntelligenceResponse,
    summary="Get combined Phase 3 intelligence",
    description="Get all Phase 3 predictions and detections for a user.",
)
async def get_phase3_intelligence(
    user_id: int,
    include_delay: bool = Query(False, description="Include delay prediction"),
    origin_lat: Optional[float] = Query(None, description="Origin lat for delay"),
    origin_lon: Optional[float] = Query(None, description="Origin lon for delay"),
    dest_lat: Optional[float] = Query(None, description="Dest lat for delay"),
    dest_lon: Optional[float] = Query(None, description="Dest lon for delay"),
    threat_service: ThreatPredictor = Depends(get_threat_service),
    drift_service: BehavioralDriftDetector = Depends(get_drift_service),
    routing_service: PredictiveRoutingService = Depends(get_routing_service),
    _api_key: str = Depends(verify_api_key),
) -> Phase3IntelligenceResponse:
    """Get combined Phase 3 intelligence for a user."""
    now = datetime.now(timezone.utc)
    
    response = Phase3IntelligenceResponse(
        user_id=user_id,
        timestamp=now,
    )
    
    recommendations = []
    
    # Get threat prediction
    try:
        threat = await threat_service.predict(
            user_id=user_id,
            risk_scores=[],
            anomalies=[],
            spoofing_results=[],
            drift_detected=False,
        )
        response.threat_prediction = threat
        response.threat_available = True
        recommendations.extend(threat.recommendations)
    except Exception as e:
        logger.warning("threat_prediction_skipped", error=str(e))
    
    # Get delay prediction if coordinates provided
    if include_delay and all([origin_lat, origin_lon, dest_lat, dest_lon]):
        try:
            delay = await routing_service.predict_delay(
                user_id=user_id,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
                destination_lat=dest_lat,
                destination_lon=dest_lon,
            )
            response.delay_prediction = delay
            response.delay_available = True
            
            if delay.predicted_delay_minutes > 15:
                recommendations.append(
                    f"Expected delay: {delay.predicted_delay_minutes:.0f} minutes"
                )
        except Exception as e:
            logger.warning("delay_prediction_skipped", error=str(e))
    
    # Determine overall risk level
    if response.threat_prediction:
        response.overall_risk_level = response.threat_prediction.threat_level.value
    
    response.recommendations = recommendations
    
    return response


# =============================================================================
# COST MONITORING ENDPOINTS
# =============================================================================

@router.get(
    "/cost/gpu",
    summary="Get GPU cost status",
    description="Get current GPU usage, spend, and NVIDIA requirement status.",
)
async def get_gpu_cost_status(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Get GPU cost monitoring status."""
    from app.utils.cost_monitor import get_cost_monitor
    
    monitor = get_cost_monitor()
    return monitor.get_summary()


@router.get(
    "/cost/api-quotas",
    summary="Get external API quota status",
    description="Get usage and quota status for OpenWeatherMap and TomTom APIs.",
)
async def get_api_quota_status(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Get external API quota status."""
    from app.utils.api_quota import get_quota_monitor
    
    monitor = get_quota_monitor()
    return monitor.get_summary()


@router.post(
    "/cost/gpu/session/start",
    summary="Start GPU session tracking",
    description="Start tracking a new GPU usage session for cost monitoring.",
)
async def start_gpu_session(
    instance_type: str = Query("g5.xlarge", description="GPU instance type"),
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """Start GPU session tracking."""
    from app.utils.cost_monitor import get_cost_monitor
    
    monitor = get_cost_monitor()
    
    # Check if allowed
    allowed, reason = monitor.check_can_use_gpu()
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail=f"GPU usage blocked: {reason}",
        )
    
    monitor.start_session(instance_type)
    
    return {
        "status": "started",
        "instance_type": instance_type,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@router.post(
    "/cost/gpu/session/end",
    summary="End GPU session tracking",
    description="End the current GPU usage session.",
)
async def end_gpu_session(
    _api_key: str = Depends(verify_api_key),
) -> dict:
    """End GPU session tracking."""
    from app.utils.cost_monitor import get_cost_monitor
    
    monitor = get_cost_monitor()
    record = monitor.end_session()
    
    if record:
        return {
            "status": "ended",
            "duration_hours": round(record.duration_hours, 2),
            "cost_usd": round(record.cost, 2),
        }
    else:
        return {
            "status": "no_active_session",
        }
