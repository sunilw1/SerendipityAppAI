"""
Safety API Routes - Crash, Fall, & Battery Monitoring
======================================================

REST API endpoints for safety detection services:
- Crash detection from accelerometer data
- Fall detection from accelerometer data
- Battery level monitoring and alerts
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.core.logging import get_logger
from app.api.deps import verify_api_key
from app.models.phase3_schemas import (
    CrashDetectionRequest,
    CrashDetectionResponse,
    FallDetectionRequest,
    FallDetectionResponse,
    BatteryReportRequest,
    BatteryAnalysisResponse,
)
from app.services.safety_detection import (
    CrashDetector,
    FallDetector,
    BatteryMonitor,
    get_crash_detector,
    get_fall_detector,
    get_battery_monitor,
)

logger = get_logger(__name__)

router = APIRouter(tags=["Phase 3 - Safety Detection"])


# =============================================================================
# CRASH DETECTION
# =============================================================================

@router.post(
    "/detect/crash",
    response_model=CrashDetectionResponse,
    summary="Detect vehicle crash",
    description="""
    Analyze accelerometer data to detect vehicle crashes.
    
    Detection algorithm:
    - Monitors for sudden G-force spikes (>4G = potential crash)
    - Considers vehicle speed at time of impact
    - Checks for post-impact motion (stationary = higher severity)
    - Classifies severity: minor, moderate, severe, critical
    
    Triggers alerts and notifies emergency contacts for moderate+ crashes.
    """,
)
async def detect_crash(
    request: CrashDetectionRequest,
    detector: CrashDetector = Depends(get_crash_detector),
    _api_key: str = Depends(verify_api_key),
) -> CrashDetectionResponse:
    """Detect vehicle crash from accelerometer data."""
    try:
        logger.info(
            "crash_detection_request",
            user_id=request.user_id,
            readings_count=len(request.readings),
            speed=request.speed_kmh,
        )

        result = detector.detect(
            user_id=request.user_id,
            readings=request.readings,
            speed_kmh=request.speed_kmh,
            lat=request.lat,
            lon=request.lon,
        )

        return CrashDetectionResponse(success=True, result=result, error=None)

    except Exception as e:
        logger.error("crash_detection_failed", user_id=request.user_id, error=str(e))
        return CrashDetectionResponse(success=False, result=None, error=str(e))


# =============================================================================
# FALL DETECTION
# =============================================================================

@router.post(
    "/detect/fall",
    response_model=FallDetectionResponse,
    summary="Detect human fall",
    description="""
    Analyze accelerometer data to detect human falls.
    
    Detection algorithm:
    - Phase 1: Detect free-fall (G-force drops below 0.4G)
    - Phase 2: Detect impact spike after free-fall (>3G)
    - Phase 3: Check if user is stationary after impact
    - Age-aware risk adjustment for elderly users
    
    Triggers alerts and notifies emergency contacts for moderate+ falls.
    """,
)
async def detect_fall(
    request: FallDetectionRequest,
    detector: FallDetector = Depends(get_fall_detector),
    _api_key: str = Depends(verify_api_key),
) -> FallDetectionResponse:
    """Detect fall from accelerometer data."""
    try:
        logger.info(
            "fall_detection_request",
            user_id=request.user_id,
            readings_count=len(request.readings),
        )

        result = detector.detect(
            user_id=request.user_id,
            readings=request.readings,
            user_age=request.user_age,
            lat=request.lat,
            lon=request.lon,
        )

        return FallDetectionResponse(success=True, result=result, error=None)

    except Exception as e:
        logger.error("fall_detection_failed", user_id=request.user_id, error=str(e))
        return FallDetectionResponse(success=False, result=None, error=str(e))


# =============================================================================
# BATTERY MONITORING
# =============================================================================

@router.post(
    "/battery/report",
    response_model=BatteryAnalysisResponse,
    summary="Report battery status",
    description="""
    Report device battery level for safety monitoring.
    
    Analysis:
    - Classifies battery status: healthy, low (<20%), critical (<10%)
    - Estimates time remaining based on drain rate
    - Triggers safety alerts when battery is low/critical
    - Recommends power-saving actions for tracking continuity
    
    Critical battery means safety tracking may be interrupted.
    """,
)
async def report_battery(
    request: BatteryReportRequest,
    monitor: BatteryMonitor = Depends(get_battery_monitor),
    _api_key: str = Depends(verify_api_key),
) -> BatteryAnalysisResponse:
    """Analyze battery status and generate alerts."""
    try:
        logger.info(
            "battery_report",
            user_id=request.user_id,
            level=request.battery_level,
            charging=request.is_charging,
        )

        result = monitor.analyze(
            user_id=request.user_id,
            battery_level=request.battery_level,
            is_charging=request.is_charging,
            drain_rate_per_hour=request.drain_rate_per_hour,
            battery_temperature=request.battery_temperature,
        )

        return BatteryAnalysisResponse(success=True, result=result, error=None)

    except Exception as e:
        logger.error("battery_analysis_failed", user_id=request.user_id, error=str(e))
        return BatteryAnalysisResponse(success=False, result=None, error=str(e))
