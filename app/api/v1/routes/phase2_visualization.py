"""
Phase 2 Visualization & Analytics Endpoints
=============================================

OPTIONAL production-grade endpoints for:
- Visualization (maps, timelines)
- Debugging and inspection
- Client demos
- Dashboard integration
- Investor/stakeholder proof of intelligence

These endpoints summarize existing Phase 2 intelligence.
They do NOT recompute anything.

NON-NEGOTIABLE:
- Read-only aggregation
- No modification to existing Phase 2 logic
- No new ML models or training
- No GPU required
- Cache-friendly
- Backend-only, battery-safe
"""

from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.api.deps import get_phase2_svc, verify_api_key
from app.services.phase2_service import Phase2Service
from app.models.phase2_schemas import RiskLevel

router = APIRouter()


# =============================================================================
# RESPONSE MODELS
# =============================================================================

class TimelineLocation(BaseModel):
    """Location point in timeline."""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")


class TimelinePoint(BaseModel):
    """
    Single point in the behavioral timeline.
    
    UI-ready format for visualization.
    """
    timestamp: datetime = Field(..., description="Event timestamp")
    location: TimelineLocation = Field(..., description="Geographic location")
    confidence: float = Field(..., ge=0, le=1, description="Phase 1 confidence score")
    risk_score: float = Field(..., ge=0, le=1, description="Phase 2 risk score")
    risk_level: str = Field(..., description="Human-readable risk level")
    anomalies: List[str] = Field(default_factory=list, description="Detected anomaly types")
    spoofing: bool = Field(..., description="Spoofing likelihood exceeds threshold")
    spoofing_likelihood: float = Field(..., ge=0, le=1, description="Spoofing probability")
    is_trusted: bool = Field(..., description="Point is trusted for downstream use")
    explanation: str = Field(..., description="Human-readable explanation")


class BehavioralTimelineResponse(BaseModel):
    """
    Complete behavioral timeline for a user.
    
    Ordered chronologically, ready for map/timeline visualization.
    """
    success: bool = Field(..., description="Query success")
    user_id: int = Field(..., description="User identifier")
    generated_at: datetime = Field(..., description="Response generation time")
    session_id: Optional[str] = Field(None, description="Session filter if applied")
    mode: str = Field(
        default="fast",
        description="Processing mode: 'fast' (confidence-based) or 'full_phase2' (ML-based)"
    )
    
    # Timeline data
    total_points: int = Field(..., ge=0, description="Total points in timeline")
    timeline: List[TimelinePoint] = Field(
        default_factory=list,
        description="Chronologically ordered timeline points"
    )
    
    # Summary statistics
    time_range: Dict[str, str] = Field(
        default_factory=dict,
        description="Start and end times"
    )
    avg_confidence: float = Field(..., ge=0, le=1, description="Average confidence")
    avg_risk: float = Field(..., ge=0, le=1, description="Average risk score")
    anomaly_count: int = Field(..., ge=0, description="Total anomalous points")
    spoofing_count: int = Field(..., ge=0, description="Points with spoofing detected")


class RiskDistribution(BaseModel):
    """Risk score distribution buckets."""
    minimal: int = Field(default=0, description="Risk < 0.1")
    low: int = Field(default=0, description="Risk 0.1-0.25")
    moderate: int = Field(default=0, description="Risk 0.25-0.45")
    elevated: int = Field(default=0, description="Risk 0.45-0.65")
    high: int = Field(default=0, description="Risk 0.65-0.85")
    critical: int = Field(default=0, description="Risk >= 0.85")


class ConfidenceDistribution(BaseModel):
    """Confidence score distribution buckets."""
    excellent: int = Field(default=0, description="Confidence >= 0.9")
    good: int = Field(default=0, description="Confidence 0.75-0.9")
    moderate: int = Field(default=0, description="Confidence 0.5-0.75")
    low: int = Field(default=0, description="Confidence 0.25-0.5")
    unreliable: int = Field(default=0, description="Confidence < 0.25")


class AnomalyBreakdown(BaseModel):
    """Breakdown of anomaly types."""
    total_anomalies: int = Field(default=0, description="Total anomalous points")
    by_type: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by anomaly type"
    )
    rate: float = Field(default=0.0, ge=0, le=1, description="Anomaly rate")


class SpoofingBreakdown(BaseModel):
    """Breakdown of spoofing indicators."""
    total_spoofing_events: int = Field(default=0, description="Points with spoofing")
    avg_likelihood: float = Field(default=0.0, ge=0, le=1, description="Average likelihood")
    max_likelihood: float = Field(default=0.0, ge=0, le=1, description="Maximum likelihood")
    indicators: Dict[str, int] = Field(
        default_factory=dict,
        description="Count by indicator type"
    )


class SessionMetrics(BaseModel):
    """Per-session summary metrics."""
    session_id: str = Field(..., description="Session identifier")
    start_time: datetime = Field(..., description="Session start")
    end_time: datetime = Field(..., description="Session end")
    duration_seconds: float = Field(..., ge=0, description="Session duration")
    total_points: int = Field(..., ge=0, description="Points in session")
    risk_score: float = Field(..., ge=0, le=1, description="Session risk score")
    risk_level: str = Field(..., description="Risk level")
    trust_rate: float = Field(..., ge=0, le=1, description="Percentage trusted")
    is_valid: bool = Field(..., description="Session appears valid")


class BehavioralMetricsResponse(BaseModel):
    """
    Comprehensive behavioral metrics for dashboards.
    
    Aggregates Phase 2 intelligence into summary statistics.
    """
    success: bool = Field(..., description="Query success")
    user_id: int = Field(..., description="User identifier")
    generated_at: datetime = Field(..., description="Response generation time")
    
    # Observation window
    observation_window: str = Field(..., description="Window description")
    start_time: Optional[datetime] = Field(None, description="Window start")
    end_time: Optional[datetime] = Field(None, description="Window end")
    
    # Core metrics
    total_sessions: int = Field(..., ge=0, description="Sessions analyzed")
    total_points: int = Field(..., ge=0, description="Total location points")
    
    # Risk metrics
    avg_risk_score: float = Field(..., ge=0, le=1, description="Average risk")
    max_risk_score: float = Field(..., ge=0, le=1, description="Maximum risk")
    risk_distribution: RiskDistribution = Field(..., description="Risk buckets")
    
    # Confidence metrics
    avg_confidence: float = Field(..., ge=0, le=1, description="Average confidence")
    confidence_distribution: ConfidenceDistribution = Field(
        ...,
        description="Confidence buckets"
    )
    
    # Anomaly metrics
    anomalies: AnomalyBreakdown = Field(..., description="Anomaly breakdown")
    
    # Spoofing metrics
    spoofing: SpoofingBreakdown = Field(..., description="Spoofing breakdown")
    
    # Trust metrics
    trusted_points: int = Field(..., ge=0, description="Trusted points")
    trust_rate: float = Field(..., ge=0, le=1, description="Overall trust rate")
    
    # Session summaries
    sessions: List[SessionMetrics] = Field(
        default_factory=list,
        description="Per-session summaries"
    )
    
    # Profile status
    profile_status: Dict[str, Any] = Field(
        default_factory=dict,
        description="Behavioral profile status"
    )


class HeatmapPoint(BaseModel):
    """Single point for heatmap visualization."""
    lat: float = Field(..., description="Latitude")
    lon: float = Field(..., description="Longitude")
    weight: float = Field(..., ge=0, le=1, description="Heat weight (0-1)")
    category: str = Field(..., description="Point category")


class HeatmapResponse(BaseModel):
    """Risk/anomaly heatmap data for map visualization."""
    success: bool = Field(..., description="Query success")
    user_id: int = Field(..., description="User identifier")
    map_type: str = Field(..., description="Heatmap type (risk/anomaly/spoofing)")
    points: List[HeatmapPoint] = Field(
        default_factory=list,
        description="Heatmap points"
    )
    bounds: Dict[str, float] = Field(
        default_factory=dict,
        description="Geographic bounds"
    )


# =============================================================================
# TIMELINE ENDPOINT (HIGH PRIORITY)
# =============================================================================

@router.get(
    "/timeline/{user_id}",
    response_model=BehavioralTimelineResponse,
    summary="Get behavioral timeline",
    description="Return chronological, UI-ready timeline of movement intelligence",
    tags=["Phase 2 - Visualization"],
)
async def get_behavioral_timeline(
    user_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session"),
    limit: int = Query(0, ge=0, description="Maximum points (0 = no limit)"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Minimum confidence"),
    include_trusted_only: bool = Query(False, description="Only trusted points"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> BehavioralTimelineResponse:
    """
    Get behavioral timeline for visualization.
    
    Returns a chronologically ordered timeline of location intelligence,
    ready for map/timeline visualization in dashboards.
    
    FAST MODE: Uses Phase 1 outputs directly with confidence as risk proxy.
    This returns data in ~7 seconds instead of 7+ minutes.
    
    Args:
        user_id: User identifier
        session_id: Optional session filter
        limit: Maximum points to return
        min_confidence: Filter by minimum confidence
        include_trusted_only: Only return trusted points
        
    Returns:
        BehavioralTimelineResponse with ordered timeline
    """
    try:
        # Check if full Phase 2 data is available (user clicked "Process Full Phase 2")
        if service.has_cached_phase2_data(user_id):
            viz_data = service.get_visualization_data_full(
                user_id=user_id,
                limit=limit,
                min_confidence=min_confidence,
            )
            if viz_data is None:
                # Fall back to fast mode if full data retrieval failed
                viz_data = service.get_visualization_data_fast(
                    user_id=user_id,
                    limit=limit,
                    min_confidence=min_confidence,
                )
        else:
            # Use FAST visualization method - skips expensive session processing
            viz_data = service.get_visualization_data_fast(
                user_id=user_id,
                limit=limit,
                min_confidence=min_confidence,
            )
        
        if not viz_data["points"]:
            return BehavioralTimelineResponse(
                success=True,
                user_id=user_id,
                generated_at=datetime.now(timezone.utc),
                session_id=session_id,
                total_points=0,
                timeline=[],
                time_range={},
                avg_confidence=0.0,
                avg_risk=0.0,
                anomaly_count=0,
                spoofing_count=0,
            )
        
        # Convert to timeline points
        timeline_points: List[TimelinePoint] = []
        
        for point in viz_data["points"]:
            # Apply session filter
            if session_id and point.get("session_id") != session_id:
                continue
            
            # Apply trusted filter
            if include_trusted_only and not point["is_trusted"]:
                continue
            
            timeline_points.append(TimelinePoint(
                timestamp=datetime.fromisoformat(point["timestamp"]),
                location=TimelineLocation(
                    lat=point["lat"],
                    lon=point["lon"],
                ),
                confidence=point["confidence"],
                risk_score=point["risk_score"],
                risk_level=point["risk_level"],
                anomalies=point["anomalies"],
                spoofing=point["spoofing"],
                spoofing_likelihood=point["spoofing_likelihood"],
                is_trusted=point["is_trusted"],
                explanation=point["explanation"],
            ))
        
        # Already sorted from the fast method
        # Apply limit only if specified
        if limit > 0:
            timeline_points = timeline_points[:limit]
        
        # Use pre-computed stats from fast method
        stats = viz_data["stats"]
        time_range = viz_data.get("time_range", {})
        if time_range.get("start"):
            time_range = {
                "start": time_range["start"],
                "end": time_range["end"],
            }
        else:
            time_range = {}
        
        return BehavioralTimelineResponse(
            success=True,
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            session_id=session_id,
            mode=viz_data.get("mode", "fast"),
            total_points=len(timeline_points),
            timeline=timeline_points,
            time_range=time_range,
            avg_confidence=round(stats.get("avg_confidence", 0.0), 4),
            avg_risk=round(stats.get("avg_risk", 0.0), 4),
            anomaly_count=stats.get("anomaly_count", 0),
            spoofing_count=stats.get("spoofing_count", 0),
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Timeline generation failed: {str(e)}",
        )


# =============================================================================
# METRICS ENDPOINT (MEDIUM PRIORITY)
# =============================================================================

@router.get(
    "/metrics/{user_id}",
    response_model=BehavioralMetricsResponse,
    summary="Get behavioral metrics",
    description="Return summary statistics for dashboards and reporting",
    tags=["Phase 2 - Analytics"],
)
async def get_behavioral_metrics(
    user_id: int,
    window: str = Query(
        "all",
        description="Time window: 'all', 'last_7_days', 'last_30_days', 'last_90_days'"
    ),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> BehavioralMetricsResponse:
    """
    Get behavioral metrics for dashboards.
    
    Returns comprehensive summary statistics suitable for
    reporting and dashboard integration.
    
    FAST MODE: Uses Phase 1 outputs directly with confidence as risk proxy.
    
    Args:
        user_id: User identifier
        window: Time window for filtering
        
    Returns:
        BehavioralMetricsResponse with comprehensive metrics
    """
    try:
        # Check if full Phase 2 data is available
        if service.has_cached_phase2_data(user_id):
            viz_data = service.get_visualization_data_full(
                user_id=user_id,
                limit=0,  # No limit - get all points
                min_confidence=0.0,
            )
            if viz_data is None:
                viz_data = service.get_visualization_data_fast(
                    user_id=user_id,
                    limit=0,  # No limit
                    min_confidence=0.0,
                )
        else:
            # Use FAST visualization method - skips expensive session processing
            viz_data = service.get_visualization_data_fast(
                user_id=user_id,
                limit=0,  # No limit - get all points for accurate metrics
                min_confidence=0.0,
            )
        
        stats = viz_data.get("stats", {})
        
        # Determine window description
        now = datetime.now(timezone.utc)
        if window == "last_7_days":
            window_desc = "last_7_days"
        elif window == "last_30_days":
            window_desc = "last_30_days"
        elif window == "last_90_days":
            window_desc = "last_90_days"
        else:
            window_desc = "all_time"
        
        if not viz_data["points"]:
            return _empty_metrics_response(user_id, window_desc, None)
        
        # Get stats from fast method
        total_points = viz_data["total_points"]
        avg_confidence = stats.get("avg_confidence", 0.0)
        avg_risk = stats.get("avg_risk", 0.0)
        
        # Count trusted points
        trusted_points = sum(1 for p in viz_data["points"] if p["is_trusted"])
        
        # Risk distribution from stats
        risk_stats = stats.get("risk_distribution", {})
        risk_dist = RiskDistribution(
            minimal=risk_stats.get("minimal", 0),
            low=risk_stats.get("low", 0),
            moderate=risk_stats.get("moderate", 0),
            elevated=risk_stats.get("elevated", 0),
            high=risk_stats.get("high", 0),
            critical=risk_stats.get("critical", 0),
        )
        
        # Confidence distribution from stats
        conf_stats = stats.get("confidence_distribution", {})
        conf_dist = ConfidenceDistribution(
            excellent=conf_stats.get("excellent", 0),
            good=conf_stats.get("good", 0),
            moderate=conf_stats.get("moderate", 0),
            low=conf_stats.get("low", 0),
            unreliable=conf_stats.get("unreliable", 0),
        )
        
        # Anomaly count from stats (low confidence points are flagged as anomalies in fast mode)
        total_anomalies = stats.get("anomaly_count", 0)
        anomaly_types: Dict[str, int] = {"low_confidence": total_anomalies}
        anomaly_rate = total_anomalies / total_points if total_points > 0 else 0.0
        
        # Spoofing metrics from stats
        spoofing_count = stats.get("spoofing_count", 0)
        max_spoofing = 1.0 if spoofing_count > 0 else 0.0
        avg_spoofing = spoofing_count / total_points if total_points > 0 else 0.0
        
        spoofing_indicators: Dict[str, int] = {}
        if spoofing_count > 0:
            spoofing_indicators["very_low_confidence"] = spoofing_count
        
        # Session summaries (create from grouped points)
        total_sessions = stats.get("total_sessions", 0)
        session_metrics: List[SessionMetrics] = []  # Simplified - no per-session details in fast mode
        
        # Profile status - not available in fast mode
        profile_status = {}
        
        # Time range from fast method
        time_range = viz_data.get("time_range", {})
        start_time = None
        end_time = None
        if time_range.get("start"):
            start_time = datetime.fromisoformat(time_range["start"])
            end_time = datetime.fromisoformat(time_range["end"])
        
        # Max risk is 1 - min confidence
        max_risk = max((p["risk_score"] for p in viz_data["points"]), default=0.0)
        
        return BehavioralMetricsResponse(
            success=True,
            user_id=user_id,
            generated_at=datetime.now(timezone.utc),
            observation_window=window_desc,
            start_time=start_time,
            end_time=end_time,
            total_sessions=total_sessions,
            total_points=total_points,
            avg_risk_score=round(avg_risk, 4),
            max_risk_score=round(max_risk, 4),
            risk_distribution=risk_dist,
            avg_confidence=round(avg_confidence, 4),
            confidence_distribution=conf_dist,
            anomalies=AnomalyBreakdown(
                total_anomalies=total_anomalies,
                by_type=anomaly_types,
                rate=round(anomaly_rate, 4),
            ),
            spoofing=SpoofingBreakdown(
                total_spoofing_events=spoofing_count,
                avg_likelihood=round(avg_spoofing, 4),
                max_likelihood=round(max_spoofing, 4),
                indicators=spoofing_indicators,
            ),
            trusted_points=trusted_points,
            trust_rate=round(trusted_points / total_points, 4) if total_points > 0 else 0.0,
            sessions=session_metrics,
            profile_status=profile_status,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Metrics generation failed: {str(e)}",
        )


# =============================================================================
# HEATMAP ENDPOINT (BONUS)
# =============================================================================

@router.get(
    "/heatmap/{user_id}",
    response_model=HeatmapResponse,
    summary="Get risk heatmap",
    description="Return geographic heatmap data for map visualization",
    tags=["Phase 2 - Visualization"],
)
async def get_risk_heatmap(
    user_id: int,
    map_type: str = Query(
        "risk",
        description="Heatmap type: 'risk', 'anomaly', 'spoofing'"
    ),
    session_id: Optional[str] = Query(None, description="Filter by session"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> HeatmapResponse:
    """
    Get heatmap data for map visualization.
    
    FAST MODE: Uses Phase 1 outputs directly.
    
    Args:
        user_id: User identifier
        map_type: Type of heatmap (risk/anomaly/spoofing)
        session_id: Optional session filter
        
    Returns:
        HeatmapResponse with weighted points
    """
    try:
        # Check if full Phase 2 data is available
        if service.has_cached_phase2_data(user_id):
            viz_data = service.get_visualization_data_full(
                user_id=user_id,
                limit=0,  # No limit
                min_confidence=0.0,
            )
            if viz_data is None:
                viz_data = service.get_visualization_data_fast(
                    user_id=user_id,
                    limit=0,  # No limit
                    min_confidence=0.0,
                )
        else:
            # Use FAST visualization method
            viz_data = service.get_visualization_data_fast(
                user_id=user_id,
                limit=0,  # No limit - get all points
                min_confidence=0.0,
            )
        
        heatmap_points: List[HeatmapPoint] = []
        min_lat, max_lat = 90.0, -90.0
        min_lon, max_lon = 180.0, -180.0
        
        for point in viz_data["points"]:
            # Apply session filter
            if session_id and point.get("session_id") != session_id:
                continue
            
            lat = point["lat"]
            lon = point["lon"]
            
            # Update bounds
            min_lat = min(min_lat, lat)
            max_lat = max(max_lat, lat)
            min_lon = min(min_lon, lon)
            max_lon = max(max_lon, lon)
            
            # Calculate weight based on map type
            if map_type == "risk":
                weight = point["risk_score"]
                category = point["risk_level"]
            elif map_type == "anomaly":
                weight = 1.0 if point["anomalies"] else 0.1
                category = "anomalous" if point["anomalies"] else "normal"
            elif map_type == "spoofing":
                weight = point["spoofing_likelihood"]
                category = "spoofed" if point["spoofing"] else "genuine"
            else:
                weight = 0.5
                category = "unknown"
            
            heatmap_points.append(HeatmapPoint(
                lat=lat,
                lon=lon,
                weight=round(weight, 4),
                category=category,
            ))
        
        return HeatmapResponse(
            success=True,
            user_id=user_id,
            map_type=map_type,
            points=heatmap_points,
            bounds={
                "min_lat": min_lat if heatmap_points else 0.0,
                "max_lat": max_lat if heatmap_points else 0.0,
                "min_lon": min_lon if heatmap_points else 0.0,
                "max_lon": max_lon if heatmap_points else 0.0,
            },
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Heatmap generation failed: {str(e)}",
        )


# =============================================================================
# SUMMARY DASHBOARD ENDPOINT (BONUS)
# =============================================================================

@router.get(
    "/dashboard/{user_id}",
    response_model=dict,
    summary="Get dashboard summary",
    description="Return all-in-one dashboard data",
    tags=["Phase 2 - Analytics"],
)
async def get_dashboard_summary(
    user_id: int,
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get all-in-one dashboard summary.
    
    FAST MODE: Uses Phase 1 outputs directly.
    
    Args:
        user_id: User identifier
        
    Returns:
        Consolidated dashboard data
    """
    try:
        # Check if full Phase 2 data is available
        if service.has_cached_phase2_data(user_id):
            viz_data = service.get_visualization_data_full(
                user_id=user_id,
                limit=0,  # No limit
                min_confidence=0.0,
            )
            if viz_data is None:
                viz_data = service.get_visualization_data_fast(
                    user_id=user_id,
                    limit=0,  # No limit
                    min_confidence=0.0,
                )
        else:
            # Use FAST visualization method
            viz_data = service.get_visualization_data_fast(
                user_id=user_id,
                limit=0,  # No limit
                min_confidence=0.0,
            )
        
        if not viz_data["points"]:
            return {
                "success": True,
                "user_id": user_id,
                "status": "no_data",
                "message": "No data available for this user",
            }
        
        stats = viz_data["stats"]
        
        # Calculate key metrics from fast stats
        total_points = viz_data["total_points"]
        trusted_points = sum(1 for p in viz_data["points"] if p["is_trusted"])
        total_anomalies = stats.get("anomaly_count", 0)
        avg_risk = stats.get("avg_risk", 0.0)
        
        # Count high risk points
        risk_dist = stats.get("risk_distribution", {})
        high_risk_count = risk_dist.get("high", 0) + risk_dist.get("critical", 0)
        
        spoofing_count = stats.get("spoofing_count", 0)
        
        return {
            "success": True,
            "user_id": user_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            
            # Top-level fields for easy access
            "total_points": total_points,
            "total_sessions": stats.get("total_sessions", 0),
            "routes_count": 0,  # Not available in fast mode
            "stops_count": 0,   # Not available in fast mode
            "avg_risk": round(avg_risk, 4),
            "avg_confidence": round(stats.get("avg_confidence", 0.0), 4),
            
            # Health score (inverse of risk)
            "health_score": round((1.0 - avg_risk) * 100, 1),
            "health_status": _get_health_status(avg_risk),
            
            # Key metrics
            "metrics": {
                "total_sessions": stats.get("total_sessions", 0),
                "total_points": total_points,
                "trusted_points": trusted_points,
                "trust_rate": round(trusted_points / total_points * 100, 1) if total_points > 0 else 0.0,
                "avg_risk_score": round(avg_risk * 100, 1),
                "high_risk_points": high_risk_count,
                "total_anomalies": total_anomalies,
                "spoofing_events": spoofing_count,
            },
            
            # Profile status (not available in fast mode)
            "profile": {
                "is_mature": False,
                "confidence": round(stats.get("avg_confidence", 0.0) * 100, 1),
                "routes_learned": 0,
                "stops_learned": 0,
            },
            
            # Recent sessions (simplified in fast mode)
            "recent_sessions": [],
            
            # Concerns based on stats
            "concerns": _generate_concerns_fast(stats),
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard generation failed: {str(e)}",
        )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _generate_point_explanation(
    confidence: float,
    risk_score: float,
    anomalies: List[str],
    spoofing: bool,
    is_trusted: bool,
) -> str:
    """Generate human-readable explanation for a timeline point."""
    parts = []
    
    # Confidence assessment
    if confidence >= 0.9:
        parts.append("Excellent data quality")
    elif confidence >= 0.75:
        parts.append("Good data quality")
    elif confidence >= 0.5:
        parts.append("Moderate data quality")
    else:
        parts.append("Low data quality")
    
    # Risk assessment
    if risk_score < 0.25:
        parts.append("normal movement pattern")
    elif risk_score < 0.5:
        parts.append("minor deviations detected")
    elif risk_score < 0.75:
        parts.append("significant behavioral deviation")
    else:
        parts.append("high-risk movement detected")
    
    # Anomaly note
    if anomalies:
        parts.append(f"({', '.join(anomalies[:2])})")
    
    # Spoofing note
    if spoofing:
        parts.append("⚠️ Potential spoofing")
    
    # Trust note
    if not is_trusted:
        parts.append("(untrusted)")
    
    return ". ".join(parts[:3])


def _empty_metrics_response(
    user_id: int,
    window_desc: str,
    profile: Optional[Any],
) -> BehavioralMetricsResponse:
    """Create empty metrics response."""
    return BehavioralMetricsResponse(
        success=True,
        user_id=user_id,
        generated_at=datetime.now(timezone.utc),
        observation_window=window_desc,
        start_time=None,
        end_time=None,
        total_sessions=0,
        total_points=0,
        avg_risk_score=0.0,
        max_risk_score=0.0,
        risk_distribution=RiskDistribution(),
        avg_confidence=0.0,
        confidence_distribution=ConfidenceDistribution(),
        anomalies=AnomalyBreakdown(),
        spoofing=SpoofingBreakdown(),
        trusted_points=0,
        trust_rate=0.0,
        sessions=[],
        profile_status={
            "is_mature": profile.is_mature if profile else False,
            "confidence": profile.profile_confidence if profile else 0.0,
        } if profile else {},
    )


def _get_health_status(avg_risk: float) -> str:
    """Get health status label from risk score."""
    if avg_risk < 0.15:
        return "excellent"
    elif avg_risk < 0.3:
        return "good"
    elif avg_risk < 0.5:
        return "fair"
    elif avg_risk < 0.7:
        return "concerning"
    else:
        return "critical"


def _generate_concerns(sessions: List[Any], profile: Optional[Any]) -> List[str]:
    """Generate list of concerns for dashboard."""
    concerns = []
    
    if not profile or not profile.is_mature:
        concerns.append("Profile not yet mature - more data needed for reliable analysis")
    
    high_risk = sum(
        1 for s in sessions
        if s.risk_score.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
    )
    if high_risk > 0:
        concerns.append(f"{high_risk} high-risk sessions detected")
    
    spoofing = sum(1 for s in sessions if s.spoofing_likelihood >= 0.5)
    if spoofing > 0:
        concerns.append(f"{spoofing} sessions with potential spoofing")
    
    total_anomalies = sum(s.anomaly_count for s in sessions)
    total_points = sum(s.total_points for s in sessions)
    if total_points > 0 and total_anomalies / total_points > 0.1:
        concerns.append(f"High anomaly rate ({total_anomalies / total_points * 100:.1f}%)")
    
    if not concerns:
        concerns.append("No significant concerns")
    
    return concerns


def _generate_concerns_fast(stats: Dict[str, Any]) -> List[str]:
    """Generate list of concerns from fast stats."""
    concerns = []
    
    # Check risk distribution
    risk_dist = stats.get("risk_distribution", {})
    high_risk = risk_dist.get("high", 0) + risk_dist.get("critical", 0)
    if high_risk > 0:
        concerns.append(f"{high_risk} high-risk points detected")
    
    # Check spoofing
    spoofing_count = stats.get("spoofing_count", 0)
    if spoofing_count > 0:
        concerns.append(f"{spoofing_count} points with potential spoofing")
    
    # Check anomaly rate
    anomaly_count = stats.get("anomaly_count", 0)
    total_sessions = stats.get("total_sessions", 0)
    if total_sessions > 0:
        # Rough anomaly rate
        avg_confidence = stats.get("avg_confidence", 0.5)
        if avg_confidence < 0.7:
            concerns.append(f"Below-average confidence ({avg_confidence*100:.1f}%)")
    
    if not concerns:
        concerns.append("No significant concerns")
    
    return concerns
