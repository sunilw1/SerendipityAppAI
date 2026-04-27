"""
Phase 2 API Routes
===================

Endpoints for Phase 2 behavioral intelligence and threat detection.

Provides:
- Risk score + confidence
- Anomaly flags + explanation
- Spoofing detection
- Validated movement signals
- Behavioral profiles

All output is intelligence-enriched with explanations.
Raw data is never exposed directly.

NOTE: These APIs return SCORES + EXPLANATIONS, not hard decisions.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_phase2_svc, verify_api_key
from app.services.phase2_service import Phase2Service
from app.models.phase2_schemas import (
    RiskAnalysisRequest,
    RiskAnalysisResponse,
    AnomalyDetectionRequest,
    AnomalyDetectionResponse,
    SpoofingDetectionResponse,
    Phase2Intelligence,
    SessionIntelligence,
    RiskLevel,
)
from app.scoring.explainability import ExplainabilityEngine

router = APIRouter()
explainability = ExplainabilityEngine()


@router.get(
    "/risk/{user_id}",
    response_model=RiskAnalysisResponse,
    summary="Get risk analysis",
    description="Get comprehensive risk analysis for a user's data",
)
async def get_risk_analysis(
    user_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session"),
    include_locations: bool = Query(True, description="Include location-level risks"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> RiskAnalysisResponse:
    """
    Get risk analysis for a user.
    
    Returns:
    - Session-level risk scores
    - Location-level risk scores (optional)
    - Risk reasons and explanations
    
    All scores are continuous (0-1), not binary decisions.
    """
    try:
        return service.get_risk_analysis(
            user_id=user_id,
            session_id=session_id,
            include_locations=include_locations,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Risk analysis failed: {str(e)}",
        )


@router.get(
    "/anomalies/{user_id}",
    response_model=AnomalyDetectionResponse,
    summary="Get anomaly detection",
    description="Get anomaly detection results with explanations",
)
async def get_anomaly_detection(
    user_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session"),
    sensitivity: float = Query(
        0.5, ge=0.0, le=1.0,
        description="Detection sensitivity (higher = more detections)"
    ),
    include_model_scores: bool = Query(
        False,
        description="Include raw model scores"
    ),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> AnomalyDetectionResponse:
    """
    Get anomaly detection results.
    
    Detects:
    - Route deviations
    - Speed anomalies
    - Impossible jumps
    - Low confidence sequences
    - Movement inconsistencies
    
    Returns anomaly scores with explanations.
    """
    try:
        return service.get_anomaly_detection(
            user_id=user_id,
            session_id=session_id,
            sensitivity=sensitivity,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Anomaly detection failed: {str(e)}",
        )


@router.get(
    "/spoofing/{user_id}",
    response_model=SpoofingDetectionResponse,
    summary="Get spoofing detection",
    description="Get GPS spoofing and tampering detection results",
)
async def get_spoofing_detection(
    user_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> SpoofingDetectionResponse:
    """
    Get spoofing and tampering detection results.
    
    Detects:
    - GPS spoofing (teleportation, impossible speeds)
    - Data tampering (timestamp manipulation, integrity issues)
    - Synthetic patterns (perfect lines, identical coordinates)
    
    Returns PROBABILISTIC LIKELIHOOD, not binary flags.
    """
    try:
        return service.get_spoofing_detection(
            user_id=user_id,
            session_id=session_id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Spoofing detection failed: {str(e)}",
        )


@router.get(
    "/profile/{user_id}",
    response_model=dict,
    summary="Get behavioral profile",
    description="Get learned behavioral profile for a user",
)
async def get_behavioral_profile(
    user_id: int,
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get learned behavioral profile.
    
    Profiles include:
    - Route patterns
    - Time-of-day behavior
    - Stop patterns
    - Movement consistency metrics
    - Profile maturity indicators
    """
    summary = service.get_profile_summary(user_id)
    
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No profile found for user {user_id}. Process user data first.",
        )
    
    return {
        "success": True,
        **summary,
    }


@router.post(
    "/process/{user_id}",
    response_model=dict,
    summary="Process user through Phase 2",
    description="Run complete Phase 2 analysis on user data",
)
async def process_user_phase2(
    user_id: int,
    force_relearn: bool = Query(
        False,
        description="Force relearning of behavioral profile"
    ),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Process a user's data through the complete Phase 2 pipeline.
    
    This:
    1. Loads Phase 1 processed data
    2. Learns behavioral profile
    3. Runs anomaly detection
    4. Runs spoofing detection
    5. Calculates risk scores
    6. Generates explanations
    
    Returns summary of Phase 2 analysis.
    """
    try:
        profile, sessions = service.process_user(
            user_id=user_id,
            force_relearn=force_relearn,
        )
        
        # Summary statistics
        total_points = sum(s.total_points for s in sessions)
        trusted_points = sum(s.trusted_points for s in sessions)
        anomalous_sessions = sum(1 for s in sessions if s.anomaly_count > 0)
        high_risk_sessions = sum(
            1 for s in sessions 
            if s.risk_score.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "processed_at": datetime.utcnow().isoformat(),
            "profile": {
                "is_mature": profile.is_mature,
                "confidence": profile.profile_confidence,
                "routes_learned": len(profile.known_routes),
                "stops_learned": len(profile.frequent_stops),
            },
            "sessions": {
                "total": len(sessions),
                "with_anomalies": anomalous_sessions,
                "high_risk": high_risk_sessions,
            },
            "points": {
                "total": total_points,
                "trusted": trusted_points,
                "trust_rate": trusted_points / total_points if total_points > 0 else 0.0,
            },
            "recommendations": [
                "Review high-risk sessions" if high_risk_sessions > 0 else None,
                "Investigate anomalies" if anomalous_sessions > len(sessions) * 0.2 else None,
            ],
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase 2 processing failed: {str(e)}",
        )


@router.get(
    "/session/{session_id}",
    response_model=dict,
    summary="Get session intelligence",
    description="Get Phase 2 intelligence for a specific session",
)
async def get_session_intelligence(
    session_id: str,
    user_id: int = Query(..., description="User ID"),
    include_explanation: bool = Query(True, description="Include detailed explanation"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get complete Phase 2 intelligence for a session.
    
    Returns:
    - Session risk score with explanation
    - Anomaly summary
    - Spoofing assessment
    - Trust metrics
    - Recommendations
    """
    # Check if session is cached
    session_intel = service._session_cache.get(session_id)
    
    if session_intel is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session {session_id} not found. Process user data first.",
        )
    
    response = {
        "success": True,
        "session_id": session_intel.session_id,
        "user_id": session_intel.user_id,
        "time_range": {
            "start": session_intel.start_time.isoformat(),
            "end": session_intel.end_time.isoformat(),
            "duration_seconds": session_intel.duration_seconds,
        },
        "risk": {
            "score": session_intel.risk_score.risk_score,
            "level": session_intel.risk_score.risk_level.value,
            "reasons": session_intel.risk_score.reasons,
        },
        "trust": {
            "total_points": session_intel.total_points,
            "trusted_points": session_intel.trusted_points,
            "trust_rate": session_intel.trust_rate,
        },
        "anomalies": {
            "count": session_intel.anomaly_count,
            "types": session_intel.anomaly_types,
        },
        "spoofing": {
            "likelihood": session_intel.spoofing_likelihood,
            "indicators": session_intel.spoofing_indicators,
        },
        "verdict": {
            "is_valid": session_intel.is_session_valid,
            "confidence": session_intel.validation_confidence,
        },
        "recommendations": session_intel.recommendations,
    }
    
    if include_explanation:
        response["explanation"] = explainability.explain_session_risk(
            session_intel.risk_score,
            verbose=True,
        )
    
    return response


@router.get(
    "/intelligence/{user_id}",
    response_model=dict,
    summary="Get location intelligence",
    description="Get Phase 2 enriched location intelligence",
)
async def get_location_intelligence(
    user_id: int,
    session_id: Optional[str] = Query(None, description="Filter by session"),
    min_trust: float = Query(0.0, ge=0.0, le=1.0, description="Minimum trust score"),
    trusted_only: bool = Query(False, description="Return only trusted locations"),
    limit: int = Query(100, ge=1, le=10000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get Phase 2 enriched location intelligence.
    
    Returns clean, validated movement signals with:
    - Phase 1 confidence
    - Phase 2 risk score
    - Anomaly indicators
    - Spoofing assessment
    - Trust determination
    
    Use trusted_only=true for downstream consumption.
    """
    try:
        # Ensure user is processed
        if service.get_profile(user_id) is None:
            service.process_user(user_id)
        
        # Get sessions
        sessions = [
            s for s in service._session_cache.values()
            if s.user_id == user_id
        ]
        
        if session_id:
            sessions = [s for s in sessions if s.session_id == session_id]
        
        if not sessions:
            raise ValueError(f"No sessions found for user {user_id}")
        
        # Collect location data from sessions
        # Note: In production, would store location intelligence
        # For now, return session-level summary
        
        locations = []
        for session in sessions:
            # Session summary as location proxy
            location_summary = {
                "session_id": session.session_id,
                "start_time": session.start_time.isoformat(),
                "end_time": session.end_time.isoformat(),
                "total_points": session.total_points,
                "trusted_points": session.trusted_points,
                "trust_rate": session.trust_rate,
                "risk_score": session.risk_score.risk_score,
                "risk_level": session.risk_score.risk_level.value,
                "is_trusted": session.is_session_valid,
            }
            
            if trusted_only and not session.is_session_valid:
                continue
            
            if session.trust_rate >= min_trust:
                locations.append(location_summary)
        
        # Pagination
        total = len(locations)
        locations = locations[offset:offset + limit]
        
        return {
            "success": True,
            "user_id": user_id,
            "total_sessions": total,
            "returned_sessions": len(locations),
            "filters": {
                "min_trust": min_trust,
                "trusted_only": trusted_only,
            },
            "data": locations,
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get intelligence: {str(e)}",
        )


@router.get(
    "/explain/risk/{event_id}",
    response_model=dict,
    summary="Get risk explanation",
    description="Get detailed explanation for a risk score",
)
async def explain_risk_score(
    event_id: str,
    user_id: int = Query(..., description="User ID"),
    verbose: bool = Query(False, description="Include technical details"),
    service: Phase2Service = Depends(get_phase2_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get detailed explanation for a risk score.
    
    MANDATORY for Phase 2: Every risk score must be explainable.
    
    Returns:
    - Why the risk score was assigned
    - Which signals contributed
    - Relative weights
    - Human-readable narrative
    """
    # This would look up the cached risk score for the event
    # For now, return explanation structure
    
    return {
        "success": True,
        "event_id": event_id,
        "message": "Risk explanation endpoint - implement full lookup",
        "explanation_structure": {
            "summary": "Human-readable summary",
            "risk_score": 0.0,
            "risk_level": "minimal",
            "primary_reasons": [],
            "contributing_factors": [],
            "narrative": "Full narrative explanation",
        },
    }
