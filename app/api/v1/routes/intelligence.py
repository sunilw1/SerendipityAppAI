"""
Intelligence Routes
====================

Endpoints for accessing location intelligence.

Provides:
- Confidence-scored location data
- Data quality reports
- User baselines
- Trip summaries

All output is standardized and intelligence-enriched.
Raw data is never exposed directly.
"""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_intelligence_svc, get_baseline_svc, verify_api_key
from app.services.intelligence_service import IntelligenceService
from app.services.baseline_service import BaselineService
from app.models.schemas import (
    LocationIntelligence,
    DataQualityReport,
    IntelligenceResponse,
)

router = APIRouter()


@router.get(
    "/location/{user_id}",
    response_model=IntelligenceResponse,
    summary="Get location intelligence",
    description="Get confidence-scored location data for a user",
)
async def get_location_intelligence(
    user_id: int,
    trip_id: Optional[int] = Query(None, description="Filter by trip"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Min confidence"),
    limit: int = Query(100, ge=1, le=10000, description="Max results"),
    offset: int = Query(0, ge=0, description="Pagination offset"),
    include_quality_report: bool = Query(True, description="Include quality report"),
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> IntelligenceResponse:
    """
    Get location intelligence for a user.
    
    Returns confidence-scored, standardized location data.
    Can be filtered by trip and minimum confidence.
    
    Args:
        user_id: User identifier
        trip_id: Optional trip filter
        min_confidence: Minimum confidence score
        limit: Maximum results
        offset: Pagination offset
        include_quality_report: Include data quality summary
        
    Returns:
        IntelligenceResponse with location data
    """
    try:
        intelligence, total = service.get_location_intelligence(
            user_id=user_id,
            trip_id=trip_id,
            min_confidence=min_confidence,
            limit=limit,
            offset=offset,
        )
        
        quality_report = None
        if include_quality_report:
            quality_report = service.get_data_quality_report(user_id, trip_id)
        
        return IntelligenceResponse(
            success=True,
            user_id=user_id,
            total_count=total,
            returned_count=len(intelligence),
            data=intelligence,
            quality_report=quality_report,
        )
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query failed: {str(e)}",
        )


@router.get(
    "/quality/{user_id}",
    response_model=DataQualityReport,
    summary="Get data quality report",
    description="Get data quality analysis for a user's data",
)
async def get_quality_report(
    user_id: int,
    trip_id: Optional[int] = Query(None, description="Filter by trip"),
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> DataQualityReport:
    """
    Get data quality report for a user.
    
    Includes:
    - Event counts and validity rates
    - Quality flag breakdown
    - Confidence score distribution
    
    Args:
        user_id: User identifier
        trip_id: Optional trip filter
        
    Returns:
        DataQualityReport
    """
    try:
        return service.get_data_quality_report(user_id, trip_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Report generation failed: {str(e)}",
        )


@router.get(
    "/baseline/{user_id}",
    response_model=dict,
    summary="Get user baseline",
    description="Get learned behavior baseline for a user",
)
async def get_user_baseline(
    user_id: int,
    service: BaselineService = Depends(get_baseline_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get learned baseline for a user.
    
    Baselines include:
    - Speed patterns
    - Activity distribution
    - Update frequency patterns
    - Typical confidence
    
    Args:
        user_id: User identifier
        
    Returns:
        Baseline summary
    """
    summary = service.get_baseline_summary(user_id)
    
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No baseline found for user {user_id}. Process user data first.",
        )
    
    return {
        "success": True,
        **summary,
    }


@router.get(
    "/baselines",
    response_model=dict,
    summary="List all baselines",
    description="Get list of users with computed baselines",
)
async def list_baselines(
    service: BaselineService = Depends(get_baseline_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    List all users with computed baselines.
    
    Returns:
        List of user IDs with baselines
    """
    user_ids = service.list_users_with_baselines()
    
    return {
        "success": True,
        "count": len(user_ids),
        "user_ids": user_ids,
    }


@router.get(
    "/trip/{trip_id}/summary",
    response_model=dict,
    summary="Get trip summary",
    description="Get summary statistics for a processed trip",
)
async def get_trip_summary(
    trip_id: int,
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get summary for a processed trip.
    
    Args:
        trip_id: Trip identifier
        
    Returns:
        Trip summary with statistics
    """
    try:
        trip = service.process_trip(trip_id)
        
        summary = trip.summary
        
        return {
            "success": True,
            "trip_id": summary.trip_id,
            "user_id": summary.user_id,
            "start_time": summary.start_time.isoformat(),
            "end_time": summary.end_time.isoformat(),
            "duration_seconds": summary.duration_seconds,
            "distance_meters": round(summary.distance_meters, 1),
            "start_location": {
                "lat": summary.start_location.lat,
                "lon": summary.start_location.lon,
            },
            "end_location": {
                "lat": summary.end_location.lat,
                "lon": summary.end_location.lon,
            },
            "total_points": summary.total_points,
            "valid_points": summary.valid_points,
            "average_confidence": round(summary.average_confidence, 4),
            "primary_activity": summary.primary_activity.value,
            "activity_breakdown": {
                k: round(v, 4) for k, v in summary.activity_breakdown.items()
            },
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get summary: {str(e)}",
        )


@router.get(
    "/confidence/distribution/{user_id}",
    response_model=dict,
    summary="Get confidence distribution",
    description="Get distribution of confidence scores for a user",
)
async def get_confidence_distribution(
    user_id: int,
    trip_id: Optional[int] = Query(None, description="Filter by trip"),
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Get confidence score distribution for a user.
    
    Args:
        user_id: User identifier
        trip_id: Optional trip filter
        
    Returns:
        Confidence distribution statistics
    """
    try:
        report = service.get_data_quality_report(user_id, trip_id)
        
        return {
            "success": True,
            "user_id": user_id,
            "trip_id": trip_id,
            "total_events": report.total_events,
            "distribution": {
                "high_confidence": {
                    "threshold": ">=0.75",
                    "count": report.high_confidence_count,
                    "percentage": round(report.high_confidence_count / report.total_events, 4) if report.total_events > 0 else 0,
                },
                "medium_confidence": {
                    "threshold": "0.5-0.75",
                    "count": report.medium_confidence_count,
                    "percentage": round(report.medium_confidence_count / report.total_events, 4) if report.total_events > 0 else 0,
                },
                "low_confidence": {
                    "threshold": "<0.5",
                    "count": report.low_confidence_count,
                    "percentage": round(report.low_confidence_count / report.total_events, 4) if report.total_events > 0 else 0,
                },
            },
            "statistics": {
                "mean": round(report.confidence_mean, 4),
                "std": round(report.confidence_std, 4),
                "min": round(report.confidence_min, 4),
                "max": round(report.confidence_max, 4),
            },
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compute distribution: {str(e)}",
        )
