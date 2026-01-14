"""
Data Ingestion Routes
======================

Endpoints for data ingestion and processing.

Phase 1 Capabilities:
- Process data from local dataset
- Validate incoming events
- Return processing statistics

Phase 2+ will add:
- Real-time event ingestion
- Webhook integrations
- Streaming processing
"""

import time
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import get_intelligence_svc, verify_api_key
from app.services.intelligence_service import IntelligenceService
from app.models.schemas import IngestResponse

router = APIRouter()


@router.post(
    "/process/user/{user_id}",
    response_model=dict,
    summary="Process user data",
    description="Process all tracking data for a user from the dataset",
)
async def process_user_data(
    user_id: int,
    limit: Optional[int] = Query(None, ge=1, le=100000, description="Max events"),
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Process all tracking data for a specific user.
    
    This will:
    1. Load user data from dataset
    2. Normalize and clean events
    3. Compute features
    4. Score confidence
    5. Learn user baseline
    
    Args:
        user_id: User identifier
        limit: Maximum events to process
        
    Returns:
        Processing summary
    """
    start_time = time.time()
    
    try:
        trips, baseline = service.process_user_data(user_id, limit)
        
        # Summary statistics
        total_events = sum(len(t.events) for t in trips)
        avg_confidence = (
            sum(e.confidence.overall for t in trips for e in t.events) / total_events
            if total_events > 0 else 0
        )
        
        return {
            "success": True,
            "user_id": user_id,
            "trips_processed": len(trips),
            "events_processed": total_events,
            "average_confidence": round(avg_confidence, 4),
            "baseline_mature": baseline.is_mature,
            "baseline_points": baseline.points_analyzed,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@router.post(
    "/process/trip/{trip_id}",
    response_model=dict,
    summary="Process single trip",
    description="Process a single trip from the dataset",
)
async def process_trip(
    trip_id: int,
    service: IntelligenceService = Depends(get_intelligence_svc),
    _: bool = Depends(verify_api_key),
) -> dict:
    """
    Process a single trip.
    
    Args:
        trip_id: Global trip identifier
        
    Returns:
        Trip processing summary
    """
    start_time = time.time()
    
    try:
        trip = service.process_trip(trip_id)
        
        return {
            "success": True,
            "trip_id": trip_id,
            "user_id": trip.summary.user_id,
            "events_processed": len(trip.events),
            "duration_seconds": trip.summary.duration_seconds,
            "distance_meters": round(trip.summary.distance_meters, 1),
            "average_confidence": round(trip.summary.average_confidence, 4),
            "primary_activity": trip.summary.primary_activity.value,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
        }
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Processing failed: {str(e)}",
        )


@router.get(
    "/dataset/stats",
    response_model=dict,
    summary="Get dataset statistics",
    description="Get statistics about the loaded dataset",
)
async def get_dataset_stats(
    service: IntelligenceService = Depends(get_intelligence_svc),
) -> dict:
    """
    Get statistics about the dataset.
    
    Returns:
        Dataset statistics including row counts, users, trips
    """
    try:
        stats = service.get_dataset_stats()
        return {
            "success": True,
            **stats,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )


@router.get(
    "/dataset/users",
    response_model=dict,
    summary="List users in dataset",
    description="Get list of unique user IDs in the dataset",
)
async def list_users(
    service: IntelligenceService = Depends(get_intelligence_svc),
) -> dict:
    """
    Get list of unique users in the dataset.
    
    Returns:
        List of user IDs
    """
    try:
        user_ids = service.ingestion.get_user_ids()
        return {
            "success": True,
            "user_count": len(user_ids),
            "user_ids": user_ids,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}",
        )


@router.get(
    "/dataset/trips",
    response_model=dict,
    summary="List trips in dataset",
    description="Get list of trip IDs, optionally filtered by user",
)
async def list_trips(
    user_id: Optional[int] = Query(None, description="Filter by user ID"),
    service: IntelligenceService = Depends(get_intelligence_svc),
) -> dict:
    """
    Get list of trips in the dataset.
    
    Args:
        user_id: Optional user ID filter
        
    Returns:
        List of trip IDs
    """
    try:
        trip_ids = service.ingestion.get_trip_ids(user_id)
        return {
            "success": True,
            "user_id": user_id,
            "trip_count": len(trip_ids),
            "trip_ids": trip_ids[:100],  # Limit response size
            "truncated": len(trip_ids) > 100,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list trips: {str(e)}",
        )
