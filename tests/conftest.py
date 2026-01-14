"""
Pytest Configuration and Fixtures
===================================

Shared fixtures for all tests.
"""

import pytest
from datetime import datetime, timezone
from typing import List

from app.models.schemas import (
    RawTrackingEvent,
    NormalizedTrackingEvent,
    LocationPoint,
    ComputedFeatures,
)
from app.core.constants import ActivityType


@pytest.fixture
def sample_raw_event() -> RawTrackingEvent:
    """Create a sample raw tracking event."""
    return RawTrackingEvent(
        user_id=1,
        trip_index_for_user=1,
        trip_global_index=1,
        point_index=0,
        trip_title="Test Trip",
        trip_session="10:00 - 10:30",
        trip_start=datetime(2024, 1, 1, 10, 0, 0),
        trip_end=datetime(2024, 1, 1, 10, 30, 0),
        trip_duration_minutes=30,
        trip_distance_km=5.0,
        timestamp=datetime(2024, 1, 1, 10, 0, 0),
        lat=34.0522,
        lon=-118.2437,
        speed=5.0,
        is_moving=1,
        activity_type="walking",
        activity_confidence=80,
        coords_accuracy=10.0,
        odometer=1000.0,
        live_address="Los Angeles, CA",
    )


@pytest.fixture
def sample_raw_events() -> List[RawTrackingEvent]:
    """Create a sequence of sample raw events."""
    base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    
    events = []
    for i in range(10):
        events.append(RawTrackingEvent(
            user_id=1,
            trip_index_for_user=1,
            trip_global_index=1,
            point_index=i,
            trip_title="Test Trip",
            trip_session="10:00 - 10:30",
            trip_start=base_time,
            trip_end=datetime(2024, 1, 1, 10, 30, 0, tzinfo=timezone.utc),
            trip_duration_minutes=30,
            trip_distance_km=5.0,
            timestamp=datetime(2024, 1, 1, 10, 0, i * 2, tzinfo=timezone.utc),
            lat=34.0522 + i * 0.0001,
            lon=-118.2437 + i * 0.0001,
            speed=5.0 + i * 0.5,
            is_moving=1,
            activity_type="walking",
            activity_confidence=80,
            coords_accuracy=10.0,
            odometer=1000.0 + i * 10,
            live_address="Los Angeles, CA",
        ))
    
    return events


@pytest.fixture
def sample_normalized_event() -> NormalizedTrackingEvent:
    """Create a sample normalized event."""
    return NormalizedTrackingEvent(
        event_id="u1_t1_p0",
        user_id=1,
        trip_id=1,
        point_index=0,
        timestamp=datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc),
        timestamp_unix=1704103200.0,
        location=LocationPoint(
            lat=34.0522,
            lon=-118.2437,
            accuracy_meters=10.0,
        ),
        speed_ms=5.0,
        is_moving=True,
        activity_type=ActivityType.WALKING,
        activity_confidence=80,
        gps_accuracy_meters=10.0,
        quality_flags=[],
        is_valid=True,
        raw_event_hash="abc123",
    )


@pytest.fixture
def sample_features() -> ComputedFeatures:
    """Create sample computed features."""
    return ComputedFeatures(
        time_delta_seconds=2.0,
        update_rate_hz=0.5,
        distance_meters=10.0,
        bearing_degrees=45.0,
        bearing_change_degrees=5.0,
        calculated_speed_ms=5.0,
        speed_difference_ms=0.0,
        acceleration_ms2=0.5,
        is_stop=False,
        stop_duration_seconds=None,
        has_time_gap=False,
        gap_duration_seconds=None,
    )
