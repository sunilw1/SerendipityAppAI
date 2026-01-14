"""
Feature Engineering Module
============================

Computes derived features from tracking data for:
- Confidence scoring
- Baseline behavior learning
- Future anomaly detection (Phase 2+)

Feature Categories:
1. Temporal - Time deltas, update rates, gaps
2. Spatial - Distances, bearings, trajectories
3. Kinematic - Speed, acceleration, jerk
4. Stop Detection - Stationary periods
5. Activity - Activity transitions, consistency

Design Principles:
- All features are explainable
- No complex embeddings in Phase 1
- Vectorized operations for performance
- Handle missing data gracefully
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from app.core.constants import TimeConstants, ActivityType
from app.core.logging import get_logger
from app.models.schemas import ComputedFeatures, NormalizedTrackingEvent
from app.utils.geo import (
    haversine_distance,
    haversine_distance_vectorized,
    calculate_bearing,
    calculate_bearing_change,
    calculate_speed,
    calculate_acceleration,
)
from app.utils.time import calculate_time_delta

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Computes features for tracking events.
    
    Features are computed relative to the previous event
    in a sequence (e.g., within a trip).
    """
    
    def __init__(
        self,
        min_stop_speed_ms: float = 0.5,
        min_stop_duration_seconds: float = TimeConstants.MIN_STOP_DURATION,
        max_normal_gap_seconds: float = TimeConstants.MAX_NORMAL_GAP,
    ):
        """
        Initialize feature engineer.
        
        Args:
            min_stop_speed_ms: Speed threshold for stop detection
            min_stop_duration_seconds: Minimum duration to qualify as stop
            max_normal_gap_seconds: Maximum normal time gap
        """
        self.min_stop_speed_ms = min_stop_speed_ms
        self.min_stop_duration_seconds = min_stop_duration_seconds
        self.max_normal_gap_seconds = max_normal_gap_seconds
    
    def compute_features(
        self,
        current: NormalizedTrackingEvent,
        previous: Optional[NormalizedTrackingEvent] = None,
        previous_features: Optional[ComputedFeatures] = None
    ) -> ComputedFeatures:
        """
        Compute features for a single event.
        
        Args:
            current: Current event
            previous: Previous event (None for first event)
            previous_features: Features from previous event (for running stats)
            
        Returns:
            ComputedFeatures for current event
        """
        if previous is None:
            # First event in sequence - no relative features
            return ComputedFeatures(
                time_delta_seconds=None,
                update_rate_hz=None,
                distance_meters=None,
                bearing_degrees=None,
                bearing_change_degrees=None,
                calculated_speed_ms=None,
                speed_difference_ms=None,
                acceleration_ms2=None,
                is_stop=not current.is_moving,
                stop_duration_seconds=None,
                has_time_gap=False,
                gap_duration_seconds=None,
            )
        
        # Temporal features
        time_delta = current.timestamp_unix - previous.timestamp_unix
        update_rate = 1.0 / time_delta if time_delta > 0 else None
        
        # Time gap detection
        has_gap = time_delta > self.max_normal_gap_seconds
        gap_duration = time_delta if has_gap else None
        
        # Spatial features
        distance = haversine_distance(
            previous.location.lat, previous.location.lon,
            current.location.lat, current.location.lon
        )
        
        bearing = calculate_bearing(
            previous.location.lat, previous.location.lon,
            current.location.lat, current.location.lon
        )
        
        # Bearing change
        bearing_change = None
        if previous_features and previous_features.bearing_degrees is not None:
            bearing_change = calculate_bearing_change(
                previous_features.bearing_degrees,
                bearing
            )
        
        # Kinematic features
        calculated_speed = calculate_speed(distance, time_delta) if time_delta > 0 else None
        
        speed_diff = None
        if calculated_speed is not None and current.speed_ms is not None:
            speed_diff = current.speed_ms - calculated_speed
        
        acceleration = None
        if (
            previous.speed_ms is not None and 
            current.speed_ms is not None and
            time_delta > 0
        ):
            acceleration = calculate_acceleration(
                previous.speed_ms,
                current.speed_ms,
                time_delta
            )
        
        # Stop detection
        is_stop = self._detect_stop(current, calculated_speed)
        
        # Stop duration (if continuing a stop)
        stop_duration = None
        if is_stop and previous_features and previous_features.is_stop:
            if previous_features.stop_duration_seconds is not None:
                stop_duration = previous_features.stop_duration_seconds + time_delta
            else:
                stop_duration = time_delta
        elif is_stop:
            stop_duration = 0.0  # Just started stopping
        
        return ComputedFeatures(
            time_delta_seconds=time_delta,
            update_rate_hz=update_rate,
            distance_meters=distance,
            bearing_degrees=bearing,
            bearing_change_degrees=bearing_change,
            calculated_speed_ms=calculated_speed,
            speed_difference_ms=speed_diff,
            acceleration_ms2=acceleration,
            is_stop=is_stop,
            stop_duration_seconds=stop_duration,
            has_time_gap=has_gap,
            gap_duration_seconds=gap_duration,
        )
    
    def compute_features_batch(
        self,
        events: List[NormalizedTrackingEvent]
    ) -> List[ComputedFeatures]:
        """
        Compute features for a batch of events.
        
        Events should be sorted by timestamp.
        
        Args:
            events: List of normalized events
            
        Returns:
            List of computed features (same length as events)
        """
        if not events:
            return []
        
        features = []
        prev_event = None
        prev_features = None
        
        for event in events:
            feat = self.compute_features(event, prev_event, prev_features)
            features.append(feat)
            prev_event = event
            prev_features = feat
        
        return features
    
    def compute_features_vectorized(
        self,
        events: List[NormalizedTrackingEvent]
    ) -> List[ComputedFeatures]:
        """
        Compute features using vectorized operations.
        
        Faster for large batches, but requires all events in memory.
        
        Args:
            events: List of normalized events
            
        Returns:
            List of computed features
        """
        n = len(events)
        if n == 0:
            return []
        
        # Extract arrays
        timestamps = np.array([e.timestamp_unix for e in events])
        lats = np.array([e.location.lat for e in events])
        lons = np.array([e.location.lon for e in events])
        speeds = np.array([e.speed_ms if e.speed_ms is not None else np.nan for e in events])
        is_moving = np.array([e.is_moving for e in events])
        
        # Time deltas
        time_deltas = np.empty(n)
        time_deltas[0] = np.nan
        time_deltas[1:] = np.diff(timestamps)
        
        # Update rates
        update_rates = np.where(time_deltas > 0, 1.0 / time_deltas, np.nan)
        
        # Distances
        distances = np.empty(n)
        distances[0] = np.nan
        distances[1:] = haversine_distance_vectorized(
            lats[:-1], lons[:-1],
            lats[1:], lons[1:]
        )
        
        # Calculated speeds
        calc_speeds = np.where(time_deltas > 0, distances / time_deltas, np.nan)
        
        # Speed differences
        speed_diffs = speeds - calc_speeds
        
        # Accelerations
        accelerations = np.empty(n)
        accelerations[0] = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            accelerations[1:] = np.diff(speeds) / time_deltas[1:]
        
        # Time gaps
        has_gaps = time_deltas > self.max_normal_gap_seconds
        gap_durations = np.where(has_gaps, time_deltas, np.nan)
        
        # Stops (simplified - actual implementation uses more context)
        is_stops = ~is_moving | (calc_speeds < self.min_stop_speed_ms)
        
        # Build feature objects
        features = []
        for i in range(n):
            features.append(ComputedFeatures(
                time_delta_seconds=None if np.isnan(time_deltas[i]) else float(time_deltas[i]),
                update_rate_hz=None if np.isnan(update_rates[i]) else float(update_rates[i]),
                distance_meters=None if np.isnan(distances[i]) else float(distances[i]),
                bearing_degrees=None,  # Computed separately if needed
                bearing_change_degrees=None,
                calculated_speed_ms=None if np.isnan(calc_speeds[i]) else float(calc_speeds[i]),
                speed_difference_ms=None if np.isnan(speed_diffs[i]) else float(speed_diffs[i]),
                acceleration_ms2=None if np.isnan(accelerations[i]) else float(accelerations[i]),
                is_stop=bool(is_stops[i]),
                stop_duration_seconds=None,  # Requires sequential computation
                has_time_gap=bool(has_gaps[i]),
                gap_duration_seconds=None if np.isnan(gap_durations[i]) else float(gap_durations[i]),
            ))
        
        return features
    
    def _detect_stop(
        self,
        event: NormalizedTrackingEvent,
        calculated_speed: Optional[float]
    ) -> bool:
        """
        Detect if an event represents a stop.
        
        Uses multiple signals: is_moving flag, reported speed,
        calculated speed, and activity type.
        """
        # Explicit not moving
        if not event.is_moving:
            return True
        
        # Activity is "still"
        if event.activity_type == ActivityType.STILL:
            return True
        
        # Very low speed
        if event.speed_ms is not None and event.speed_ms < self.min_stop_speed_ms:
            return True
        
        if calculated_speed is not None and calculated_speed < self.min_stop_speed_ms:
            return True
        
        return False


class TripFeatureEngineer:
    """
    Computes trip-level aggregate features.
    
    Provides summary statistics for a complete trip.
    """
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
    
    def compute_trip_features(
        self,
        events: List[NormalizedTrackingEvent],
        features: Optional[List[ComputedFeatures]] = None
    ) -> dict:
        """
        Compute aggregate features for a trip.
        
        Args:
            events: List of events in the trip
            features: Pre-computed features (optional)
            
        Returns:
            Dict of trip-level features
        """
        if not events:
            return {}
        
        # Compute features if not provided
        if features is None:
            features = self.feature_engineer.compute_features_batch(events)
        
        # Temporal aggregates
        total_duration = events[-1].timestamp_unix - events[0].timestamp_unix
        
        time_deltas = [f.time_delta_seconds for f in features if f.time_delta_seconds is not None]
        avg_update_rate = np.mean([f.update_rate_hz for f in features if f.update_rate_hz is not None]) if time_deltas else None
        
        # Spatial aggregates
        distances = [f.distance_meters for f in features if f.distance_meters is not None]
        total_distance = sum(distances)
        
        # Speed aggregates
        speeds = [e.speed_ms for e in events if e.speed_ms is not None]
        avg_speed = np.mean(speeds) if speeds else None
        max_speed = max(speeds) if speeds else None
        
        # Stop detection
        stops = [f for f in features if f.is_stop]
        num_stops = len(stops)
        total_stop_duration = sum(
            f.stop_duration_seconds for f in stops 
            if f.stop_duration_seconds is not None
        )
        
        # Activity breakdown
        activity_counts = {}
        for event in events:
            act = event.activity_type.value
            activity_counts[act] = activity_counts.get(act, 0) + 1
        
        total_events = len(events)
        activity_percentages = {k: v / total_events for k, v in activity_counts.items()}
        
        # Gap analysis
        gaps = [f for f in features if f.has_time_gap]
        num_gaps = len(gaps)
        total_gap_duration = sum(
            f.gap_duration_seconds for f in gaps 
            if f.gap_duration_seconds is not None
        )
        
        return {
            # Temporal
            "duration_seconds": total_duration,
            "num_points": len(events),
            "avg_update_rate_hz": avg_update_rate,
            
            # Spatial
            "total_distance_meters": total_distance,
            
            # Speed
            "avg_speed_ms": avg_speed,
            "max_speed_ms": max_speed,
            
            # Stops
            "num_stops": num_stops,
            "total_stop_duration_seconds": total_stop_duration,
            
            # Activity
            "activity_breakdown": activity_percentages,
            "primary_activity": max(activity_counts, key=activity_counts.get) if activity_counts else None,
            
            # Gaps
            "num_gaps": num_gaps,
            "total_gap_duration_seconds": total_gap_duration,
        }


def compute_features(
    events: List[NormalizedTrackingEvent]
) -> List[ComputedFeatures]:
    """
    Convenience function to compute features.
    
    Args:
        events: List of normalized events
        
    Returns:
        List of computed features
    """
    engineer = FeatureEngineer()
    return engineer.compute_features_batch(events)
