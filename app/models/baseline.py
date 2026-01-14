"""
Baseline Behavior Learning
===========================

Learns normal behavior patterns from historical tracking data.

Phase 1 Scope (Observe-Only):
- Statistical baselines for movement patterns
- Activity distribution analysis
- Update frequency patterns
- Stop/movement behavior

No anomaly detection or alerting in Phase 1.
Baselines are purely descriptive.

Design Principles:
- Learn from high-confidence data only
- Maintain separate baselines per user
- Support incremental updates
- Provide clear statistical summaries
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np

from app.core.constants import ActivityType, TimeConstants
from app.core.logging import get_logger
from app.models.schemas import (
    BaselineMetrics,
    UserBaseline,
    ComputedFeatures,
    NormalizedTrackingEvent,
    ConfidenceScore,
)
from app.utils.metrics import (
    compute_baseline_stats,
    compute_activity_distribution,
)

logger = get_logger(__name__)


class BaselineLearner:
    """
    Learns behavioral baselines from tracking data.
    
    Processes trip data to build statistical profiles
    of normal user behavior.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        min_trips: int = 5,
        min_points: int = 100,
    ):
        """
        Initialize baseline learner.
        
        Args:
            min_confidence: Minimum confidence score to include in baseline
            min_trips: Minimum trips required for reliable baseline
            min_points: Minimum points required for reliable baseline
        """
        self.min_confidence = min_confidence
        self.min_trips = min_trips
        self.min_points = min_points
    
    def learn_baseline(
        self,
        user_id: int,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
        trip_ids: Optional[List[int]] = None,
    ) -> UserBaseline:
        """
        Learn baseline behavior from user's tracking data.
        
        Args:
            user_id: User identifier
            events: Normalized tracking events
            features: Computed features
            scores: Confidence scores
            trip_ids: Optional list of trip IDs (for trip count)
            
        Returns:
            UserBaseline with learned patterns
        """
        if len(events) != len(features) or len(events) != len(scores):
            raise ValueError("Events, features, and scores must have same length")
        
        # Filter by confidence
        high_conf_indices = [
            i for i, s in enumerate(scores) 
            if s.overall >= self.min_confidence
        ]
        
        filtered_events = [events[i] for i in high_conf_indices]
        filtered_features = [features[i] for i in high_conf_indices]
        filtered_scores = [scores[i] for i in high_conf_indices]
        
        # Determine trip count
        if trip_ids is not None:
            trips_analyzed = len(set(trip_ids))
        else:
            trips_analyzed = len(set(e.trip_id for e in filtered_events))
        
        points_analyzed = len(filtered_events)
        
        # Calculate date range
        if filtered_events:
            timestamps = [e.timestamp for e in filtered_events]
            date_range_days = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        else:
            date_range_days = 0.0
        
        # Learn speed baseline
        speed_values = [
            e.speed_ms for e in filtered_events 
            if e.speed_ms is not None
        ]
        speed_baseline = compute_baseline_stats(speed_values, "speed_ms")
        
        # Learn update frequency baseline
        update_rates = [
            f.update_rate_hz for f in filtered_features 
            if f.update_rate_hz is not None
        ]
        update_freq_baseline = compute_baseline_stats(update_rates, "update_rate_hz")
        
        # Learn trip duration baseline (need to compute from data)
        trip_durations = self._compute_trip_durations(filtered_events)
        trip_duration_baseline = compute_baseline_stats(trip_durations, "trip_duration_seconds")
        
        # Learn stop duration baseline
        stop_durations = [
            f.stop_duration_seconds for f in filtered_features
            if f.is_stop and f.stop_duration_seconds is not None and f.stop_duration_seconds > 0
        ]
        stop_duration_baseline = compute_baseline_stats(stop_durations, "stop_duration_seconds")
        
        # Activity distribution
        activities = [e.activity_type.value for e in filtered_events]
        activity_distribution = compute_activity_distribution(activities)
        
        # Typical confidence
        confidence_values = [s.overall for s in filtered_scores]
        typical_confidence = np.median(confidence_values) if confidence_values else 0.5
        
        now = datetime.now(timezone.utc)
        
        return UserBaseline(
            user_id=user_id,
            created_at=now,
            updated_at=now,
            trips_analyzed=trips_analyzed,
            points_analyzed=points_analyzed,
            date_range_days=date_range_days,
            speed_baseline=speed_baseline,
            update_frequency_baseline=update_freq_baseline,
            trip_duration_baseline=trip_duration_baseline,
            stop_duration_baseline=stop_duration_baseline,
            activity_distribution=activity_distribution,
            typical_confidence=typical_confidence,
        )
    
    def update_baseline(
        self,
        existing: UserBaseline,
        new_events: List[NormalizedTrackingEvent],
        new_features: List[ComputedFeatures],
        new_scores: List[ConfidenceScore],
    ) -> UserBaseline:
        """
        Incrementally update an existing baseline with new data.
        
        Uses exponential weighting to blend old and new statistics.
        
        Args:
            existing: Existing user baseline
            new_events: New tracking events
            new_features: New computed features
            new_scores: New confidence scores
            
        Returns:
            Updated UserBaseline
        """
        # Learn baseline from new data
        new_baseline = self.learn_baseline(
            existing.user_id,
            new_events,
            new_features,
            new_scores,
        )
        
        # Weight based on relative sample sizes
        total_old = existing.points_analyzed
        total_new = new_baseline.points_analyzed
        total = total_old + total_new
        
        if total == 0:
            return existing
        
        weight_old = total_old / total
        weight_new = total_new / total
        
        # Blend speed baseline
        speed_baseline = self._blend_baselines(
            existing.speed_baseline,
            new_baseline.speed_baseline,
            weight_old,
            weight_new,
        )
        
        # Blend update frequency baseline
        update_freq_baseline = self._blend_baselines(
            existing.update_frequency_baseline,
            new_baseline.update_frequency_baseline,
            weight_old,
            weight_new,
        )
        
        # Blend trip duration baseline
        trip_duration_baseline = self._blend_baselines(
            existing.trip_duration_baseline,
            new_baseline.trip_duration_baseline,
            weight_old,
            weight_new,
        )
        
        # Blend stop duration baseline
        stop_duration_baseline = self._blend_baselines(
            existing.stop_duration_baseline,
            new_baseline.stop_duration_baseline,
            weight_old,
            weight_new,
        )
        
        # Blend activity distribution
        activity_distribution = self._blend_distributions(
            existing.activity_distribution,
            new_baseline.activity_distribution,
            weight_old,
            weight_new,
        )
        
        # Blend typical confidence
        typical_confidence = (
            weight_old * existing.typical_confidence +
            weight_new * new_baseline.typical_confidence
        )
        
        return UserBaseline(
            user_id=existing.user_id,
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
            trips_analyzed=existing.trips_analyzed + new_baseline.trips_analyzed,
            points_analyzed=total,
            date_range_days=max(existing.date_range_days, new_baseline.date_range_days),
            speed_baseline=speed_baseline,
            update_frequency_baseline=update_freq_baseline,
            trip_duration_baseline=trip_duration_baseline,
            stop_duration_baseline=stop_duration_baseline,
            activity_distribution=activity_distribution,
            typical_confidence=typical_confidence,
        )
    
    def _compute_trip_durations(
        self,
        events: List[NormalizedTrackingEvent]
    ) -> List[float]:
        """Compute duration for each trip in the data."""
        if not events:
            return []
        
        # Group by trip
        trips: Dict[int, List[NormalizedTrackingEvent]] = {}
        for event in events:
            if event.trip_id not in trips:
                trips[event.trip_id] = []
            trips[event.trip_id].append(event)
        
        # Compute durations
        durations = []
        for trip_events in trips.values():
            if len(trip_events) >= 2:
                sorted_events = sorted(trip_events, key=lambda e: e.timestamp_unix)
                duration = sorted_events[-1].timestamp_unix - sorted_events[0].timestamp_unix
                durations.append(duration)
        
        return durations
    
    def _blend_baselines(
        self,
        old: BaselineMetrics,
        new: BaselineMetrics,
        weight_old: float,
        weight_new: float,
    ) -> BaselineMetrics:
        """Blend two baseline metrics using weighted average."""
        return BaselineMetrics(
            metric_name=old.metric_name,
            mean=weight_old * old.mean + weight_new * new.mean,
            median=weight_old * old.median + weight_new * new.median,
            std=np.sqrt(weight_old * old.std**2 + weight_new * new.std**2),
            min_value=min(old.min_value, new.min_value),
            max_value=max(old.max_value, new.max_value),
            p5=weight_old * old.p5 + weight_new * new.p5,
            p25=weight_old * old.p25 + weight_new * new.p25,
            p75=weight_old * old.p75 + weight_new * new.p75,
            p95=weight_old * old.p95 + weight_new * new.p95,
            sample_size=old.sample_size + new.sample_size,
        )
    
    def _blend_distributions(
        self,
        old: Dict[str, float],
        new: Dict[str, float],
        weight_old: float,
        weight_new: float,
    ) -> Dict[str, float]:
        """Blend two probability distributions."""
        all_keys = set(old.keys()) | set(new.keys())
        
        blended = {}
        for key in all_keys:
            old_val = old.get(key, 0.0)
            new_val = new.get(key, 0.0)
            blended[key] = weight_old * old_val + weight_new * new_val
        
        return blended


class BaselineAnalyzer:
    """
    Analyzes data against learned baselines.
    
    Phase 1: Provides descriptive statistics only.
    Phase 2+: Will support anomaly detection.
    """
    
    def __init__(self, baseline: UserBaseline):
        """
        Initialize analyzer with a baseline.
        
        Args:
            baseline: Learned user baseline
        """
        self.baseline = baseline
    
    def analyze_event(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
    ) -> Dict[str, any]:
        """
        Analyze an event against the baseline.
        
        Returns descriptive statistics, not anomaly flags.
        
        Args:
            event: Normalized event
            features: Computed features
            
        Returns:
            Dict of analysis results
        """
        results = {
            "user_id": event.user_id,
            "event_id": event.event_id,
            "baseline_mature": self.baseline.is_mature,
        }
        
        # Speed analysis
        if event.speed_ms is not None:
            speed_percentile = self._compute_percentile(
                event.speed_ms,
                self.baseline.speed_baseline
            )
            results["speed_percentile"] = speed_percentile
            results["speed_vs_mean_ratio"] = (
                event.speed_ms / self.baseline.speed_baseline.mean
                if self.baseline.speed_baseline.mean > 0 else None
            )
        
        # Update frequency analysis
        if features.update_rate_hz is not None:
            freq_percentile = self._compute_percentile(
                features.update_rate_hz,
                self.baseline.update_frequency_baseline
            )
            results["update_freq_percentile"] = freq_percentile
        
        # Activity analysis
        activity = event.activity_type.value
        if activity in self.baseline.activity_distribution:
            results["activity_typical_pct"] = self.baseline.activity_distribution[activity]
        else:
            results["activity_typical_pct"] = 0.0
        
        return results
    
    def analyze_trip(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
    ) -> Dict[str, any]:
        """
        Analyze a trip against the baseline.
        
        Args:
            events: Trip events
            features: Computed features
            
        Returns:
            Dict of trip-level analysis
        """
        if not events:
            return {}
        
        results = {
            "trip_id": events[0].trip_id,
            "user_id": events[0].user_id,
            "num_events": len(events),
            "baseline_mature": self.baseline.is_mature,
        }
        
        # Trip duration
        duration = events[-1].timestamp_unix - events[0].timestamp_unix
        duration_percentile = self._compute_percentile(
            duration,
            self.baseline.trip_duration_baseline
        )
        results["duration_seconds"] = duration
        results["duration_percentile"] = duration_percentile
        
        # Speed statistics
        speeds = [e.speed_ms for e in events if e.speed_ms is not None]
        if speeds:
            avg_speed = np.mean(speeds)
            results["avg_speed_ms"] = avg_speed
            results["avg_speed_percentile"] = self._compute_percentile(
                avg_speed,
                self.baseline.speed_baseline
            )
        
        # Activity breakdown vs baseline
        activities = [e.activity_type.value for e in events]
        trip_distribution = compute_activity_distribution(activities)
        
        distribution_diff = {}
        for activity, trip_pct in trip_distribution.items():
            baseline_pct = self.baseline.activity_distribution.get(activity, 0.0)
            distribution_diff[activity] = trip_pct - baseline_pct
        
        results["activity_distribution_diff"] = distribution_diff
        
        return results
    
    def _compute_percentile(
        self,
        value: float,
        baseline: BaselineMetrics
    ) -> Optional[float]:
        """
        Estimate what percentile a value falls in.
        
        Uses linear interpolation between known percentiles.
        """
        if baseline.sample_size == 0:
            return None
        
        percentiles = [
            (0, baseline.min_value),
            (5, baseline.p5),
            (25, baseline.p25),
            (50, baseline.median),
            (75, baseline.p75),
            (95, baseline.p95),
            (100, baseline.max_value),
        ]
        
        for i in range(len(percentiles) - 1):
            p1, v1 = percentiles[i]
            p2, v2 = percentiles[i + 1]
            
            if v1 <= value <= v2:
                if v2 == v1:
                    return (p1 + p2) / 2
                ratio = (value - v1) / (v2 - v1)
                return p1 + ratio * (p2 - p1)
        
        # Value outside known range
        if value < baseline.min_value:
            return 0.0
        return 100.0


def learn_user_baseline(
    user_id: int,
    events: List[NormalizedTrackingEvent],
    features: List[ComputedFeatures],
    scores: List[ConfidenceScore],
) -> UserBaseline:
    """
    Convenience function to learn a user baseline.
    
    Args:
        user_id: User identifier
        events: Normalized events
        features: Computed features
        scores: Confidence scores
        
    Returns:
        UserBaseline
    """
    learner = BaselineLearner()
    return learner.learn_baseline(user_id, events, features, scores)
