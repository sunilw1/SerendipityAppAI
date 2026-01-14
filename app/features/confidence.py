"""
Confidence Scoring Module
===========================

Computes confidence scores for tracking data.

The confidence score represents the reliability of a location point,
considering:
1. GPS accuracy
2. Speed validity
3. Acceleration validity
4. Temporal consistency
5. Activity consistency
6. Signal continuity

Score Range: [0, 1] where 1 = highest confidence

Design Principles:
- Transparent, explainable scoring
- Weighted combination of factors
- Conservative scoring (safety-first)
- No machine learning in Phase 1
"""

from typing import List, Optional, Tuple

import numpy as np

from app.core.constants import (
    ConfidenceWeights,
    DataQualityFlag,
    GPSAccuracyTiers,
    SpeedLimits,
    AccelerationLimits,
    TimeConstants,
    ActivityType,
)
from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    ComputedFeatures,
    ConfidenceScore,
    NormalizedTrackingEvent,
)

logger = get_logger(__name__)


class ConfidenceScorer:
    """
    Computes confidence scores for tracking events.
    
    Each component score is in [0, 1], and the overall score
    is a weighted combination of components.
    """
    
    def __init__(
        self,
        weights: Optional[dict] = None,
        max_speed_ms: float = settings.max_realistic_speed_ms,
        max_acceleration_ms2: float = settings.max_realistic_acceleration_ms2,
    ):
        """
        Initialize confidence scorer.
        
        Args:
            weights: Custom weights for components (optional)
            max_speed_ms: Maximum realistic speed
            max_acceleration_ms2: Maximum realistic acceleration
        """
        self.weights = weights or {
            "gps_accuracy": ConfidenceWeights.GPS_ACCURACY,
            "speed_validity": ConfidenceWeights.SPEED_VALIDITY,
            "acceleration_validity": ConfidenceWeights.ACCELERATION_VALIDITY,
            "temporal_consistency": ConfidenceWeights.TEMPORAL_CONSISTENCY,
            "activity_consistency": ConfidenceWeights.ACTIVITY_CONSISTENCY,
            "signal_continuity": ConfidenceWeights.SIGNAL_CONTINUITY,
        }
        self.max_speed_ms = max_speed_ms
        self.max_acceleration_ms2 = max_acceleration_ms2
    
    def score(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures
    ) -> ConfidenceScore:
        """
        Compute confidence score for a single event.
        
        Args:
            event: Normalized tracking event
            features: Computed features for the event
            
        Returns:
            ConfidenceScore with component breakdown
        """
        flags = []
        
        # 1. GPS Accuracy Score
        gps_score = self._score_gps_accuracy(event, flags)
        
        # 2. Speed Validity Score
        speed_score = self._score_speed_validity(event, features, flags)
        
        # 3. Acceleration Validity Score
        accel_score = self._score_acceleration_validity(features, flags)
        
        # 4. Temporal Consistency Score
        temporal_score = self._score_temporal_consistency(features, flags)
        
        # 5. Activity Consistency Score
        activity_score = self._score_activity_consistency(event, features, flags)
        
        # 6. Signal Continuity Score
        continuity_score = self._score_signal_continuity(event, features, flags)
        
        # Compute weighted overall score
        overall = (
            self.weights["gps_accuracy"] * gps_score +
            self.weights["speed_validity"] * speed_score +
            self.weights["acceleration_validity"] * accel_score +
            self.weights["temporal_consistency"] * temporal_score +
            self.weights["activity_consistency"] * activity_score +
            self.weights["signal_continuity"] * continuity_score
        )
        
        # Penalize for quality flags
        flag_penalty = self._compute_flag_penalty(event.quality_flags)
        overall = max(0.0, overall - flag_penalty)
        
        return ConfidenceScore(
            overall=round(overall, 4),
            gps_accuracy_score=round(gps_score, 4),
            speed_validity_score=round(speed_score, 4),
            acceleration_validity_score=round(accel_score, 4),
            temporal_consistency_score=round(temporal_score, 4),
            activity_consistency_score=round(activity_score, 4),
            signal_continuity_score=round(continuity_score, 4),
            flags=flags,
        )
    
    def score_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures]
    ) -> List[ConfidenceScore]:
        """
        Score a batch of events.
        
        Args:
            events: List of normalized events
            features: Corresponding computed features
            
        Returns:
            List of confidence scores
        """
        if len(events) != len(features):
            raise ValueError("Events and features must have same length")
        
        return [self.score(e, f) for e, f in zip(events, features)]
    
    def _score_gps_accuracy(
        self,
        event: NormalizedTrackingEvent,
        flags: List[str]
    ) -> float:
        """
        Score GPS accuracy.
        
        Excellent accuracy (< 5m) → 1.0
        Good accuracy (< 10m) → 0.9
        Moderate accuracy (< 25m) → 0.7
        Poor accuracy (< 50m) → 0.5
        Unreliable (> 100m) → 0.1
        Missing → 0.3 (conservative default)
        """
        accuracy = event.gps_accuracy_meters
        
        if accuracy is None:
            flags.append("missing_gps_accuracy")
            return 0.3  # Conservative default
        
        if accuracy <= GPSAccuracyTiers.EXCELLENT:
            return 1.0
        elif accuracy <= GPSAccuracyTiers.GOOD:
            return 0.9
        elif accuracy <= GPSAccuracyTiers.MODERATE:
            return 0.7
        elif accuracy <= GPSAccuracyTiers.POOR:
            flags.append("poor_gps_accuracy")
            return 0.5
        elif accuracy <= GPSAccuracyTiers.UNRELIABLE:
            flags.append("unreliable_gps_accuracy")
            return 0.3
        else:
            flags.append("very_poor_gps_accuracy")
            return 0.1
    
    def _score_speed_validity(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        flags: List[str]
    ) -> float:
        """
        Score speed validity.
        
        Considers:
        - Is reported speed physically possible?
        - Does calculated speed match reported speed?
        - Is speed appropriate for activity type?
        """
        score = 1.0
        
        # Check reported speed
        if event.speed_ms is not None:
            if event.speed_ms > self.max_speed_ms:
                flags.append("impossible_speed")
                score = min(score, 0.1)
            elif event.speed_ms > self.max_speed_ms * 0.8:
                flags.append("very_high_speed")
                score = min(score, 0.5)
            
            # Check against activity type
            max_for_activity = event.activity_type.max_speed_ms
            if event.speed_ms > max_for_activity * 1.5:
                flags.append("speed_activity_mismatch")
                score = min(score, 0.6)
        else:
            # Missing speed
            score = min(score, 0.7)
        
        # Check speed difference (reported vs calculated)
        if features.speed_difference_ms is not None:
            diff_ratio = abs(features.speed_difference_ms)
            if features.calculated_speed_ms and features.calculated_speed_ms > 1.0:
                diff_ratio = diff_ratio / features.calculated_speed_ms
                
                if diff_ratio > 0.5:  # 50% difference
                    flags.append("large_speed_discrepancy")
                    score = min(score, 0.5)
                elif diff_ratio > 0.25:  # 25% difference
                    score = min(score, 0.7)
        
        return score
    
    def _score_acceleration_validity(
        self,
        features: ComputedFeatures,
        flags: List[str]
    ) -> float:
        """
        Score acceleration validity.
        
        High acceleration values may indicate:
        - GPS jumps
        - Sensor errors
        - Data processing issues
        """
        if features.acceleration_ms2 is None:
            return 0.8  # First point or missing data
        
        accel = abs(features.acceleration_ms2)
        
        if accel > self.max_acceleration_ms2:
            flags.append("impossible_acceleration")
            return 0.2
        elif accel > AccelerationLimits.VEHICLE_EMERGENCY:
            flags.append("extreme_acceleration")
            return 0.4
        elif accel > AccelerationLimits.VEHICLE_NORMAL:
            return 0.7
        elif accel > AccelerationLimits.HUMAN_COMFORTABLE:
            return 0.85
        else:
            return 1.0
    
    def _score_temporal_consistency(
        self,
        features: ComputedFeatures,
        flags: List[str]
    ) -> float:
        """
        Score temporal consistency.
        
        Considers:
        - Time gap since previous point
        - Update frequency
        - Timestamp ordering issues
        """
        if features.time_delta_seconds is None:
            return 0.8  # First point
        
        delta = features.time_delta_seconds
        
        # Check for gaps
        if features.has_time_gap:
            flags.append(f"time_gap_{int(delta)}s")
            if delta > TimeConstants.MAX_ACCEPTABLE_GAP:
                return 0.2
            elif delta > TimeConstants.MAX_NORMAL_GAP * 2:
                return 0.4
            else:
                return 0.6
        
        # Check for too-rapid updates
        if delta < TimeConstants.MIN_UPDATE_INTERVAL:
            flags.append("rapid_update")
            return 0.6
        
        # Normal update rate
        if delta <= 5.0:  # <= 5 seconds
            return 1.0
        elif delta <= 15.0:  # <= 15 seconds
            return 0.9
        elif delta <= 30.0:  # <= 30 seconds
            return 0.8
        else:
            return 0.7
    
    def _score_activity_consistency(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        flags: List[str]
    ) -> float:
        """
        Score activity type consistency.
        
        Checks if activity type matches movement characteristics.
        """
        score = 1.0
        
        activity = event.activity_type
        is_moving = event.is_moving
        speed = event.speed_ms or 0.0
        calc_speed = features.calculated_speed_ms or 0.0
        
        # Activity confidence factor
        confidence_factor = event.activity_confidence / 100.0
        
        # Still but moving
        if activity == ActivityType.STILL:
            if is_moving and speed > 1.0:
                flags.append("still_but_moving")
                score = min(score, 0.5)
            elif calc_speed > 2.0:
                flags.append("still_but_displaced")
                score = min(score, 0.6)
        
        # Moving activity but stationary
        elif activity in [ActivityType.RUNNING, ActivityType.WALKING, ActivityType.IN_VEHICLE]:
            if not is_moving and speed < 0.5 and calc_speed < 0.5:
                flags.append("active_but_stationary")
                score = min(score, 0.6)
        
        # Weight by activity confidence
        score = score * (0.5 + 0.5 * confidence_factor)
        
        return score
    
    def _score_signal_continuity(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        flags: List[str]
    ) -> float:
        """
        Score signal continuity.
        
        Checks for:
        - Teleportation
        - GPS drift
        - Smooth trajectory
        """
        score = 1.0
        
        # Check for teleportation flag
        if DataQualityFlag.TELEPORTATION.value in event.quality_flags:
            flags.append("teleportation_detected")
            return 0.1
        
        # Check for GPS drift
        if DataQualityFlag.GPS_DRIFT.value in event.quality_flags:
            flags.append("gps_drift_detected")
            score = min(score, 0.5)
        
        # Large bearing changes can indicate jumps
        if features.bearing_change_degrees is not None:
            if abs(features.bearing_change_degrees) > 150:
                # Near-180 degree turn at speed is unusual
                if features.calculated_speed_ms and features.calculated_speed_ms > 5.0:
                    flags.append("sharp_direction_change")
                    score = min(score, 0.7)
        
        return score
    
    def _compute_flag_penalty(
        self,
        quality_flags: List[str]
    ) -> float:
        """
        Compute penalty based on quality flags.
        
        Severe flags get higher penalties.
        """
        penalty = 0.0
        
        severe_flags = {
            DataQualityFlag.INVALID_COORDINATES.value: 0.5,
            DataQualityFlag.TELEPORTATION.value: 0.4,
            DataQualityFlag.TIMESTAMP_OUT_OF_ORDER.value: 0.3,
        }
        
        moderate_flags = {
            DataQualityFlag.UNREALISTIC_SPEED.value: 0.15,
            DataQualityFlag.UNREALISTIC_ACCELERATION.value: 0.1,
            DataQualityFlag.GPS_DRIFT.value: 0.1,
            DataQualityFlag.LOW_GPS_ACCURACY.value: 0.05,
        }
        
        minor_flags = {
            DataQualityFlag.DUPLICATE_TIMESTAMP.value: 0.05,
            DataQualityFlag.LARGE_TIME_GAP.value: 0.05,
            DataQualityFlag.MISSING_SPEED.value: 0.02,
            DataQualityFlag.MISSING_ACCURACY.value: 0.02,
        }
        
        for flag in quality_flags:
            if flag in severe_flags:
                penalty += severe_flags[flag]
            elif flag in moderate_flags:
                penalty += moderate_flags[flag]
            elif flag in minor_flags:
                penalty += minor_flags[flag]
        
        return min(penalty, 0.8)  # Cap at 0.8 to avoid zero scores


class BatchConfidenceScorer:
    """
    Scores confidence for batches with statistical context.
    
    Uses batch statistics to inform scoring (e.g., relative
    accuracy within the batch).
    """
    
    def __init__(self):
        self.scorer = ConfidenceScorer()
    
    def score_trip(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures]
    ) -> Tuple[List[ConfidenceScore], dict]:
        """
        Score all events in a trip with statistical context.
        
        Args:
            events: Normalized events
            features: Computed features
            
        Returns:
            Tuple of (confidence scores, trip statistics)
        """
        # Individual scoring
        scores = self.scorer.score_batch(events, features)
        
        # Trip-level statistics
        overall_scores = [s.overall for s in scores]
        stats = {
            "mean_confidence": np.mean(overall_scores),
            "median_confidence": np.median(overall_scores),
            "min_confidence": np.min(overall_scores),
            "max_confidence": np.max(overall_scores),
            "std_confidence": np.std(overall_scores),
            "high_confidence_pct": np.mean([s >= 0.75 for s in overall_scores]),
            "low_confidence_pct": np.mean([s < 0.5 for s in overall_scores]),
            "total_events": len(events),
        }
        
        return scores, stats


def score_confidence(
    event: NormalizedTrackingEvent,
    features: ComputedFeatures
) -> ConfidenceScore:
    """
    Convenience function to score confidence.
    
    Args:
        event: Normalized event
        features: Computed features
        
    Returns:
        ConfidenceScore
    """
    scorer = ConfidenceScorer()
    return scorer.score(event, features)


def score_confidence_batch(
    events: List[NormalizedTrackingEvent],
    features: List[ComputedFeatures]
) -> List[ConfidenceScore]:
    """
    Convenience function to score confidence for a batch.
    
    Args:
        events: List of normalized events
        features: List of computed features
        
    Returns:
        List of confidence scores
    """
    scorer = ConfidenceScorer()
    return scorer.score_batch(events, features)
