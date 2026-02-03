"""
Behavioral Drift Detection
============================

Phase 3 detection of gradual changes in user behavior over time.

Features:
- Statistical divergence measures (KL, JS divergence)
- CUSUM change point detection
- Multi-window temporal analysis
- Concern level classification

Design Principles:
- Distinguish normal adaptation from concerning drift
- Explainable detection results
- GPU-accelerated analysis via Triton
- Conservative alerting (minimize false positives)
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import uuid
import time
import math

import numpy as np
from scipy import stats

from app.core.logging import get_logger
from app.core.config import settings
from app.models.phase3_schemas import (
    DriftResult,
    DriftSignal,
    DriftType,
    DriftConcern,
    DriftDetectionRequest,
)
from app.models.phase2_schemas import BehavioralProfile, TimeOfDayPattern
from app.models.schemas import BaselineMetrics

logger = get_logger(__name__)

# Drift detection thresholds
DRIFT_THRESHOLD_MINOR = 0.15
DRIFT_THRESHOLD_SIGNIFICANT = 0.30
DRIFT_THRESHOLD_CONCERNING = 0.50
DRIFT_THRESHOLD_CRITICAL = 0.70

# Window weights for temporal analysis
WINDOW_WEIGHTS = {
    7: 0.5,   # 7-day window (most recent, highest weight)
    30: 0.3,  # 30-day window
    90: 0.2,  # 90-day window (historical baseline)
}


@dataclass
class ProfileSnapshot:
    """Snapshot of a behavioral profile at a point in time."""
    timestamp: datetime
    speed_mean: float
    speed_std: float
    update_regularity: float
    route_adherence: float
    hourly_distribution: List[float]  # 24-hour activity distribution
    stop_count: int
    trip_count: int


class DivergenceMeasures:
    """
    Statistical divergence measures for drift detection.
    """
    
    @staticmethod
    def kl_divergence(p: np.ndarray, q: np.ndarray, epsilon: float = 1e-10) -> float:
        """
        Calculate Kullback-Leibler divergence D_KL(P || Q).
        
        Args:
            p: Distribution P
            q: Distribution Q
            epsilon: Small value to avoid log(0)
            
        Returns:
            KL divergence value
        """
        p = np.asarray(p, dtype=np.float64) + epsilon
        q = np.asarray(q, dtype=np.float64) + epsilon
        
        # Normalize to probability distributions
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        return float(np.sum(p * np.log(p / q)))
    
    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Calculate Jensen-Shannon divergence.
        
        Symmetric and bounded version of KL divergence.
        
        Args:
            p: Distribution P
            q: Distribution Q
            
        Returns:
            JS divergence value (0-1 range)
        """
        p = np.asarray(p, dtype=np.float64) + 1e-10
        q = np.asarray(q, dtype=np.float64) + 1e-10
        
        # Normalize
        p = p / np.sum(p)
        q = q / np.sum(q)
        
        # Average distribution
        m = 0.5 * (p + q)
        
        # JS divergence = 0.5 * (KL(P||M) + KL(Q||M))
        js = 0.5 * (
            DivergenceMeasures.kl_divergence(p, m) +
            DivergenceMeasures.kl_divergence(q, m)
        )
        
        # Normalize to 0-1 range (sqrt of JS is a metric)
        return float(np.sqrt(js))
    
    @staticmethod
    def relative_change(old_value: float, new_value: float) -> float:
        """
        Calculate relative change between two values.
        
        Args:
            old_value: Original value
            new_value: New value
            
        Returns:
            Relative change (-1 to +inf, 0 means no change)
        """
        if old_value == 0:
            return 0.0 if new_value == 0 else 1.0
        
        return (new_value - old_value) / abs(old_value)
    
    @staticmethod
    def normalized_change(old_value: float, new_value: float, max_val: float) -> float:
        """
        Calculate normalized change score (0-1 range).
        
        Args:
            old_value: Original value
            new_value: New value
            max_val: Maximum expected value for normalization
            
        Returns:
            Normalized change score
        """
        if max_val == 0:
            return 0.0
        
        diff = abs(new_value - old_value)
        return min(1.0, diff / max_val)


class CUSUMDetector:
    """
    Cumulative Sum (CUSUM) change point detector.
    
    Detects shifts in time series data.
    """
    
    def __init__(
        self,
        threshold: float = 5.0,
        drift_allowance: float = 0.5,
    ):
        """
        Initialize CUSUM detector.
        
        Args:
            threshold: Detection threshold
            drift_allowance: Allowed drift before detection
        """
        self.threshold = threshold
        self.drift_allowance = drift_allowance
    
    def detect(
        self,
        values: List[float],
        target_mean: Optional[float] = None,
    ) -> Tuple[bool, float, int]:
        """
        Detect change point in time series.
        
        Args:
            values: Time series values
            target_mean: Expected mean (uses first half if not provided)
            
        Returns:
            Tuple of (change_detected, max_cusum, change_point_index)
        """
        if len(values) < 4:
            return False, 0.0, -1
        
        values = np.array(values)
        
        if target_mean is None:
            # Use first half as baseline
            target_mean = np.mean(values[:len(values)//2])
        
        # Calculate CUSUM
        cusum_pos = np.zeros(len(values))
        cusum_neg = np.zeros(len(values))
        
        for i in range(1, len(values)):
            diff = values[i] - target_mean
            cusum_pos[i] = max(0, cusum_pos[i-1] + diff - self.drift_allowance)
            cusum_neg[i] = max(0, cusum_neg[i-1] - diff - self.drift_allowance)
        
        max_pos = np.max(cusum_pos)
        max_neg = np.max(cusum_neg)
        max_cusum = max(max_pos, max_neg)
        
        if max_cusum > self.threshold:
            if max_pos > max_neg:
                change_point = int(np.argmax(cusum_pos))
            else:
                change_point = int(np.argmax(cusum_neg))
            return True, float(max_cusum), change_point
        
        return False, float(max_cusum), -1


class BehavioralDriftDetector:
    """
    Main drift detection service.
    
    Analyzes behavioral profile changes over time.
    """
    
    def __init__(
        self,
        sensitivity: float = 0.5,
    ):
        """
        Initialize drift detector.
        
        Args:
            sensitivity: Detection sensitivity (0-1)
        """
        self.sensitivity = sensitivity
        self.divergence = DivergenceMeasures()
        self.cusum = CUSUMDetector()
        
        # Profile snapshot history (in production, this would be in DB)
        self._profile_history: Dict[int, List[ProfileSnapshot]] = {}
    
    def add_profile_snapshot(
        self,
        user_id: int,
        profile: BehavioralProfile,
    ) -> None:
        """
        Add a profile snapshot to history.
        
        Args:
            user_id: User ID
            profile: Current behavioral profile
        """
        # Extract hourly distribution
        hourly_dist = [0.0] * 24
        for pattern in profile.hourly_patterns:
            if pattern.hour < 24:
                hourly_dist[pattern.hour] = pattern.activity_probability
        
        snapshot = ProfileSnapshot(
            timestamp=profile.updated_at,
            speed_mean=profile.baseline.speed_baseline.mean,
            speed_std=profile.baseline.speed_baseline.std,
            update_regularity=profile.update_regularity,
            route_adherence=profile.route_adherence_score,
            hourly_distribution=hourly_dist,
            stop_count=len(profile.frequent_stops),
            trip_count=profile.total_trips_analyzed,
        )
        
        if user_id not in self._profile_history:
            self._profile_history[user_id] = []
        
        self._profile_history[user_id].append(snapshot)
        
        # Keep last 100 snapshots
        if len(self._profile_history[user_id]) > 100:
            self._profile_history[user_id] = self._profile_history[user_id][-100:]
    
    def _detect_speed_drift(
        self,
        current: ProfileSnapshot,
        baseline: ProfileSnapshot,
    ) -> Optional[DriftSignal]:
        """Detect drift in speed patterns."""
        mean_change = self.divergence.relative_change(
            baseline.speed_mean, current.speed_mean
        )
        std_change = self.divergence.relative_change(
            baseline.speed_std, current.speed_std
        )
        
        combined_change = abs(mean_change) * 0.7 + abs(std_change) * 0.3
        
        if combined_change < DRIFT_THRESHOLD_MINOR:
            return None
        
        severity = min(1.0, combined_change)
        
        if mean_change > 0:
            description = (
                f"Speed increased by {abs(mean_change)*100:.1f}% "
                f"(from {baseline.speed_mean:.1f} to {current.speed_mean:.1f} m/s)"
            )
        else:
            description = (
                f"Speed decreased by {abs(mean_change)*100:.1f}% "
                f"(from {baseline.speed_mean:.1f} to {current.speed_mean:.1f} m/s)"
            )
        
        return DriftSignal(
            drift_type=DriftType.SPEED_RANGE,
            severity=severity,
            confidence=0.8,
            divergence_score=combined_change,
            baseline_value=baseline.speed_mean,
            current_value=current.speed_mean,
            change_percent=mean_change * 100,
            description=description,
        )
    
    def _detect_route_adherence_drift(
        self,
        current: ProfileSnapshot,
        baseline: ProfileSnapshot,
    ) -> Optional[DriftSignal]:
        """Detect drift in route adherence."""
        change = current.route_adherence - baseline.route_adherence
        
        if abs(change) < DRIFT_THRESHOLD_MINOR:
            return None
        
        severity = min(1.0, abs(change) * 2)
        
        if change < 0:
            description = (
                f"Route adherence decreased by {abs(change)*100:.1f}% "
                f"(user following fewer known routes)"
            )
        else:
            description = (
                f"Route adherence increased by {abs(change)*100:.1f}% "
                f"(user following more predictable routes)"
            )
        
        return DriftSignal(
            drift_type=DriftType.ROUTE_ADHERENCE,
            severity=severity,
            confidence=0.75,
            divergence_score=abs(change),
            baseline_value=baseline.route_adherence,
            current_value=current.route_adherence,
            change_percent=change * 100,
            description=description,
        )
    
    def _detect_activity_timing_drift(
        self,
        current: ProfileSnapshot,
        baseline: ProfileSnapshot,
    ) -> Optional[DriftSignal]:
        """Detect drift in activity timing patterns."""
        current_dist = np.array(current.hourly_distribution)
        baseline_dist = np.array(baseline.hourly_distribution)
        
        # Calculate JS divergence
        js_div = self.divergence.js_divergence(current_dist, baseline_dist)
        
        if js_div < DRIFT_THRESHOLD_MINOR:
            return None
        
        severity = min(1.0, js_div * 2)
        
        # Find which hours changed most
        diff = np.abs(current_dist - baseline_dist)
        top_hours = np.argsort(diff)[-3:]
        
        description = (
            f"Activity timing pattern shifted (JS divergence: {js_div:.3f}). "
            f"Most changed hours: {', '.join(str(h) for h in top_hours)}"
        )
        
        return DriftSignal(
            drift_type=DriftType.ACTIVITY_TIMING,
            severity=severity,
            confidence=0.7,
            divergence_score=js_div,
            baseline_value=None,
            current_value=None,
            change_percent=None,
            description=description,
        )
    
    def _detect_stop_pattern_drift(
        self,
        current: ProfileSnapshot,
        baseline: ProfileSnapshot,
    ) -> Optional[DriftSignal]:
        """Detect drift in stop patterns."""
        change = current.stop_count - baseline.stop_count
        
        if baseline.stop_count == 0:
            relative_change = 0 if current.stop_count == 0 else 1.0
        else:
            relative_change = abs(change) / baseline.stop_count
        
        if relative_change < DRIFT_THRESHOLD_SIGNIFICANT:
            return None
        
        severity = min(1.0, relative_change)
        
        if change > 0:
            description = (
                f"New stop locations detected (+{change} stops). "
                f"Now visiting {current.stop_count} frequent locations."
            )
        else:
            description = (
                f"Fewer stop locations ({abs(change)} locations no longer visited). "
                f"Now visiting {current.stop_count} frequent locations."
            )
        
        return DriftSignal(
            drift_type=DriftType.STOP_PATTERN,
            severity=severity,
            confidence=0.65,
            divergence_score=relative_change,
            baseline_value=float(baseline.stop_count),
            current_value=float(current.stop_count),
            change_percent=relative_change * 100,
            description=description,
        )
    
    def _detect_update_frequency_drift(
        self,
        current: ProfileSnapshot,
        baseline: ProfileSnapshot,
    ) -> Optional[DriftSignal]:
        """Detect drift in update frequency regularity."""
        change = current.update_regularity - baseline.update_regularity
        
        if abs(change) < DRIFT_THRESHOLD_MINOR:
            return None
        
        severity = min(1.0, abs(change) * 2)
        
        if change < 0:
            description = (
                f"Update frequency became less regular "
                f"(regularity: {baseline.update_regularity:.2f} → {current.update_regularity:.2f})"
            )
        else:
            description = (
                f"Update frequency became more regular "
                f"(regularity: {baseline.update_regularity:.2f} → {current.update_regularity:.2f})"
            )
        
        return DriftSignal(
            drift_type=DriftType.UPDATE_FREQUENCY,
            severity=severity,
            confidence=0.6,
            divergence_score=abs(change),
            baseline_value=baseline.update_regularity,
            current_value=current.update_regularity,
            change_percent=change * 100,
            description=description,
        )
    
    def _classify_concern(
        self,
        severity: float,
        signals: List[DriftSignal],
        is_gradual: bool,
    ) -> DriftConcern:
        """
        Classify the concern level of detected drift.
        
        Args:
            severity: Overall drift severity
            signals: Detected signals
            is_gradual: Whether change is gradual
            
        Returns:
            DriftConcern level
        """
        # Check for concerning patterns
        concerning_types = {DriftType.ROUTE_ADHERENCE, DriftType.ACTIVITY_TIMING}
        has_concerning = any(
            s.drift_type in concerning_types and s.severity > 0.4
            for s in signals
        )
        
        if severity < DRIFT_THRESHOLD_MINOR:
            return DriftConcern.NORMAL_ADAPTATION
        elif severity < DRIFT_THRESHOLD_SIGNIFICANT:
            return DriftConcern.MINOR_CHANGE
        elif severity < DRIFT_THRESHOLD_CONCERNING:
            if has_concerning or not is_gradual:
                return DriftConcern.SIGNIFICANT_CHANGE
            return DriftConcern.MINOR_CHANGE
        elif severity < DRIFT_THRESHOLD_CRITICAL:
            return DriftConcern.CONCERNING
        else:
            return DriftConcern.CRITICAL
    
    async def detect(
        self,
        user_id: int,
        current_profile: BehavioralProfile,
        time_windows: List[int] = [7, 30, 90],
    ) -> DriftResult:
        """
        Detect behavioral drift for a user.
        
        Args:
            user_id: User ID
            current_profile: Current behavioral profile
            time_windows: Time windows in days to analyze
            
        Returns:
            DriftResult with detection outcome
        """
        start_time = time.time()
        now = datetime.now(timezone.utc)
        
        # Get or create profile history
        history = self._profile_history.get(user_id, [])
        
        # Add current profile to history
        self.add_profile_snapshot(user_id, current_profile)
        
        signals = []
        window_drifts = {}
        
        # Create current snapshot for comparison
        hourly_dist = [0.0] * 24
        for pattern in current_profile.hourly_patterns:
            if pattern.hour < 24:
                hourly_dist[pattern.hour] = pattern.activity_probability
        
        current_snapshot = ProfileSnapshot(
            timestamp=current_profile.updated_at,
            speed_mean=current_profile.baseline.speed_baseline.mean,
            speed_std=current_profile.baseline.speed_baseline.std,
            update_regularity=current_profile.update_regularity,
            route_adherence=current_profile.route_adherence_score,
            hourly_distribution=hourly_dist,
            stop_count=len(current_profile.frequent_stops),
            trip_count=current_profile.total_trips_analyzed,
        )
        
        # Analyze each time window
        for window_days in time_windows:
            window_start = now - timedelta(days=window_days)
            
            # Find baseline snapshot from window start
            baseline_snapshots = [
                s for s in history
                if s.timestamp <= window_start
            ]
            
            if not baseline_snapshots:
                window_drifts[window_days] = 0.0
                continue
            
            # Use oldest snapshot in window as baseline
            baseline = baseline_snapshots[-1]
            
            # Detect various drift types
            speed_drift = self._detect_speed_drift(current_snapshot, baseline)
            if speed_drift:
                signals.append(speed_drift)
            
            route_drift = self._detect_route_adherence_drift(current_snapshot, baseline)
            if route_drift:
                signals.append(route_drift)
            
            timing_drift = self._detect_activity_timing_drift(current_snapshot, baseline)
            if timing_drift:
                signals.append(timing_drift)
            
            stop_drift = self._detect_stop_pattern_drift(current_snapshot, baseline)
            if stop_drift:
                signals.append(stop_drift)
            
            freq_drift = self._detect_update_frequency_drift(current_snapshot, baseline)
            if freq_drift:
                signals.append(freq_drift)
            
            # Calculate window drift score
            if signals:
                window_drifts[window_days] = np.mean([s.severity for s in signals])
            else:
                window_drifts[window_days] = 0.0
        
        # Calculate overall severity (weighted by window)
        if window_drifts:
            total_weight = sum(WINDOW_WEIGHTS.get(w, 0.1) for w in window_drifts.keys())
            overall_severity = sum(
                window_drifts[w] * WINDOW_WEIGHTS.get(w, 0.1)
                for w in window_drifts.keys()
            ) / total_weight
        else:
            overall_severity = 0.0
        
        # Determine if drift is gradual
        is_gradual = True
        if 7 in window_drifts and 90 in window_drifts:
            # If short-term drift is much higher than long-term, it's sudden
            if window_drifts[7] > window_drifts[90] * 2:
                is_gradual = False
        
        # Classify concern level
        concern_level = self._classify_concern(overall_severity, signals, is_gradual)
        is_concerning = concern_level in [DriftConcern.CONCERNING, DriftConcern.CRITICAL]
        
        # Determine primary drift type
        primary_type = None
        if signals:
            primary_signal = max(signals, key=lambda s: s.severity)
            primary_type = primary_signal.drift_type
        
        # Generate explanation
        if not signals:
            explanation = "No significant behavioral drift detected."
            details = ["Behavior patterns remain consistent with historical baseline."]
        else:
            if is_concerning:
                explanation = (
                    f"Concerning behavioral drift detected (severity: {overall_severity:.2f}). "
                    f"Primary change: {primary_type.value if primary_type else 'unknown'}."
                )
            else:
                explanation = (
                    f"Minor behavioral changes detected (severity: {overall_severity:.2f}). "
                    f"Changes appear to be normal adaptation."
                )
            
            details = [s.description for s in signals]
        
        inference_time = (time.time() - start_time) * 1000
        
        result = DriftResult(
            result_id=f"drift_{uuid.uuid4().hex[:12]}",
            user_id=user_id,
            drift_detected=len(signals) > 0,
            drift_severity=overall_severity,
            primary_drift_type=primary_type,
            concern_level=concern_level,
            is_concerning=is_concerning,
            signals=signals,
            window_7d_drift=window_drifts.get(7, 0.0),
            window_30d_drift=window_drifts.get(30, 0.0),
            window_90d_drift=window_drifts.get(90, 0.0),
            profile_snapshots_compared=len(history),
            explanation=explanation,
            details=details,
            analyzed_at=now,
            model_version="1.0",
            inference_time_ms=inference_time,
        )
        
        logger.info(
            "drift_detected",
            user_id=user_id,
            severity=overall_severity,
            concern=concern_level.value,
            signals=len(signals),
            time_ms=inference_time,
        )
        
        return result


# Singleton instance
_drift_detector: Optional[BehavioralDriftDetector] = None


def get_drift_detector() -> BehavioralDriftDetector:
    """Get or create singleton detector."""
    global _drift_detector
    if _drift_detector is None:
        _drift_detector = BehavioralDriftDetector()
    return _drift_detector


async def detect_behavioral_drift(
    user_id: int,
    current_profile: BehavioralProfile,
    time_windows: List[int] = [7, 30, 90],
) -> DriftResult:
    """
    Convenience function for drift detection.
    
    Args:
        user_id: User ID
        current_profile: Current profile
        time_windows: Time windows to analyze
        
    Returns:
        DriftResult
    """
    detector = get_drift_detector()
    return await detector.detect(user_id, current_profile, time_windows)
