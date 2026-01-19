"""
Spoofing & Tampering Detection Module
======================================

Pattern-based detection of GPS spoofing and data tampering.

Detects:
1. Physical impossibility checks (teleportation, impossible speeds)
2. Repeated teleport patterns
3. Inconsistent movement vs timestamps
4. Accuracy degradation patterns
5. Session integrity violations
6. Synthetic/artificial patterns

Design Principles:
- Outputs probabilistic likelihood, NOT binary flags
- Backend-only computation
- Explainable detections
- Configurable thresholds
- Low false positive rate
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set
import numpy as np

from app.core.logging import get_logger
from app.core.constants import SpeedLimits, AccelerationLimits, TimeConstants
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
)
from app.models.phase2_schemas import (
    SpoofingIndicator,
    SpoofingSignal,
    SpoofingResult,
)
from app.utils.geo import haversine_distance

logger = get_logger(__name__)


class PhysicalConstraintChecker:
    """
    Checks for physically impossible movements.
    
    Detects:
    - Teleportation (impossible distance in time)
    - Impossible speeds
    - Impossible accelerations
    """
    
    def __init__(
        self,
        max_realistic_speed_ms: float = SpeedLimits.AIRCRAFT,
        max_realistic_acceleration_ms2: float = AccelerationLimits.MAXIMUM_REALISTIC,
        teleport_threshold_factor: float = 1.2,
    ):
        """
        Initialize physical constraint checker.
        
        Args:
            max_realistic_speed_ms: Maximum possible speed
            max_realistic_acceleration_ms2: Maximum possible acceleration
            teleport_threshold_factor: Factor above max speed for teleportation
        """
        self.max_speed = max_realistic_speed_ms
        self.max_accel = max_realistic_acceleration_ms2
        self.teleport_factor = teleport_threshold_factor
    
    def check(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        prev_event: Optional[NormalizedTrackingEvent] = None,
    ) -> List[SpoofingSignal]:
        """Check for physical impossibilities."""
        signals = []
        
        # Check reported speed
        if event.speed_ms is not None and event.speed_ms > self.max_speed:
            signals.append(SpoofingSignal(
                indicator=SpoofingIndicator.IMPOSSIBLE_SPEED,
                likelihood=min(1.0, event.speed_ms / (self.max_speed * 2)),
                confidence=0.95,
                description=f"Reported speed {event.speed_ms:.1f} m/s exceeds maximum realistic {self.max_speed:.0f} m/s",
                evidence={
                    "reported_speed_ms": event.speed_ms,
                    "max_realistic_ms": self.max_speed,
                    "ratio": event.speed_ms / self.max_speed,
                },
            ))
        
        # Check calculated speed (teleportation)
        if features.calculated_speed_ms is not None:
            if features.calculated_speed_ms > self.max_speed * self.teleport_factor:
                teleport_ratio = features.calculated_speed_ms / self.max_speed
                signals.append(SpoofingSignal(
                    indicator=SpoofingIndicator.TELEPORTATION,
                    likelihood=min(1.0, (teleport_ratio - 1.0) / 2.0),
                    confidence=0.9,
                    description=f"Movement implies {features.calculated_speed_ms:.1f} m/s - likely teleportation",
                    evidence={
                        "calculated_speed_ms": features.calculated_speed_ms,
                        "distance_meters": features.distance_meters,
                        "time_delta_seconds": features.time_delta_seconds,
                        "teleport_ratio": teleport_ratio,
                    },
                ))
        
        # Check acceleration
        if features.acceleration_ms2 is not None:
            if abs(features.acceleration_ms2) > self.max_accel:
                accel_ratio = abs(features.acceleration_ms2) / self.max_accel
                signals.append(SpoofingSignal(
                    indicator=SpoofingIndicator.IMPOSSIBLE_SPEED,
                    likelihood=min(1.0, (accel_ratio - 1.0)),
                    confidence=0.85,
                    description=f"Acceleration {features.acceleration_ms2:.1f} m/s² exceeds realistic {self.max_accel:.0f} m/s²",
                    evidence={
                        "acceleration_ms2": features.acceleration_ms2,
                        "max_realistic_ms2": self.max_accel,
                        "ratio": accel_ratio,
                    },
                ))
        
        return signals


class TeleportPatternDetector:
    """
    Detects repeated teleportation patterns.
    
    Multiple teleportation events suggest systematic spoofing
    rather than occasional GPS glitches.
    """
    
    def __init__(
        self,
        window_size: int = 10,
        teleport_speed_threshold: float = SpeedLimits.AIRCRAFT * 1.2,
        min_teleports_for_pattern: int = 3,
    ):
        """
        Initialize teleport pattern detector.
        
        Args:
            window_size: Number of events to consider
            teleport_speed_threshold: Speed threshold for teleportation
            min_teleports_for_pattern: Minimum teleports for pattern
        """
        self.window_size = window_size
        self.teleport_threshold = teleport_speed_threshold
        self.min_teleports = min_teleports_for_pattern
    
    def detect(
        self,
        features: List[ComputedFeatures],
    ) -> List[Tuple[int, SpoofingSignal]]:
        """
        Detect teleport patterns in a sequence.
        
        Returns:
            List of (index, signal) tuples
        """
        results = []
        
        # Find teleportation events
        teleport_indices = []
        for i, feat in enumerate(features):
            if feat.calculated_speed_ms is not None:
                if feat.calculated_speed_ms > self.teleport_threshold:
                    teleport_indices.append(i)
        
        if len(teleport_indices) < self.min_teleports:
            return results
        
        # Check for patterns (clusters of teleports)
        for i in range(len(teleport_indices) - self.min_teleports + 1):
            window_teleports = []
            for j in range(i, min(i + self.window_size, len(teleport_indices))):
                if teleport_indices[j] - teleport_indices[i] <= self.window_size:
                    window_teleports.append(teleport_indices[j])
            
            if len(window_teleports) >= self.min_teleports:
                # Pattern detected
                likelihood = min(1.0, len(window_teleports) / self.window_size)
                
                for idx in window_teleports:
                    results.append((idx, SpoofingSignal(
                        indicator=SpoofingIndicator.REPEATED_TELEPORT,
                        likelihood=likelihood,
                        confidence=0.85,
                        description=f"Repeated teleportation pattern: {len(window_teleports)} in {self.window_size} events",
                        evidence={
                            "teleport_count": len(window_teleports),
                            "window_size": self.window_size,
                            "pattern_density": len(window_teleports) / self.window_size,
                        },
                    )))
        
        return results


class TimestampConsistencyChecker:
    """
    Checks for timestamp manipulation or clock drift.
    
    Detects:
    - Out of order timestamps
    - Impossible time gaps
    - Clock drift patterns
    """
    
    def __init__(
        self,
        max_backward_drift_seconds: float = 1.0,
        max_clock_drift_rate: float = 0.01,  # 1% drift
    ):
        """
        Initialize timestamp checker.
        
        Args:
            max_backward_drift_seconds: Max allowed backward time
            max_clock_drift_rate: Max allowed clock drift rate
        """
        self.max_backward = max_backward_drift_seconds
        self.max_drift_rate = max_clock_drift_rate
    
    def check(
        self,
        events: List[NormalizedTrackingEvent],
    ) -> List[Tuple[int, SpoofingSignal]]:
        """Check for timestamp inconsistencies."""
        results = []
        
        for i in range(1, len(events)):
            prev_ts = events[i-1].timestamp_unix
            curr_ts = events[i].timestamp_unix
            
            time_diff = curr_ts - prev_ts
            
            # Backward timestamp
            if time_diff < -self.max_backward:
                results.append((i, SpoofingSignal(
                    indicator=SpoofingIndicator.TIMESTAMP_MANIPULATION,
                    likelihood=min(1.0, abs(time_diff) / 60),  # Scale to 1 at 60s backward
                    confidence=0.9,
                    description=f"Backward timestamp: {time_diff:.1f}s earlier than previous",
                    evidence={
                        "time_diff_seconds": time_diff,
                        "previous_timestamp": prev_ts,
                        "current_timestamp": curr_ts,
                    },
                )))
            
            # Very large gap might indicate manipulation
            if time_diff > TimeConstants.MAX_ACCEPTABLE_GAP * 10:  # 50 minutes
                results.append((i, SpoofingSignal(
                    indicator=SpoofingIndicator.TIMESTAMP_MANIPULATION,
                    likelihood=0.3,  # Lower confidence - could be legitimate
                    confidence=0.5,
                    description=f"Very large time gap: {time_diff/60:.1f} minutes",
                    evidence={
                        "time_gap_seconds": time_diff,
                        "time_gap_minutes": time_diff / 60,
                    },
                )))
        
        return results


class AccuracyDegradationDetector:
    """
    Detects suspicious accuracy degradation patterns.
    
    Sudden or systematic accuracy changes can indicate
    spoofing or signal manipulation.
    """
    
    def __init__(
        self,
        degradation_ratio_threshold: float = 5.0,
        sustained_poor_accuracy_m: float = 100.0,
        sustained_window: int = 5,
    ):
        """
        Initialize accuracy degradation detector.
        
        Args:
            degradation_ratio_threshold: Ratio for sudden degradation
            sustained_poor_accuracy_m: Threshold for poor accuracy
            sustained_window: Window for sustained poor accuracy
        """
        self.degradation_ratio = degradation_ratio_threshold
        self.poor_accuracy = sustained_poor_accuracy_m
        self.sustained_window = sustained_window
    
    def detect(
        self,
        events: List[NormalizedTrackingEvent],
    ) -> List[Tuple[int, SpoofingSignal]]:
        """Detect accuracy degradation patterns."""
        results = []
        
        accuracies = [
            e.gps_accuracy_meters if e.gps_accuracy_meters is not None else 50.0
            for e in events
        ]
        
        # Detect sudden degradation
        for i in range(1, len(accuracies)):
            if accuracies[i-1] > 0:
                ratio = accuracies[i] / accuracies[i-1]
                if ratio >= self.degradation_ratio:
                    results.append((i, SpoofingSignal(
                        indicator=SpoofingIndicator.ACCURACY_DEGRADATION,
                        likelihood=min(1.0, (ratio - 1.0) / 10.0),
                        confidence=0.6,
                        description=f"Sudden accuracy degradation: {accuracies[i-1]:.0f}m → {accuracies[i]:.0f}m",
                        evidence={
                            "previous_accuracy_m": accuracies[i-1],
                            "current_accuracy_m": accuracies[i],
                            "degradation_ratio": ratio,
                        },
                    )))
        
        # Detect sustained poor accuracy
        for i in range(len(accuracies) - self.sustained_window + 1):
            window = accuracies[i:i + self.sustained_window]
            if all(a >= self.poor_accuracy for a in window):
                # Only flag the start of the poor accuracy window
                results.append((i, SpoofingSignal(
                    indicator=SpoofingIndicator.ACCURACY_DEGRADATION,
                    likelihood=0.4,
                    confidence=0.5,
                    description=f"Sustained poor accuracy: {self.sustained_window} points above {self.poor_accuracy}m",
                    evidence={
                        "window_size": self.sustained_window,
                        "avg_accuracy_m": np.mean(window),
                        "threshold_m": self.poor_accuracy,
                    },
                )))
        
        return results


class SyntheticPatternDetector:
    """
    Detects artificial/synthetic movement patterns.
    
    Looks for:
    - Perfect straight lines
    - Identical coordinates
    - Unnaturally regular patterns
    """
    
    def __init__(
        self,
        identical_coord_threshold: int = 3,
        perfect_line_tolerance: float = 0.001,  # Degrees
        regularity_cv_threshold: float = 0.05,  # Coefficient of variation
    ):
        """
        Initialize synthetic pattern detector.
        
        Args:
            identical_coord_threshold: Min identical coords for flag
            perfect_line_tolerance: Tolerance for line detection
            regularity_cv_threshold: CV threshold for regularity
        """
        self.identical_threshold = identical_coord_threshold
        self.line_tolerance = perfect_line_tolerance
        self.regularity_threshold = regularity_cv_threshold
    
    def detect(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
    ) -> List[Tuple[int, SpoofingSignal]]:
        """Detect synthetic patterns."""
        results = []
        
        # Detect identical coordinates
        coord_counts: Dict[Tuple[float, float], List[int]] = defaultdict(list)
        for i, event in enumerate(events):
            coord = (round(event.location.lat, 6), round(event.location.lon, 6))
            coord_counts[coord].append(i)
        
        for coord, indices in coord_counts.items():
            if len(indices) >= self.identical_threshold:
                # Check if they're consecutive (legitimate stop) or spread out (suspicious)
                is_consecutive = all(
                    indices[j] - indices[j-1] == 1 
                    for j in range(1, len(indices))
                )
                
                if not is_consecutive:
                    for idx in indices:
                        results.append((idx, SpoofingSignal(
                            indicator=SpoofingIndicator.IDENTICAL_COORDINATES,
                            likelihood=min(1.0, len(indices) / 10.0),
                            confidence=0.7,
                            description=f"Identical coordinates repeated {len(indices)} times (non-consecutive)",
                            evidence={
                                "lat": coord[0],
                                "lon": coord[1],
                                "occurrence_count": len(indices),
                                "is_consecutive": is_consecutive,
                            },
                        )))
        
        # Detect perfect straight lines
        if len(events) >= 5:
            for i in range(len(events) - 4):
                window_events = events[i:i+5]
                
                # Check if points form a perfect line
                lats = [e.location.lat for e in window_events]
                lons = [e.location.lon for e in window_events]
                
                if len(set(lats)) > 2 and len(set(lons)) > 2:  # Not just horizontal/vertical
                    # Linear regression residuals
                    try:
                        lat_fit = np.polyfit(range(5), lats, 1)
                        lon_fit = np.polyfit(range(5), lons, 1)
                        
                        lat_residuals = np.abs(
                            np.array(lats) - np.polyval(lat_fit, range(5))
                        )
                        lon_residuals = np.abs(
                            np.array(lons) - np.polyval(lon_fit, range(5))
                        )
                        
                        max_residual = max(lat_residuals.max(), lon_residuals.max())
                        
                        if max_residual < self.line_tolerance:
                            results.append((i + 2, SpoofingSignal(
                                indicator=SpoofingIndicator.PERFECT_LINE_PATH,
                                likelihood=1.0 - max_residual / self.line_tolerance,
                                confidence=0.6,
                                description="Unnaturally perfect straight line path",
                                evidence={
                                    "window_size": 5,
                                    "max_residual_degrees": max_residual,
                                    "tolerance_degrees": self.line_tolerance,
                                },
                            )))
                    except Exception:
                        pass
        
        # Detect unnaturally regular timing
        if len(features) >= 10:
            time_deltas = [
                f.time_delta_seconds for f in features 
                if f.time_delta_seconds is not None and f.time_delta_seconds > 0
            ]
            
            if len(time_deltas) >= 5:
                mean_delta = np.mean(time_deltas)
                std_delta = np.std(time_deltas)
                
                if mean_delta > 0:
                    cv = std_delta / mean_delta
                    
                    if cv < self.regularity_threshold:
                        # Unnaturally regular - flag middle point
                        mid_idx = len(features) // 2
                        results.append((mid_idx, SpoofingSignal(
                            indicator=SpoofingIndicator.SYNTHETIC_PATTERN,
                            likelihood=1.0 - cv / self.regularity_threshold,
                            confidence=0.5,
                            description=f"Unnaturally regular timing (CV={cv:.3f})",
                            evidence={
                                "mean_delta_seconds": mean_delta,
                                "std_delta_seconds": std_delta,
                                "coefficient_of_variation": cv,
                                "threshold": self.regularity_threshold,
                            },
                        )))
        
        return results


class SessionIntegrityChecker:
    """
    Checks session-level integrity.
    
    Looks for:
    - Gaps in sequence numbers
    - Missing data patterns
    - Inconsistent session metadata
    """
    
    def check(
        self,
        events: List[NormalizedTrackingEvent],
    ) -> List[SpoofingSignal]:
        """Check session integrity."""
        signals = []
        
        if len(events) < 2:
            return signals
        
        # Check point index sequence
        indices = [e.point_index for e in events]
        expected_indices = list(range(min(indices), max(indices) + 1))
        
        missing_indices = set(expected_indices) - set(indices)
        if missing_indices:
            gap_ratio = len(missing_indices) / len(expected_indices)
            signals.append(SpoofingSignal(
                indicator=SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
                likelihood=min(1.0, gap_ratio * 2),
                confidence=0.7,
                description=f"Missing point indices: {len(missing_indices)} gaps in sequence",
                evidence={
                    "missing_count": len(missing_indices),
                    "total_expected": len(expected_indices),
                    "gap_ratio": gap_ratio,
                },
            ))
        
        # Check for duplicate indices
        duplicate_indices = [
            idx for idx in set(indices) 
            if indices.count(idx) > 1
        ]
        if duplicate_indices:
            signals.append(SpoofingSignal(
                indicator=SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
                likelihood=min(1.0, len(duplicate_indices) / len(indices)),
                confidence=0.8,
                description=f"Duplicate point indices: {len(duplicate_indices)} duplicated",
                evidence={
                    "duplicate_count": len(duplicate_indices),
                    "duplicate_indices": duplicate_indices[:10],  # First 10
                },
            ))
        
        return signals


class SpoofingDetector:
    """
    Main spoofing detection orchestrator.
    
    Combines all detection methods and produces
    unified spoofing likelihood scores.
    """
    
    def __init__(
        self,
        physical_weight: float = 0.3,
        pattern_weight: float = 0.25,
        timestamp_weight: float = 0.2,
        accuracy_weight: float = 0.1,
        synthetic_weight: float = 0.15,
    ):
        """
        Initialize spoofing detector.
        
        Args:
            physical_weight: Weight for physical constraint checks
            pattern_weight: Weight for pattern detection
            timestamp_weight: Weight for timestamp checks
            accuracy_weight: Weight for accuracy degradation
            synthetic_weight: Weight for synthetic pattern detection
        """
        self.weights = {
            "physical": physical_weight,
            "pattern": pattern_weight,
            "timestamp": timestamp_weight,
            "accuracy": accuracy_weight,
            "synthetic": synthetic_weight,
        }
        
        self.physical_checker = PhysicalConstraintChecker()
        self.teleport_detector = TeleportPatternDetector()
        self.timestamp_checker = TimestampConsistencyChecker()
        self.accuracy_detector = AccuracyDegradationDetector()
        self.synthetic_detector = SyntheticPatternDetector()
        self.integrity_checker = SessionIntegrityChecker()
    
    def detect_point(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        prev_event: Optional[NormalizedTrackingEvent] = None,
    ) -> SpoofingResult:
        """Detect spoofing for a single point."""
        all_signals = []
        
        # Physical constraints
        phys_signals = self.physical_checker.check(event, features, prev_event)
        all_signals.extend(phys_signals)
        
        # Calculate likelihoods
        spoofing_scores = [s.likelihood for s in all_signals]
        
        if spoofing_scores:
            spoofing_likelihood = max(spoofing_scores)
        else:
            spoofing_likelihood = 0.0
        
        # Tampering is a subset of spoofing signals
        tampering_indicators = {
            SpoofingIndicator.TIMESTAMP_MANIPULATION,
            SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
        }
        tampering_scores = [
            s.likelihood for s in all_signals 
            if s.indicator in tampering_indicators
        ]
        tampering_likelihood = max(tampering_scores) if tampering_scores else 0.0
        
        return SpoofingResult(
            scope="point",
            scope_id=event.event_id,
            timestamp=event.timestamp,
            spoofing_likelihood=spoofing_likelihood,
            tampering_likelihood=tampering_likelihood,
            indicators=all_signals,
            session_integrity_score=1.0,  # Single point, no session context
        )
    
    def detect_session(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        session_id: str,
    ) -> SpoofingResult:
        """Detect spoofing for an entire session."""
        if not events:
            return SpoofingResult(
                scope="session",
                scope_id=session_id,
                timestamp=datetime.now(timezone.utc),
                spoofing_likelihood=0.0,
                tampering_likelihood=0.0,
                indicators=[],
                session_integrity_score=1.0,
            )
        
        all_signals = []
        point_signals: Dict[int, List[SpoofingSignal]] = defaultdict(list)
        
        # Physical constraints for each point
        for i, (event, feat) in enumerate(zip(events, features)):
            prev_event = events[i-1] if i > 0 else None
            signals = self.physical_checker.check(event, feat, prev_event)
            point_signals[i].extend(signals)
            all_signals.extend(signals)
        
        # Teleport patterns
        teleport_results = self.teleport_detector.detect(features)
        for idx, signal in teleport_results:
            point_signals[idx].append(signal)
            all_signals.append(signal)
        
        # Timestamp consistency
        timestamp_results = self.timestamp_checker.check(events)
        for idx, signal in timestamp_results:
            point_signals[idx].append(signal)
            all_signals.append(signal)
        
        # Accuracy degradation
        accuracy_results = self.accuracy_detector.detect(events)
        for idx, signal in accuracy_results:
            point_signals[idx].append(signal)
            all_signals.append(signal)
        
        # Synthetic patterns
        synthetic_results = self.synthetic_detector.detect(events, features)
        for idx, signal in synthetic_results:
            point_signals[idx].append(signal)
            all_signals.append(signal)
        
        # Session integrity
        integrity_signals = self.integrity_checker.check(events)
        all_signals.extend(integrity_signals)
        
        # Calculate session-level scores
        if all_signals:
            # Group by type and calculate weighted average
            type_scores: Dict[str, List[float]] = defaultdict(list)
            
            for signal in all_signals:
                if signal.indicator in {
                    SpoofingIndicator.TELEPORTATION,
                    SpoofingIndicator.IMPOSSIBLE_SPEED,
                    SpoofingIndicator.REPEATED_TELEPORT,
                }:
                    type_scores["physical"].append(signal.likelihood)
                elif signal.indicator in {
                    SpoofingIndicator.TIMESTAMP_MANIPULATION,
                    SpoofingIndicator.CLOCK_DRIFT,
                }:
                    type_scores["timestamp"].append(signal.likelihood)
                elif signal.indicator == SpoofingIndicator.ACCURACY_DEGRADATION:
                    type_scores["accuracy"].append(signal.likelihood)
                elif signal.indicator in {
                    SpoofingIndicator.SYNTHETIC_PATTERN,
                    SpoofingIndicator.PERFECT_LINE_PATH,
                    SpoofingIndicator.IDENTICAL_COORDINATES,
                }:
                    type_scores["synthetic"].append(signal.likelihood)
                else:
                    type_scores["pattern"].append(signal.likelihood)
            
            # Weighted combination
            weighted_score = 0.0
            total_weight = 0.0
            
            for type_name, weight in self.weights.items():
                if type_name in type_scores:
                    type_max = max(type_scores[type_name])
                    weighted_score += weight * type_max
                    total_weight += weight
            
            if total_weight > 0:
                spoofing_likelihood = weighted_score / total_weight
            else:
                spoofing_likelihood = 0.0
        else:
            spoofing_likelihood = 0.0
        
        # Tampering likelihood
        tampering_indicators = {
            SpoofingIndicator.TIMESTAMP_MANIPULATION,
            SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
        }
        tampering_scores = [
            s.likelihood for s in all_signals 
            if s.indicator in tampering_indicators
        ]
        tampering_likelihood = max(tampering_scores) if tampering_scores else 0.0
        
        # Session integrity score
        integrity_scores = [
            1.0 - s.likelihood for s in integrity_signals
        ]
        session_integrity = min(integrity_scores) if integrity_scores else 1.0
        
        return SpoofingResult(
            scope="session",
            scope_id=session_id,
            timestamp=events[0].timestamp if events else datetime.now(timezone.utc),
            spoofing_likelihood=min(1.0, spoofing_likelihood),
            tampering_likelihood=min(1.0, tampering_likelihood),
            indicators=all_signals,
            session_integrity_score=session_integrity,
        )


class TamperingDetector:
    """
    Specialized detector for data tampering.
    
    Focuses on data manipulation patterns rather than
    GPS-level spoofing.
    """
    
    def __init__(self):
        """Initialize tampering detector."""
        self.timestamp_checker = TimestampConsistencyChecker()
        self.integrity_checker = SessionIntegrityChecker()
    
    def detect(
        self,
        events: List[NormalizedTrackingEvent],
        session_id: str,
    ) -> SpoofingResult:
        """Detect tampering in a session."""
        all_signals = []
        
        # Timestamp checks
        timestamp_results = self.timestamp_checker.check(events)
        for idx, signal in timestamp_results:
            all_signals.append(signal)
        
        # Integrity checks
        integrity_signals = self.integrity_checker.check(events)
        all_signals.extend(integrity_signals)
        
        # Additional data-level tampering checks
        # Check for suspicious data patterns
        
        # All null accuracy values
        if all(e.gps_accuracy_meters is None for e in events):
            all_signals.append(SpoofingSignal(
                indicator=SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
                likelihood=0.5,
                confidence=0.6,
                description="All accuracy values are null - possible data stripping",
                evidence={"null_accuracy_count": len(events)},
            ))
        
        # All null speed values
        if all(e.speed_ms is None for e in events):
            all_signals.append(SpoofingSignal(
                indicator=SpoofingIndicator.SESSION_INTEGRITY_VIOLATION,
                likelihood=0.4,
                confidence=0.5,
                description="All speed values are null - possible data stripping",
                evidence={"null_speed_count": len(events)},
            ))
        
        # Calculate likelihood
        if all_signals:
            tampering_likelihood = max(s.likelihood for s in all_signals)
        else:
            tampering_likelihood = 0.0
        
        # Integrity score
        integrity_scores = [1.0 - s.likelihood for s in integrity_signals]
        session_integrity = min(integrity_scores) if integrity_scores else 1.0
        
        return SpoofingResult(
            scope="session",
            scope_id=session_id,
            timestamp=events[0].timestamp if events else datetime.now(timezone.utc),
            spoofing_likelihood=tampering_likelihood * 0.5,  # Tampering is a type of spoofing
            tampering_likelihood=tampering_likelihood,
            indicators=all_signals,
            session_integrity_score=session_integrity,
        )
