"""
Anomaly Detection Module
=========================

ML-based anomaly detection for location tracking data.

Implements:
1. Isolation Forest - unsupervised anomaly detection
2. Statistical z-score - deviation from learned baseline
3. HBOS (Histogram-Based Outlier Score) - fast statistical detection
4. Ensemble detector - combines multiple methods

Design Principles:
- Lightweight and explainable models
- Robust to noise in GPS data
- No labels required (unsupervised)
- Backend-only, no on-device computation
- All scores are probabilistic (0-1), not binary
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats

from app.core.logging import get_logger
from app.core.constants import SpeedLimits, AccelerationLimits
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    UserBaseline,
)
from app.models.phase2_schemas import (
    AnomalyType,
    AnomalySignal,
    AnomalyResult,
    BehavioralProfile,
)

logger = get_logger(__name__)


class AnomalyDetector(ABC):
    """Abstract base class for anomaly detectors."""
    
    @abstractmethod
    def fit(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> None:
        """Fit the detector on training data."""
        pass
    
    @abstractmethod
    def detect(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> Tuple[float, List[AnomalySignal]]:
        """
        Detect anomalies in a single event.
        
        Returns:
            Tuple of (anomaly_score, list of signals)
        """
        pass
    
    @abstractmethod
    def detect_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> List[Tuple[float, List[AnomalySignal]]]:
        """Detect anomalies in a batch of events."""
        pass


class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest based anomaly detector.
    
    Isolation Forest isolates anomalies by randomly selecting
    features and split values. Anomalies are easier to isolate
    and thus have shorter paths in the tree.
    
    Good for:
    - High-dimensional data
    - No assumptions about data distribution
    - Detecting outliers in feature space
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        """
        Initialize Isolation Forest detector.
        
        Args:
            contamination: Expected proportion of anomalies
            n_estimators: Number of trees in the forest
            max_samples: Samples to draw for each tree
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            max_samples=max_samples,
            random_state=random_state,
            warm_start=False,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = [
            "speed_ms",
            "calculated_speed_ms",
            "acceleration_ms2",
            "time_delta_seconds",
            "distance_meters",
            "gps_accuracy_meters",
            "confidence_overall",
        ]
    
    def _extract_features(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> np.ndarray:
        """Extract feature vector for Isolation Forest."""
        return np.array([
            event.speed_ms if event.speed_ms is not None else 0.0,
            features.calculated_speed_ms if features.calculated_speed_ms is not None else 0.0,
            features.acceleration_ms2 if features.acceleration_ms2 is not None else 0.0,
            features.time_delta_seconds if features.time_delta_seconds is not None else 0.0,
            features.distance_meters if features.distance_meters is not None else 0.0,
            event.gps_accuracy_meters if event.gps_accuracy_meters is not None else 50.0,
            score.overall,
        ])
    
    def fit(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> None:
        """Fit the Isolation Forest on training data."""
        if len(events) < 10:
            logger.warning("insufficient_data_for_iforest", count=len(events))
            return
        
        # Extract feature matrix
        X = np.array([
            self._extract_features(e, f, s)
            for e, f, s in zip(events, features, scores)
        ])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit Isolation Forest
        self.model.fit(X_scaled)
        self.is_fitted = True
        
        logger.info("iforest_fitted", n_samples=len(events))
    
    def detect(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> Tuple[float, List[AnomalySignal]]:
        """Detect anomaly for a single event."""
        if not self.is_fitted:
            return 0.0, []
        
        # Extract and scale features
        X = self._extract_features(event, features, score).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        
        # Get anomaly score (convert from [-1, 1] to [0, 1])
        # Isolation Forest returns -1 for anomalies, 1 for normal
        raw_score = self.model.decision_function(X_scaled)[0]
        # Normalize to 0-1 range where 1 = anomalous
        anomaly_score = max(0.0, min(1.0, 0.5 - raw_score / 2))
        
        signals = []
        if anomaly_score > 0.5:
            signals.append(AnomalySignal(
                anomaly_type=AnomalyType.MOVEMENT_CONSISTENCY_ANOMALY,
                severity=anomaly_score,
                confidence=0.7,  # Isolation Forest confidence
                description="Unusual pattern detected by Isolation Forest",
                observed_value=raw_score,
                expected_range=(-0.5, 0.5),
                contribution_weight=0.5,
            ))
        
        return anomaly_score, signals
    
    def detect_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> List[Tuple[float, List[AnomalySignal]]]:
        """Detect anomalies in batch."""
        if not self.is_fitted:
            return [(0.0, []) for _ in events]
        
        X = np.array([
            self._extract_features(e, f, s)
            for e, f, s in zip(events, features, scores)
        ])
        X_scaled = self.scaler.transform(X)
        
        raw_scores = self.model.decision_function(X_scaled)
        
        results = []
        for i, raw_score in enumerate(raw_scores):
            anomaly_score = max(0.0, min(1.0, 0.5 - raw_score / 2))
            
            signals = []
            if anomaly_score > 0.5:
                signals.append(AnomalySignal(
                    anomaly_type=AnomalyType.MOVEMENT_CONSISTENCY_ANOMALY,
                    severity=anomaly_score,
                    confidence=0.7,
                    description="Unusual pattern detected by Isolation Forest",
                    observed_value=float(raw_score),
                    expected_range=(-0.5, 0.5),
                    contribution_weight=0.5,
                ))
            
            results.append((anomaly_score, signals))
        
        return results


class StatisticalDetector(AnomalyDetector):
    """
    Statistical z-score based anomaly detector.
    
    Compares each feature against learned baseline using
    z-scores. Simple, fast, and highly explainable.
    
    Good for:
    - Single-feature anomalies
    - Interpretable results
    - When baseline is well-established
    """
    
    def __init__(
        self,
        z_threshold: float = 3.0,
        min_samples: int = 30,
    ):
        """
        Initialize statistical detector.
        
        Args:
            z_threshold: Z-score threshold for anomaly
            min_samples: Minimum samples for reliable statistics
        """
        self.z_threshold = z_threshold
        self.min_samples = min_samples
        self.baseline_stats: Dict[str, Tuple[float, float]] = {}
        self.is_fitted = False
    
    def fit(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> None:
        """Compute baseline statistics from training data."""
        if len(events) < self.min_samples:
            logger.warning("insufficient_data_for_stats", count=len(events))
            return
        
        # Compute statistics for each feature
        speeds = [e.speed_ms for e in events if e.speed_ms is not None]
        calc_speeds = [f.calculated_speed_ms for f in features if f.calculated_speed_ms]
        accelerations = [f.acceleration_ms2 for f in features if f.acceleration_ms2]
        time_deltas = [f.time_delta_seconds for f in features if f.time_delta_seconds]
        distances = [f.distance_meters for f in features if f.distance_meters]
        
        if speeds:
            self.baseline_stats["speed"] = (np.mean(speeds), np.std(speeds))
        if calc_speeds:
            self.baseline_stats["calc_speed"] = (np.mean(calc_speeds), np.std(calc_speeds))
        if accelerations:
            self.baseline_stats["acceleration"] = (np.mean(accelerations), np.std(accelerations))
        if time_deltas:
            self.baseline_stats["time_delta"] = (np.mean(time_deltas), np.std(time_deltas))
        if distances:
            self.baseline_stats["distance"] = (np.mean(distances), np.std(distances))
        
        self.is_fitted = True
        logger.info("statistical_detector_fitted", features=list(self.baseline_stats.keys()))
    
    def fit_from_baseline(self, baseline: UserBaseline) -> None:
        """Initialize from a Phase 1 baseline."""
        self.baseline_stats["speed"] = (
            baseline.speed_baseline.mean,
            baseline.speed_baseline.std,
        )
        self.baseline_stats["time_delta"] = (
            1.0 / baseline.update_frequency_baseline.mean if baseline.update_frequency_baseline.mean > 0 else 5.0,
            1.0 / baseline.update_frequency_baseline.std if baseline.update_frequency_baseline.std > 0 else 2.0,
        )
        self.is_fitted = True
    
    def _compute_z_score(
        self,
        value: Optional[float],
        stat_key: str
    ) -> Optional[float]:
        """Compute z-score for a value against baseline."""
        if value is None or stat_key not in self.baseline_stats:
            return None
        
        mean, std = self.baseline_stats[stat_key]
        if std == 0:
            return 0.0 if value == mean else 5.0
        
        return (value - mean) / std
    
    def detect(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> Tuple[float, List[AnomalySignal]]:
        """Detect anomalies using z-scores."""
        if not self.is_fitted:
            return 0.0, []
        
        signals = []
        max_z = 0.0
        
        # Speed z-score
        speed_z = self._compute_z_score(event.speed_ms, "speed")
        if speed_z is not None and abs(speed_z) > self.z_threshold:
            max_z = max(max_z, abs(speed_z))
            mean, std = self.baseline_stats["speed"]
            signals.append(AnomalySignal(
                anomaly_type=AnomalyType.SPEED_ANOMALY,
                severity=min(1.0, abs(speed_z) / (self.z_threshold * 2)),
                confidence=0.8,
                description=f"Speed deviation: {speed_z:.1f} std from baseline",
                observed_value=event.speed_ms,
                expected_range=(mean - 2*std, mean + 2*std),
                z_score=speed_z,
                contribution_weight=0.4,
            ))
        
        # Acceleration z-score
        accel_z = self._compute_z_score(features.acceleration_ms2, "acceleration")
        if accel_z is not None and abs(accel_z) > self.z_threshold:
            max_z = max(max_z, abs(accel_z))
            signals.append(AnomalySignal(
                anomaly_type=AnomalyType.ACCELERATION_ANOMALY,
                severity=min(1.0, abs(accel_z) / (self.z_threshold * 2)),
                confidence=0.75,
                description=f"Acceleration deviation: {accel_z:.1f} std from baseline",
                observed_value=features.acceleration_ms2,
                z_score=accel_z,
                contribution_weight=0.3,
            ))
        
        # Time delta z-score (gap detection)
        delta_z = self._compute_z_score(features.time_delta_seconds, "time_delta")
        if delta_z is not None and delta_z > self.z_threshold:  # Only large gaps
            max_z = max(max_z, delta_z)
            signals.append(AnomalySignal(
                anomaly_type=AnomalyType.UPDATE_FREQUENCY_ANOMALY,
                severity=min(1.0, delta_z / (self.z_threshold * 2)),
                confidence=0.7,
                description=f"Update gap: {delta_z:.1f} std from baseline",
                observed_value=features.time_delta_seconds,
                z_score=delta_z,
                contribution_weight=0.3,
            ))
        
        # Convert max z-score to anomaly score
        if max_z <= self.z_threshold:
            anomaly_score = max_z / (self.z_threshold * 2)
        else:
            anomaly_score = min(1.0, 0.5 + (max_z - self.z_threshold) / (self.z_threshold * 2))
        
        return anomaly_score, signals
    
    def detect_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> List[Tuple[float, List[AnomalySignal]]]:
        """Detect anomalies in batch."""
        return [
            self.detect(e, f, s)
            for e, f, s in zip(events, features, scores)
        ]


class HBOSDetector(AnomalyDetector):
    """
    Histogram-Based Outlier Score (HBOS) detector.
    
    Fast statistical method that assumes feature independence.
    Scores each feature based on histogram density.
    
    Good for:
    - Large datasets
    - Real-time detection
    - When features are relatively independent
    """
    
    def __init__(
        self,
        n_bins: int = 20,
        alpha: float = 0.1,
    ):
        """
        Initialize HBOS detector.
        
        Args:
            n_bins: Number of histogram bins
            alpha: Regularization for empty bins
        """
        self.n_bins = n_bins
        self.alpha = alpha
        self.histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.is_fitted = False
    
    def fit(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> None:
        """Build histograms from training data."""
        if len(events) < 50:
            logger.warning("insufficient_data_for_hbos", count=len(events))
            return
        
        # Extract feature arrays
        feature_data = {
            "speed": [e.speed_ms for e in events if e.speed_ms is not None],
            "calc_speed": [f.calculated_speed_ms for f in features if f.calculated_speed_ms],
            "acceleration": [f.acceleration_ms2 for f in features if f.acceleration_ms2],
            "distance": [f.distance_meters for f in features if f.distance_meters],
        }
        
        # Build histograms
        for name, values in feature_data.items():
            if len(values) >= 10:
                hist, bin_edges = np.histogram(values, bins=self.n_bins, density=True)
                # Add regularization
                hist = hist + self.alpha
                hist = hist / hist.sum()
                self.histograms[name] = (hist, bin_edges)
        
        self.is_fitted = True
        logger.info("hbos_fitted", features=list(self.histograms.keys()))
    
    def _score_value(
        self,
        value: Optional[float],
        hist_key: str
    ) -> float:
        """Score a single value using its histogram."""
        if value is None or hist_key not in self.histograms:
            return 0.5  # Neutral score
        
        hist, bins = self.histograms[hist_key]
        
        # Find bin
        bin_idx = np.searchsorted(bins[1:], value)
        bin_idx = min(bin_idx, len(hist) - 1)
        
        # Get density (lower density = higher anomaly)
        density = hist[bin_idx]
        
        # Convert to anomaly score (0-1)
        # Higher score = more anomalous (lower density)
        max_density = hist.max()
        anomaly_score = 1.0 - (density / max_density)
        
        return anomaly_score
    
    def detect(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> Tuple[float, List[AnomalySignal]]:
        """Detect anomalies using HBOS."""
        if not self.is_fitted:
            return 0.0, []
        
        # Score each feature
        scores_dict = {
            "speed": self._score_value(event.speed_ms, "speed"),
            "calc_speed": self._score_value(features.calculated_speed_ms, "calc_speed"),
            "acceleration": self._score_value(features.acceleration_ms2, "acceleration"),
            "distance": self._score_value(features.distance_meters, "distance"),
        }
        
        # Combine scores (product for independence assumption)
        valid_scores = [s for s in scores_dict.values() if s > 0]
        if not valid_scores:
            return 0.0, []
        
        # Use average for more stable results
        combined_score = np.mean(valid_scores)
        
        signals = []
        for name, s in scores_dict.items():
            if s > 0.7:  # High anomaly
                anomaly_type = {
                    "speed": AnomalyType.SPEED_ANOMALY,
                    "calc_speed": AnomalyType.SPEED_ANOMALY,
                    "acceleration": AnomalyType.ACCELERATION_ANOMALY,
                    "distance": AnomalyType.IMPOSSIBLE_JUMP,
                }.get(name, AnomalyType.MOVEMENT_CONSISTENCY_ANOMALY)
                
                signals.append(AnomalySignal(
                    anomaly_type=anomaly_type,
                    severity=s,
                    confidence=0.6,  # HBOS is less confident
                    description=f"Low density in {name} histogram",
                    contribution_weight=0.25,
                ))
        
        return combined_score, signals
    
    def detect_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> List[Tuple[float, List[AnomalySignal]]]:
        """Detect anomalies in batch."""
        return [
            self.detect(e, f, s)
            for e, f, s in zip(events, features, scores)
        ]


class EnsembleAnomalyDetector:
    """
    Ensemble detector combining multiple detection methods.
    
    Combines:
    - Isolation Forest (pattern-based)
    - Statistical z-score (deviation-based)
    - HBOS (histogram-based)
    
    Also adds physical constraint checks.
    """
    
    def __init__(
        self,
        isolation_weight: float = 0.3,
        statistical_weight: float = 0.4,
        hbos_weight: float = 0.2,
        physical_weight: float = 0.1,
        anomaly_threshold: float = 0.5,
    ):
        """
        Initialize ensemble detector.
        
        Args:
            isolation_weight: Weight for Isolation Forest
            statistical_weight: Weight for statistical detector
            hbos_weight: Weight for HBOS
            physical_weight: Weight for physical constraint checks
            anomaly_threshold: Threshold for marking as anomalous
        """
        self.weights = {
            "isolation": isolation_weight,
            "statistical": statistical_weight,
            "hbos": hbos_weight,
            "physical": physical_weight,
        }
        self.threshold = anomaly_threshold
        
        self.iforest = IsolationForestDetector()
        self.statistical = StatisticalDetector()
        self.hbos = HBOSDetector()
        
        self.is_fitted = False
    
    def fit(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
        baseline: Optional[UserBaseline] = None,
    ) -> None:
        """Fit all detectors."""
        self.iforest.fit(events, features, scores)
        
        if baseline:
            self.statistical.fit_from_baseline(baseline)
        else:
            self.statistical.fit(events, features, scores)
        
        self.hbos.fit(events, features, scores)
        
        self.is_fitted = True
        logger.info("ensemble_detector_fitted")
    
    def _check_physical_constraints(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
    ) -> Tuple[float, List[AnomalySignal]]:
        """Check physical impossibilities."""
        signals = []
        max_score = 0.0
        
        # Impossible speed
        if event.speed_ms is not None and event.speed_ms > SpeedLimits.AIRCRAFT:
            score = 1.0
            max_score = max(max_score, score)
            signals.append(AnomalySignal(
                anomaly_type=AnomalyType.IMPOSSIBLE_JUMP,
                severity=score,
                confidence=1.0,
                description=f"Impossible speed: {event.speed_ms:.1f} m/s",
                observed_value=event.speed_ms,
                expected_range=(0, SpeedLimits.AIRCRAFT),
                contribution_weight=1.0,
            ))
        
        # Impossible acceleration
        if features.acceleration_ms2 is not None:
            if abs(features.acceleration_ms2) > AccelerationLimits.MAXIMUM_REALISTIC * 2:
                score = 1.0
                max_score = max(max_score, score)
                signals.append(AnomalySignal(
                    anomaly_type=AnomalyType.ACCELERATION_ANOMALY,
                    severity=score,
                    confidence=1.0,
                    description=f"Impossible acceleration: {features.acceleration_ms2:.1f} m/s²",
                    observed_value=features.acceleration_ms2,
                    expected_range=(
                        -AccelerationLimits.MAXIMUM_REALISTIC,
                        AccelerationLimits.MAXIMUM_REALISTIC
                    ),
                    contribution_weight=1.0,
                ))
        
        # Teleportation detection
        if features.calculated_speed_ms is not None:
            if features.calculated_speed_ms > SpeedLimits.AIRCRAFT * 1.5:
                score = 1.0
                max_score = max(max_score, score)
                signals.append(AnomalySignal(
                    anomaly_type=AnomalyType.IMPOSSIBLE_JUMP,
                    severity=score,
                    confidence=1.0,
                    description=f"Teleportation: calculated speed {features.calculated_speed_ms:.1f} m/s",
                    observed_value=features.calculated_speed_ms,
                    expected_range=(0, SpeedLimits.AIRCRAFT),
                    contribution_weight=1.0,
                ))
        
        return max_score, signals
    
    def detect(
        self,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
        score: ConfidenceScore,
    ) -> AnomalyResult:
        """Detect anomalies using ensemble."""
        all_signals = []
        
        # Physical constraints (always checked)
        phys_score, phys_signals = self._check_physical_constraints(event, features)
        all_signals.extend(phys_signals)
        
        # If physical constraint violated, immediately flag
        if phys_score >= 1.0:
            return AnomalyResult(
                event_id=event.event_id,
                timestamp=event.timestamp,
                anomaly_score=1.0,
                is_anomalous=True,
                signals=phys_signals,
                isolation_forest_score=None,
                statistical_score=None,
            )
        
        # ML detectors
        iso_score, iso_signals = self.iforest.detect(event, features, score)
        stat_score, stat_signals = self.statistical.detect(event, features, score)
        hbos_score, hbos_signals = self.hbos.detect(event, features, score)
        
        all_signals.extend(iso_signals)
        all_signals.extend(stat_signals)
        all_signals.extend(hbos_signals)
        
        # Weighted combination
        combined_score = (
            self.weights["isolation"] * iso_score +
            self.weights["statistical"] * stat_score +
            self.weights["hbos"] * hbos_score +
            self.weights["physical"] * phys_score
        )
        
        is_anomalous = combined_score >= self.threshold
        
        return AnomalyResult(
            event_id=event.event_id,
            timestamp=event.timestamp,
            anomaly_score=min(1.0, combined_score),
            is_anomalous=is_anomalous,
            signals=all_signals,
            isolation_forest_score=iso_score,
            statistical_score=stat_score,
        )
    
    def detect_batch(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> List[AnomalyResult]:
        """Detect anomalies in batch."""
        return [
            self.detect(e, f, s)
            for e, f, s in zip(events, features, scores)
        ]
    
    def detect_sequence_anomalies(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
        window_size: int = 5,
        low_confidence_threshold: float = 0.5,
    ) -> List[AnomalyResult]:
        """
        Detect anomalies including sequence patterns.
        
        Adds detection for:
        - Low confidence sequences
        - Repeated pattern anomalies
        """
        results = self.detect_batch(events, features, scores)
        
        # Detect low-confidence sequences
        for i in range(len(scores) - window_size + 1):
            window_scores = scores[i:i + window_size]
            avg_confidence = np.mean([s.overall for s in window_scores])
            
            if avg_confidence < low_confidence_threshold:
                # Mark all events in window
                for j in range(i, i + window_size):
                    if j < len(results):
                        existing_result = results[j]
                        new_signal = AnomalySignal(
                            anomaly_type=AnomalyType.LOW_CONFIDENCE_SEQUENCE,
                            severity=1.0 - avg_confidence,
                            confidence=0.8,
                            description=f"Low confidence sequence: avg {avg_confidence:.2f}",
                            observed_value=avg_confidence,
                            expected_range=(low_confidence_threshold, 1.0),
                            contribution_weight=0.3,
                        )
                        
                        # Update result with new signal
                        new_signals = list(existing_result.signals) + [new_signal]
                        new_score = min(1.0, existing_result.anomaly_score + 0.1)
                        
                        results[j] = AnomalyResult(
                            event_id=existing_result.event_id,
                            timestamp=existing_result.timestamp,
                            anomaly_score=new_score,
                            is_anomalous=existing_result.is_anomalous or new_score >= self.threshold,
                            signals=new_signals,
                            isolation_forest_score=existing_result.isolation_forest_score,
                            statistical_score=existing_result.statistical_score,
                        )
        
        return results
