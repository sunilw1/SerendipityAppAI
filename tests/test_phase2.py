"""
Phase 2 Tests
==============

Tests for Phase 2 behavioral intelligence and threat detection.

Covers:
- Behavioral profile learning
- Anomaly detection
- Spoofing detection
- Risk scoring
- Explainability
"""

import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import List

from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    LocationPoint,
    UserBaseline,
    BaselineMetrics,
)
from app.models.phase2_schemas import (
    AnomalyType,
    SpoofingIndicator,
    RiskLevel,
    BehavioralProfile,
    AnomalyResult,
    SpoofingResult,
    LocationRiskScore,
    SessionRiskScore,
)
from app.core.constants import ActivityType


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_events() -> List[NormalizedTrackingEvent]:
    """Create sample normalized events for testing."""
    base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
    events = []
    
    for i in range(20):
        event = NormalizedTrackingEvent(
            event_id=f"event_{i}",
            user_id=1,
            trip_id=100,
            point_index=i,
            timestamp=base_time + timedelta(seconds=i * 5),
            timestamp_unix=(base_time + timedelta(seconds=i * 5)).timestamp(),
            location=LocationPoint(
                lat=37.7749 + i * 0.0001,
                lon=-122.4194 + i * 0.0001,
            ),
            speed_ms=5.0 + np.random.randn() * 0.5,
            is_moving=True,
            activity_type=ActivityType.IN_VEHICLE,
            activity_confidence=80,
            gps_accuracy_meters=10.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash=f"hash_{i}",
        )
        events.append(event)
    
    return events


@pytest.fixture
def sample_features(sample_events) -> List[ComputedFeatures]:
    """Create sample computed features."""
    features = []
    
    for i, event in enumerate(sample_events):
        if i == 0:
            feat = ComputedFeatures(
                time_delta_seconds=None,
                update_rate_hz=None,
                distance_meters=None,
                calculated_speed_ms=None,
                is_stop=False,
                has_time_gap=False,
            )
        else:
            feat = ComputedFeatures(
                time_delta_seconds=5.0,
                update_rate_hz=0.2,
                distance_meters=50.0 + np.random.randn() * 5,
                calculated_speed_ms=10.0 + np.random.randn() * 1,
                acceleration_ms2=0.5 + np.random.randn() * 0.1,
                bearing_degrees=45.0,
                is_stop=False,
                has_time_gap=False,
            )
        features.append(feat)
    
    return features


@pytest.fixture
def sample_scores(sample_events) -> List[ConfidenceScore]:
    """Create sample confidence scores."""
    return [
        ConfidenceScore(
            overall=0.85 + np.random.randn() * 0.05,
            gps_accuracy_score=0.9,
            speed_validity_score=0.85,
            acceleration_validity_score=0.9,
            temporal_consistency_score=0.85,
            activity_consistency_score=0.8,
            signal_continuity_score=0.9,
            flags=[],
        )
        for _ in sample_events
    ]


@pytest.fixture
def sample_baseline() -> UserBaseline:
    """Create sample user baseline."""
    now = datetime.now(timezone.utc)
    
    return UserBaseline(
        user_id=1,
        created_at=now,
        updated_at=now,
        trips_analyzed=10,
        points_analyzed=500,
        date_range_days=30.0,
        speed_baseline=BaselineMetrics(
            metric_name="speed_ms",
            mean=5.0,
            median=4.5,
            std=2.0,
            min_value=0.0,
            max_value=15.0,
            p5=0.5,
            p25=3.0,
            p75=7.0,
            p95=12.0,
            sample_size=500,
        ),
        update_frequency_baseline=BaselineMetrics(
            metric_name="update_rate_hz",
            mean=0.2,
            median=0.2,
            std=0.05,
            min_value=0.1,
            max_value=0.5,
            p5=0.15,
            p25=0.18,
            p75=0.22,
            p95=0.3,
            sample_size=500,
        ),
        trip_duration_baseline=BaselineMetrics(
            metric_name="trip_duration_seconds",
            mean=1800.0,
            median=1500.0,
            std=600.0,
            min_value=300.0,
            max_value=3600.0,
            p5=400.0,
            p25=1000.0,
            p75=2200.0,
            p95=3200.0,
            sample_size=10,
        ),
        stop_duration_baseline=BaselineMetrics(
            metric_name="stop_duration_seconds",
            mean=120.0,
            median=90.0,
            std=60.0,
            min_value=30.0,
            max_value=300.0,
            p5=35.0,
            p25=60.0,
            p75=150.0,
            p95=250.0,
            sample_size=50,
        ),
        activity_distribution={
            "in_vehicle": 0.6,
            "walking": 0.2,
            "still": 0.15,
            "unknown": 0.05,
        },
        typical_confidence=0.85,
    )


# =============================================================================
# ANOMALY DETECTION TESTS
# =============================================================================

class TestAnomalyDetection:
    """Tests for anomaly detection."""
    
    def test_statistical_detector_fit(self, sample_events, sample_features, sample_scores):
        """Test statistical detector fitting."""
        from app.detection.anomaly import StatisticalDetector
        
        detector = StatisticalDetector()
        detector.fit(sample_events, sample_features, sample_scores)
        
        assert detector.is_fitted
        assert "speed" in detector.baseline_stats
    
    def test_statistical_detector_detect(self, sample_events, sample_features, sample_scores):
        """Test statistical detection."""
        from app.detection.anomaly import StatisticalDetector
        
        detector = StatisticalDetector()
        detector.fit(sample_events, sample_features, sample_scores)
        
        score, signals = detector.detect(
            sample_events[5],
            sample_features[5],
            sample_scores[5],
        )
        
        assert 0.0 <= score <= 1.0
        assert isinstance(signals, list)
    
    def test_isolation_forest_detector(self, sample_events, sample_features, sample_scores):
        """Test Isolation Forest detector."""
        from app.detection.anomaly import IsolationForestDetector
        
        detector = IsolationForestDetector()
        detector.fit(sample_events, sample_features, sample_scores)
        
        assert detector.is_fitted
        
        score, signals = detector.detect(
            sample_events[5],
            sample_features[5],
            sample_scores[5],
        )
        
        assert 0.0 <= score <= 1.0
    
    def test_ensemble_detector(self, sample_events, sample_features, sample_scores, sample_baseline):
        """Test ensemble anomaly detector."""
        from app.detection.anomaly import EnsembleAnomalyDetector
        
        detector = EnsembleAnomalyDetector()
        detector.fit(sample_events, sample_features, sample_scores, sample_baseline)
        
        assert detector.is_fitted
        
        result = detector.detect(
            sample_events[5],
            sample_features[5],
            sample_scores[5],
        )
        
        assert isinstance(result, AnomalyResult)
        assert 0.0 <= result.anomaly_score <= 1.0
        assert isinstance(result.is_anomalous, bool)
    
    def test_detect_impossible_speed(self):
        """Test detection of impossible speed."""
        from app.detection.anomaly import EnsembleAnomalyDetector
        
        detector = EnsembleAnomalyDetector()
        
        # Create event with impossible speed
        event = NormalizedTrackingEvent(
            event_id="impossible",
            user_id=1,
            trip_id=100,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=37.7749, lon=-122.4194),
            speed_ms=500.0,  # Impossible speed
            is_moving=True,
            activity_type=ActivityType.IN_VEHICLE,
            activity_confidence=80,
            gps_accuracy_meters=10.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="hash_impossible",
        )
        
        features = ComputedFeatures(
            calculated_speed_ms=500.0,
            is_stop=False,
            has_time_gap=False,
        )
        
        score = ConfidenceScore(
            overall=0.5,
            gps_accuracy_score=0.9,
            speed_validity_score=0.1,
            acceleration_validity_score=0.9,
            temporal_consistency_score=0.85,
            activity_consistency_score=0.8,
            signal_continuity_score=0.9,
            flags=["impossible_speed"],
        )
        
        result = detector.detect(event, features, score)
        
        assert result.anomaly_score == 1.0
        assert result.is_anomalous


# =============================================================================
# SPOOFING DETECTION TESTS
# =============================================================================

class TestSpoofingDetection:
    """Tests for spoofing detection."""
    
    def test_physical_constraint_checker(self, sample_events, sample_features):
        """Test physical constraint checking."""
        from app.detection.spoofing import PhysicalConstraintChecker
        
        checker = PhysicalConstraintChecker()
        
        signals = checker.check(sample_events[5], sample_features[5])
        
        assert isinstance(signals, list)
    
    def test_teleportation_detection(self):
        """Test teleportation detection."""
        from app.detection.spoofing import PhysicalConstraintChecker
        
        checker = PhysicalConstraintChecker()
        
        # Create teleportation scenario
        event = NormalizedTrackingEvent(
            event_id="teleport",
            user_id=1,
            trip_id=100,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=37.7749, lon=-122.4194),
            speed_ms=10.0,
            is_moving=True,
            activity_type=ActivityType.IN_VEHICLE,
            activity_confidence=80,
            gps_accuracy_meters=10.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="hash_teleport",
        )
        
        features = ComputedFeatures(
            time_delta_seconds=5.0,
            distance_meters=5000.0,  # 5km in 5 seconds = 1000 m/s
            calculated_speed_ms=1000.0,
            is_stop=False,
            has_time_gap=False,
        )
        
        signals = checker.check(event, features)
        
        assert len(signals) > 0
        assert any(s.indicator == SpoofingIndicator.TELEPORTATION for s in signals)
    
    def test_spoofing_detector_session(self, sample_events, sample_features):
        """Test session-level spoofing detection."""
        from app.detection.spoofing import SpoofingDetector
        
        detector = SpoofingDetector()
        
        result = detector.detect_session(
            sample_events,
            sample_features,
            "test_session",
        )
        
        assert isinstance(result, SpoofingResult)
        assert 0.0 <= result.spoofing_likelihood <= 1.0
        assert 0.0 <= result.tampering_likelihood <= 1.0
        assert 0.0 <= result.session_integrity_score <= 1.0


# =============================================================================
# RISK SCORING TESTS
# =============================================================================

class TestRiskScoring:
    """Tests for risk scoring."""
    
    def test_location_risk_calculator(
        self, sample_events, sample_scores, sample_baseline
    ):
        """Test location risk calculation."""
        from app.scoring.risk_engine import LocationRiskCalculator
        
        calculator = LocationRiskCalculator()
        
        risk = calculator.calculate(
            event=sample_events[5],
            confidence=sample_scores[5],
            anomaly=None,
            spoofing=None,
            baseline=sample_baseline,
        )
        
        assert isinstance(risk, LocationRiskScore)
        assert 0.0 <= risk.risk_score <= 1.0
        assert risk.risk_level in RiskLevel
        assert len(risk.contributors) > 0
    
    def test_session_risk_calculator(self, sample_events, sample_scores, sample_baseline):
        """Test session risk calculation."""
        from app.scoring.risk_engine import (
            LocationRiskCalculator,
            SessionRiskCalculator,
        )
        
        loc_calculator = LocationRiskCalculator()
        sess_calculator = SessionRiskCalculator()
        
        # Calculate location risks
        location_risks = [
            loc_calculator.calculate(
                event=event,
                confidence=score,
                anomaly=None,
                spoofing=None,
                baseline=sample_baseline,
            )
            for event, score in zip(sample_events, sample_scores)
        ]
        
        # Calculate session risk
        session_risk = sess_calculator.calculate(
            session_id="test_session",
            user_id=1,
            location_risks=location_risks,
            session_spoofing=None,
        )
        
        assert isinstance(session_risk, SessionRiskScore)
        assert 0.0 <= session_risk.risk_score <= 1.0
        assert session_risk.total_points == len(sample_events)
    
    def test_risk_scoring_engine(
        self, sample_events, sample_features, sample_scores, sample_baseline
    ):
        """Test complete risk scoring engine."""
        from app.detection.anomaly import EnsembleAnomalyDetector
        from app.detection.spoofing import SpoofingDetector
        from app.scoring.risk_engine import RiskScoringEngine
        
        # Set up detectors
        anomaly_detector = EnsembleAnomalyDetector()
        anomaly_detector.fit(sample_events, sample_features, sample_scores, sample_baseline)
        
        spoofing_detector = SpoofingDetector()
        
        # Create engine
        engine = RiskScoringEngine()
        
        # Detect and score
        anomaly = anomaly_detector.detect(
            sample_events[5], sample_features[5], sample_scores[5]
        )
        spoofing = spoofing_detector.detect_point(
            sample_events[5], sample_features[5]
        )
        
        location_risk = engine.score_location(
            event=sample_events[5],
            confidence=sample_scores[5],
            anomaly=anomaly,
            spoofing=spoofing,
            baseline=sample_baseline,
        )
        
        assert isinstance(location_risk, LocationRiskScore)
        assert 0.0 <= location_risk.risk_score <= 1.0


# =============================================================================
# EXPLAINABILITY TESTS
# =============================================================================

class TestExplainability:
    """Tests for explainability."""
    
    def test_risk_explainer(self, sample_events, sample_scores, sample_baseline):
        """Test risk explanation generation."""
        from app.scoring.risk_engine import LocationRiskCalculator
        from app.scoring.explainability import RiskExplainer
        
        calculator = LocationRiskCalculator()
        explainer = RiskExplainer()
        
        risk = calculator.calculate(
            event=sample_events[5],
            confidence=sample_scores[5],
            anomaly=None,
            spoofing=None,
            baseline=sample_baseline,
        )
        
        explanation = explainer.explain_location_risk(risk)
        
        assert "summary" in explanation
        assert "risk_score" in explanation
        assert "risk_level" in explanation
        assert "narrative" in explanation
        assert "contributing_factors" in explanation
    
    def test_explainability_engine(self, sample_events, sample_scores, sample_baseline):
        """Test complete explainability engine."""
        from app.scoring.risk_engine import LocationRiskCalculator
        from app.scoring.explainability import ExplainabilityEngine
        
        calculator = LocationRiskCalculator()
        engine = ExplainabilityEngine()
        
        risk = calculator.calculate(
            event=sample_events[5],
            confidence=sample_scores[5],
            anomaly=None,
            spoofing=None,
            baseline=sample_baseline,
        )
        
        full_explanation = engine.generate_full_explanation(
            location_risk=risk,
            anomaly=None,
            spoofing=None,
            verbose=True,
        )
        
        assert "executive_summary" in full_explanation
        assert "overall_risk" in full_explanation


# =============================================================================
# BEHAVIORAL PROFILE TESTS
# =============================================================================

class TestBehavioralProfile:
    """Tests for behavioral profile learning."""
    
    def test_time_pattern_learner(self, sample_events, sample_scores):
        """Test time pattern learning."""
        from app.models.behavioral_profile import TimePatternLearner
        
        learner = TimePatternLearner()
        patterns = learner.learn_patterns(sample_events, sample_scores)
        
        assert len(patterns) == 24  # One for each hour
        assert all(0 <= p.activity_probability <= 1 for p in patterns)
    
    def test_stop_pattern_learner(self, sample_events, sample_features):
        """Test stop pattern learning."""
        from app.models.behavioral_profile import StopPatternLearner
        
        # Create events with stops
        events = list(sample_events)
        features = list(sample_features)
        
        # Mark some as stops
        for i in range(5, 10):
            features[i] = ComputedFeatures(
                time_delta_seconds=5.0,
                distance_meters=0.5,
                calculated_speed_ms=0.1,
                is_stop=True,
                stop_duration_seconds=60.0,
                has_time_gap=False,
            )
        
        learner = StopPatternLearner(min_stop_frequency=1)
        stops = learner.learn_stops(events, features)
        
        # May or may not find stops depending on clustering
        assert isinstance(stops, list)


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestPhase2Integration:
    """Integration tests for Phase 2 pipeline."""
    
    def test_full_pipeline(
        self, sample_events, sample_features, sample_scores, sample_baseline
    ):
        """Test complete Phase 2 pipeline."""
        from app.detection.anomaly import EnsembleAnomalyDetector
        from app.detection.spoofing import SpoofingDetector
        from app.scoring.risk_engine import RiskScoringEngine
        from app.scoring.explainability import ExplainabilityEngine
        
        # Initialize components
        anomaly_detector = EnsembleAnomalyDetector()
        spoofing_detector = SpoofingDetector()
        risk_engine = RiskScoringEngine()
        explainability = ExplainabilityEngine()
        
        # Fit anomaly detector
        anomaly_detector.fit(
            sample_events, sample_features, sample_scores, sample_baseline
        )
        
        # Process each event
        location_risks = []
        anomalies = []
        
        for event, features, score in zip(sample_events, sample_features, sample_scores):
            # Detect anomalies
            anomaly = anomaly_detector.detect(event, features, score)
            anomalies.append(anomaly)
            
            # Detect spoofing
            spoofing = spoofing_detector.detect_point(event, features)
            
            # Score risk
            risk = risk_engine.score_location(
                event=event,
                confidence=score,
                anomaly=anomaly,
                spoofing=spoofing,
                baseline=sample_baseline,
            )
            location_risks.append(risk)
        
        # Session-level spoofing
        session_spoofing = spoofing_detector.detect_session(
            sample_events, sample_features, "test_session"
        )
        
        # Session risk
        session_risk = risk_engine.score_session(
            session_id="test_session",
            user_id=1,
            location_risks=location_risks,
            session_spoofing=session_spoofing,
        )
        
        # Assertions
        assert len(location_risks) == len(sample_events)
        assert all(0.0 <= r.risk_score <= 1.0 for r in location_risks)
        assert 0.0 <= session_risk.risk_score <= 1.0
        assert session_risk.total_points == len(sample_events)
        
        # Explain session risk
        explanation = explainability.explain_session_risk(session_risk)
        assert "summary" in explanation
        assert "narrative" in explanation


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
