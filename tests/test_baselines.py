"""
Tests for Baseline Learning
=============================

Tests for the baseline behavior learning system.
"""

import pytest
from datetime import datetime, timezone
from typing import List

from app.models.baseline import (
    BaselineLearner,
    BaselineAnalyzer,
    learn_user_baseline,
)
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    UserBaseline,
    LocationPoint,
)
from app.core.constants import ActivityType


class TestBaselineLearner:
    """Tests for BaselineLearner."""
    
    @pytest.fixture
    def learner(self) -> BaselineLearner:
        """Create a learner instance."""
        return BaselineLearner(min_confidence=0.5)
    
    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample data for testing."""
        events = []
        features = []
        scores = []
        
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        for i in range(100):
            event = NormalizedTrackingEvent(
                event_id=f"test_{i}",
                user_id=1,
                trip_id=i // 20,  # 5 trips
                point_index=i % 20,
                timestamp=datetime(
                    2024, 1, 1, 10, i // 60, i % 60,
                    tzinfo=timezone.utc
                ),
                timestamp_unix=base_time.timestamp() + i * 2,
                location=LocationPoint(
                    lat=34.0 + i * 0.0001,
                    lon=-118.0 + i * 0.0001,
                ),
                speed_ms=5.0 + (i % 10) * 0.5,
                is_moving=True,
                activity_type=ActivityType.WALKING if i % 3 else ActivityType.IN_VEHICLE,
                activity_confidence=80,
                gps_accuracy_meters=10.0,
                quality_flags=[],
                is_valid=True,
                raw_event_hash=f"hash_{i}",
            )
            events.append(event)
            
            feat = ComputedFeatures(
                time_delta_seconds=2.0,
                update_rate_hz=0.5,
                distance_meters=10.0,
                calculated_speed_ms=5.0,
            )
            features.append(feat)
            
            score = ConfidenceScore(
                overall=0.8,
                gps_accuracy_score=0.9,
                speed_validity_score=0.8,
                acceleration_validity_score=0.8,
                temporal_consistency_score=0.9,
                activity_consistency_score=0.8,
                signal_continuity_score=0.9,
            )
            scores.append(score)
        
        return events, features, scores
    
    def test_learn_baseline(
        self,
        learner: BaselineLearner,
        sample_data: tuple
    ):
        """Test basic baseline learning."""
        events, features, scores = sample_data
        
        baseline = learner.learn_baseline(1, events, features, scores)
        
        assert isinstance(baseline, UserBaseline)
        assert baseline.user_id == 1
        assert baseline.points_analyzed > 0
    
    def test_baseline_speed_stats(
        self,
        learner: BaselineLearner,
        sample_data: tuple
    ):
        """Test speed baseline statistics."""
        events, features, scores = sample_data
        
        baseline = learner.learn_baseline(1, events, features, scores)
        
        assert baseline.speed_baseline.mean > 0
        assert baseline.speed_baseline.std >= 0
        assert baseline.speed_baseline.min_value <= baseline.speed_baseline.max_value
    
    def test_baseline_activity_distribution(
        self,
        learner: BaselineLearner,
        sample_data: tuple
    ):
        """Test activity distribution is computed."""
        events, features, scores = sample_data
        
        baseline = learner.learn_baseline(1, events, features, scores)
        
        assert len(baseline.activity_distribution) > 0
        assert sum(baseline.activity_distribution.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_baseline_maturity(
        self,
        learner: BaselineLearner,
        sample_data: tuple
    ):
        """Test baseline maturity check."""
        events, features, scores = sample_data
        
        # Full data should be mature
        full_baseline = learner.learn_baseline(1, events, features, scores)
        assert full_baseline.is_mature
        
        # Small data should not be mature
        small_baseline = learner.learn_baseline(
            1, events[:10], features[:10], scores[:10]
        )
        assert not small_baseline.is_mature
    
    def test_update_baseline(
        self,
        learner: BaselineLearner,
        sample_data: tuple
    ):
        """Test baseline update with new data."""
        events, features, scores = sample_data
        
        # Initial baseline
        initial = learner.learn_baseline(
            1, events[:50], features[:50], scores[:50]
        )
        
        # Update with new data
        updated = learner.update_baseline(
            initial,
            events[50:],
            features[50:],
            scores[50:],
        )
        
        assert updated.points_analyzed > initial.points_analyzed
        assert updated.updated_at >= initial.updated_at


class TestBaselineAnalyzer:
    """Tests for BaselineAnalyzer."""
    
    @pytest.fixture
    def baseline_and_analyzer(self, sample_data) -> tuple:
        """Create baseline and analyzer."""
        events, features, scores = sample_data
        
        learner = BaselineLearner()
        baseline = learner.learn_baseline(1, events, features, scores)
        analyzer = BaselineAnalyzer(baseline)
        
        return baseline, analyzer
    
    @pytest.fixture
    def sample_data(self) -> tuple:
        """Create sample data."""
        events = []
        features = []
        scores = []
        
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        for i in range(100):
            event = NormalizedTrackingEvent(
                event_id=f"test_{i}",
                user_id=1,
                trip_id=i // 20,
                point_index=i % 20,
                timestamp=datetime(
                    2024, 1, 1, 10, i // 60, i % 60,
                    tzinfo=timezone.utc
                ),
                timestamp_unix=base_time.timestamp() + i * 2,
                location=LocationPoint(
                    lat=34.0 + i * 0.0001,
                    lon=-118.0 + i * 0.0001,
                ),
                speed_ms=5.0,
                is_moving=True,
                activity_type=ActivityType.WALKING,
                activity_confidence=80,
                gps_accuracy_meters=10.0,
                quality_flags=[],
                is_valid=True,
                raw_event_hash=f"hash_{i}",
            )
            events.append(event)
            features.append(ComputedFeatures(update_rate_hz=0.5))
            scores.append(ConfidenceScore(
                overall=0.8,
                gps_accuracy_score=0.9,
                speed_validity_score=0.8,
                acceleration_validity_score=0.8,
                temporal_consistency_score=0.9,
                activity_consistency_score=0.8,
                signal_continuity_score=0.9,
            ))
        
        return events, features, scores
    
    def test_analyze_event(self, baseline_and_analyzer, sample_data):
        """Test event analysis against baseline."""
        baseline, analyzer = baseline_and_analyzer
        events, features, scores = sample_data
        
        result = analyzer.analyze_event(events[0], features[0])
        
        assert "user_id" in result
        assert "baseline_mature" in result
    
    def test_analyze_trip(self, baseline_and_analyzer, sample_data):
        """Test trip analysis against baseline."""
        baseline, analyzer = baseline_and_analyzer
        events, features, scores = sample_data
        
        result = analyzer.analyze_trip(events[:20], features[:20])
        
        assert "trip_id" in result
        assert "duration_seconds" in result


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_learn_user_baseline(self):
        """Test learn_user_baseline function."""
        base_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        
        events = [
            NormalizedTrackingEvent(
                event_id=f"test_{i}",
                user_id=1,
                trip_id=0,
                point_index=i,
                timestamp=datetime(2024, 1, 1, 10, 0, i, tzinfo=timezone.utc),
                timestamp_unix=base_time.timestamp() + i,
                location=LocationPoint(lat=34.0, lon=-118.0),
                speed_ms=5.0,
                is_moving=True,
                activity_type=ActivityType.WALKING,
                activity_confidence=80,
                gps_accuracy_meters=10.0,
                quality_flags=[],
                is_valid=True,
                raw_event_hash=f"hash_{i}",
            )
            for i in range(50)
        ]
        
        features = [ComputedFeatures() for _ in range(50)]
        scores = [
            ConfidenceScore(
                overall=0.8,
                gps_accuracy_score=0.9,
                speed_validity_score=0.8,
                acceleration_validity_score=0.8,
                temporal_consistency_score=0.9,
                activity_consistency_score=0.8,
                signal_continuity_score=0.9,
            )
            for _ in range(50)
        ]
        
        baseline = learn_user_baseline(1, events, features, scores)
        
        assert isinstance(baseline, UserBaseline)
        assert baseline.user_id == 1
