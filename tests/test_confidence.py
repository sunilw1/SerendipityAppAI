"""
Tests for Confidence Scoring
==============================

Tests for the confidence scoring system.
"""

import pytest
from datetime import datetime, timezone

from app.features.confidence import (
    ConfidenceScorer,
    score_confidence,
    score_confidence_batch,
)
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    LocationPoint,
    ConfidenceScore,
)
from app.core.constants import ActivityType, DataQualityFlag


class TestConfidenceScorer:
    """Tests for ConfidenceScorer."""
    
    @pytest.fixture
    def scorer(self) -> ConfidenceScorer:
        """Create a scorer instance."""
        return ConfidenceScorer()
    
    def test_score_range(
        self,
        scorer: ConfidenceScorer,
        sample_normalized_event: NormalizedTrackingEvent,
        sample_features: ComputedFeatures
    ):
        """Test confidence score is in valid range."""
        score = scorer.score(sample_normalized_event, sample_features)
        
        assert 0.0 <= score.overall <= 1.0
        assert 0.0 <= score.gps_accuracy_score <= 1.0
        assert 0.0 <= score.speed_validity_score <= 1.0
    
    def test_high_quality_event(self, scorer: ConfidenceScorer):
        """Test high quality event gets high score."""
        event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0, accuracy_meters=5.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=5.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        features = ComputedFeatures(
            time_delta_seconds=2.0,
            update_rate_hz=0.5,
            distance_meters=6.0,
            calculated_speed_ms=3.0,
            speed_difference_ms=0.0,
            acceleration_ms2=0.5,
            is_stop=False,
        )
        
        score = scorer.score(event, features)
        
        # High quality should score above 0.7
        assert score.overall >= 0.7
    
    def test_low_gps_accuracy_penalty(self, scorer: ConfidenceScorer):
        """Test low GPS accuracy reduces score."""
        # Good accuracy event
        good_event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0, accuracy_meters=5.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=5.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        # Poor accuracy event
        poor_event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0, accuracy_meters=150.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=150.0,  # Very poor
            quality_flags=[DataQualityFlag.LOW_GPS_ACCURACY.value],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        features = ComputedFeatures(
            time_delta_seconds=2.0,
            calculated_speed_ms=3.0,
        )
        
        good_score = scorer.score(good_event, features)
        poor_score = scorer.score(poor_event, features)
        
        assert good_score.gps_accuracy_score > poor_score.gps_accuracy_score
        assert good_score.overall > poor_score.overall
    
    def test_quality_flags_reduce_score(self, scorer: ConfidenceScorer):
        """Test quality flags reduce overall score."""
        # Event without flags
        clean_event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=10.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        # Event with flags
        flagged_event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=10.0,
            quality_flags=[
                DataQualityFlag.UNREALISTIC_SPEED.value,
                DataQualityFlag.GPS_DRIFT.value,
            ],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        features = ComputedFeatures()
        
        clean_score = scorer.score(clean_event, features)
        flagged_score = scorer.score(flagged_event, features)
        
        assert clean_score.overall > flagged_score.overall
    
    def test_confidence_level_labels(self, scorer: ConfidenceScorer):
        """Test confidence level labels are correct."""
        # Create events with different quality levels
        features = ComputedFeatures()
        
        # Excellent quality
        excellent_event = NormalizedTrackingEvent(
            event_id="test",
            user_id=1,
            trip_id=1,
            point_index=0,
            timestamp=datetime.now(timezone.utc),
            timestamp_unix=datetime.now(timezone.utc).timestamp(),
            location=LocationPoint(lat=34.0, lon=-118.0, accuracy_meters=3.0),
            speed_ms=3.0,
            is_moving=True,
            activity_type=ActivityType.WALKING,
            activity_confidence=100,
            gps_accuracy_meters=3.0,
            quality_flags=[],
            is_valid=True,
            raw_event_hash="abc",
        )
        
        score = scorer.score(excellent_event, features)
        
        # Should be one of the valid labels
        assert score.confidence_level in ["excellent", "good", "moderate", "low", "unreliable"]


class TestConvenienceFunctions:
    """Tests for convenience functions."""
    
    def test_score_confidence(
        self,
        sample_normalized_event: NormalizedTrackingEvent,
        sample_features: ComputedFeatures
    ):
        """Test score_confidence convenience function."""
        score = score_confidence(sample_normalized_event, sample_features)
        
        assert isinstance(score, ConfidenceScore)
        assert 0.0 <= score.overall <= 1.0
    
    def test_score_confidence_batch(
        self,
        sample_normalized_event: NormalizedTrackingEvent,
        sample_features: ComputedFeatures
    ):
        """Test batch scoring."""
        events = [sample_normalized_event] * 5
        features = [sample_features] * 5
        
        scores = score_confidence_batch(events, features)
        
        assert len(scores) == 5
        for score in scores:
            assert isinstance(score, ConfidenceScore)
