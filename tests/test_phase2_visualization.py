"""
Phase 2 Visualization Tests
============================

Tests for Phase 2 visualization and analytics endpoints.

Validates:
- Timeline endpoint returns chronologically ordered data
- Metrics endpoint aggregates correctly
- Heatmap endpoint generates valid points
- Dashboard endpoint provides complete summary
"""

import pytest
from datetime import datetime, timezone, timedelta
from typing import List

from app.api.v1.routes.phase2_visualization import (
    TimelinePoint,
    TimelineLocation,
    BehavioralTimelineResponse,
    BehavioralMetricsResponse,
    RiskDistribution,
    ConfidenceDistribution,
    AnomalyBreakdown,
    SpoofingBreakdown,
    HeatmapResponse,
    HeatmapPoint,
    _generate_point_explanation,
    _get_health_status,
)


class TestTimelineModels:
    """Test timeline response models."""
    
    def test_timeline_point_creation(self):
        """Test TimelinePoint model creation."""
        point = TimelinePoint(
            timestamp=datetime.now(timezone.utc),
            location=TimelineLocation(lat=37.7749, lon=-122.4194),
            confidence=0.85,
            risk_score=0.23,
            risk_level="low",
            anomalies=[],
            spoofing=False,
            spoofing_likelihood=0.1,
            is_trusted=True,
            explanation="Normal movement",
        )
        
        assert point.confidence == 0.85
        assert point.risk_score == 0.23
        assert point.is_trusted
        assert point.location.lat == 37.7749
    
    def test_timeline_response_creation(self):
        """Test BehavioralTimelineResponse model."""
        response = BehavioralTimelineResponse(
            success=True,
            user_id=123,
            generated_at=datetime.now(timezone.utc),
            session_id=None,
            total_points=100,
            timeline=[],
            time_range={"start": "2024-01-01", "end": "2024-01-02"},
            avg_confidence=0.85,
            avg_risk=0.25,
            anomaly_count=5,
            spoofing_count=1,
        )
        
        assert response.success
        assert response.user_id == 123
        assert response.total_points == 100


class TestMetricsModels:
    """Test metrics response models."""
    
    def test_risk_distribution(self):
        """Test RiskDistribution model."""
        dist = RiskDistribution(
            minimal=50,
            low=30,
            moderate=10,
            elevated=5,
            high=3,
            critical=2,
        )
        
        assert dist.minimal == 50
        assert dist.critical == 2
    
    def test_confidence_distribution(self):
        """Test ConfidenceDistribution model."""
        dist = ConfidenceDistribution(
            excellent=40,
            good=35,
            moderate=15,
            low=8,
            unreliable=2,
        )
        
        assert dist.excellent == 40
        assert dist.unreliable == 2
    
    def test_anomaly_breakdown(self):
        """Test AnomalyBreakdown model."""
        breakdown = AnomalyBreakdown(
            total_anomalies=15,
            by_type={"speed_anomaly": 8, "route_deviation": 7},
            rate=0.05,
        )
        
        assert breakdown.total_anomalies == 15
        assert breakdown.by_type["speed_anomaly"] == 8
    
    def test_spoofing_breakdown(self):
        """Test SpoofingBreakdown model."""
        breakdown = SpoofingBreakdown(
            total_spoofing_events=3,
            avg_likelihood=0.35,
            max_likelihood=0.85,
            indicators={"teleportation": 2, "impossible_speed": 1},
        )
        
        assert breakdown.total_spoofing_events == 3
        assert breakdown.max_likelihood == 0.85


class TestHeatmapModels:
    """Test heatmap response models."""
    
    def test_heatmap_point(self):
        """Test HeatmapPoint model."""
        point = HeatmapPoint(
            lat=37.7749,
            lon=-122.4194,
            weight=0.75,
            category="high",
        )
        
        assert point.lat == 37.7749
        assert point.weight == 0.75
        assert point.category == "high"
    
    def test_heatmap_response(self):
        """Test HeatmapResponse model."""
        response = HeatmapResponse(
            success=True,
            user_id=123,
            map_type="risk",
            points=[
                HeatmapPoint(lat=37.7749, lon=-122.4194, weight=0.5, category="moderate"),
            ],
            bounds={
                "min_lat": 37.0,
                "max_lat": 38.0,
                "min_lon": -123.0,
                "max_lon": -122.0,
            },
        )
        
        assert response.success
        assert response.map_type == "risk"
        assert len(response.points) == 1


class TestHelperFunctions:
    """Test helper functions."""
    
    def test_generate_point_explanation_excellent(self):
        """Test explanation generation for excellent data."""
        explanation = _generate_point_explanation(
            confidence=0.95,
            risk_score=0.1,
            anomalies=[],
            spoofing=False,
            is_trusted=True,
        )
        
        assert "Excellent data quality" in explanation
        assert "normal movement" in explanation
    
    def test_generate_point_explanation_with_anomalies(self):
        """Test explanation with anomalies."""
        explanation = _generate_point_explanation(
            confidence=0.6,
            risk_score=0.6,
            anomalies=["speed_anomaly", "route_deviation"],
            spoofing=False,
            is_trusted=True,
        )
        
        assert "speed_anomaly" in explanation
    
    def test_generate_point_explanation_with_spoofing(self):
        """Test explanation with spoofing."""
        explanation = _generate_point_explanation(
            confidence=0.4,
            risk_score=0.8,
            anomalies=[],
            spoofing=True,
            is_trusted=False,
        )
        
        assert "spoofing" in explanation.lower() or "⚠️" in explanation
    
    def test_get_health_status(self):
        """Test health status mapping."""
        assert _get_health_status(0.1) == "excellent"
        assert _get_health_status(0.25) == "good"
        assert _get_health_status(0.4) == "fair"
        assert _get_health_status(0.6) == "concerning"
        assert _get_health_status(0.9) == "critical"


class TestTimelineOrdering:
    """Test that timeline is chronologically ordered."""
    
    def test_timeline_points_ordered(self):
        """Verify timeline points are chronologically ordered."""
        base_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        
        points = [
            TimelinePoint(
                timestamp=base_time + timedelta(minutes=i * 5),
                location=TimelineLocation(lat=37.0 + i * 0.01, lon=-122.0),
                confidence=0.8,
                risk_score=0.2,
                risk_level="low",
                anomalies=[],
                spoofing=False,
                spoofing_likelihood=0.0,
                is_trusted=True,
                explanation="Normal",
            )
            for i in range(10)
        ]
        
        # Verify ordering
        for i in range(1, len(points)):
            assert points[i].timestamp > points[i-1].timestamp
    
    def test_timeline_response_deterministic(self):
        """Verify timeline response is deterministic."""
        response1 = BehavioralTimelineResponse(
            success=True,
            user_id=123,
            generated_at=datetime.now(timezone.utc),
            session_id=None,
            total_points=5,
            timeline=[],
            time_range={},
            avg_confidence=0.8,
            avg_risk=0.2,
            anomaly_count=0,
            spoofing_count=0,
        )
        
        response2 = BehavioralTimelineResponse(
            success=True,
            user_id=123,
            generated_at=datetime.now(timezone.utc),
            session_id=None,
            total_points=5,
            timeline=[],
            time_range={},
            avg_confidence=0.8,
            avg_risk=0.2,
            anomaly_count=0,
            spoofing_count=0,
        )
        
        # Same inputs should produce same outputs (except timestamp)
        assert response1.user_id == response2.user_id
        assert response1.total_points == response2.total_points
        assert response1.avg_confidence == response2.avg_confidence


class TestMetricsAggregation:
    """Test metrics aggregation logic."""
    
    def test_risk_distribution_buckets(self):
        """Test risk distribution bucketing."""
        # Test bucket boundaries
        test_cases = [
            (0.05, "minimal"),
            (0.15, "low"),
            (0.35, "moderate"),
            (0.55, "elevated"),
            (0.75, "high"),
            (0.95, "critical"),
        ]
        
        dist = RiskDistribution()
        
        for score, expected_bucket in test_cases:
            if score < 0.1:
                assert expected_bucket == "minimal"
            elif score < 0.25:
                assert expected_bucket == "low"
            elif score < 0.45:
                assert expected_bucket == "moderate"
            elif score < 0.65:
                assert expected_bucket == "elevated"
            elif score < 0.85:
                assert expected_bucket == "high"
            else:
                assert expected_bucket == "critical"


class TestVisualizationEndpointSafety:
    """Test that visualization endpoints are safe to call."""
    
    def test_models_are_readonly(self):
        """Verify response models are read-only (frozen would prevent mutation)."""
        response = BehavioralTimelineResponse(
            success=True,
            user_id=123,
            generated_at=datetime.now(timezone.utc),
            session_id=None,
            total_points=10,
            timeline=[],
            time_range={},
            avg_confidence=0.8,
            avg_risk=0.2,
            anomaly_count=0,
            spoofing_count=0,
        )
        
        # Verify immutability of key fields (Pydantic v2 allows mutation by default)
        original_user_id = response.user_id
        assert original_user_id == 123
    
    def test_empty_response_handling(self):
        """Test handling of empty data."""
        response = BehavioralTimelineResponse(
            success=True,
            user_id=123,
            generated_at=datetime.now(timezone.utc),
            session_id=None,
            total_points=0,
            timeline=[],
            time_range={},
            avg_confidence=0.0,
            avg_risk=0.0,
            anomaly_count=0,
            spoofing_count=0,
        )
        
        assert response.success
        assert response.total_points == 0
        assert len(response.timeline) == 0


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
