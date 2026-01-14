"""
Baseline Service
=================

Service for managing user behavior baselines.

Provides:
- Baseline computation from user data
- Baseline storage and retrieval
- Baseline updates with new data
- Baseline comparison and analysis
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

from app.core.logging import get_logger
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
)
from app.models.baseline import (
    BaselineLearner,
    BaselineAnalyzer,
    UserBaseline,
)

logger = get_logger(__name__)


class BaselineService:
    """
    Service for managing user behavior baselines.
    
    In Phase 1, baselines are:
    - Descriptive (observe-only)
    - Stored in memory (use database in production)
    - Updated incrementally
    """
    
    def __init__(self):
        """Initialize baseline service."""
        self.learner = BaselineLearner()
        # In-memory storage (replace with database in production)
        self._baselines: Dict[int, UserBaseline] = {}
        self._analyzers: Dict[int, BaselineAnalyzer] = {}
    
    def compute_baseline(
        self,
        user_id: int,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> UserBaseline:
        """
        Compute and store baseline for a user.
        
        Args:
            user_id: User identifier
            events: Normalized events
            features: Computed features
            scores: Confidence scores
            
        Returns:
            Computed UserBaseline
        """
        logger.info(
            "computing_baseline",
            user_id=user_id,
            num_events=len(events)
        )
        
        baseline = self.learner.learn_baseline(
            user_id,
            events,
            features,
            scores,
        )
        
        self._baselines[user_id] = baseline
        self._analyzers[user_id] = BaselineAnalyzer(baseline)
        
        logger.info(
            "baseline_computed",
            user_id=user_id,
            trips_analyzed=baseline.trips_analyzed,
            points_analyzed=baseline.points_analyzed,
            typical_confidence=round(baseline.typical_confidence, 3),
        )
        
        return baseline
    
    def update_baseline(
        self,
        user_id: int,
        new_events: List[NormalizedTrackingEvent],
        new_features: List[ComputedFeatures],
        new_scores: List[ConfidenceScore],
    ) -> Optional[UserBaseline]:
        """
        Update existing baseline with new data.
        
        Args:
            user_id: User identifier
            new_events: New normalized events
            new_features: New computed features
            new_scores: New confidence scores
            
        Returns:
            Updated baseline, or None if no existing baseline
        """
        existing = self._baselines.get(user_id)
        
        if existing is None:
            logger.warning(
                "no_existing_baseline",
                user_id=user_id,
                action="creating_new"
            )
            return self.compute_baseline(
                user_id, new_events, new_features, new_scores
            )
        
        updated = self.learner.update_baseline(
            existing,
            new_events,
            new_features,
            new_scores,
        )
        
        self._baselines[user_id] = updated
        self._analyzers[user_id] = BaselineAnalyzer(updated)
        
        logger.info(
            "baseline_updated",
            user_id=user_id,
            total_points=updated.points_analyzed,
        )
        
        return updated
    
    def get_baseline(self, user_id: int) -> Optional[UserBaseline]:
        """
        Get baseline for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            UserBaseline or None
        """
        return self._baselines.get(user_id)
    
    def has_baseline(self, user_id: int) -> bool:
        """Check if user has a baseline."""
        return user_id in self._baselines
    
    def analyze_event(
        self,
        user_id: int,
        event: NormalizedTrackingEvent,
        features: ComputedFeatures,
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze an event against user baseline.
        
        Args:
            user_id: User identifier
            event: Normalized event
            features: Computed features
            
        Returns:
            Analysis results, or None if no baseline
        """
        analyzer = self._analyzers.get(user_id)
        
        if analyzer is None:
            return None
        
        return analyzer.analyze_event(event, features)
    
    def analyze_trip(
        self,
        user_id: int,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
    ) -> Optional[Dict[str, Any]]:
        """
        Analyze a trip against user baseline.
        
        Args:
            user_id: User identifier
            events: Trip events
            features: Computed features
            
        Returns:
            Analysis results, or None if no baseline
        """
        analyzer = self._analyzers.get(user_id)
        
        if analyzer is None:
            return None
        
        return analyzer.analyze_trip(events, features)
    
    def get_baseline_summary(self, user_id: int) -> Optional[Dict[str, Any]]:
        """
        Get a summary of user baseline.
        
        Args:
            user_id: User identifier
            
        Returns:
            Summary dict, or None if no baseline
        """
        baseline = self._baselines.get(user_id)
        
        if baseline is None:
            return None
        
        return {
            "user_id": baseline.user_id,
            "created_at": baseline.created_at.isoformat(),
            "updated_at": baseline.updated_at.isoformat(),
            "is_mature": baseline.is_mature,
            "trips_analyzed": baseline.trips_analyzed,
            "points_analyzed": baseline.points_analyzed,
            "date_range_days": round(baseline.date_range_days, 1),
            "typical_confidence": round(baseline.typical_confidence, 3),
            "speed": {
                "mean_ms": round(baseline.speed_baseline.mean, 2),
                "p95_ms": round(baseline.speed_baseline.p95, 2),
            },
            "activity_distribution": {
                k: round(v, 3) for k, v in baseline.activity_distribution.items()
            },
        }
    
    def list_users_with_baselines(self) -> List[int]:
        """Get list of user IDs with baselines."""
        return list(self._baselines.keys())
    
    def delete_baseline(self, user_id: int) -> bool:
        """
        Delete baseline for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            True if deleted, False if not found
        """
        if user_id in self._baselines:
            del self._baselines[user_id]
            if user_id in self._analyzers:
                del self._analyzers[user_id]
            return True
        return False


# Global service instance
_service: Optional[BaselineService] = None


def get_baseline_service() -> BaselineService:
    """Get or create the global baseline service."""
    global _service
    if _service is None:
        _service = BaselineService()
    return _service
