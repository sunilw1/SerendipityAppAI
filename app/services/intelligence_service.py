"""
Intelligence Service
=====================

Main service for processing tracking data and producing intelligence.

This service orchestrates:
1. Data ingestion
2. Normalization
3. Cleaning
4. Feature engineering
5. Confidence scoring
6. Intelligence output

Provides high-level APIs for the FastAPI routes.
"""

import time
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime

from app.core.config import settings
from app.core.logging import get_logger, log_processing_metrics
from app.data.ingestion import CSVIngestionPipeline, load_dataset
from app.data.normalization import EventNormalizer, TripNormalizer
from app.data.cleaning import DataCleaner, TripCleaner
from app.data.validation import validate_trip
from app.features.engineering import FeatureEngineer, TripFeatureEngineer, compute_features
from app.features.confidence import ConfidenceScorer, BatchConfidenceScorer
from app.models.schemas import (
    RawTrackingEvent,
    NormalizedTrackingEvent,
    ProcessedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    LocationIntelligence,
    DataQualityReport,
    Trip,
    TripSummary,
    LocationPoint,
)
from app.models.baseline import BaselineLearner, BaselineAnalyzer, UserBaseline

logger = get_logger(__name__)


class IntelligenceService:
    """
    Core intelligence service for the Serendipity backend.
    
    Provides end-to-end processing of tracking data:
    Raw → Normalized → Cleaned → Featured → Scored → Intelligence
    """
    
    def __init__(self):
        """Initialize the intelligence service with all components."""
        self.ingestion = CSVIngestionPipeline()
        self.normalizer = EventNormalizer()
        self.trip_normalizer = TripNormalizer()
        self.cleaner = TripCleaner()
        self.feature_engineer = FeatureEngineer()
        self.trip_feature_engineer = TripFeatureEngineer()
        self.confidence_scorer = ConfidenceScorer()
        self.batch_scorer = BatchConfidenceScorer()
        self.baseline_learner = BaselineLearner()
        
        # In-memory cache for baselines (use database in production)
        self._baselines: Dict[int, UserBaseline] = {}
        self._processed_cache: Dict[int, List[ProcessedTrackingEvent]] = {}
    
    def process_trip(
        self,
        trip_id: int,
        raw_events: Optional[List[RawTrackingEvent]] = None
    ) -> Trip:
        """
        Process a complete trip end-to-end.
        
        Args:
            trip_id: Global trip ID
            raw_events: Optional pre-loaded events (loads from file if None)
            
        Returns:
            Processed Trip with all events and summary
        """
        start_time = time.time()
        
        # Load events if not provided
        if raw_events is None:
            raw_events = self.ingestion.load_events_by_trip(trip_id)
        
        if not raw_events:
            raise ValueError(f"No events found for trip {trip_id}")
        
        logger.info("processing_trip", trip_id=trip_id, num_events=len(raw_events))
        
        # 1. Normalize
        normalized = self.trip_normalizer.normalize_trip(raw_events)
        
        # 2. Clean
        cleaned, cleaning_stats = self.cleaner.clean_trip(normalized)
        
        # 3. Feature engineering
        features = self.feature_engineer.compute_features_batch(cleaned)
        
        # 4. Confidence scoring
        scores, score_stats = self.batch_scorer.score_trip(cleaned, features)
        
        # 5. Create processed events
        processed_events = [
            ProcessedTrackingEvent(
                event=event,
                features=feat,
                confidence=score,
            )
            for event, feat, score in zip(cleaned, features, scores)
        ]
        
        # 6. Create trip summary
        summary = self._create_trip_summary(
            trip_id,
            cleaned,
            features,
            scores,
        )
        
        duration_ms = (time.time() - start_time) * 1000
        log_processing_metrics(
            logger,
            "process_trip",
            len(processed_events),
            duration_ms,
            extra={"trip_id": trip_id}
        )
        
        return Trip(
            summary=summary,
            events=processed_events,
        )
    
    def process_user_data(
        self,
        user_id: int,
        limit: Optional[int] = None
    ) -> Tuple[List[Trip], UserBaseline]:
        """
        Process all data for a user and learn baseline.
        
        Args:
            user_id: User identifier
            limit: Optional limit on events per user
            
        Returns:
            Tuple of (processed trips, learned baseline)
        """
        start_time = time.time()
        
        # Load user events
        raw_events = self.ingestion.load_events_by_user(user_id, limit)
        
        if not raw_events:
            raise ValueError(f"No events found for user {user_id}")
        
        logger.info("processing_user", user_id=user_id, num_events=len(raw_events))
        
        # Group by trip
        trips_raw: Dict[int, List[RawTrackingEvent]] = {}
        for event in raw_events:
            if event.trip_global_index not in trips_raw:
                trips_raw[event.trip_global_index] = []
            trips_raw[event.trip_global_index].append(event)
        
        # Process each trip
        trips = []
        all_events = []
        all_features = []
        all_scores = []
        
        for trip_id, trip_events in trips_raw.items():
            try:
                trip = self.process_trip(trip_id, trip_events)
                trips.append(trip)
                
                for pe in trip.events:
                    all_events.append(pe.event)
                    all_features.append(pe.features)
                    all_scores.append(pe.confidence)
                    
            except Exception as e:
                logger.warning(
                    "trip_processing_failed",
                    trip_id=trip_id,
                    error=str(e)
                )
        
        # Learn baseline
        baseline = self.baseline_learner.learn_baseline(
            user_id,
            all_events,
            all_features,
            all_scores,
        )
        
        # Cache baseline
        self._baselines[user_id] = baseline
        
        # Cache processed events for data quality reports
        all_processed = []
        for trip in trips:
            all_processed.extend(trip.events)
        self._processed_cache[user_id] = all_processed
        
        duration_ms = (time.time() - start_time) * 1000
        log_processing_metrics(
            logger,
            "process_user",
            len(all_events),
            duration_ms,
            extra={"user_id": user_id, "num_trips": len(trips)}
        )
        
        return trips, baseline
    
    def get_location_intelligence(
        self,
        user_id: int,
        trip_id: Optional[int] = None,
        min_confidence: float = 0.0,
        limit: int = 100,
        offset: int = 0,
    ) -> Tuple[List[LocationIntelligence], int]:
        """
        Get location intelligence data for a user.
        
        Args:
            user_id: User identifier
            trip_id: Optional specific trip
            min_confidence: Minimum confidence threshold
            limit: Maximum results
            offset: Pagination offset
            
        Returns:
            Tuple of (intelligence list, total count)
        """
        # Load and process if not cached
        if user_id not in self._processed_cache:
            trips, _ = self.process_user_data(user_id)
            
            # Flatten to events
            all_events = []
            for trip in trips:
                all_events.extend(trip.events)
            
            self._processed_cache[user_id] = all_events
        
        processed = self._processed_cache[user_id]
        
        # Filter by trip if specified
        if trip_id is not None:
            processed = [e for e in processed if e.event.trip_id == trip_id]
        
        # Filter by confidence
        processed = [e for e in processed if e.confidence.overall >= min_confidence]
        
        total = len(processed)
        
        # Paginate
        processed = processed[offset:offset + limit]
        
        # Convert to intelligence format
        intelligence = [
            self._to_intelligence(pe) for pe in processed
        ]
        
        return intelligence, total
    
    def get_data_quality_report(
        self,
        user_id: int,
        trip_id: Optional[int] = None,
    ) -> DataQualityReport:
        """
        Generate a data quality report.
        
        Args:
            user_id: User identifier
            trip_id: Optional specific trip
            
        Returns:
            DataQualityReport
        """
        # Ensure data is processed
        if user_id not in self._processed_cache:
            self.process_user_data(user_id)
        
        processed = self._processed_cache[user_id]
        
        if trip_id is not None:
            processed = [e for e in processed if e.event.trip_id == trip_id]
        
        if not processed:
            return DataQualityReport(
                user_id=user_id,
                trip_id=trip_id,
                total_events=0,
                valid_events=0,
                flagged_events=0,
                flag_counts={},
                confidence_mean=0.0,
                confidence_std=0.0,
                confidence_min=0.0,
                confidence_max=0.0,
                high_confidence_count=0,
                medium_confidence_count=0,
                low_confidence_count=0,
            )
        
        # Compute statistics
        total_events = len(processed)
        valid_events = sum(1 for e in processed if e.event.is_valid)
        
        # Flag counts
        flag_counts: Dict[str, int] = {}
        flagged_events = 0
        for pe in processed:
            if pe.event.quality_flags:
                flagged_events += 1
                for flag in pe.event.quality_flags:
                    flag_counts[flag] = flag_counts.get(flag, 0) + 1
        
        # Confidence statistics
        confidences = [e.confidence.overall for e in processed]
        
        high_conf = sum(1 for c in confidences if c >= 0.75)
        medium_conf = sum(1 for c in confidences if 0.5 <= c < 0.75)
        low_conf = sum(1 for c in confidences if c < 0.5)
        
        import numpy as np
        
        return DataQualityReport(
            user_id=user_id,
            trip_id=trip_id,
            total_events=total_events,
            valid_events=valid_events,
            flagged_events=flagged_events,
            flag_counts=flag_counts,
            confidence_mean=float(np.mean(confidences)),
            confidence_std=float(np.std(confidences)),
            confidence_min=float(np.min(confidences)),
            confidence_max=float(np.max(confidences)),
            high_confidence_count=high_conf,
            medium_confidence_count=medium_conf,
            low_confidence_count=low_conf,
        )
    
    def get_user_baseline(self, user_id: int) -> Optional[UserBaseline]:
        """Get cached baseline for a user."""
        return self._baselines.get(user_id)
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """Get statistics about the dataset."""
        try:
            return self.ingestion.get_file_stats()
        except Exception as e:
            logger.error("failed_to_get_stats", error=str(e))
            return {}
    
    def _create_trip_summary(
        self,
        trip_id: int,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        scores: List[ConfidenceScore],
    ) -> TripSummary:
        """Create a trip summary from processed data."""
        if not events:
            raise ValueError("Cannot create summary from empty events")
        
        user_id = events[0].user_id
        
        # Temporal
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration_seconds = events[-1].timestamp_unix - events[0].timestamp_unix
        
        # Spatial
        distances = [f.distance_meters for f in features if f.distance_meters is not None]
        total_distance = sum(distances)
        
        start_location = events[0].location
        end_location = events[-1].location
        
        # Quality
        total_points = len(events)
        valid_points = sum(1 for e in events if e.is_valid)
        avg_confidence = sum(s.overall for s in scores) / len(scores)
        
        # Activity
        from app.utils.metrics import compute_activity_distribution
        from app.core.constants import ActivityType
        
        activities = [e.activity_type.value for e in events]
        activity_breakdown = compute_activity_distribution(activities)
        
        primary_activity_str = max(activity_breakdown, key=activity_breakdown.get)
        primary_activity = ActivityType.from_string(primary_activity_str)
        
        return TripSummary(
            trip_id=trip_id,
            user_id=user_id,
            start_time=start_time,
            end_time=end_time,
            duration_seconds=duration_seconds,
            distance_meters=total_distance,
            start_location=start_location,
            end_location=end_location,
            total_points=total_points,
            valid_points=valid_points,
            average_confidence=avg_confidence,
            primary_activity=primary_activity,
            activity_breakdown=activity_breakdown,
        )
    
    def _to_intelligence(
        self,
        pe: ProcessedTrackingEvent
    ) -> LocationIntelligence:
        """Convert ProcessedTrackingEvent to LocationIntelligence."""
        return LocationIntelligence(
            user_id=pe.event.user_id,
            event_id=pe.event.event_id,
            timestamp=pe.event.timestamp,
            latitude=pe.event.location.lat,
            longitude=pe.event.location.lon,
            accuracy_meters=pe.event.gps_accuracy_meters,
            speed_ms=pe.event.speed_ms,
            is_moving=pe.event.is_moving,
            activity=pe.event.activity_type.value,
            confidence=pe.confidence.overall,
            confidence_level=pe.confidence.confidence_level,
            has_quality_issues=len(pe.event.quality_flags) > 0,
            quality_issue_count=len(pe.event.quality_flags),
        )


# Global service instance
_service: Optional[IntelligenceService] = None


def get_intelligence_service() -> IntelligenceService:
    """Get or create the global intelligence service."""
    global _service
    if _service is None:
        _service = IntelligenceService()
    return _service
