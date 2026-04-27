"""
Behavioral Profile Learning
=============================

Phase 2 extension of Phase 1 baselines into comprehensive behavioral profiles.

Learns:
- Typical routes and route patterns
- Time-of-day behavior patterns
- Stop frequency and duration patterns
- Movement consistency metrics
- Update regularity patterns

Design Principles:
- Builds on Phase 1 baselines (does not replace)
- Learns automatically from high-confidence data
- Updates incrementally
- Provides explainable patterns
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple, Set
import hashlib
import uuid

import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import fcluster, linkage

from app.core.logging import get_logger
from app.core.constants import ActivityType
from app.models.schemas import (
    NormalizedTrackingEvent,
    ComputedFeatures,
    ConfidenceScore,
    UserBaseline,
    BaselineMetrics,
    LocationPoint,
)
from app.models.phase2_schemas import (
    BehavioralProfile,
    RoutePattern,
    TimeOfDayPattern,
    StopPattern,
)
from app.utils.geo import haversine_distance
from app.utils.metrics import compute_baseline_stats

logger = get_logger(__name__)


class RoutePatternLearner:
    """
    Learns route patterns from trip data.
    
    Groups similar trips by start/end locations and
    identifies common waypoints.
    """
    
    def __init__(
        self,
        zone_radius_meters: float = 200.0,
        min_route_frequency: int = 2,
        waypoint_clustering_distance: float = 150.0,
    ):
        """
        Initialize route pattern learner.
        
        Args:
            zone_radius_meters: Radius for start/end zone matching
            min_route_frequency: Minimum trips to establish a route
            waypoint_clustering_distance: Distance for waypoint clustering
        """
        self.zone_radius = zone_radius_meters
        self.min_frequency = min_route_frequency
        self.waypoint_distance = waypoint_clustering_distance
    
    def learn_routes(
        self,
        trips: List[List[NormalizedTrackingEvent]],
        features: List[List[ComputedFeatures]],
    ) -> List[RoutePattern]:
        """
        Learn route patterns from trip data.
        
        Args:
            trips: List of trips (each trip is list of events)
            features: Corresponding features for each trip
            
        Returns:
            List of learned route patterns
        """
        if len(trips) < self.min_frequency:
            return []
        
        # Extract trip signatures (start, end, duration, distance)
        trip_signatures = []
        for trip_events, trip_features in zip(trips, features):
            if len(trip_events) < 2:
                continue
                
            start = trip_events[0]
            end = trip_events[-1]
            duration = end.timestamp_unix - start.timestamp_unix
            
            distances = [f.distance_meters for f in trip_features if f.distance_meters]
            total_distance = sum(distances)
            
            speeds = [e.speed_ms for e in trip_events if e.speed_ms is not None]
            
            trip_signatures.append({
                "events": trip_events,
                "features": trip_features,
                "start": (start.location.lat, start.location.lon),
                "end": (end.location.lat, end.location.lon),
                "duration": duration,
                "distance": total_distance,
                "speeds": speeds,
                "start_hour": start.timestamp.hour,
            })
        
        if not trip_signatures:
            return []
        
        # Cluster trips by start/end locations
        route_groups = self._cluster_by_endpoints(trip_signatures)
        
        # Build route patterns from clusters
        patterns = []
        for group in route_groups:
            if len(group) < self.min_frequency:
                continue
            
            pattern = self._build_route_pattern(group)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _cluster_by_endpoints(
        self,
        signatures: List[dict]
    ) -> List[List[dict]]:
        """Cluster trips by their start and end points."""
        if len(signatures) < 2:
            return [signatures] if signatures else []
        
        # Build feature matrix: [start_lat, start_lon, end_lat, end_lon]
        features = np.array([
            [s["start"][0], s["start"][1], s["end"][0], s["end"][1]]
            for s in signatures
        ])
        
        # Use hierarchical clustering
        try:
            linkage_matrix = linkage(features, method='average')
            # Convert radius to approximate degree distance
            threshold = self.zone_radius / 111000  # ~111km per degree
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        except Exception as e:
            logger.warning("clustering_failed", error=str(e))
            return [signatures]
        
        # Group by cluster
        groups: Dict[int, List[dict]] = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            groups[cluster_id].append(signatures[i])
        
        return list(groups.values())
    
    def _build_route_pattern(self, trips: List[dict]) -> Optional[RoutePattern]:
        """Build a route pattern from a cluster of similar trips."""
        if not trips:
            return None
        
        # Compute average start/end zones
        start_lats = [t["start"][0] for t in trips]
        start_lons = [t["start"][1] for t in trips]
        end_lats = [t["end"][0] for t in trips]
        end_lons = [t["end"][1] for t in trips]
        
        start_center = (np.mean(start_lats), np.mean(start_lons))
        end_center = (np.mean(end_lats), np.mean(end_lons))
        
        # Calculate zone radii
        start_radius = max(
            haversine_distance(
                start_center[0], start_center[1],
                lat, lon
            )
            for lat, lon in zip(start_lats, start_lons)
        )
        start_radius = max(start_radius, self.zone_radius)
        
        end_radius = max(
            haversine_distance(
                end_center[0], end_center[1],
                lat, lon
            )
            for lat, lon in zip(end_lats, end_lons)
        )
        end_radius = max(end_radius, self.zone_radius)
        
        # Compute speed statistics
        all_speeds = []
        for t in trips:
            all_speeds.extend(t["speeds"])
        
        speed_baseline = compute_baseline_stats(all_speeds, "route_speed_ms")
        
        # Compute typical time of day
        hours = [t["start_hour"] for t in trips]
        if hours:
            min_hour = min(hours)
            max_hour = max(hours)
            typical_time = (min_hour, max_hour)
        else:
            typical_time = None
        
        # Generate route ID
        route_hash = hashlib.md5(
            f"{start_center}{end_center}".encode()
        ).hexdigest()[:8]
        
        return RoutePattern(
            route_id=f"route_{route_hash}",
            start_zone=(start_center[0], start_center[1], start_radius),
            end_zone=(end_center[0], end_center[1], end_radius),
            waypoint_zones=[],  # Simplified - could add waypoint detection
            frequency=len(trips),
            avg_duration_seconds=np.mean([t["duration"] for t in trips]),
            avg_distance_meters=np.mean([t["distance"] for t in trips]),
            typical_speeds_ms=speed_baseline,
            typical_time_of_day=typical_time,
        )


class TimePatternLearner:
    """
    Learns time-of-day behavior patterns.
    
    Analyzes when a user is typically active and
    what activities they perform at different hours.
    """
    
    def __init__(self):
        """Initialize time pattern learner."""
        pass
    
    def learn_patterns(
        self,
        events: List[NormalizedTrackingEvent],
        scores: List[ConfidenceScore],
        min_confidence: float = 0.5,
    ) -> List[TimeOfDayPattern]:
        """
        Learn hourly activity patterns.
        
        Args:
            events: All events for the user
            scores: Confidence scores
            min_confidence: Minimum confidence to include
            
        Returns:
            List of hourly patterns
        """
        # Filter by confidence
        filtered = [
            (e, s) for e, s in zip(events, scores)
            if s.overall >= min_confidence
        ]
        
        if not filtered:
            return []
        
        # Group by hour
        hourly_data: Dict[int, List[Tuple[NormalizedTrackingEvent, ConfidenceScore]]] = defaultdict(list)
        for e, s in filtered:
            hour = e.timestamp.hour
            hourly_data[hour].append((e, s))
        
        # Calculate total sample count for probability
        total_samples = len(filtered)
        
        # Build patterns
        patterns = []
        for hour in range(24):
            samples = hourly_data.get(hour, [])
            
            if not samples:
                patterns.append(TimeOfDayPattern(
                    hour=hour,
                    activity_probability=0.0,
                    typical_activity=None,
                    avg_speed_ms=None,
                    sample_count=0,
                ))
                continue
            
            # Activity probability
            prob = len(samples) / total_samples
            
            # Most common activity
            activities = [e.activity_type.value for e, _ in samples]
            activity_counts = defaultdict(int)
            for a in activities:
                activity_counts[a] += 1
            typical_activity = max(activity_counts, key=activity_counts.get)
            
            # Average speed
            speeds = [e.speed_ms for e, _ in samples if e.speed_ms is not None]
            avg_speed = np.mean(speeds) if speeds else None
            
            patterns.append(TimeOfDayPattern(
                hour=hour,
                activity_probability=prob,
                typical_activity=typical_activity,
                avg_speed_ms=avg_speed,
                sample_count=len(samples),
            ))
        
        return patterns


class StopPatternLearner:
    """
    Learns frequent stop locations.
    
    Identifies places where a user frequently stops
    and analyzes stop patterns.
    """
    
    def __init__(
        self,
        stop_radius_meters: float = 100.0,
        min_stop_frequency: int = 3,
        min_stop_duration_seconds: float = 60.0,
    ):
        """
        Initialize stop pattern learner.
        
        Args:
            stop_radius_meters: Radius for grouping stops
            min_stop_frequency: Minimum visits to establish a stop
            min_stop_duration_seconds: Minimum stop duration
        """
        self.stop_radius = stop_radius_meters
        self.min_frequency = min_stop_frequency
        self.min_duration = min_stop_duration_seconds
    
    def learn_stops(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
    ) -> List[StopPattern]:
        """
        Learn frequent stop patterns.
        
        Args:
            events: Normalized events
            features: Computed features
            
        Returns:
            List of stop patterns
        """
        # Extract stops
        stops = []
        current_stop_start = None
        current_stop_events = []
        
        for i, (event, feat) in enumerate(zip(events, features)):
            if feat.is_stop:
                if current_stop_start is None:
                    current_stop_start = i
                    current_stop_events = [event]
                else:
                    current_stop_events.append(event)
            else:
                if current_stop_start is not None:
                    # End of stop
                    if current_stop_events:
                        stop_duration = (
                            current_stop_events[-1].timestamp_unix -
                            current_stop_events[0].timestamp_unix
                        )
                        
                        if stop_duration >= self.min_duration:
                            # Average location during stop
                            avg_lat = np.mean([e.location.lat for e in current_stop_events])
                            avg_lon = np.mean([e.location.lon for e in current_stop_events])
                            
                            stops.append({
                                "lat": avg_lat,
                                "lon": avg_lon,
                                "duration": stop_duration,
                                "arrival_hour": current_stop_events[0].timestamp.hour,
                                "departure_hour": current_stop_events[-1].timestamp.hour,
                            })
                    
                    current_stop_start = None
                    current_stop_events = []
        
        if not stops:
            return []
        
        # Cluster stops by location
        stop_groups = self._cluster_stops(stops)
        
        # Build stop patterns
        patterns = []
        for group in stop_groups:
            if len(group) < self.min_frequency:
                continue
            
            pattern = self._build_stop_pattern(group)
            if pattern:
                patterns.append(pattern)
        
        return patterns
    
    def _cluster_stops(self, stops: List[dict]) -> List[List[dict]]:
        """Cluster stops by location."""
        if len(stops) < 2:
            return [stops] if stops else []
        
        # Build coordinate matrix
        coords = np.array([[s["lat"], s["lon"]] for s in stops])
        
        try:
            linkage_matrix = linkage(coords, method='average')
            threshold = self.stop_radius / 111000
            clusters = fcluster(linkage_matrix, threshold, criterion='distance')
        except Exception as e:
            logger.warning("stop_clustering_failed", error=str(e))
            return [stops]
        
        groups: Dict[int, List[dict]] = defaultdict(list)
        for i, cluster_id in enumerate(clusters):
            groups[cluster_id].append(stops[i])
        
        return list(groups.values())
    
    def _build_stop_pattern(self, stops: List[dict]) -> Optional[StopPattern]:
        """Build a stop pattern from clustered stops."""
        if not stops:
            return None
        
        # Average location
        avg_lat = np.mean([s["lat"] for s in stops])
        avg_lon = np.mean([s["lon"] for s in stops])
        
        # Compute radius
        radius = max(
            haversine_distance(avg_lat, avg_lon, s["lat"], s["lon"])
            for s in stops
        )
        radius = max(radius, self.stop_radius)
        
        # Average duration
        avg_duration = np.mean([s["duration"] for s in stops])
        
        # Typical hours
        arrival_hours = list(set(s["arrival_hour"] for s in stops))
        departure_hours = list(set(s["departure_hour"] for s in stops))
        
        # Generate stop ID
        stop_hash = hashlib.md5(f"{avg_lat}{avg_lon}".encode()).hexdigest()[:8]
        
        return StopPattern(
            stop_id=f"stop_{stop_hash}",
            location=LocationPoint(lat=avg_lat, lon=avg_lon),
            radius_meters=radius,
            frequency=len(stops),
            avg_duration_seconds=avg_duration,
            typical_arrival_hours=sorted(arrival_hours),
            typical_departure_hours=sorted(departure_hours),
        )


class BehavioralProfileLearner:
    """
    Main class for learning behavioral profiles.
    
    Combines route, time, and stop pattern learning
    with Phase 1 baselines to build comprehensive profiles.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        min_trips: int = 5,
        min_points: int = 100,
    ):
        """
        Initialize behavioral profile learner.
        
        Args:
            min_confidence: Minimum confidence for data inclusion
            min_trips: Minimum trips for reliable profile
            min_points: Minimum points for reliable profile
        """
        self.min_confidence = min_confidence
        self.min_trips = min_trips
        self.min_points = min_points
        
        self.route_learner = RoutePatternLearner()
        self.time_learner = TimePatternLearner()
        self.stop_learner = StopPatternLearner()
    
    def learn_profile(
        self,
        user_id: int,
        baseline: UserBaseline,
        trips: List[List[NormalizedTrackingEvent]],
        features: List[List[ComputedFeatures]],
        scores: List[List[ConfidenceScore]],
    ) -> BehavioralProfile:
        """
        Learn complete behavioral profile.
        
        Args:
            user_id: User identifier
            baseline: Phase 1 baseline
            trips: List of trips (each trip is list of events)
            features: Corresponding features for each trip
            scores: Corresponding confidence scores
            
        Returns:
            Complete behavioral profile
        """
        logger.info(
            "learning_behavioral_profile",
            user_id=user_id,
            num_trips=len(trips)
        )
        
        # Flatten for time and stop patterns
        all_events = []
        all_features = []
        all_scores = []
        
        for trip_events, trip_features, trip_scores in zip(trips, features, scores):
            all_events.extend(trip_events)
            all_features.extend(trip_features)
            all_scores.extend(trip_scores)
        
        # Learn route patterns
        routes = self.route_learner.learn_routes(trips, features)
        
        # Learn time patterns
        time_patterns = self.time_learner.learn_patterns(
            all_events, all_scores, self.min_confidence
        )
        
        # Learn stop patterns
        stop_patterns = self.stop_learner.learn_stops(all_events, all_features)
        
        # Calculate consistency metrics
        speed_consistency = self._calculate_speed_consistency(
            all_events, all_features, baseline
        )
        update_regularity = self._calculate_update_regularity(all_features)
        route_adherence = self._calculate_route_adherence(trips, routes)
        
        # Calculate profile confidence
        profile_confidence = self._calculate_profile_confidence(
            len(trips),
            len(all_events),
            baseline,
        )
        
        now = datetime.now(timezone.utc)
        
        return BehavioralProfile(
            user_id=user_id,
            profile_version="2.0",
            created_at=now,
            updated_at=now,
            baseline=baseline,
            known_routes=routes,
            route_adherence_score=route_adherence,
            hourly_patterns=time_patterns,
            frequent_stops=stop_patterns,
            speed_consistency=speed_consistency,
            update_regularity=update_regularity,
            total_trips_analyzed=len(trips),
            total_points_analyzed=len(all_events),
            profile_confidence=profile_confidence,
        )
    
    def update_profile(
        self,
        existing: BehavioralProfile,
        new_trips: List[List[NormalizedTrackingEvent]],
        new_features: List[List[ComputedFeatures]],
        new_scores: List[List[ConfidenceScore]],
        updated_baseline: UserBaseline,
    ) -> BehavioralProfile:
        """
        Incrementally update a profile with new data.
        
        Args:
            existing: Existing profile
            new_trips: New trip data
            new_features: New features
            new_scores: New scores
            updated_baseline: Updated Phase 1 baseline
            
        Returns:
            Updated behavioral profile
        """
        # For now, re-learn with combined data
        # In production, implement true incremental learning
        
        # This is a simplified version - full implementation would
        # merge patterns intelligently
        
        all_events = []
        all_features = []
        all_scores = []
        
        for trip_events, trip_features, trip_scores in zip(
            new_trips, new_features, new_scores
        ):
            all_events.extend(trip_events)
            all_features.extend(trip_features)
            all_scores.extend(trip_scores)
        
        # Learn new patterns
        new_routes = self.route_learner.learn_routes(new_trips, new_features)
        new_time = self.time_learner.learn_patterns(
            all_events, all_scores, self.min_confidence
        )
        new_stops = self.stop_learner.learn_stops(all_events, all_features)
        
        # Merge with existing (simplified - take union)
        merged_routes = self._merge_routes(existing.known_routes, new_routes)
        merged_stops = self._merge_stops(existing.frequent_stops, new_stops)
        
        # Recalculate consistency
        speed_consistency = self._calculate_speed_consistency(
            all_events, all_features, updated_baseline
        )
        update_regularity = self._calculate_update_regularity(all_features)
        route_adherence = self._calculate_route_adherence(new_trips, merged_routes)
        
        total_trips = existing.total_trips_analyzed + len(new_trips)
        total_points = existing.total_points_analyzed + len(all_events)
        
        profile_confidence = self._calculate_profile_confidence(
            total_trips,
            total_points,
            updated_baseline,
        )
        
        return BehavioralProfile(
            user_id=existing.user_id,
            profile_version="2.0",
            created_at=existing.created_at,
            updated_at=datetime.now(timezone.utc),
            baseline=updated_baseline,
            known_routes=merged_routes,
            route_adherence_score=route_adherence,
            hourly_patterns=new_time,  # Simplified - would blend in production
            frequent_stops=merged_stops,
            speed_consistency=speed_consistency,
            update_regularity=update_regularity,
            total_trips_analyzed=total_trips,
            total_points_analyzed=total_points,
            profile_confidence=profile_confidence,
        )
    
    def _calculate_speed_consistency(
        self,
        events: List[NormalizedTrackingEvent],
        features: List[ComputedFeatures],
        baseline: UserBaseline,
    ) -> float:
        """Calculate how consistent speeds are with baseline."""
        if not baseline.speed_baseline.sample_size:
            return 0.5
        
        speeds = [e.speed_ms for e in events if e.speed_ms is not None]
        if not speeds:
            return 0.5
        
        baseline_mean = baseline.speed_baseline.mean
        baseline_std = baseline.speed_baseline.std
        
        if baseline_std == 0:
            return 1.0 if np.std(speeds) < 1.0 else 0.5
        
        # Calculate coefficient of variation ratio
        observed_std = np.std(speeds)
        observed_mean = np.mean(speeds)
        
        if observed_mean == 0:
            return 0.5
        
        observed_cv = observed_std / observed_mean
        baseline_cv = baseline_std / baseline_mean if baseline_mean > 0 else 1.0
        
        # Consistency = how close CVs are
        cv_ratio = min(observed_cv, baseline_cv) / max(observed_cv, baseline_cv)
        return cv_ratio
    
    def _calculate_update_regularity(
        self,
        features: List[ComputedFeatures],
    ) -> float:
        """Calculate how regular update intervals are."""
        deltas = [f.time_delta_seconds for f in features if f.time_delta_seconds]
        
        if len(deltas) < 2:
            return 0.5
        
        # Calculate coefficient of variation
        mean_delta = np.mean(deltas)
        std_delta = np.std(deltas)
        
        if mean_delta == 0:
            return 0.5
        
        cv = std_delta / mean_delta
        
        # Lower CV = more regular
        regularity = 1.0 / (1.0 + cv)
        return min(1.0, regularity)
    
    def _calculate_route_adherence(
        self,
        trips: List[List[NormalizedTrackingEvent]],
        routes: List[RoutePattern],
    ) -> float:
        """Calculate how often trips follow known routes."""
        if not trips or not routes:
            return 0.0
        
        matching = 0
        for trip in trips:
            if len(trip) < 2:
                continue
            
            start = (trip[0].location.lat, trip[0].location.lon)
            end = (trip[-1].location.lat, trip[-1].location.lon)
            
            for route in routes:
                start_match = haversine_distance(
                    start[0], start[1],
                    route.start_zone[0], route.start_zone[1]
                ) <= route.start_zone[2]
                
                end_match = haversine_distance(
                    end[0], end[1],
                    route.end_zone[0], route.end_zone[1]
                ) <= route.end_zone[2]
                
                if start_match and end_match:
                    matching += 1
                    break
        
        return matching / len(trips)
    
    def _calculate_profile_confidence(
        self,
        num_trips: int,
        num_points: int,
        baseline: UserBaseline,
    ) -> float:
        """Calculate overall profile reliability."""
        trip_factor = min(1.0, num_trips / 20)  # Max at 20 trips
        point_factor = min(1.0, num_points / 1000)  # Max at 1000 points
        baseline_factor = 1.0 if baseline.is_mature else 0.5
        
        return (trip_factor * 0.3 + point_factor * 0.3 + baseline_factor * 0.4)
    
    def _merge_routes(
        self,
        existing: List[RoutePattern],
        new: List[RoutePattern],
    ) -> List[RoutePattern]:
        """Merge route patterns (simplified - takes union with deduplication)."""
        # In production, would intelligently merge similar routes
        route_ids = {r.route_id for r in existing}
        merged = list(existing)
        
        for route in new:
            if route.route_id not in route_ids:
                merged.append(route)
        
        return merged
    
    def _merge_stops(
        self,
        existing: List[StopPattern],
        new: List[StopPattern],
    ) -> List[StopPattern]:
        """Merge stop patterns."""
        stop_ids = {s.stop_id for s in existing}
        merged = list(existing)
        
        for stop in new:
            if stop.stop_id not in stop_ids:
                merged.append(stop)
        
        return merged
