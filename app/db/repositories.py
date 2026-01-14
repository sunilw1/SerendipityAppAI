"""
Database Repositories
======================

Repository pattern for data access.

Phase 1: Placeholder for future implementation.
Phase 2+: Full CRUD operations for tracking data.

Design Principles:
- Repository pattern for clean data access
- Async operations throughout
- Batch operations for performance
- Clear separation from business logic
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, List, Optional, TypeVar, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.models.schemas import (
    NormalizedTrackingEvent,
    ProcessedTrackingEvent,
    UserBaseline,
)

logger = get_logger(__name__)

# Generic type for repository entities
T = TypeVar("T")


class BaseRepository(ABC, Generic[T]):
    """
    Abstract base repository.
    
    Defines the interface for all repositories.
    """
    
    @abstractmethod
    async def get(self, id: str) -> Optional[T]:
        """Get entity by ID."""
        pass
    
    @abstractmethod
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        """Get all entities with pagination."""
        pass
    
    @abstractmethod
    async def create(self, entity: T) -> T:
        """Create a new entity."""
        pass
    
    @abstractmethod
    async def update(self, entity: T) -> T:
        """Update an existing entity."""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> bool:
        """Delete an entity by ID."""
        pass


class InMemoryRepository(BaseRepository[T]):
    """
    In-memory repository for Phase 1.
    
    Provides a simple dict-based storage for development.
    Will be replaced with database repositories in Phase 2.
    """
    
    def __init__(self):
        self._storage: Dict[str, T] = {}
    
    async def get(self, id: str) -> Optional[T]:
        return self._storage.get(id)
    
    async def get_all(self, limit: int = 100, offset: int = 0) -> List[T]:
        all_items = list(self._storage.values())
        return all_items[offset:offset + limit]
    
    async def create(self, entity: T) -> T:
        # Assumes entity has an 'id' attribute or is a dict with 'id'
        entity_id = getattr(entity, 'id', None) or entity.get('id') if isinstance(entity, dict) else id(entity)
        self._storage[str(entity_id)] = entity
        return entity
    
    async def update(self, entity: T) -> T:
        return await self.create(entity)  # Same as create for in-memory
    
    async def delete(self, id: str) -> bool:
        if id in self._storage:
            del self._storage[id]
            return True
        return False
    
    async def count(self) -> int:
        return len(self._storage)
    
    async def clear(self) -> None:
        self._storage.clear()


class TrackingEventRepository(InMemoryRepository[ProcessedTrackingEvent]):
    """
    Repository for tracking events.
    
    Phase 1: In-memory storage
    Phase 2+: Database with time-series optimization
    """
    
    async def get_by_user(
        self,
        user_id: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
    ) -> List[ProcessedTrackingEvent]:
        """Get events for a user within time range."""
        events = [
            e for e in self._storage.values()
            if e.event.user_id == user_id
        ]
        
        if start_time:
            events = [e for e in events if e.event.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.event.timestamp <= end_time]
        
        # Sort by timestamp
        events.sort(key=lambda e: e.event.timestamp)
        
        return events[:limit]
    
    async def get_by_trip(
        self,
        trip_id: int,
    ) -> List[ProcessedTrackingEvent]:
        """Get all events for a trip."""
        events = [
            e for e in self._storage.values()
            if e.event.trip_id == trip_id
        ]
        
        events.sort(key=lambda e: e.event.timestamp)
        return events
    
    async def bulk_create(
        self,
        events: List[ProcessedTrackingEvent],
    ) -> int:
        """Create multiple events at once."""
        count = 0
        for event in events:
            await self.create(event)
            count += 1
        return count


class BaselineRepository(InMemoryRepository[UserBaseline]):
    """
    Repository for user baselines.
    
    Phase 1: In-memory storage
    Phase 2+: Database with caching
    """
    
    async def get_by_user(self, user_id: int) -> Optional[UserBaseline]:
        """Get baseline for a specific user."""
        for baseline in self._storage.values():
            if baseline.user_id == user_id:
                return baseline
        return None
    
    async def upsert(self, baseline: UserBaseline) -> UserBaseline:
        """Create or update baseline for user."""
        # Remove existing baseline for user
        to_remove = [
            k for k, v in self._storage.items()
            if v.user_id == baseline.user_id
        ]
        for k in to_remove:
            del self._storage[k]
        
        # Add new baseline
        key = f"baseline_{baseline.user_id}"
        self._storage[key] = baseline
        return baseline


# Repository instances (singletons)
_event_repo: Optional[TrackingEventRepository] = None
_baseline_repo: Optional[BaselineRepository] = None


def get_event_repository() -> TrackingEventRepository:
    """Get or create event repository."""
    global _event_repo
    if _event_repo is None:
        _event_repo = TrackingEventRepository()
    return _event_repo


def get_baseline_repository() -> BaselineRepository:
    """Get or create baseline repository."""
    global _baseline_repo
    if _baseline_repo is None:
        _baseline_repo = BaselineRepository()
    return _baseline_repo
