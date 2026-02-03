"""
External API Quota Monitor
===========================

Monitors usage of external APIs (OpenWeatherMap, TomTom).

Features:
- Track API call counts
- Monitor rate limits
- Alert on approaching quotas
- Daily/monthly usage reports

Design Principles:
- Non-blocking tracking
- Per-API granularity
- Clear quota visibility
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class QuotaStatus(str, Enum):
    """API quota status."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class APIQuotaConfig:
    """Configuration for an API's quota limits."""
    name: str
    daily_limit: int
    monthly_limit: int
    requests_per_minute: int = 60
    cost_per_call_usd: float = 0.0


# Default quota configurations
API_QUOTAS = {
    "openweathermap": APIQuotaConfig(
        name="OpenWeatherMap",
        daily_limit=1000,        # Free tier: 1000/day
        monthly_limit=30000,     # ~1000/day * 30 days
        requests_per_minute=60,
        cost_per_call_usd=0.0,   # Free tier
    ),
    "tomtom": APIQuotaConfig(
        name="TomTom Traffic",
        daily_limit=2500,        # Free tier: 2500/day
        monthly_limit=75000,     # ~2500/day * 30 days
        requests_per_minute=100,
        cost_per_call_usd=0.0,   # Free tier
    ),
}


@dataclass
class APIUsageStats:
    """Usage statistics for an API."""
    api_name: str
    calls_today: int = 0
    calls_this_month: int = 0
    calls_last_minute: int = 0
    
    daily_limit: int = 0
    monthly_limit: int = 0
    rate_limit_per_minute: int = 60
    
    last_call_time: Optional[datetime] = None
    errors_today: int = 0
    avg_latency_ms: float = 0.0
    
    @property
    def daily_usage_percent(self) -> float:
        if self.daily_limit == 0:
            return 0.0
        return (self.calls_today / self.daily_limit) * 100
    
    @property
    def monthly_usage_percent(self) -> float:
        if self.monthly_limit == 0:
            return 0.0
        return (self.calls_this_month / self.monthly_limit) * 100
    
    @property
    def status(self) -> QuotaStatus:
        daily_pct = self.daily_usage_percent
        monthly_pct = self.monthly_usage_percent
        
        if daily_pct >= 100 or monthly_pct >= 100:
            return QuotaStatus.EXCEEDED
        elif daily_pct >= 90 or monthly_pct >= 90:
            return QuotaStatus.CRITICAL
        elif daily_pct >= 75 or monthly_pct >= 75:
            return QuotaStatus.WARNING
        return QuotaStatus.OK


class APIQuotaMonitor:
    """
    Monitors external API usage and quotas.
    
    Tracks calls to OpenWeatherMap, TomTom, and other APIs.
    """
    
    def __init__(self):
        """Initialize quota monitor."""
        self._call_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: {"today": 0, "month": 0}
        )
        self._minute_calls: Dict[str, List[datetime]] = defaultdict(list)
        self._errors: Dict[str, int] = defaultdict(int)
        self._latencies: Dict[str, List[float]] = defaultdict(list)
        self._last_call: Dict[str, datetime] = {}
        
        self._day_start = self._get_day_start()
        self._month_start = self._get_month_start()
    
    def _get_day_start(self) -> datetime:
        """Get start of current day (UTC)."""
        now = datetime.now(timezone.utc)
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    
    def _get_month_start(self) -> datetime:
        """Get start of current month."""
        now = datetime.now(timezone.utc)
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _check_day_rollover(self) -> None:
        """Reset daily counts if day changed."""
        current_day = self._get_day_start()
        if current_day > self._day_start:
            for api in self._call_counts:
                self._call_counts[api]["today"] = 0
                self._errors[api] = 0
            self._day_start = current_day
    
    def _check_month_rollover(self) -> None:
        """Reset monthly counts if month changed."""
        current_month = self._get_month_start()
        if current_month > self._month_start:
            for api in self._call_counts:
                self._call_counts[api]["month"] = 0
            self._month_start = current_month
    
    def _prune_minute_calls(self, api: str) -> None:
        """Remove calls older than 1 minute."""
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(minutes=1)
        self._minute_calls[api] = [
            t for t in self._minute_calls[api] if t > cutoff
        ]
    
    def record_call(
        self,
        api: str,
        success: bool = True,
        latency_ms: Optional[float] = None,
    ) -> None:
        """
        Record an API call.
        
        Args:
            api: API identifier (e.g., "openweathermap", "tomtom")
            success: Whether call succeeded
            latency_ms: Call latency in milliseconds
        """
        self._check_day_rollover()
        self._check_month_rollover()
        
        now = datetime.now(timezone.utc)
        
        # Update counts
        self._call_counts[api]["today"] += 1
        self._call_counts[api]["month"] += 1
        
        # Track for rate limiting
        self._minute_calls[api].append(now)
        self._prune_minute_calls(api)
        
        # Track errors
        if not success:
            self._errors[api] += 1
        
        # Track latency
        if latency_ms is not None:
            self._latencies[api].append(latency_ms)
            # Keep last 100 latencies
            if len(self._latencies[api]) > 100:
                self._latencies[api] = self._latencies[api][-100:]
        
        self._last_call[api] = now
        
        # Log warnings if approaching limits
        config = API_QUOTAS.get(api)
        if config:
            daily_pct = (self._call_counts[api]["today"] / config.daily_limit) * 100
            if daily_pct >= 90:
                logger.warning(
                    "api_quota_critical",
                    api=api,
                    daily_usage_percent=daily_pct,
                )
            elif daily_pct >= 75:
                logger.warning(
                    "api_quota_warning",
                    api=api,
                    daily_usage_percent=daily_pct,
                )
    
    def can_call(self, api: str) -> tuple[bool, str]:
        """
        Check if an API call is allowed under current quotas.
        
        Args:
            api: API identifier
            
        Returns:
            Tuple of (allowed, reason)
        """
        self._check_day_rollover()
        self._prune_minute_calls(api)
        
        config = API_QUOTAS.get(api)
        if not config:
            return True, "No quota configured"
        
        # Check daily limit
        if self._call_counts[api]["today"] >= config.daily_limit:
            return False, f"Daily limit reached ({config.daily_limit})"
        
        # Check monthly limit
        if self._call_counts[api]["month"] >= config.monthly_limit:
            return False, f"Monthly limit reached ({config.monthly_limit})"
        
        # Check rate limit
        if len(self._minute_calls[api]) >= config.requests_per_minute:
            return False, f"Rate limit reached ({config.requests_per_minute}/min)"
        
        return True, "OK"
    
    def get_stats(self, api: str) -> APIUsageStats:
        """
        Get usage statistics for an API.
        
        Args:
            api: API identifier
            
        Returns:
            APIUsageStats
        """
        self._check_day_rollover()
        self._check_month_rollover()
        self._prune_minute_calls(api)
        
        config = API_QUOTAS.get(api, APIQuotaConfig(
            name=api,
            daily_limit=10000,
            monthly_limit=300000,
        ))
        
        latencies = self._latencies.get(api, [])
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        
        return APIUsageStats(
            api_name=config.name,
            calls_today=self._call_counts[api]["today"],
            calls_this_month=self._call_counts[api]["month"],
            calls_last_minute=len(self._minute_calls[api]),
            daily_limit=config.daily_limit,
            monthly_limit=config.monthly_limit,
            rate_limit_per_minute=config.requests_per_minute,
            last_call_time=self._last_call.get(api),
            errors_today=self._errors.get(api, 0),
            avg_latency_ms=avg_latency,
        )
    
    def get_all_stats(self) -> Dict[str, APIUsageStats]:
        """Get statistics for all tracked APIs."""
        stats = {}
        for api in API_QUOTAS:
            stats[api] = self.get_stats(api)
        return stats
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary for API responses."""
        all_stats = self.get_all_stats()
        
        return {
            "apis": {
                api: {
                    "name": stats.api_name,
                    "status": stats.status.value,
                    "daily": {
                        "calls": stats.calls_today,
                        "limit": stats.daily_limit,
                        "percent": round(stats.daily_usage_percent, 1),
                    },
                    "monthly": {
                        "calls": stats.calls_this_month,
                        "limit": stats.monthly_limit,
                        "percent": round(stats.monthly_usage_percent, 1),
                    },
                    "rate_limit": {
                        "calls_last_minute": stats.calls_last_minute,
                        "limit_per_minute": stats.rate_limit_per_minute,
                    },
                    "errors_today": stats.errors_today,
                    "avg_latency_ms": round(stats.avg_latency_ms, 2),
                }
                for api, stats in all_stats.items()
            },
            "checked_at": datetime.now(timezone.utc).isoformat(),
        }


# Singleton instance
_quota_monitor: Optional[APIQuotaMonitor] = None


def get_quota_monitor() -> APIQuotaMonitor:
    """Get or create singleton quota monitor."""
    global _quota_monitor
    if _quota_monitor is None:
        _quota_monitor = APIQuotaMonitor()
    return _quota_monitor
