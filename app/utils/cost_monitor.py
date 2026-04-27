"""
GPU Cost Monitor
=================

Monitors and enforces GPU cost guardrails.

Features:
- Track GPU usage hours
- Calculate estimated spend
- Alert when approaching limits
- Enforce spend caps

Design Principles:
- Non-blocking cost checks
- Clear spend visibility
- Configurable thresholds
- Safe defaults
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass, field
from enum import Enum

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)


class CostAlertLevel(str, Enum):
    """Cost alert severity levels."""
    OK = "ok"
    WARNING = "warning"
    CRITICAL = "critical"
    EXCEEDED = "exceeded"


@dataclass
class GPUUsageRecord:
    """Record of GPU usage."""
    start_time: datetime
    end_time: Optional[datetime] = None
    instance_type: str = "g5.xlarge"
    hourly_cost: float = 1.006
    
    @property
    def duration_hours(self) -> float:
        """Get usage duration in hours."""
        end = self.end_time or datetime.now(timezone.utc)
        delta = end - self.start_time
        return delta.total_seconds() / 3600
    
    @property
    def cost(self) -> float:
        """Calculate cost for this usage period."""
        return self.duration_hours * self.hourly_cost


@dataclass
class CostStatus:
    """Current cost status and projections."""
    current_month: str
    hours_used: float
    hours_remaining: float
    hours_limit: float
    
    spend_current_usd: float
    spend_limit_usd: float
    spend_min_target_usd: float
    spend_projected_usd: float
    
    alert_level: CostAlertLevel
    alert_message: str
    
    on_track_for_minimum: bool
    days_remaining_in_month: int
    
    recommendations: List[str] = field(default_factory=list)


class GPUCostMonitor:
    """
    Monitors GPU costs and enforces guardrails.
    
    Tracks:
    - Total GPU hours used this month
    - Current and projected spend
    - Alerts when approaching limits
    """
    
    def __init__(
        self,
        max_monthly_hours: Optional[int] = None,
        max_monthly_spend: Optional[float] = None,
        min_monthly_spend: Optional[float] = None,
        hourly_cost: Optional[float] = None,
        alert_threshold_percent: Optional[float] = None,
    ):
        """
        Initialize cost monitor.
        
        Args:
            max_monthly_hours: Maximum GPU hours per month
            max_monthly_spend: Maximum spend in USD
            min_monthly_spend: Minimum spend target (NVIDIA requirement)
            hourly_cost: Hourly instance cost
            alert_threshold_percent: Alert threshold percentage
        """
        self.max_hours = max_monthly_hours or settings.gpu_max_monthly_hours
        self.max_spend = max_monthly_spend or settings.gpu_max_monthly_spend_usd
        self.min_spend = min_monthly_spend or settings.gpu_min_monthly_spend_usd
        self.hourly_cost = hourly_cost or settings.gpu_instance_hourly_cost_usd
        self.alert_threshold = alert_threshold_percent or settings.gpu_alert_threshold_percent
        
        # Usage tracking
        self._usage_records: List[GPUUsageRecord] = []
        self._current_session: Optional[GPUUsageRecord] = None
        self._month_start: datetime = self._get_month_start()
    
    def _get_month_start(self) -> datetime:
        """Get start of current month."""
        now = datetime.now(timezone.utc)
        return now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    
    def _get_days_in_month(self) -> int:
        """Get number of days in current month."""
        now = datetime.now(timezone.utc)
        if now.month == 12:
            next_month = now.replace(year=now.year + 1, month=1, day=1)
        else:
            next_month = now.replace(month=now.month + 1, day=1)
        return (next_month - now.replace(day=1)).days
    
    def _get_days_remaining(self) -> int:
        """Get days remaining in current month."""
        now = datetime.now(timezone.utc)
        days_in_month = self._get_days_in_month()
        return days_in_month - now.day + 1
    
    def start_session(self, instance_type: str = "g5.xlarge") -> None:
        """
        Start tracking a GPU session.
        
        Args:
            instance_type: Type of GPU instance
        """
        if self._current_session is not None:
            logger.warning("session_already_active", action="ending_previous")
            self.end_session()
        
        self._current_session = GPUUsageRecord(
            start_time=datetime.now(timezone.utc),
            instance_type=instance_type,
            hourly_cost=self.hourly_cost,
        )
        
        logger.info(
            "gpu_session_started",
            instance_type=instance_type,
            hourly_cost=self.hourly_cost,
        )
    
    def end_session(self) -> Optional[GPUUsageRecord]:
        """
        End current GPU session.
        
        Returns:
            The completed usage record
        """
        if self._current_session is None:
            return None
        
        self._current_session.end_time = datetime.now(timezone.utc)
        record = self._current_session
        self._usage_records.append(record)
        self._current_session = None
        
        logger.info(
            "gpu_session_ended",
            duration_hours=record.duration_hours,
            cost_usd=record.cost,
        )
        
        return record
    
    def add_usage(self, hours: float, cost: Optional[float] = None) -> None:
        """
        Manually add GPU usage (for historical data or external tracking).
        
        Args:
            hours: Hours of GPU usage
            cost: Cost in USD (calculated from hours if not provided)
        """
        now = datetime.now(timezone.utc)
        record = GPUUsageRecord(
            start_time=now - timedelta(hours=hours),
            end_time=now,
            hourly_cost=self.hourly_cost if cost is None else cost / hours,
        )
        self._usage_records.append(record)
    
    def get_monthly_hours(self) -> float:
        """Get total GPU hours used this month."""
        total = 0.0
        
        for record in self._usage_records:
            if record.start_time >= self._month_start:
                total += record.duration_hours
        
        # Add current session if active
        if self._current_session:
            total += self._current_session.duration_hours
        
        return total
    
    def get_monthly_spend(self) -> float:
        """Get total GPU spend this month in USD."""
        total = 0.0
        
        for record in self._usage_records:
            if record.start_time >= self._month_start:
                total += record.cost
        
        # Add current session if active
        if self._current_session:
            total += self._current_session.cost
        
        return total
    
    def get_projected_spend(self) -> float:
        """
        Project total spend for the month based on current usage rate.
        
        Returns:
            Projected monthly spend in USD
        """
        now = datetime.now(timezone.utc)
        days_elapsed = (now - self._month_start).days + 1
        days_in_month = self._get_days_in_month()
        
        current_spend = self.get_monthly_spend()
        
        if days_elapsed > 0:
            daily_rate = current_spend / days_elapsed
            return daily_rate * days_in_month
        
        return 0.0
    
    def check_can_use_gpu(self) -> Tuple[bool, str]:
        """
        Check if GPU usage is allowed under current guardrails.
        
        Returns:
            Tuple of (allowed, reason)
        """
        if not settings.gpu_cost_guardrail_enabled:
            return True, "Guardrails disabled"
        
        current_hours = self.get_monthly_hours()
        current_spend = self.get_monthly_spend()
        
        if current_hours >= self.max_hours:
            return False, f"Monthly hour limit reached ({current_hours:.1f}/{self.max_hours}h)"
        
        if current_spend >= self.max_spend:
            return False, f"Monthly spend limit reached (${current_spend:.2f}/${self.max_spend:.2f})"
        
        return True, "Within limits"
    
    def get_status(self) -> CostStatus:
        """
        Get comprehensive cost status.
        
        Returns:
            CostStatus with current and projected metrics
        """
        now = datetime.now(timezone.utc)
        current_hours = self.get_monthly_hours()
        current_spend = self.get_monthly_spend()
        projected_spend = self.get_projected_spend()
        days_remaining = self._get_days_remaining()
        
        # Calculate hours remaining
        hours_remaining = max(0, self.max_hours - current_hours)
        
        # Determine alert level
        spend_percent = (current_spend / self.max_spend) * 100 if self.max_spend > 0 else 0
        
        if spend_percent >= 100:
            alert_level = CostAlertLevel.EXCEEDED
            alert_message = f"Monthly spend limit exceeded: ${current_spend:.2f}"
        elif spend_percent >= self.alert_threshold:
            alert_level = CostAlertLevel.CRITICAL
            alert_message = f"Approaching spend limit: {spend_percent:.1f}% used"
        elif spend_percent >= self.alert_threshold * 0.75:
            alert_level = CostAlertLevel.WARNING
            alert_message = f"Spend at {spend_percent:.1f}% of limit"
        else:
            alert_level = CostAlertLevel.OK
            alert_message = "Spend within normal range"
        
        # Check if on track for minimum NVIDIA requirement
        on_track = projected_spend >= self.min_spend
        
        # Generate recommendations
        recommendations = []
        
        if not on_track:
            hours_needed = (self.min_spend - current_spend) / self.hourly_cost
            recommendations.append(
                f"Need {hours_needed:.1f} more GPU hours to meet ${self.min_spend:.0f} minimum"
            )
            if days_remaining > 0:
                daily_hours_needed = hours_needed / days_remaining
                recommendations.append(
                    f"Run GPU ~{daily_hours_needed:.1f} hours/day for remaining {days_remaining} days"
                )
        
        if alert_level == CostAlertLevel.EXCEEDED:
            recommendations.append("Consider stopping non-essential GPU workloads")
        
        if projected_spend > self.max_spend * 1.1:
            recommendations.append(
                f"Projected overspend: ${projected_spend:.2f} vs ${self.max_spend:.2f} limit"
            )
        
        return CostStatus(
            current_month=now.strftime("%B %Y"),
            hours_used=current_hours,
            hours_remaining=hours_remaining,
            hours_limit=float(self.max_hours),
            spend_current_usd=current_spend,
            spend_limit_usd=self.max_spend,
            spend_min_target_usd=self.min_spend,
            spend_projected_usd=projected_spend,
            alert_level=alert_level,
            alert_message=alert_message,
            on_track_for_minimum=on_track,
            days_remaining_in_month=days_remaining,
            recommendations=recommendations,
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary as dictionary for API responses."""
        status = self.get_status()
        return {
            "month": status.current_month,
            "hours": {
                "used": round(status.hours_used, 2),
                "remaining": round(status.hours_remaining, 2),
                "limit": status.hours_limit,
            },
            "spend_usd": {
                "current": round(status.spend_current_usd, 2),
                "projected": round(status.spend_projected_usd, 2),
                "limit": status.spend_limit_usd,
                "min_target": status.spend_min_target_usd,
            },
            "alert": {
                "level": status.alert_level.value,
                "message": status.alert_message,
            },
            "nvidia_requirement": {
                "on_track": status.on_track_for_minimum,
                "min_spend": status.spend_min_target_usd,
            },
            "days_remaining": status.days_remaining_in_month,
            "recommendations": status.recommendations,
        }


# Singleton instance
_cost_monitor: Optional[GPUCostMonitor] = None


def get_cost_monitor() -> GPUCostMonitor:
    """Get or create singleton cost monitor."""
    global _cost_monitor
    if _cost_monitor is None:
        _cost_monitor = GPUCostMonitor()
    return _cost_monitor
