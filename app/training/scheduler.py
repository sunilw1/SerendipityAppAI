"""
Training Scheduler
===================

Cron-based job scheduling for model retraining.

Features:
- Periodic retraining triggers
- Feature statistics updates
- Model validation before deployment
- Slack/email notifications (optional)

Design Principles:
- Configurable schedules
- Idempotent job execution
- Clear job status reporting
- Graceful error handling
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid

from app.core.logging import get_logger
from app.core.config import settings
from app.training.trainer import ModelTrainer, get_trainer, TrainingResult
from app.training.deployer import ModelDeployer, get_deployer, DeploymentResult

logger = get_logger(__name__)


class JobStatus(str, Enum):
    """Job execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobType(str, Enum):
    """Types of scheduled jobs."""
    FEATURE_STATS_UPDATE = "feature_stats_update"
    DELAY_MODEL_RETRAIN = "delay_model_retrain"
    THREAT_MODEL_RETRAIN = "threat_model_retrain"
    DRIFT_MODEL_RETRAIN = "drift_model_retrain"
    FULL_RETRAIN = "full_retrain"


@dataclass
class JobResult:
    """Result from a scheduled job."""
    job_id: str
    job_type: JobType
    status: JobStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: float = 0.0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScheduledJob:
    """Definition of a scheduled job."""
    job_type: JobType
    cron_expression: str
    enabled: bool = True
    last_run: Optional[datetime] = None
    next_run: Optional[datetime] = None
    
    def should_run(self, now: datetime) -> bool:
        """Check if job should run now."""
        if not self.enabled:
            return False
        
        if self.next_run is None:
            return True
        
        return now >= self.next_run


class TrainingScheduler:
    """
    Manages scheduled model retraining jobs.
    
    Default Schedule:
    - Daily: Feature statistics update
    - Weekly: Delay prediction model retrain
    - Monthly: Full model retraining
    """
    
    def __init__(
        self,
        trainer: Optional[ModelTrainer] = None,
        deployer: Optional[ModelDeployer] = None,
    ):
        """
        Initialize scheduler.
        
        Args:
            trainer: Model trainer instance
            deployer: Model deployer instance
        """
        self.trainer = trainer or get_trainer()
        self.deployer = deployer or get_deployer()
        
        self._jobs: Dict[JobType, ScheduledJob] = {}
        self._job_history: List[JobResult] = []
        self._running = False
        self._task: Optional[asyncio.Task] = None
        
        # Initialize default jobs
        self._init_default_jobs()
    
    def _init_default_jobs(self) -> None:
        """Initialize default scheduled jobs."""
        self._jobs = {
            JobType.FEATURE_STATS_UPDATE: ScheduledJob(
                job_type=JobType.FEATURE_STATS_UPDATE,
                cron_expression="0 3 * * *",  # Daily at 3 AM
                enabled=True,
            ),
            JobType.DELAY_MODEL_RETRAIN: ScheduledJob(
                job_type=JobType.DELAY_MODEL_RETRAIN,
                cron_expression="0 2 * * 0",  # Weekly Sunday at 2 AM
                enabled=True,
            ),
            JobType.THREAT_MODEL_RETRAIN: ScheduledJob(
                job_type=JobType.THREAT_MODEL_RETRAIN,
                cron_expression="0 2 * * 0",  # Weekly Sunday at 2 AM
                enabled=True,
            ),
            JobType.FULL_RETRAIN: ScheduledJob(
                job_type=JobType.FULL_RETRAIN,
                cron_expression="0 1 1 * *",  # Monthly 1st at 1 AM
                enabled=True,
            ),
        }
    
    def _parse_cron_interval(self, cron: str) -> timedelta:
        """
        Parse cron expression to approximate interval.
        
        This is a simplified parser for demonstration.
        In production, use a library like croniter.
        """
        parts = cron.split()
        if len(parts) != 5:
            return timedelta(days=1)
        
        minute, hour, day, month, dow = parts
        
        # Simple heuristics
        if day == "1" and month == "*":
            return timedelta(days=30)  # Monthly
        elif dow != "*":
            return timedelta(days=7)   # Weekly
        elif hour != "*":
            return timedelta(days=1)   # Daily
        else:
            return timedelta(hours=1)  # Hourly
    
    def _calculate_next_run(self, job: ScheduledJob) -> datetime:
        """Calculate next run time for a job."""
        interval = self._parse_cron_interval(job.cron_expression)
        
        if job.last_run:
            return job.last_run + interval
        else:
            return datetime.now(timezone.utc)
    
    async def _run_job(self, job_type: JobType) -> JobResult:
        """Execute a scheduled job."""
        job_id = f"job_{uuid.uuid4().hex[:8]}"
        started_at = datetime.now(timezone.utc)
        
        logger.info("job_started", job_id=job_id, job_type=job_type.value)
        
        result = JobResult(
            job_id=job_id,
            job_type=job_type,
            status=JobStatus.RUNNING,
            started_at=started_at,
        )
        
        try:
            if job_type == JobType.FEATURE_STATS_UPDATE:
                await self._update_feature_stats()
                result.details["action"] = "feature_stats_updated"
                
            elif job_type == JobType.DELAY_MODEL_RETRAIN:
                train_result = await self._retrain_delay_model()
                result.details["training"] = {
                    "success": train_result.success,
                    "model_path": train_result.model_path,
                }
                
            elif job_type == JobType.THREAT_MODEL_RETRAIN:
                train_result = await self._retrain_threat_model()
                result.details["training"] = {
                    "success": train_result.success,
                    "model_path": train_result.model_path,
                }
                
            elif job_type == JobType.FULL_RETRAIN:
                await self._full_retrain()
                result.details["action"] = "full_retrain_completed"
            
            result.status = JobStatus.COMPLETED
            
        except Exception as e:
            logger.error("job_failed", job_id=job_id, error=str(e))
            result.status = JobStatus.FAILED
            result.error = str(e)
        
        result.completed_at = datetime.now(timezone.utc)
        result.duration_seconds = (
            result.completed_at - result.started_at
        ).total_seconds()
        
        # Update job last run
        if job_type in self._jobs:
            self._jobs[job_type].last_run = result.completed_at
            self._jobs[job_type].next_run = self._calculate_next_run(
                self._jobs[job_type]
            )
        
        # Store in history
        self._job_history.append(result)
        if len(self._job_history) > 100:
            self._job_history = self._job_history[-100:]
        
        logger.info(
            "job_completed",
            job_id=job_id,
            status=result.status.value,
            duration=result.duration_seconds,
        )
        
        return result
    
    async def _update_feature_stats(self) -> None:
        """Update feature statistics from recent data."""
        # In production, this would:
        # 1. Query recent tracking data
        # 2. Compute feature statistics
        # 3. Store updated statistics
        logger.info("feature_stats_update_placeholder")
    
    async def _retrain_delay_model(self) -> TrainingResult:
        """Retrain delay prediction model."""
        import numpy as np
        
        # Generate synthetic training data for demonstration
        # In production, this would query historical data
        n_samples = 1000
        X = np.random.randn(n_samples, 8).astype(np.float32)
        y = np.random.exponential(10, n_samples).astype(np.float32)
        
        split = int(n_samples * 0.8)
        
        result = await self.trainer.train_delay_predictor(
            X[:split], y[:split],
            X[split:], y[split:],
        )
        
        if result.success and result.onnx_path:
            deploy_result = await self.deployer.deploy(
                model_name="delay_predictor",
                version=datetime.now().strftime("%Y%m%d%H%M"),
                artifact_path=result.onnx_path,
            )
            logger.info("delay_model_deployed", success=deploy_result.success)
        
        return result
    
    async def _retrain_threat_model(self) -> TrainingResult:
        """Retrain threat classification model."""
        import numpy as np
        
        # Synthetic data
        n_samples = 1000
        X = np.random.randn(n_samples, 16).astype(np.float32)
        y = (np.random.rand(n_samples) > 0.9).astype(np.float32)  # 10% positive
        
        split = int(n_samples * 0.8)
        
        result = await self.trainer.train_threat_classifier(
            X[:split], y[:split],
            X[split:], y[split:],
        )
        
        if result.success and result.onnx_path:
            await self.deployer.deploy(
                model_name="threat_classifier",
                version=datetime.now().strftime("%Y%m%d%H%M"),
                artifact_path=result.onnx_path,
            )
        
        return result
    
    async def _full_retrain(self) -> None:
        """Full retraining of all models."""
        await self._retrain_delay_model()
        await self._retrain_threat_model()
        logger.info("full_retrain_completed")
    
    async def trigger_job(self, job_type: JobType) -> JobResult:
        """
        Manually trigger a job.
        
        Args:
            job_type: Type of job to run
            
        Returns:
            JobResult
        """
        return await self._run_job(job_type)
    
    async def start(self) -> None:
        """Start the scheduler background task."""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("scheduler_started")
    
    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("scheduler_stopped")
    
    async def _run_loop(self) -> None:
        """Main scheduler loop."""
        while self._running:
            now = datetime.now(timezone.utc)
            
            for job_type, job in self._jobs.items():
                if job.should_run(now):
                    try:
                        await self._run_job(job_type)
                    except Exception as e:
                        logger.error(
                            "scheduler_job_error",
                            job_type=job_type.value,
                            error=str(e),
                        )
            
            # Check every minute
            await asyncio.sleep(60)
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status."""
        return {
            "running": self._running,
            "jobs": {
                job_type.value: {
                    "enabled": job.enabled,
                    "cron": job.cron_expression,
                    "last_run": job.last_run.isoformat() if job.last_run else None,
                    "next_run": job.next_run.isoformat() if job.next_run else None,
                }
                for job_type, job in self._jobs.items()
            },
            "recent_history": [
                {
                    "job_id": r.job_id,
                    "job_type": r.job_type.value,
                    "status": r.status.value,
                    "started_at": r.started_at.isoformat(),
                    "duration": r.duration_seconds,
                }
                for r in self._job_history[-10:]
            ],
        }


# Singleton instance
_scheduler: Optional[TrainingScheduler] = None


def get_scheduler() -> TrainingScheduler:
    """Get or create singleton scheduler."""
    global _scheduler
    if _scheduler is None:
        _scheduler = TrainingScheduler()
    return _scheduler
