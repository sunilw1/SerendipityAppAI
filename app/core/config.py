"""
Application Configuration
==========================

Centralized configuration management using Pydantic Settings.
Loads from environment variables with sensible defaults for development.

Design Decisions:
- All config is typed and validated at startup
- Secrets are never logged or exposed
- Environment-specific overrides via .env files
- Immutable after initialization for thread safety
"""

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    
    All settings have sensible defaults for local development.
    Production deployments should override via environment variables.
    """
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
    
    # -------------------------------------------------------------------------
    # Application Settings
    # -------------------------------------------------------------------------
    app_name: str = "serendipity-ai-backend"
    app_env: str = Field(default="development", description="Environment: development, staging, production")
    debug: bool = Field(default=True, description="Enable debug mode")
    api_version: str = "v1"
    
    # -------------------------------------------------------------------------
    # Server Settings
    # -------------------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = Field(default=4, ge=1, le=32)
    
    # -------------------------------------------------------------------------
    # Database Settings
    # -------------------------------------------------------------------------
    database_url: str = Field(
        default="mysql+aiomysql://root:password@localhost:3306/serendipity",
        description="MySQL connection string"
    )
    database_pool_size: int = Field(default=10, ge=1, le=100)
    database_max_overflow: int = Field(default=20, ge=0, le=100)
    
    # -------------------------------------------------------------------------
    # Data Ingestion Settings
    # -------------------------------------------------------------------------
    dataset_path: Path = Field(default=Path("../dataset"))
    raw_data_file: str = "trips-500-for-different_users.csv"
    ingestion_batch_size: int = Field(default=1000, ge=100, le=10000)
    max_records_per_request: int = Field(default=10000, ge=100, le=100000)
    
    # -------------------------------------------------------------------------
    # Confidence Scoring Thresholds
    # -------------------------------------------------------------------------
    # Physical constraints
    max_realistic_speed_ms: float = Field(
        default=100.0,  # ~360 km/h (high-speed rail)
        description="Maximum realistic speed in m/s"
    )
    max_realistic_acceleration_ms2: float = Field(
        default=15.0,  # ~1.5g
        description="Maximum realistic acceleration in m/s²"
    )
    min_gps_accuracy_meters: float = Field(
        default=3.0,
        description="Best expected GPS accuracy"
    )
    max_gps_accuracy_meters: float = Field(
        default=100.0,
        description="Maximum acceptable GPS accuracy"
    )
    
    # Time gap thresholds
    min_update_interval_seconds: float = Field(
        default=0.5,
        description="Minimum realistic update interval"
    )
    max_update_gap_seconds: float = Field(
        default=300.0,  # 5 minutes
        description="Maximum acceptable gap between updates"
    )
    
    # -------------------------------------------------------------------------
    # Baseline Learning Settings
    # -------------------------------------------------------------------------
    min_trips_for_baseline: int = Field(default=5, ge=1)
    min_points_for_baseline: int = Field(default=100, ge=10)
    baseline_refresh_interval_hours: int = Field(default=24, ge=1)
    
    # -------------------------------------------------------------------------
    # Logging Settings
    # -------------------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="json or text")
    log_output: str = Field(default="stdout")
    
    # -------------------------------------------------------------------------
    # NVIDIA / GPU Settings (Phase 2+)
    # -------------------------------------------------------------------------
    cuda_enabled: bool = Field(default=False, description="Enable CUDA acceleration")
    tensorrt_enabled: bool = Field(default=False, description="Enable TensorRT optimization")
    triton_url: str = Field(default="localhost:8001", description="Triton Inference Server URL")
    triton_model_repository: str = "/models"
    
    # -------------------------------------------------------------------------
    # Security Settings
    # -------------------------------------------------------------------------
    api_secret_key: str = Field(default="dev-secret-key-change-in-production")
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:8080")
    
    # -------------------------------------------------------------------------
    # Storage Settings
    # -------------------------------------------------------------------------
    s3_bucket: str = "serendipity-models"
    s3_region: str = "us-west-2"
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env.lower() == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env.lower() == "development"
    
    @property
    def dataset_file_path(self) -> Path:
        """Full path to the dataset file."""
        return self.dataset_path / self.raw_data_file
    
    @property
    def allowed_origins_list(self) -> List[str]:
        """Parse allowed origins into a list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {v}. Must be one of {valid_levels}")
        return v.upper()


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached application settings.
    
    Uses lru_cache to ensure settings are only loaded once.
    This provides a singleton-like pattern for configuration.
    
    Returns:
        Settings: Application configuration instance
    """
    return Settings()


# Convenience alias for direct import
settings = get_settings()
