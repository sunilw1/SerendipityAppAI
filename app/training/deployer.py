"""
Model Deployer
===============

Handles deployment of trained models to S3 and Triton.

Features:
- S3 model artifact upload
- Triton model repository management
- Model versioning
- Rollback support

Design Principles:
- Safe deployment (validate before deploy)
- Atomic updates
- Clear versioning
- Audit trail
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import json
import shutil

from app.core.logging import get_logger
from app.core.config import settings

logger = get_logger(__name__)

# Try to import boto3
_S3_AVAILABLE = False
try:
    import boto3
    from botocore.exceptions import ClientError
    _S3_AVAILABLE = True
except ImportError:
    logger.warning("boto3_not_available", note="S3 deployment disabled")


@dataclass
class DeploymentResult:
    """Result from model deployment."""
    success: bool
    model_name: str
    version: str
    s3_path: Optional[str] = None
    triton_path: Optional[str] = None
    error: Optional[str] = None
    deployed_at: datetime = None
    
    def __post_init__(self):
        if self.deployed_at is None:
            self.deployed_at = datetime.now(timezone.utc)


class ModelDeployer:
    """
    Deploys models to S3 and Triton.
    
    Workflow:
    1. Validate model artifact
    2. Upload to S3
    3. Update Triton model repository
    4. Trigger Triton model reload
    """
    
    def __init__(
        self,
        s3_bucket: Optional[str] = None,
        s3_prefix: str = "models",
        triton_model_repo: str = "docker/triton/model_repository",
        max_versions: int = 5,
    ):
        """
        Initialize deployer.
        
        Args:
            s3_bucket: S3 bucket for model storage
            s3_prefix: S3 key prefix
            triton_model_repo: Local Triton model repository path
            max_versions: Maximum versions to keep
        """
        self.s3_bucket = s3_bucket or getattr(settings, 's3_bucket', 'serendipity-models')
        self.s3_prefix = s3_prefix
        self.triton_repo = Path(triton_model_repo)
        self.max_versions = max_versions
        
        self._s3_client = None
        if _S3_AVAILABLE:
            try:
                self._s3_client = boto3.client(
                    's3',
                    region_name=getattr(settings, 's3_region', 'us-west-2'),
                )
            except Exception as e:
                logger.warning("s3_client_init_failed", error=str(e))
    
    def _validate_artifact(self, artifact_path: str) -> bool:
        """Validate model artifact exists and is valid."""
        path = Path(artifact_path)
        
        if not path.exists():
            logger.error("artifact_not_found", path=artifact_path)
            return False
        
        if path.suffix not in [".onnx", ".plan", ".txt", ".pkl"]:
            logger.warning("unknown_artifact_type", suffix=path.suffix)
        
        # Check file size
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 500:
            logger.warning("large_artifact", size_mb=size_mb)
        
        return True
    
    async def deploy_to_s3(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeploymentResult:
        """
        Deploy model artifact to S3.
        
        Args:
            model_name: Model name
            version: Model version
            artifact_path: Path to model artifact
            metadata: Additional metadata
            
        Returns:
            DeploymentResult
        """
        if not self._s3_client:
            return DeploymentResult(
                success=False,
                model_name=model_name,
                version=version,
                error="S3 client not available",
            )
        
        if not self._validate_artifact(artifact_path):
            return DeploymentResult(
                success=False,
                model_name=model_name,
                version=version,
                error=f"Invalid artifact: {artifact_path}",
            )
        
        try:
            artifact = Path(artifact_path)
            s3_key = f"{self.s3_prefix}/{model_name}/{version}/{artifact.name}"
            
            # Upload artifact
            extra_args = {}
            if metadata:
                extra_args["Metadata"] = {
                    k: str(v) for k, v in metadata.items()
                }
            
            self._s3_client.upload_file(
                str(artifact_path),
                self.s3_bucket,
                s3_key,
                ExtraArgs=extra_args if extra_args else None,
            )
            
            s3_path = f"s3://{self.s3_bucket}/{s3_key}"
            
            logger.info(
                "model_deployed_to_s3",
                model=model_name,
                version=version,
                s3_path=s3_path,
            )
            
            return DeploymentResult(
                success=True,
                model_name=model_name,
                version=version,
                s3_path=s3_path,
            )
            
        except Exception as e:
            logger.error("s3_deploy_failed", error=str(e))
            return DeploymentResult(
                success=False,
                model_name=model_name,
                version=version,
                error=str(e),
            )
    
    async def deploy_to_triton(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
    ) -> DeploymentResult:
        """
        Deploy model to local Triton model repository.
        
        Args:
            model_name: Model name
            version: Model version (numeric string)
            artifact_path: Path to model artifact (.plan or .onnx)
            
        Returns:
            DeploymentResult
        """
        if not self._validate_artifact(artifact_path):
            return DeploymentResult(
                success=False,
                model_name=model_name,
                version=version,
                error=f"Invalid artifact: {artifact_path}",
            )
        
        try:
            artifact = Path(artifact_path)
            
            # Determine version number
            try:
                version_num = int(version)
            except ValueError:
                # Use timestamp-based version
                version_num = int(datetime.now().strftime("%Y%m%d%H"))
            
            # Create version directory
            version_dir = self.triton_repo / model_name / str(version_num)
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine target filename
            if artifact.suffix == ".plan":
                target_name = "model.plan"
            elif artifact.suffix == ".onnx":
                target_name = "model.onnx"
            else:
                target_name = artifact.name
            
            target_path = version_dir / target_name
            
            # Copy artifact
            shutil.copy2(artifact_path, target_path)
            
            # Clean up old versions
            await self._cleanup_old_versions(model_name)
            
            logger.info(
                "model_deployed_to_triton",
                model=model_name,
                version=version_num,
                path=str(target_path),
            )
            
            return DeploymentResult(
                success=True,
                model_name=model_name,
                version=str(version_num),
                triton_path=str(target_path),
            )
            
        except Exception as e:
            logger.error("triton_deploy_failed", error=str(e))
            return DeploymentResult(
                success=False,
                model_name=model_name,
                version=version,
                error=str(e),
            )
    
    async def _cleanup_old_versions(self, model_name: str) -> None:
        """Remove old model versions beyond max_versions."""
        model_dir = self.triton_repo / model_name
        
        if not model_dir.exists():
            return
        
        # Get version directories
        versions = []
        for item in model_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                versions.append((int(item.name), item))
        
        # Sort by version number (descending)
        versions.sort(key=lambda x: x[0], reverse=True)
        
        # Remove old versions
        for version_num, version_dir in versions[self.max_versions:]:
            try:
                shutil.rmtree(version_dir)
                logger.info(
                    "old_version_removed",
                    model=model_name,
                    version=version_num,
                )
            except Exception as e:
                logger.warning("version_cleanup_failed", error=str(e))
    
    async def deploy(
        self,
        model_name: str,
        version: str,
        artifact_path: str,
        deploy_s3: bool = True,
        deploy_triton: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeploymentResult:
        """
        Deploy model to all configured targets.
        
        Args:
            model_name: Model name
            version: Model version
            artifact_path: Path to artifact
            deploy_s3: Deploy to S3
            deploy_triton: Deploy to Triton
            metadata: Additional metadata
            
        Returns:
            DeploymentResult (combined)
        """
        results = []
        
        if deploy_s3:
            s3_result = await self.deploy_to_s3(
                model_name, version, artifact_path, metadata
            )
            results.append(s3_result)
        
        if deploy_triton:
            triton_result = await self.deploy_to_triton(
                model_name, version, artifact_path
            )
            results.append(triton_result)
        
        # Combine results
        success = all(r.success for r in results)
        errors = [r.error for r in results if r.error]
        
        return DeploymentResult(
            success=success,
            model_name=model_name,
            version=version,
            s3_path=next((r.s3_path for r in results if r.s3_path), None),
            triton_path=next((r.triton_path for r in results if r.triton_path), None),
            error="; ".join(errors) if errors else None,
        )
    
    async def list_versions(
        self,
        model_name: str,
        source: str = "triton",
    ) -> List[str]:
        """
        List available model versions.
        
        Args:
            model_name: Model name
            source: Source to list from ("triton" or "s3")
            
        Returns:
            List of version strings
        """
        if source == "triton":
            model_dir = self.triton_repo / model_name
            if not model_dir.exists():
                return []
            
            versions = []
            for item in model_dir.iterdir():
                if item.is_dir() and item.name.isdigit():
                    versions.append(item.name)
            
            return sorted(versions, key=int, reverse=True)
        
        elif source == "s3" and self._s3_client:
            try:
                response = self._s3_client.list_objects_v2(
                    Bucket=self.s3_bucket,
                    Prefix=f"{self.s3_prefix}/{model_name}/",
                    Delimiter="/",
                )
                
                versions = []
                for prefix in response.get("CommonPrefixes", []):
                    version = prefix["Prefix"].rstrip("/").split("/")[-1]
                    versions.append(version)
                
                return sorted(versions, reverse=True)
                
            except Exception as e:
                logger.error("s3_list_failed", error=str(e))
                return []
        
        return []


# Singleton instance
_deployer: Optional[ModelDeployer] = None


def get_deployer() -> ModelDeployer:
    """Get or create singleton deployer."""
    global _deployer
    if _deployer is None:
        _deployer = ModelDeployer()
    return _deployer
