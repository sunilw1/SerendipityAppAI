#!/usr/bin/env python3
"""
Model Optimization Script
==========================

Converts trained models to TensorRT engines for Triton deployment.

Features:
- ONNX export from scikit-learn/LightGBM
- TensorRT engine generation
- FP16/FP32 precision support
- Validation of converted models

Usage:
    python scripts/optimize_models.py --model delay_predictor --fp16
    python scripts/optimize_models.py --all --output-dir docker/triton/model_repository
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import json
import time

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are available."""
    deps = {}
    
    try:
        import onnx
        deps["onnx"] = True
    except ImportError:
        deps["onnx"] = False
    
    try:
        import onnxmltools
        deps["onnxmltools"] = True
    except ImportError:
        deps["onnxmltools"] = False
    
    try:
        import tensorrt as trt
        deps["tensorrt"] = True
        deps["tensorrt_version"] = trt.__version__
    except ImportError:
        deps["tensorrt"] = False
    
    try:
        import torch
        deps["torch"] = True
    except ImportError:
        deps["torch"] = False
    
    try:
        import lightgbm
        deps["lightgbm"] = True
    except ImportError:
        deps["lightgbm"] = False
    
    return deps


def convert_lightgbm_to_onnx(
    model_path: str,
    output_path: str,
    input_dim: int,
    model_name: str = "model",
) -> bool:
    """
    Convert LightGBM model to ONNX format.
    
    Args:
        model_path: Path to LightGBM model file
        output_path: Output ONNX file path
        input_dim: Input feature dimension
        model_name: Model name in ONNX
        
    Returns:
        Whether conversion succeeded
    """
    try:
        import lightgbm as lgb
        from onnxmltools import convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        
        # Load LightGBM model
        model = lgb.Booster(model_file=model_path)
        
        # Define input type
        initial_types = [
            ("input", FloatTensorType([None, input_dim]))
        ]
        
        # Convert to ONNX
        onnx_model = convert_lightgbm(
            model,
            name=model_name,
            initial_types=initial_types,
            target_opset=13,
        )
        
        # Save ONNX model
        import onnx
        onnx.save(onnx_model, output_path)
        
        print(f"[OK] Converted LightGBM model to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] LightGBM to ONNX conversion failed: {e}")
        return False


def convert_sklearn_to_onnx(
    model_path: str,
    output_path: str,
    input_dim: int,
    model_name: str = "model",
) -> bool:
    """
    Convert scikit-learn model to ONNX format.
    
    Args:
        model_path: Path to pickle/joblib model file
        output_path: Output ONNX file path
        input_dim: Input feature dimension
        model_name: Model name
        
    Returns:
        Whether conversion succeeded
    """
    try:
        import joblib
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Load sklearn model
        model = joblib.load(model_path)
        
        # Define input type
        initial_types = [
            ("input", FloatTensorType([None, input_dim]))
        ]
        
        # Convert to ONNX
        onnx_model = convert_sklearn(
            model,
            name=model_name,
            initial_types=initial_types,
            target_opset=13,
        )
        
        # Save
        import onnx
        onnx.save(onnx_model, output_path)
        
        print(f"[OK] Converted sklearn model to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] sklearn to ONNX conversion failed: {e}")
        return False


def convert_pytorch_to_onnx(
    model: "torch.nn.Module",
    output_path: str,
    input_shape: Tuple[int, ...],
    input_names: list = ["input"],
    output_names: list = ["output"],
) -> bool:
    """
    Convert PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: Output ONNX file path
        input_shape: Input tensor shape (batch, features)
        input_names: Input tensor names
        output_names: Output tensor names
        
    Returns:
        Whether conversion succeeded
    """
    try:
        import torch
        
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                input_names[0]: {0: "batch_size"},
                output_names[0]: {0: "batch_size"},
            },
            opset_version=13,
        )
        
        print(f"[OK] Converted PyTorch model to ONNX: {output_path}")
        return True
        
    except Exception as e:
        print(f"[ERROR] PyTorch to ONNX conversion failed: {e}")
        return False


def optimize_onnx_to_tensorrt(
    onnx_path: str,
    output_path: str,
    fp16: bool = True,
    max_batch_size: int = 64,
    workspace_size_mb: int = 1024,
) -> bool:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Input ONNX file path
        output_path: Output TensorRT engine path
        fp16: Enable FP16 precision
        max_batch_size: Maximum batch size
        workspace_size_mb: Workspace size in MB
        
    Returns:
        Whether conversion succeeded
    """
    try:
        import tensorrt as trt
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Create builder
        builder = trt.Builder(TRT_LOGGER)
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)
        
        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    print(f"[ERROR] {parser.get_error(i)}")
                return False
        
        # Configure builder
        config = builder.create_builder_config()
        config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            workspace_size_mb * 1024 * 1024
        )
        
        # Enable FP16 if requested
        if fp16 and builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("[INFO] FP16 precision enabled")
        
        # Set optimization profile for dynamic batching
        profile = builder.create_optimization_profile()
        
        for i in range(network.num_inputs):
            input_tensor = network.get_input(i)
            input_shape = input_tensor.shape
            
            # Handle dynamic batch dimension
            min_shape = [1] + list(input_shape[1:])
            opt_shape = [max_batch_size // 2] + list(input_shape[1:])
            max_shape = [max_batch_size] + list(input_shape[1:])
            
            profile.set_shape(
                input_tensor.name,
                min_shape,
                opt_shape,
                max_shape,
            )
        
        config.add_optimization_profile(profile)
        
        # Build engine
        print("[INFO] Building TensorRT engine (this may take a few minutes)...")
        start_time = time.time()
        
        serialized_engine = builder.build_serialized_network(network, config)
        
        if serialized_engine is None:
            print("[ERROR] Failed to build TensorRT engine")
            return False
        
        # Save engine
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(serialized_engine)
        
        build_time = time.time() - start_time
        print(f"[OK] Built TensorRT engine: {output_path}")
        print(f"[INFO] Build time: {build_time:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] TensorRT conversion failed: {e}")
        return False


def validate_tensorrt_engine(
    engine_path: str,
    input_shape: Tuple[int, ...],
    num_tests: int = 10,
) -> bool:
    """
    Validate TensorRT engine with random inputs.
    
    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input shape for testing
        num_tests: Number of test runs
        
    Returns:
        Whether validation passed
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit
        
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        
        context = engine.create_execution_context()
        
        # Allocate buffers
        input_data = np.random.randn(*input_shape).astype(np.float32)
        output_shape = (input_shape[0], 1)  # Assuming single output
        output_data = np.empty(output_shape, dtype=np.float32)
        
        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        d_output = cuda.mem_alloc(output_data.nbytes)
        
        # Run inference tests
        times = []
        for i in range(num_tests):
            # Copy input
            cuda.memcpy_htod(d_input, input_data)
            
            # Set input shape
            context.set_input_shape(engine.get_tensor_name(0), input_shape)
            
            # Run
            start = time.time()
            context.execute_v2([int(d_input), int(d_output)])
            times.append((time.time() - start) * 1000)
            
            # Copy output
            cuda.memcpy_dtoh(output_data, d_output)
        
        avg_time = np.mean(times)
        print(f"[OK] Validation passed ({num_tests} runs)")
        print(f"[INFO] Average inference time: {avg_time:.2f} ms")
        
        return True
        
    except Exception as e:
        print(f"[WARNING] Validation failed (may work on GPU system): {e}")
        return False


def create_dummy_model(
    model_type: str,
    output_dir: str,
    input_dim: int,
) -> Optional[str]:
    """
    Create a dummy model for testing the optimization pipeline.
    
    Args:
        model_type: Model type (delay_predictor, threat_classifier, drift_detector)
        output_dir: Output directory
        input_dim: Input dimension
        
    Returns:
        Path to created model or None
    """
    try:
        import torch
        import torch.nn as nn
        
        class DummyModel(nn.Module):
            def __init__(self, input_dim: int, output_dim: int = 1):
                super().__init__()
                self.fc1 = nn.Linear(input_dim, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, output_dim)
                self.relu = nn.ReLU()
                self.sigmoid = nn.Sigmoid()
            
            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x
        
        model = DummyModel(input_dim)
        
        # Save as ONNX
        os.makedirs(output_dir, exist_ok=True)
        onnx_path = os.path.join(output_dir, f"{model_type}.onnx")
        
        dummy_input = torch.randn(1, input_dim)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={
                "input": {0: "batch_size"},
                "output": {0: "batch_size"},
            },
            opset_version=13,
        )
        
        print(f"[OK] Created dummy model: {onnx_path}")
        return onnx_path
        
    except Exception as e:
        print(f"[ERROR] Failed to create dummy model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert models to TensorRT for Triton deployment"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["delay_predictor", "threat_classifier", "drift_detector"],
        help="Model to optimize",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize all models",
    )
    parser.add_argument(
        "--input-model",
        type=str,
        help="Input model path (ONNX, LightGBM, or sklearn)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="docker/triton/model_repository",
        help="Output directory for TensorRT engines",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Enable FP16 precision",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=64,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--workspace-mb",
        type=int,
        default=1024,
        help="TensorRT workspace size in MB",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create dummy models for testing",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated engines",
    )
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check dependencies and exit",
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    deps = check_dependencies()
    
    if args.check_deps:
        print("\nDependency Status:")
        print("-" * 40)
        for dep, available in deps.items():
            status = "OK" if available else "MISSING"
            if dep == "tensorrt_version" and isinstance(available, str):
                print(f"  TensorRT version: {available}")
            else:
                print(f"  {dep}: {status}")
        return
    
    # Model configurations
    model_configs = {
        "delay_predictor": {"input_dim": 8, "output_dim": 1},
        "threat_classifier": {"input_dim": 16, "output_dim": 1},
        "drift_detector": {"input_dim": 64, "output_dim": 7},  # 32+32 in, 1+6 out
    }
    
    models_to_process = []
    if args.all:
        models_to_process = list(model_configs.keys())
    elif args.model:
        models_to_process = [args.model]
    else:
        parser.print_help()
        return
    
    for model_name in models_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {model_name}")
        print("=" * 60)
        
        config = model_configs[model_name]
        output_dir = Path(args.output_dir) / model_name / "1"
        output_path = output_dir / "model.plan"
        
        # Create dummy model if requested
        if args.create_dummy:
            onnx_path = create_dummy_model(
                model_name,
                str(output_dir.parent),
                config["input_dim"],
            )
            if not onnx_path:
                continue
        elif args.input_model:
            onnx_path = args.input_model
        else:
            onnx_path = str(output_dir.parent / f"{model_name}.onnx")
        
        # Check if ONNX exists
        if not os.path.exists(onnx_path):
            print(f"[WARNING] ONNX model not found: {onnx_path}")
            print("[INFO] Use --create-dummy to create a test model")
            continue
        
        # Convert to TensorRT
        if deps.get("tensorrt"):
            success = optimize_onnx_to_tensorrt(
                onnx_path,
                str(output_path),
                fp16=args.fp16,
                max_batch_size=args.max_batch_size,
                workspace_size_mb=args.workspace_mb,
            )
            
            if success and args.validate:
                validate_tensorrt_engine(
                    str(output_path),
                    (1, config["input_dim"]),
                )
        else:
            print("[WARNING] TensorRT not available")
            print("[INFO] ONNX model created, but TensorRT conversion skipped")
    
    print("\n" + "=" * 60)
    print("Optimization complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
