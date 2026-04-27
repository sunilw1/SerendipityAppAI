#!/usr/bin/env python3
"""
Create Placeholder Models for Triton
=====================================

Creates simple ONNX models that Triton can load.
These are placeholders until real trained models are available.
"""

import os
import numpy as np

def create_onnx_model(model_name: str, input_dim: int, output_dim: int, output_dir: str):
    """Create a simple ONNX model for Triton."""
    try:
        import onnx
        from onnx import helper, TensorProto
    except ImportError:
        print("ONNX not installed. Installing...")
        os.system("pip install onnx -q")
        import onnx
        from onnx import helper, TensorProto
    
    # Create a simple linear model: output = input @ weights + bias
    # This is just a placeholder - real models will replace these
    
    weights = np.random.randn(input_dim, output_dim).astype(np.float32)
    bias = np.zeros(output_dim).astype(np.float32)
    
    # Create weight initializers
    weight_init = helper.make_tensor(
        'weights',
        TensorProto.FLOAT,
        [input_dim, output_dim],
        weights.flatten().tolist()
    )
    bias_init = helper.make_tensor(
        'bias',
        TensorProto.FLOAT,
        [output_dim],
        bias.tolist()
    )
    
    # Create nodes
    matmul_node = helper.make_node(
        'MatMul',
        inputs=['input', 'weights'],
        outputs=['matmul_out']
    )
    add_node = helper.make_node(
        'Add',
        inputs=['matmul_out', 'bias'],
        outputs=['output']
    )
    
    # Create graph
    graph = helper.make_graph(
        [matmul_node, add_node],
        model_name,
        [helper.make_tensor_value_info('input', TensorProto.FLOAT, [None, input_dim])],
        [helper.make_tensor_value_info('output', TensorProto.FLOAT, [None, output_dim])],
        [weight_init, bias_init]
    )
    
    # Create model
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    
    # Save model
    model_path = os.path.join(output_dir, 'model.onnx')
    onnx.save(model, model_path)
    print(f"Created {model_path}")
    return model_path


def create_triton_config(model_name: str, input_dim: int, output_dim: int, output_dir: str):
    """Create Triton config.pbtxt for ONNX model."""
    config = f'''name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 32

input [
  {{
    name: "input"
    data_type: TYPE_FP32
    dims: [{input_dim}]
  }}
]

output [
  {{
    name: "output"
    data_type: TYPE_FP32
    dims: [{output_dim}]
  }}
]

instance_group [
  {{
    count: 1
    kind: KIND_GPU
    gpus: [0]
  }}
]

dynamic_batching {{
  preferred_batch_size: [4, 8, 16]
  max_queue_delay_microseconds: 100
}}
'''
    config_path = os.path.join(output_dir, 'config.pbtxt')
    with open(config_path, 'w') as f:
        f.write(config)
    print(f"Created {config_path}")


def main():
    MODEL_REPO = os.environ.get('MODEL_REPO', '/home/ubuntu/models')
    
    models = [
        ('delay_predictor', 16, 1),      # 16 features -> 1 delay prediction
        ('threat_classifier', 12, 2),    # 12 features -> 2 classes (threat/no-threat)
        ('drift_detector', 20, 5),       # 20 features -> 5 drift signals
    ]
    
    print("Creating placeholder models for Triton...")
    print(f"Model repository: {MODEL_REPO}\n")
    
    for model_name, input_dim, output_dim in models:
        model_dir = os.path.join(MODEL_REPO, model_name)
        version_dir = os.path.join(model_dir, '1')
        
        os.makedirs(version_dir, exist_ok=True)
        
        print(f"\n=== {model_name} ===")
        create_onnx_model(model_name, input_dim, output_dim, version_dir)
        create_triton_config(model_name, input_dim, output_dim, model_dir)
    
    print("\n✓ All placeholder models created!")
    print(f"\nModel repository structure:")
    os.system(f"find {MODEL_REPO} -type f")


if __name__ == '__main__':
    main()
