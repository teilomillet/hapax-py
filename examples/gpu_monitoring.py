"""
Simple example demonstrating GPU monitoring in Hapax.

This example shows how to:
1. Set up global GPU monitoring
2. Add GPU monitoring to a specific graph
3. Get and display GPU metrics

Requirements:
- openlit>=1.33.8
- torch (for demonstration purposes)
"""
import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv
from hapax import ops, Graph, enable_gpu_monitoring, get_gpu_metrics

# Load .env file if present
load_dotenv()

# Define operations for a simple ML pipeline
@ops(name="preprocess_data", tags=["data", "preprocessing"])
def preprocess_data(data: List[float]) -> List[float]:
    """Simple preprocessing example."""
    # Normalize data
    max_val = max(data) if data else 1.0
    return [val / max_val for val in data]

@ops(name="run_model", tags=["ml", "gpu"])
def run_model(data: List[float]) -> Dict[str, Any]:
    """Run a ML model that would typically use GPU."""
    # This is a simulation - in real use, this would use a model on GPU
    try:
        import torch
        
        # Create a simple model and force it to GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Convert input to tensor and move to the selected device
        input_tensor = torch.tensor(data, device=device)
        
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(len(data), 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 10),
            torch.nn.Softmax(dim=0)
        ).to(device)
        
        # Run the model multiple times to simulate load
        results = []
        for _ in range(100):
            output = model(input_tensor)
            results.append(output.tolist())
        
        # Return the last result
        return {
            "prediction": results[-1],
            "device": str(device),
            "input_shape": list(input_tensor.shape)
        }
        
    except ImportError:
        # If torch is not available, use a dummy computation
        print("PyTorch not available, using CPU simulation")
        import time
        import random
        
        # Simulate calculation
        time.sleep(2)
        return {
            "prediction": [random.random() for _ in range(10)],
            "device": "cpu-simulation",
            "input_shape": [len(data)]
        }

def main():
    print("Hapax GPU Monitoring Example\n")
    
    # Enable global GPU monitoring
    enable_gpu_monitoring(
        otlp_endpoint=os.getenv("OPENLIT_ENDPOINT", "http://localhost:4318"),
        sample_rate_seconds=1,
        application_name="hapax_gpu_example"
    )
    
    # Create test data
    test_data = [1.0, 2.5, 3.7, 4.2, 5.0, 6.1, 7.3, 8.2, 9.5, 10.0]
    
    # Example 1: Simple operations with global GPU monitoring
    print("Example 1: Operations with global GPU monitoring")
    processed_data = preprocess_data(test_data)
    model_result = run_model(processed_data)
    
    print(f"Input data: {test_data}")
    print(f"Processed data: {processed_data}")
    print(f"Model result: {model_result}")
    
    # Display GPU metrics after the first run
    metrics = get_gpu_metrics()
    if metrics:
        print("\nGPU Metrics after first run:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo GPU metrics available (either no GPU or monitoring not enabled).")
    
    print("\n" + "-" * 60 + "\n")
    
    # Example 2: Using a graph with GPU monitoring
    print("Example 2: Graph with dedicated GPU monitoring")
    
    pipeline = (
        Graph("ml_pipeline", "Simple ML pipeline with GPU monitoring")
        .then(preprocess_data)
        .then(run_model)
        .with_gpu_monitoring(
            enabled=True,
            sample_rate_seconds=0.5,  # More frequent sampling
            custom_config={"trace_memory": True}  # Custom config option
        )
    )
    
    # Execute the pipeline
    print("Executing graph with GPU monitoring...")
    result = pipeline.execute(test_data)
    print(f"Graph execution complete.")
    print(f"Result device: {result['device']}")
    
    # Wait a moment to collect metrics
    print("Waiting for metrics collection...")
    time.sleep(3)
    
    # Display updated GPU metrics
    metrics = get_gpu_metrics()
    if metrics:
        print("\nGPU Metrics after graph execution:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")
    else:
        print("\nNo GPU metrics available (either no GPU or monitoring not enabled).")

if __name__ == "__main__":
    main() 