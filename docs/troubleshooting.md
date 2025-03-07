# Troubleshooting Guide

This guide addresses common issues you might encounter when using Hapax and provides solutions.

## Table of Contents
- [Installation Issues](#installation-issues)
- [Type Errors](#type-errors)
- [Graph Validation Errors](#graph-validation-errors)
- [Execution Errors](#execution-errors)
- [OpenLIT Integration Issues](#openlit-integration-issues)
- [GPU Monitoring Issues](#gpu-monitoring-issues)
- [Evaluation Issues](#evaluation-issues)
- [Performance Considerations](#performance-considerations)

## Installation Issues

### Cannot install hapax due to dependency conflicts

**Problem:**
```
ERROR: Cannot install hapax due to incompatible dependencies: openlit requires opentelemetry>=1.20.0
```

**Solution:**
1. Create a fresh virtual environment:
   ```bash
   python -m venv fresh_env
   source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate
   ```
2. Install with explicit version specification:
   ```bash
   pip install "hapax[all]" --upgrade
   ```

### Missing Optional Dependencies

**Problem:**
```
ImportError: Cannot import name 'HallucinationEvaluator' from 'hapax'
```

**Solution:**
Install the appropriate optional dependencies:
```bash
pip install "hapax[eval]"  # For evaluators
pip install "hapax[gpu]"   # For GPU monitoring
pip install "hapax[all]"   # For all features
```

## Type Errors

### Incompatible Types in Operation Composition

**Problem:**
```
TypeError: Cannot compose operations: output type List[str] does not match input type Dict[str, Any]
```

**Solution:**
1. Check your operation type hints:
   ```python
   @ops
   def tokenize(text: str) -> List[str]:  # Returns List[str]
       return text.split()
   
   @ops
   def count_words(tokens: List[str]) -> Dict[str, int]:  # Expects List[str]
       from collections import Counter
       return dict(Counter(tokens))
   
   # Correct composition
   pipeline = tokenize >> count_words
   ```

2. Add a transformation operation between incompatible operations:
   ```python
   @ops
   def transform(items: List[str]) -> Dict[str, Any]:
       return {"items": items}
   
   pipeline = tokenize >> transform >> dict_processor
   ```

### Runtime Type Errors

**Problem:**
```
TypeError: Expected str but got int for argument 'text'
```

**Solution:**
1. Ensure proper input types when calling operations:
   ```python
   result = text_operation("Hello")  # Correct
   # result = text_operation(123)    # Wrong
   ```

2. Add explicit type conversion where needed:
   ```python
   @ops
   def ensure_string(value: Any) -> str:
       return str(value)
   
   pipeline = ensure_string >> text_operation
   ```

## Graph Validation Errors

### Cycle Detected in Graph

**Problem:**
```
GraphValidationError: Graph contains cycles: [['op1', 'op2', 'op1']]
```

**Solution:**
1. Verify your graph structure doesn't have circular references
2. Use `Loop` explicitly for intended repetition:
   ```python
   from hapax.core.flow import Loop
   
   loop = Loop(
       "process_loop",
       process_operation,
       condition=lambda x: x.is_complete,
       max_iterations=10
   )
   ```

### Missing Operation in Graph

**Problem:**
```
GraphValidationError: Operation 'cleanup' referenced but not defined
```

**Solution:**
1. Ensure all operations are properly defined before using them in a graph
2. Check for typos in operation names
3. Verify the operation is imported in the current scope

## Execution Errors

### Branch Errors

**Problem:**
```
BranchError: Errors in branches: [('sentiment', ValueError('Invalid input'))]
```

**Solution:**
1. Handle branch errors explicitly:
   ```python
   try:
       result = pipeline.execute(input_data)
   except BranchError as e:
       print(f"Branch errors: {e.branch_errors}")
       print(f"Partial results: {e.partial_results}")
       # Use partial results or fallback strategy
   ```

2. Add validation and error handling in branch operations:
   ```python
   @ops
   def safe_sentiment(text: str) -> float:
       try:
           # Potentially risky operation
           return calculate_sentiment(text)
       except Exception:
           # Fallback
           return 0.0
   ```

### Memory Errors with Large Graphs

**Problem:**
```
MemoryError: Unable to allocate memory for graph execution
```

**Solution:**
1. Process data in smaller batches
2. Implement streaming operations
3. Add garbage collection in long-running operations:
   ```python
   @ops
   def memory_intensive(data: List[Any]) -> Any:
       import gc
       result = process_large_data(data)
       gc.collect()  # Explicitly run garbage collection
       return result
   ```

## OpenLIT Integration Issues

### Connection Refused

**Problem:**
```
ConnectionRefusedError: [Errno 111] Connection refused - OpenLIT endpoint not available
```

**Solution:**
1. Verify your OpenLIT endpoint is running:
   ```bash
   # Check if the port is open
   nc -zv localhost 4318
   ```
2. Update your endpoint configuration:
   ```python
   openlit.init(otlp_endpoint="http://localhost:4318")
   ```
3. If you don't have an OpenLIT backend, disable monitoring:
   ```python
   set_openlit_config(None)  # Disable OpenLIT integration
   ```

### Missing Metrics or Traces

**Problem:** Operations execute but metrics/traces are not appearing in your monitoring system.

**Solution:**
1. Verify OpenLIT initialization occurs before operation execution:
   ```python
   import openlit
   
   # Initialize OpenLIT first
   openlit.init(otlp_endpoint="http://localhost:4318")
   
   # Then define and use operations
   @ops
   def my_operation(x: int) -> int:
       return x + 1
   ```

2. Check OpenLIT configuration:
   ```python
   from hapax import set_openlit_config
   
   set_openlit_config({
       "trace_content": True,     # Enable content tracing
       "disable_metrics": False,  # Ensure metrics are enabled
       "otlp_endpoint": "http://localhost:4318"
   })
   ```

## GPU Monitoring Issues

### GPU Metrics Not Available

**Problem:** `get_gpu_metrics()` returns empty results or errors.

**Solution:**
1. Verify NVIDIA drivers are installed and accessible
2. Check optional dependencies:
   ```bash
   pip install "hapax[gpu]"
   ```
3. Ensure GPU monitoring is enabled:
   ```python
   from hapax import enable_gpu_monitoring
   
   enable_gpu_monitoring(sample_rate_seconds=1)
   ```

## Evaluation Issues

### API Key Not Found

**Problem:**
```
KeyError: Provider API key not found for OpenAI/Anthropic
```

**Solution:**
1. Set API keys as environment variables:
   ```bash
   export OPENAI_API_KEY=your_key_here
   export ANTHROPIC_API_KEY=your_key_here
   ```
2. Or pass explicitly:
   ```python
   evaluator = HallucinationEvaluator(
       provider="openai",
       api_key="your_key_here"
   )
   ```

### Evaluation Always Fails/Passes

**Problem:** Evaluations always return the same result regardless of content.

**Solution:**
1. Adjust threshold values:
   ```python
   @eval(evals=["toxicity"], threshold=0.7)  # Higher threshold = more permissive
   ```
2. Check your evaluator implementation or provider settings
3. Add debug logging:
   ```python
   @eval(
       evals=["toxicity"],
       openlit_config={"trace_content": True, "log_level": "DEBUG"}
   )
   ```

## Performance Considerations

### Slow Graph Execution

**Problem:** Graph execution is slower than expected.

**Solution:**
1. Profile your operations to identify bottlenecks:
   ```python
   import time
   
   @ops
   def timed_operation(data: Any) -> Any:
       start = time.time()
       result = process(data)
       duration = time.time() - start
       print(f"Operation took {duration:.2f} seconds")
       return result
   ```

2. Parallelize independent operations with Branch:
   ```python
   pipeline = (
       Graph("parallel_processing")
       .branch(
           heavy_operation_1,
           heavy_operation_2
       )
       .merge(combine_results)
   )
   ```

3. Use caching for expensive operations:
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=100)
   def expensive_computation(data: str) -> Any:
       # Expensive processing
       return result
   
   @ops
   def cached_operation(data: str) -> Any:
       return expensive_computation(data)
   ```

### High Memory Usage

**Problem:** Memory usage grows excessively during graph execution.

**Solution:**
1. Process data in smaller chunks
2. Implement generators for large datasets:
   ```python
   @ops
   def process_batches(data_source: Any) -> List[Any]:
       results = []
       for batch in get_batches(data_source, batch_size=100):
           result = process_batch(batch)
           results.append(result)
       return results
   ```

3. Add explicit cleanup in operations:
   ```python
   @ops
   def cleanup_after(data: Any) -> Any:
       result = process(data)
       # Explicit cleanup
       import gc
       gc.collect()
       return result
   ```

## Still Having Issues?

If you're still experiencing problems:

1. Check the [GitHub Issues](https://github.com/your-org/hapax/issues) for similar problems
2. Join our [Community Slack/Discord] for real-time help
3. File a detailed bug report with:
   - Hapax version
   - Python version
   - Operating system
   - Complete error traceback
   - Minimal code example to reproduce the issue 