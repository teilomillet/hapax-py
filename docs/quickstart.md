# Hapax Quick Start Guide

This guide will help you get started with Hapax in under 5 minutes.

## Installation

```bash
pip install hapax
```

## Basic Usage

### 1. Create Simple Operations

```python
from hapax import ops
from typing import List

@ops  # Uses function name as operation name
def tokenize(text: str) -> List[str]:
    return text.split()

@ops  # Uses function name as operation name
def remove_stops(words: List[str]) -> List[str]:
    stops = {'the', 'a', 'an'}
    return [w for w in words if w not in stops]
```

### 2. Compose Operations

```python
# Use >> to chain operations
pipeline = tokenize >> remove_stops

# Use the pipeline
result = pipeline("The quick brown fox")  # ['quick', 'brown', 'fox']
```

### 3. Create a Graph

```python
from hapax import graph

@graph  # Uses function name as graph name
def text_pipeline(text: str) -> List[str]:
    return tokenize >> remove_stops

# Use the graph
result = text_pipeline("The quick brown fox")
```

## Add Monitoring (Optional)

```python
import openlit

# Initialize monitoring
openlit.init(otlp_endpoint="http://127.0.0.1:4318")

# Your operations are now automatically monitored!
```

## Next Steps

- Read the [full documentation](index.md) for more features
- Check out the [OpenLit integration guide](openlit.md) for monitoring
- See the [examples directory](examples/) for more use cases

That's it! You're ready to build type-safe data processing pipelines with Hapax.
