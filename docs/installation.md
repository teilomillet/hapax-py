# Installation Guide

This guide covers how to install Hapax and its dependencies.

## System Requirements

- **Python**: 3.8 or higher
- **Operating Systems**: Linux, macOS, Windows
- **Memory**: Minimum 2GB RAM (4GB+ recommended for larger graphs)

## Basic Installation

Install the latest stable version of Hapax using pip:

```bash
pip install hapax
```

## Installation with Optional Features

### Basic Installation (Core Features)

```bash
pip install hapax
```

### With OpenLIT Monitoring

For basic monitoring capabilities:

```bash
pip install "hapax[monitoring]"
```

This installs OpenLIT with standard monitoring capabilities.

### With Advanced Evaluation Support

For LLM-based evaluations (requires API keys):

```bash
pip install "hapax[eval]"
```

This includes dependencies for OpenAI and Anthropic evaluators.

### With GPU Monitoring

For GPU monitoring capabilities:

```bash
pip install "hapax[gpu]"
```

Includes dependencies for tracking GPU usage and performance.

### Complete Installation (All Features)

```bash
pip install "hapax[all]"
```

Installs Hapax with all optional dependencies.

## Installation from Source

For the latest development version:

```bash
git clone https://github.com/teilomillet/hapax-py.git
cd hapax
pip install -e .
```

For development installation with test dependencies:

```bash
pip install -e ".[dev]"
```

## Dependencies

Hapax has the following core dependencies:

- **networkx**: For graph representation and operations
- **pydantic**: For configuration validation
- **opentelemetry-api/sdk**: For telemetry and monitoring
- **typing-extensions**: For enhanced type hints

Optional dependencies by feature:

| Feature | Dependencies |
|---------|--------------|
| Monitoring | openlit>=1.33.8 |
| GPU Monitoring | openlit>=1.33.8, nvidia-ml-py3 |
| Evaluations | openlit>=1.33.8, openai>=1.0.0 or anthropic>=0.5.0 |
| Development | pytest, mypy, flake8, black |

## Environment Variables

Hapax uses the following environment variables:

- `OPENAI_API_KEY`: For OpenAI-based evaluations
- `ANTHROPIC_API_KEY`: For Anthropic-based evaluations
- `OPENLIT_ENDPOINT`: Default OpenLIT telemetry endpoint
- `HAPAX_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Verifying Installation

To verify your installation:

```python
import hapax
print(hapax.__version__)

# Test basic functionality
from hapax import ops

@ops
def test_op(x: int) -> int:
    return x + 1

result = test_op(41)  # Should return 42
print(result)
```

## Troubleshooting Installation

### Common Issues

1. **Version Conflicts**:
   ```
   ERROR: Cannot install hapax due to version conflict with openlit
   ```
   Solution: Create a fresh virtual environment or try `pip install --upgrade hapax`.

2. **Missing Dependencies**:
   ```
   ImportError: Cannot import name 'X' from 'hapax'
   ```
   Solution: Ensure you installed the correct optional dependencies for the features you're using.

3. **OpenLIT Connection Issues**:
   ```
   ConnectionRefusedError: Failed to connect to OpenLIT endpoint
   ```
   Solution: Check that your OpenLIT endpoint is running and accessible.

For more detailed troubleshooting, see the [Troubleshooting Guide](troubleshooting.md). 