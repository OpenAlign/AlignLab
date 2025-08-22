# AlignLab Installation Guide

This guide will help you install and set up the AlignLab framework for AI alignment evaluation.

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)

## Quick Installation

### 1. Clone the Repository

```bash
git clone https://github.com/alignlab/alignlab.git
cd alignlab
```

### 2. Install All Packages

Install all AlignLab packages in development mode:

```bash
# Install core packages
pip install -e packages/alignlab-core
pip install -e packages/alignlab-cli
pip install -e packages/alignlab-guards
pip install -e packages/alignlab-evals
pip install -e packages/alignlab-dash
pip install -e packages/alignlab-agents

# Or install all at once from the root
pip install -e .
```

### 3. Verify Installation

Test that everything is working:

```bash
# Test CLI
alignlab --help

# Test core functionality
python3 examples/quickstart.py

# Run full demo
python3 examples/full_demo.py
```

## Detailed Installation

### Core Dependencies

The core framework requires these Python packages:

```bash
pip install numpy pandas pyyaml click datasets
```

### Optional Dependencies

#### For Guard Models
```bash
pip install transformers torch
```

#### For Dashboard
```bash
pip install dash plotly
```

#### For External Eval Adapters
```bash
# For lm-eval-harness
pip install lm-eval

# For OpenAI Evals
pip install openai-evals

# For PDF reports
pip install pdfkit
```

#### For Development
```bash
pip install pytest black isort mypy coverage
```

## Configuration

### Environment Variables

Set up environment variables for API access:

```bash
# OpenAI API (for OpenAI provider)
export OPENAI_API_KEY="your-openai-api-key"

# HuggingFace API (for HF provider)
export HUGGINGFACE_API_KEY="your-hf-api-key"

# Google Cloud (for Vertex AI provider)
export GOOGLE_APPLICATION_CREDENTIALS="path/to/credentials.json"
```

### Configuration Files

Create a configuration file at `~/.alignlab/config.yaml`:

```yaml
# Default model provider
default_provider: "hf"

# Model configurations
models:
  hf:
    default_model: "gpt2"
    max_tokens: 512
    temperature: 0.0
  
  openai:
    default_model: "gpt-3.5-turbo"
    max_tokens: 512
    temperature: 0.0

# Guard model settings
guards:
  llama_guard:
    model_id: "meta-llama/Llama-Guard-3-8B"
    threshold: 0.5
  
  rule_guard:
    enabled: true
    rules_file: "~/.alignlab/rules.yaml"

# Dashboard settings
dashboard:
  port: 8050
  host: "localhost"
  auto_open: true

# Evaluation settings
evaluation:
  default_seed: 42
  max_samples: 1000
  output_dir: "./results"
```

## Usage Examples

### Basic Evaluation

```bash
# List available benchmarks
alignlab benchmarks ls

# Run a benchmark
alignlab eval run truthfulqa --model gpt2 --provider hf

# Run a suite
alignlab eval run alignlab:safety_core_v1 --model gpt2 --provider hf
```

### Guard Models

```python
from alignlab_guards import LlamaGuard, RuleGuard, EnsembleGuard

# Initialize guards
llama_guard = LlamaGuard("meta-llama/Llama-Guard-3-8B")
rule_guard = RuleGuard()
ensemble = EnsembleGuard([llama_guard, rule_guard])

# Test a prompt
result = ensemble.classify_prompt("How to hack a computer?")
print(f"Safe: {result.is_safe}, Category: {result.category}")
```

### Agent Evaluation

```python
from alignlab_agents import AgentEvaluator, AgentConfig
from alignlab_core.models import HuggingFaceProvider

# Create agent evaluator
model_provider = HuggingFaceProvider("gpt2")
config = AgentConfig(model_provider=model_provider)
evaluator = AgentEvaluator(config)

# Run evaluation
scenarios = [...]  # Your test scenarios
results = evaluator.evaluate_robustness(scenarios)
```

### Dashboard

```python
from alignlab_dash import Dashboard

# Create dashboard
dashboard = Dashboard()
dashboard.add_results_from_directory("./results")
dashboard.create_dashboard()  # Opens in browser
```

## Troubleshooting

### Common Issues

#### Import Errors
If you get import errors, make sure all packages are installed:

```bash
pip install -e packages/alignlab-core
pip install -e packages/alignlab-cli
pip install -e packages/alignlab-guards
pip install -e packages/alignlab-evals
pip install -e packages/alignlab-dash
pip install -e packages/alignlab-agents
```

#### Model Provider Issues
If you can't access models:

1. Check your API keys are set correctly
2. Verify you have sufficient credits/quota
3. Try a different model provider

#### Guard Model Issues
If guard models fail to load:

1. Check you have enough GPU memory
2. Try using CPU-only mode
3. Verify the model ID is correct

#### Dashboard Issues
If the dashboard won't start:

1. Check if port 8050 is available
2. Try a different port: `dashboard = Dashboard(port=8051)`
3. Make sure you have the required dependencies

### Getting Help

- Check the [examples/](examples/) directory for usage examples
- Run `python3 examples/full_demo.py` to test all components
- Check the logs for detailed error messages
- Open an issue on GitHub if you need help

## Development Setup

### For Contributors

1. Fork the repository
2. Clone your fork
3. Install in development mode:

```bash
pip install -e .[dev]
```

4. Install pre-commit hooks:

```bash
pre-commit install
```

5. Run tests:

```bash
pytest
```

### Code Style

The project uses:
- **Black** for code formatting
- **isort** for import sorting
- **mypy** for type checking
- **pytest** for testing

Format your code:

```bash
black .
isort .
mypy .
```

## Next Steps

After installation:

1. **Explore the examples**: Check out the examples in the `examples/` directory
2. **Read the documentation**: See the `docs/` directory for detailed guides
3. **Try the benchmarks**: Run some evaluations with the included benchmarks
4. **Create custom benchmarks**: Learn how to create your own benchmarks
5. **Join the community**: Contribute to the project or ask questions

## Support

- **Documentation**: Check the `docs/` directory
- **Examples**: See the `examples/` directory
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions for questions

Happy evaluating! 🚀

