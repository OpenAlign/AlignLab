# AlignLab, by OpenAlign

@agi_atharv

AlignLab is a comprehensive alignment framework that provides researchers and practitioners with easy to use tools for aligning models across multiple dimensions: safety, truthfulness, bias, toxicity, and agentic robustness.

There is a lot more to add to AlignLab! Please work with us and contribute to the project. 
## 🏗️ Repository Layout (Monorepo)

```
alignlab/
├── packages/
│   ├── alignlab-core/            # Core Python library (datasets, runners, scoring, reports)
│   ├── alignlab-cli/             # CLI wrapping core functionality
│   ├── alignlab-evals/           # Thin adapters to lm-eval-harness, OpenAI evals, JailbreakBench, HarmBench
│   ├── alignlab-guards/          # Guard models (Llama-Guard adapters), rule engines, ensembles
│   ├── alignlab-dash/            # Streamlit/FastAPI dashboard + report viewer
│   └── alignlab-agents/          # Sandboxed agent evals (tools, browsing-locked simulators)
├── benchmarks/                   # YAML registry (sources, splits, metrics, licenses)
├── templates/                    # Cookiecutter for new evals/benchmarks
├── docs/                         # MkDocs documentation
└── examples/                     # Runnable notebooks & scripts
```

## 🎯 Core Promises

### Registry-First Design
Every benchmark is a single YAML entry with fields for taxonomy, language coverage, scoring, and version pinning. Reproducible by default.

### Adapters, Not Reinvention
First-class adapters to:
- **lm-evaluation-harness** for general QA/MC/MT (EleutherAI)
- **OpenAI Evals** for templated eval flows & rubric judging
- **JailbreakBench** (attacks/defenses, scoring, leaderboard spec)
- **HarmBench** (automated red teaming + robust refusal)

### Multilingual Support
Plug-in loaders for multilingual toxicity, truthfulness, and bias datasets.

### Guard-Stack
Unified API to run Llama-Guard-3 (and alternatives) as pre/post-filters or judges, with ensemble and calibration support.

### Agent Evals
Safe, containerized tools and scenarios with metrics like ASR (attack success rate), query efficiency, and over-refusal.

### Reports That Matter
One command yields a NeurIPS-style PDF/HTML with tables, CIs, error-buckets, and taxonomy-level breakdowns.

## 🚀 Quick Start

### Installation (Development)

```bash
# Install uv for dependency management
pipx install uv

# Create virtual environment and install packages
uv venv
uv pip install -e packages/alignlab-core -e packages/alignlab-cli \
               -e packages/alignlab-evals -e packages/alignlab-guards \
               -e packages/alignlab-agents -e packages/alignlab-dash
```

### Run a Full Safety Sweep

```bash
# Run comprehensive safety evaluation
alignlab eval run --suite alignlab:safety_core_v1 \
  --model meta-llama/Llama-3.1-8B-Instruct --provider hf \
  --guards llama_guard_3 --max-samples 200 \
  --report out/safety_core_v1

# Generate reports
alignlab report build out/safety_core_v1 --format html,pdf
```

### CLI Examples

```bash
# List available benchmarks and models
alignlab benchmarks ls --filter safety,multilingual
alignlab models ls

# Run a single benchmark
alignlab eval run truthfulqa --split validation --judge llm_rubric

# Agentic robustness (jailbreaks)
alignlab jailbreak run --suite jb_default_v1 --defense none --report out/jb
```

## 📊 Prebuilt Suites

### `alignlab:safety_core_v1`
- **Harm/Risk**: HarmBench, JailbreakBench, UltraSafety
- **Toxicity**: RealToxicityPrompts, PolygloToxicityPrompts
- **Truthfulness**: TruthfulQA + multilingual extensions
- **Bias/Fairness**: BBQ, CrowS-Pairs, StereoSet, Multilingual HolisticBias
- **Guards**: Optional pre/post Llama-Guard-3

### `alignlab:agent_robustness_v1`
- JailbreakBench canonical config
- HarmBench auto red teaming subset
- Prompt-injection battery

## 🔧 Key Modules & API

### Core Usage

```python
from alignlab.core import EvalRunner, load_benchmark

# Load and run a benchmark
bench = load_benchmark("truthfulqa")
runner = EvalRunner(model="meta-llama/Llama-3.1-8B-Instruct", provider="hf")
result = runner.run(bench, split="validation", judge="exact_match|llm_rubric")
result.to_report("reports/run_001")
```

### Guard Pipeline

```python
from alignlab.guards import LlamaGuard, RuleGuard, EnsembleGuard

guard = EnsembleGuard([
    LlamaGuard("meta-llama/Llama-Guard-3-8B"), 
    RuleGuard.from_yaml("mlc_taxonomy.yaml")
])
safe_out = guard.wrap(model).generate(prompt)
```

## 📋 Benchmark Registry

### Harm/Safety & Jailbreak
- **HarmBench** – Automated red teaming + refusal robustness
- **JailbreakBench** – NeurIPS'24 D&B artifacts, threat model, scoring
- **UltraSafety** – 3k harmful prompts with paired jailbreak prompts

### Toxicity (Mono + Multi)
- **RealToxicityPrompts** – Canonical 100k prompts
- **PolygloToxicityPrompts** – 425k prompts, 17 languages

### Truthfulness / Hallucination
- **TruthfulQA** – 817 Qs, 38 categories with MC/binary variants

### Bias/Fairness (Mono + Multi)
- **BBQ**, **CrowS-Pairs**, **StereoSet**, **Multilingual HolisticBias**

## 📝 Adding a New Benchmark

Create a YAML file in `benchmarks/`:

```yaml
# benchmarks/truthfulqa.yaml
id: truthfulqa
source:
  type: huggingface
  repo: "sylinrl/TruthfulQA"
splits: [validation]
task: freeform_qa
judges:
  - exact_match
  - llm_rubric: {model: "gpt-4o-mini", rubric: "truthfulqa_v1"}
metrics: [truthful, informative, truthful_informative]
taxonomy: [truthfulness]
license: "MIT"
version: "2025-01-15"
```

## 📊 Reporting & UX

- Per-taxonomy dashboards with macro/micro scores and CIs
- Guard deltas and hazard-class confusion matrices
- Multilingual heatmaps
- Exportable bundles (JSONL, PDF, HTML)

## 🤝 Contributing

This project follows a monorepo structure with clear separation of concerns. Each package can be developed and tested independently while maintaining integration through the core package.

## 📚 Documentation

See the `docs/` directory for comprehensive documentation, including:
- API reference
- Benchmark development guide
- Guard model integration
- Agent evaluation setup

## 📄 License

MIT License - see LICENSE file for details.
