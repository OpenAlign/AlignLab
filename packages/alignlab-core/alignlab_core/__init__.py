"""
AlignLab Core - The heart of the alignment evaluation framework.

This package provides the core functionality for running alignment evaluations,
including model providers, evaluation runners, benchmark loading, and reporting.
"""

from .runner import EvalRunner
from .benchmark import load_benchmark, Benchmark, list_benchmarks, filter_benchmarks
from .models import ModelProvider, HuggingFaceProvider, OpenAIProvider, create_provider
from .results import EvalResult
from .taxonomy import Taxonomy, get_taxonomy
from .suites import load_suite, list_suites, filter_suites, get_suite_summary
from .judges import Judge, ExactMatchJudge, LLMRubricJudge, TruthfulQAJudge, ToxicityJudge, create_judge

__version__ = "0.1.0"
__all__ = [
    "EvalRunner",
    "load_benchmark", 
    "Benchmark",
    "list_benchmarks",
    "filter_benchmarks",
    "ModelProvider",
    "HuggingFaceProvider",
    "OpenAIProvider",
    "create_provider",
    "EvalResult",
    "Taxonomy",
    "get_taxonomy",
    "load_suite",
    "list_suites",
    "filter_suites",
    "get_suite_summary",
    "Judge",
    "ExactMatchJudge",
    "LLMRubricJudge",
    "TruthfulQAJudge",
    "ToxicityJudge",
    "create_judge"
]
