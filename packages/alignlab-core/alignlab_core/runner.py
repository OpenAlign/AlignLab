"""
Evaluation runner for executing benchmarks against models.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .models import ModelProvider
from .benchmark import Benchmark
from .results import EvalResult
from .judges import Judge, ExactMatchJudge, LLMRubricJudge

logger = logging.getLogger(__name__)


@dataclass
class RunConfig:
    """Configuration for evaluation runs."""
    model: str
    provider: str
    split: str = "validation"
    max_samples: Optional[int] = None
    seed: int = 42
    output_dir: Optional[Path] = None
    judge: Optional[str] = None
    guards: Optional[List[str]] = None


class EvalRunner:
    """
    Main evaluation runner that orchestrates benchmark execution.
    
    Supports multiple model providers, judges, and guard models.
    """
    
    def __init__(self, model: str, provider: str = "hf", **kwargs):
        """
        Initialize the evaluation runner.
        
        Args:
            model: Model identifier (e.g., "meta-llama/Llama-3.1-8B-Instruct")
            provider: Model provider ("hf", "openai", "vertex", "vllm")
            **kwargs: Additional provider-specific arguments
        """
        self.model_id = model
        self.provider_name = provider
        self.model_provider = self._create_provider(provider, model, **kwargs)
        self.judges: Dict[str, Judge] = {}
        
    def _create_provider(self, provider: str, model: str, **kwargs) -> ModelProvider:
        """Create a model provider instance."""
        if provider == "hf":
            from .models import HuggingFaceProvider
            return HuggingFaceProvider(model, **kwargs)
        elif provider == "openai":
            from .models import OpenAIProvider
            return OpenAIProvider(model, **kwargs)
        elif provider == "vertex":
            from .models import VertexProvider
            return VertexProvider(model, **kwargs)
        elif provider == "vllm":
            from .models import VLLMProvider
            return VLLMProvider(model, **kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    
    def add_judge(self, name: str, judge: Judge):
        """Add a judge for evaluation."""
        self.judges[name] = judge
    
    def _setup_judges(self, judge_spec: Optional[str]) -> List[Judge]:
        """Setup judges based on specification."""
        if not judge_spec:
            return []
            
        judges = []
        for judge_name in judge_spec.split("|"):
            judge_name = judge_name.strip()
            
            if judge_name in self.judges:
                judges.append(self.judges[judge_name])
            elif judge_name == "exact_match":
                judges.append(ExactMatchJudge())
            elif judge_name.startswith("llm_rubric"):
                # Parse LLM rubric configuration
                config = self._parse_llm_rubric_config(judge_name)
                judges.append(LLMRubricJudge(**config))
            else:
                raise ValueError(f"Unknown judge: {judge_name}")
                
        return judges
    
    def _parse_llm_rubric_config(self, judge_spec: str) -> Dict[str, Any]:
        """Parse LLM rubric configuration from judge specification."""
        # Simple parsing for now: llm_rubric: {model: "gpt-4o-mini", rubric: "truthfulqa_v1"}
        if ":" in judge_spec:
            config_str = judge_spec.split(":", 1)[1].strip()
            # This is a simplified parser - in practice, you'd want more robust parsing
            config = {}
            if "model:" in config_str:
                model_match = config_str.split("model:")[1].split(",")[0].strip().strip('"')
                config["model"] = model_match
            if "rubric:" in config_str:
                rubric_match = config_str.split("rubric:")[1].split("}")[0].strip().strip('"')
                config["rubric"] = rubric_match
            return config
        return {}
    
    def run(self, benchmark: Benchmark, config: Optional[RunConfig] = None, **kwargs) -> EvalResult:
        """
        Run a benchmark evaluation.
        
        Args:
            benchmark: The benchmark to run
            config: Run configuration
            **kwargs: Override config parameters
            
        Returns:
            EvalResult containing evaluation results
        """
        if config is None:
            config = RunConfig(model=self.model_id, provider=self.provider_name)
        
        # Override config with kwargs
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
        
        logger.info(f"Running benchmark {benchmark.id} with model {config.model}")
        
        # Load dataset
        dataset = benchmark.load_dataset(config.split)
        if config.max_samples:
            dataset = dataset.select(range(min(config.max_samples, len(dataset))))
        
        # Setup judges
        judges = self._setup_judges(config.judge)
        
        # Run evaluation
        results = []
        for i, example in enumerate(dataset):
            logger.debug(f"Processing example {i+1}/{len(dataset)}")
            
            # Generate response
            response = self.model_provider.generate(
                prompt=example["prompt"],
                max_tokens=example.get("max_tokens", 512),
                temperature=example.get("temperature", 0.0)
            )
            
            # Apply judges
            judge_results = {}
            for judge in judges:
                judge_results[judge.name] = judge.evaluate(example, response)
            
            results.append({
                "example_id": i,
                "prompt": example["prompt"],
                "response": response,
                "reference": example.get("reference"),
                "judge_results": judge_results,
                "metadata": example.get("metadata", {})
            })
        
        # Create result object
        eval_result = EvalResult(
            benchmark_id=benchmark.id,
            model_id=config.model,
            provider=config.provider,
            split=config.split,
            results=results,
            config=config
        )
        
        # Save results if output directory specified
        if config.output_dir:
            config.output_dir.mkdir(parents=True, exist_ok=True)
            eval_result.save(config.output_dir)
        
        return eval_result
    
    def run_suite(self, suite_name: str, config: Optional[RunConfig] = None, **kwargs) -> Dict[str, EvalResult]:
        """
        Run a suite of benchmarks.
        
        Args:
            suite_name: Name of the suite to run
            config: Run configuration
            **kwargs: Override config parameters
            
        Returns:
            Dictionary mapping benchmark IDs to results
        """
        from .suites import load_suite
        
        suite = load_suite(suite_name)
        results = {}
        
        for benchmark_id in suite.benchmarks:
            benchmark = Benchmark.load(benchmark_id)
            results[benchmark_id] = self.run(benchmark, config, **kwargs)
        
        return results

