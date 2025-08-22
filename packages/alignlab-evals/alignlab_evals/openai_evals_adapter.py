"""
Adapter for OpenAI Evals integration.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from alignlab_core import EvalResult, Benchmark
from alignlab_core.models import ModelProvider

logger = logging.getLogger(__name__)


@dataclass
class OpenAIEvalConfig:
    """Configuration for OpenAI Evals."""
    eval_name: str
    eval_args: Optional[Dict[str, Any]] = None
    model: Optional[str] = None
    max_samples: Optional[int] = None
    output_dir: Optional[str] = None


class OpenAIEvalsAdapter:
    """Adapter for OpenAI Evals integration."""
    
    def __init__(self, model_provider: ModelProvider):
        """
        Initialize the adapter.
        
        Args:
            model_provider: Model provider instance
        """
        self.model_provider = model_provider
        self._openai_evals_available = self._check_openai_evals_availability()
    
    def _check_openai_evals_availability(self) -> bool:
        """Check if OpenAI Evals is available."""
        try:
            import evals
            return True
        except ImportError:
            logger.warning("OpenAI Evals not available. Install with: pip install openai-evals")
            return False
    
    def run_eval(self, config: OpenAIEvalConfig) -> EvalResult:
        """
        Run an OpenAI Eval.
        
        Args:
            config: Eval configuration
            
        Returns:
            EvalResult with eval results
        """
        if not self._openai_evals_available:
            raise ImportError("OpenAI Evals not available")
        
        try:
            import evals
            from evals import eval as run_eval
            
            # Prepare eval arguments
            eval_args = config.eval_args or {}
            if config.model:
                eval_args['model'] = config.model
            if config.max_samples:
                eval_args['max_samples'] = config.max_samples
            if config.output_dir:
                eval_args['output_dir'] = config.output_dir
            
            # Run the eval
            results = run_eval(
                eval_name=config.eval_name,
                **eval_args
            )
            
            # Convert results to AlignLab format
            return self._convert_results(results, config)
            
        except Exception as e:
            logger.error(f"Error running OpenAI Eval: {e}")
            raise
    
    def _convert_results(self, openai_results: Dict[str, Any], config: OpenAIEvalConfig) -> EvalResult:
        """Convert OpenAI Eval results to AlignLab format."""
        # Extract metrics from results
        metrics = {}
        if 'metrics' in openai_results:
            for metric_name, metric_value in openai_results['metrics'].items():
                if isinstance(metric_value, (int, float)):
                    metrics[metric_name] = metric_value
        
        # Create results list
        results = []
        if 'samples' in openai_results:
            for i, sample in enumerate(openai_results['samples']):
                results.append({
                    "example_id": i,
                    "prompt": sample.get('input', ''),
                    "response": sample.get('output', ''),
                    "reference": sample.get('ideal', ''),
                    "judge_results": {
                        "openai_eval": {
                            "score": sample.get('score', 0.0),
                            "category": sample.get('category', 'unknown'),
                            "rationale": sample.get('rationale', '')
                        }
                    },
                    "metadata": {
                        "eval_name": config.eval_name,
                        "sample_id": sample.get('id', i)
                    }
                })
        
        return EvalResult(
            benchmark_id=f"openai_eval_{config.eval_name}",
            model_id=getattr(self.model_provider, 'model_id', 'unknown'),
            provider=getattr(self.model_provider, 'provider_name', 'unknown'),
            split="test",
            results=results
        )
    
    def list_available_evals(self) -> List[str]:
        """List available OpenAI Evals."""
        if not self._openai_evals_available:
            return []
        
        try:
            import evals
            from evals import registry
            
            # Get available evals
            available_evals = []
            for eval_name in registry.get_evals():
                available_evals.append(eval_name)
            
            return sorted(available_evals)
            
        except Exception as e:
            logger.error(f"Error listing evals: {e}")
            return []
    
    def get_eval_info(self, eval_name: str) -> Dict[str, Any]:
        """Get information about a specific eval."""
        if not self._openai_evals_available:
            return {}
        
        try:
            import evals
            from evals import registry
            
            # Get eval class
            eval_class = registry.get_eval(eval_name)
            
            return {
                "name": eval_name,
                "description": getattr(eval_class, '__doc__', 'No description available'),
                "version": getattr(eval_class, 'VERSION', 'unknown'),
                "metrics": getattr(eval_class, 'METRICS', []),
            }
            
        except Exception as e:
            logger.error(f"Error getting eval info: {e}")
            return {}
    
    def create_eval_from_yaml(self, yaml_path: str) -> str:
        """
        Create an eval from a YAML file.
        
        Args:
            yaml_path: Path to YAML eval definition
            
        Returns:
            Eval name
        """
        if not self._openai_evals_available:
            raise ImportError("OpenAI Evals not available")
        
        try:
            import evals
            from evals import registry
            
            # Load eval from YAML
            eval_name = registry.register_eval(yaml_path)
            return eval_name
            
        except Exception as e:
            logger.error(f"Error creating eval from YAML: {e}")
            raise


def create_openai_eval_benchmark(eval_name: str, **kwargs) -> Benchmark:
    """
    Create a Benchmark from an OpenAI Eval.
    
    Args:
        eval_name: Name of the OpenAI Eval
        **kwargs: Additional benchmark configuration
        
    Returns:
        Benchmark instance
    """
    return Benchmark(
        id=f"openai_eval_{eval_name}",
        source=None,  # Would be set to use OpenAI Evals
        splits=["test"],
        task="openai_eval",
        judges=[],  # Would be configured based on eval
        metrics=["accuracy"],  # Would be configured based on eval
        taxonomy=["general"],
        license="Unknown",
        version="1.0.0",
        description=f"OpenAI Eval: {eval_name}"
    )

