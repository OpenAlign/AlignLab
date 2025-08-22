"""
Adapter for lm-evaluation-harness integration.
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
class LMEvalConfig:
    """Configuration for lm-eval-harness tasks."""
    task_name: str
    task_args: Optional[Dict[str, Any]] = None
    model_args: Optional[Dict[str, Any]] = None
    limit: Optional[int] = None
    num_fewshot: int = 0
    batch_size: Optional[str] = None
    device: Optional[str] = None


class LMEvalAdapter:
    """Adapter for lm-evaluation-harness tasks."""
    
    def __init__(self, model_provider: ModelProvider):
        """
        Initialize the adapter.
        
        Args:
            model_provider: Model provider instance
        """
        self.model_provider = model_provider
        self._lm_eval_available = self._check_lm_eval_availability()
    
    def _check_lm_eval_availability(self) -> bool:
        """Check if lm-eval-harness is available."""
        try:
            import lm_eval
            return True
        except ImportError:
            logger.warning("lm-eval-harness not available. Install with: pip install lm-eval")
            return False
    
    def run_task(self, config: LMEvalConfig) -> EvalResult:
        """
        Run a lm-eval-harness task.
        
        Args:
            config: Task configuration
            
        Returns:
            EvalResult with task results
        """
        if not self._lm_eval_available:
            raise ImportError("lm-eval-harness not available")
        
        try:
            import lm_eval
            from lm_eval import evaluator
            
            # Prepare model arguments
            model_args = config.model_args or {}
            if hasattr(self.model_provider, 'model_id'):
                model_args['pretrained'] = self.model_provider.model_id
            
            # Prepare task arguments
            task_args = config.task_args or {}
            if config.limit:
                task_args['limit'] = config.limit
            
            # Run evaluation
            results = evaluator.evaluate(
                model=self._create_lm_eval_model(),
                tasks=[config.task_name],
                num_fewshot=config.num_fewshot,
                batch_size=config.batch_size,
                device=config.device
            )
            
            # Convert results to AlignLab format
            return self._convert_results(results, config)
            
        except Exception as e:
            logger.error(f"Error running lm-eval task: {e}")
            raise
    
    def _create_lm_eval_model(self):
        """Create a lm-eval compatible model wrapper."""
        class LMEvalModelWrapper:
            def __init__(self, model_provider):
                self.model_provider = model_provider
            
            def generate_until(self, requests):
                """Generate responses for lm-eval requests."""
                results = []
                for request in requests:
                    try:
                        response = self.model_provider.generate(
                            request['prompt'],
                            max_tokens=request.get('max_gen_tokens', 512),
                            temperature=request.get('temperature', 0.0)
                        )
                        results.append(response)
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        results.append("")
                return results
            
            def loglikelihood(self, requests):
                """Compute log likelihoods (placeholder)."""
                # This would require access to model logits
                # For now, return placeholder values
                results = []
                for request in requests:
                    results.append((0.0, 0.0))  # (log_likelihood, is_greedy)
                return results
        
        return LMEvalModelWrapper(self.model_provider)
    
    def _convert_results(self, lm_eval_results: Dict[str, Any], config: LMEvalConfig) -> EvalResult:
        """Convert lm-eval results to AlignLab format."""
        # Extract metrics
        metrics = {}
        for task_name, task_results in lm_eval_results['results'].items():
            for metric_name, metric_value in task_results.items():
                if isinstance(metric_value, (int, float)):
                    metrics[f"{task_name}_{metric_name}"] = metric_value
        
        # Create mock results for compatibility
        results = []
        for i in range(min(10, config.limit or 10)):  # Mock results
            results.append({
                "example_id": i,
                "prompt": f"Example prompt {i}",
                "response": f"Example response {i}",
                "judge_results": {
                    "lm_eval": {
                        "score": 0.5,  # Placeholder
                        "category": "completed"
                    }
                },
                "metadata": {
                    "task": config.task_name,
                    "metrics": metrics
                }
            })
        
        return EvalResult(
            benchmark_id=f"lm_eval_{config.task_name}",
            model_id=getattr(self.model_provider, 'model_id', 'unknown'),
            provider=getattr(self.model_provider, 'provider_name', 'unknown'),
            split="test",
            results=results
        )
    
    def list_available_tasks(self) -> List[str]:
        """List available lm-eval tasks."""
        if not self._lm_eval_available:
            return []
        
        try:
            import lm_eval
            from lm_eval import tasks
            
            # Get available tasks
            available_tasks = []
            for task_name in tasks.ALL_TASKS:
                available_tasks.append(task_name)
            
            return sorted(available_tasks)
            
        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []
    
    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """Get information about a specific task."""
        if not self._lm_eval_available:
            return {}
        
        try:
            import lm_eval
            from lm_eval import tasks
            
            if task_name not in tasks.ALL_TASKS:
                return {}
            
            # Get task class
            task_class = tasks.get_task(task_name)
            
            return {
                "name": task_name,
                "description": getattr(task_class, '__doc__', 'No description available'),
                "version": getattr(task_class, 'VERSION', 'unknown'),
                "datasets": getattr(task_class, 'DATASET_PATH', 'unknown'),
                "metrics": getattr(task_class, 'METRICS', []),
            }
            
        except Exception as e:
            logger.error(f"Error getting task info: {e}")
            return {}


def create_lm_eval_benchmark(task_name: str, **kwargs) -> Benchmark:
    """
    Create a Benchmark from an lm-eval task.
    
    Args:
        task_name: Name of the lm-eval task
        **kwargs: Additional benchmark configuration
        
    Returns:
        Benchmark instance
    """
    # This would create a benchmark definition that uses the lm-eval adapter
    # For now, return a placeholder
    return Benchmark(
        id=f"lm_eval_{task_name}",
        source=None,  # Would be set to use lm-eval
        splits=["test"],
        task="lm_eval",
        judges=[],  # Would be configured based on task
        metrics=["accuracy"],  # Would be configured based on task
        taxonomy=["general"],
        license="Unknown",
        version="1.0.0",
        description=f"lm-eval-harness task: {task_name}"
    )

