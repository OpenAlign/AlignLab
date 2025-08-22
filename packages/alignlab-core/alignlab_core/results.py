"""
Evaluation results and reporting.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Container for evaluation results."""
    benchmark_id: str
    model_id: str
    provider: str
    split: str
    results: List[Dict[str, Any]]
    config: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def __post_init__(self):
        """Post-initialization processing."""
        self._metrics = None
        self._summary = None
    
    def save(self, output_dir: Path):
        """Save results to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save raw results
        results_file = output_dir / "results.jsonl"
        with open(results_file, "w") as f:
            for result in self.results:
                f.write(json.dumps(result) + "\n")
        
        # Save metadata
        metadata = {
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "split": self.split,
            "timestamp": self.timestamp,
            "num_examples": len(self.results),
            "metadata": self.metadata
        }
        
        if self.config:
            metadata["config"] = self.config.__dict__
        
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save summary
        summary = self.get_summary()
        summary_file = output_dir / "summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Saved results to {output_dir}")
    
    @classmethod
    def load(cls, output_dir: Path) -> "EvalResult":
        """Load results from disk."""
        output_dir = Path(output_dir)
        
        # Load metadata
        metadata_file = output_dir / "metadata.json"
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        # Load results
        results = []
        results_file = output_dir / "results.jsonl"
        with open(results_file, "r") as f:
            for line in f:
                results.append(json.loads(line.strip()))
        
        return cls(
            benchmark_id=metadata["benchmark_id"],
            model_id=metadata["model_id"],
            provider=metadata["provider"],
            split=metadata["split"],
            results=results,
            metadata=metadata.get("metadata", {}),
            timestamp=metadata["timestamp"]
        )
    
    def get_metrics(self) -> Dict[str, Any]:
        """Compute metrics from results."""
        if self._metrics is not None:
            return self._metrics
        
        metrics = {}
        
        # Extract judge results
        judge_results = {}
        for result in self.results:
            for judge_name, judge_result in result.get("judge_results", {}).items():
                if judge_name not in judge_results:
                    judge_results[judge_name] = []
                judge_results[judge_name].append(judge_result)
        
        # Compute metrics for each judge
        for judge_name, results in judge_results.items():
            metrics[judge_name] = self._compute_judge_metrics(results)
        
        # Compute overall metrics
        metrics["overall"] = self._compute_overall_metrics()
        
        self._metrics = metrics
        return metrics
    
    def _compute_judge_metrics(self, judge_results: List[Any]) -> Dict[str, Any]:
        """Compute metrics for a specific judge."""
        metrics = {}
        
        # Extract scores
        scores = []
        for result in judge_results:
            if isinstance(result, dict):
                score = result.get("score")
                if score is not None:
                    scores.append(float(score))
            elif isinstance(result, (int, float)):
                scores.append(float(result))
        
        if scores:
            metrics["mean"] = np.mean(scores)
            metrics["std"] = np.std(scores)
            metrics["min"] = np.min(scores)
            metrics["max"] = np.max(scores)
            metrics["median"] = np.median(scores)
            
            # Confidence intervals (bootstrap)
            ci_lower, ci_upper = self._bootstrap_ci(scores)
            metrics["ci_95"] = [ci_lower, ci_upper]
        
        # Extract categorical results
        categories = {}
        for result in judge_results:
            if isinstance(result, dict):
                category = result.get("category")
                if category:
                    categories[category] = categories.get(category, 0) + 1
        
        if categories:
            metrics["categories"] = categories
            total = sum(categories.values())
            metrics["category_distribution"] = {
                cat: count / total for cat, count in categories.items()
            }
        
        return metrics
    
    def _compute_overall_metrics(self) -> Dict[str, Any]:
        """Compute overall metrics across all results."""
        metrics = {
            "num_examples": len(self.results),
            "num_with_responses": sum(1 for r in self.results if r.get("response")),
            "avg_response_length": np.mean([
                len(r.get("response", "")) for r in self.results
            ])
        }
        
        return metrics
    
    def _bootstrap_ci(self, scores: List[float], confidence: float = 0.95, n_bootstrap: int = 1000) -> tuple:
        """Compute bootstrap confidence intervals."""
        if len(scores) < 2:
            return np.mean(scores), np.mean(scores)
        
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100
        
        return np.percentile(bootstrap_means, [lower_percentile, upper_percentile])
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the evaluation results."""
        if self._summary is not None:
            return self._summary
        
        metrics = self.get_metrics()
        
        summary = {
            "benchmark_id": self.benchmark_id,
            "model_id": self.model_id,
            "provider": self.provider,
            "split": self.split,
            "timestamp": self.timestamp,
            "num_examples": len(self.results),
            "metrics": metrics,
            "metadata": self.metadata
        }
        
        self._summary = summary
        return summary
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        df_data = []
        
        for result in self.results:
            row = {
                "example_id": result.get("example_id"),
                "prompt": result.get("prompt"),
                "response": result.get("response"),
                "reference": result.get("reference")
            }
            
            # Add judge results
            for judge_name, judge_result in result.get("judge_results", {}).items():
                if isinstance(judge_result, dict):
                    row[f"{judge_name}_score"] = judge_result.get("score")
                    row[f"{judge_name}_category"] = judge_result.get("category")
                    row[f"{judge_name}_rationale"] = judge_result.get("rationale")
                else:
                    row[f"{judge_name}_result"] = judge_result
            
            # Add metadata
            for key, value in result.get("metadata", {}).items():
                row[f"metadata_{key}"] = value
            
            df_data.append(row)
        
        return pd.DataFrame(df_data)
    
    def to_report(self, output_path: str, format: str = "html"):
        """Generate a report from the results."""
        from .reporting import ReportGenerator
        
        generator = ReportGenerator()
        
        if format == "html":
            return generator.generate_html_report(self, output_path)
        elif format == "pdf":
            return generator.generate_pdf_report(self, output_path)
        elif format == "markdown":
            return generator.generate_markdown_report(self, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def filter_by_category(self, judge_name: str, category: str) -> "EvalResult":
        """Filter results by judge category."""
        filtered_results = []
        
        for result in self.results:
            judge_result = result.get("judge_results", {}).get(judge_name, {})
            if isinstance(judge_result, dict) and judge_result.get("category") == category:
                filtered_results.append(result)
        
        return EvalResult(
            benchmark_id=self.benchmark_id,
            model_id=self.model_id,
            provider=self.provider,
            split=self.split,
            results=filtered_results,
            config=self.config,
            metadata=self.metadata,
            timestamp=self.timestamp
        )
    
    def get_error_examples(self, judge_name: str, max_examples: int = 10) -> List[Dict[str, Any]]:
        """Get examples where the model made errors (low scores)."""
        error_examples = []
        
        for result in self.results:
            judge_result = result.get("judge_results", {}).get(judge_name, {})
            if isinstance(judge_result, dict):
                score = judge_result.get("score")
                if score is not None and score < 0.5:  # Threshold for errors
                    error_examples.append(result)
        
        # Sort by score (lowest first)
        error_examples.sort(key=lambda x: x["judge_results"][judge_name].get("score", 1.0))
        
        return error_examples[:max_examples]

