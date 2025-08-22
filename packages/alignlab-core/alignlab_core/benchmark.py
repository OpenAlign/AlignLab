"""
Benchmark loading and management.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datasets import Dataset, load_dataset

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSource:
    """Source configuration for a benchmark."""
    type: str  # "huggingface", "script", "local"
    repo: Optional[str] = None
    path: Optional[str] = None
    config: Optional[str] = None
    revision: Optional[str] = None


@dataclass
class JudgeConfig:
    """Configuration for a judge."""
    name: str
    type: str  # "exact_match", "llm_rubric", "custom"
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Benchmark:
    """A benchmark definition."""
    id: str
    source: BenchmarkSource
    splits: List[str]
    task: str  # "freeform_qa", "multiple_choice", "classification", etc.
    judges: List[JudgeConfig]
    metrics: List[str]
    taxonomy: List[str]
    license: str
    version: str
    description: Optional[str] = None
    citation: Optional[str] = None
    
    @classmethod
    def load(cls, benchmark_id: str) -> "Benchmark":
        """Load a benchmark from the registry."""
        registry_path = Path(__file__).parent.parent.parent.parent / "benchmarks"
        benchmark_file = registry_path / f"{benchmark_id}.yaml"
        
        if not benchmark_file.exists():
            raise FileNotFoundError(f"Benchmark {benchmark_id} not found in registry")
        
        with open(benchmark_file, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Benchmark":
        """Create a benchmark from a dictionary."""
        source_data = data["source"]
        source = BenchmarkSource(
            type=source_data["type"],
            repo=source_data.get("repo"),
            path=source_data.get("path"),
            config=source_data.get("config"),
            revision=source_data.get("revision")
        )
        
        judges = []
        for judge_data in data.get("judges", []):
            if isinstance(judge_data, str):
                judges.append(JudgeConfig(name=judge_data, type=judge_data))
            elif isinstance(judge_data, dict):
                judge_name = list(judge_data.keys())[0]
                judge_config = judge_data[judge_name]
                if isinstance(judge_config, dict):
                    judges.append(JudgeConfig(
                        name=judge_name,
                        type=judge_name,
                        config=judge_config
                    ))
                else:
                    judges.append(JudgeConfig(name=judge_name, type=judge_name))
        
        return cls(
            id=data["id"],
            source=source,
            splits=data["splits"],
            task=data["task"],
            judges=judges,
            metrics=data["metrics"],
            taxonomy=data["taxonomy"],
            license=data["license"],
            version=data["version"],
            description=data.get("description"),
            citation=data.get("citation")
        )
    
    def load_dataset(self, split: str) -> Dataset:
        """Load the dataset for a specific split."""
        if split not in self.splits:
            raise ValueError(f"Split {split} not available for benchmark {self.id}")
        
        if self.source.type == "huggingface":
            return self._load_hf_dataset(split)
        elif self.source.type == "script":
            return self._load_script_dataset(split)
        elif self.source.type == "local":
            return self._load_local_dataset(split)
        else:
            raise ValueError(f"Unsupported source type: {self.source.type}")
    
    def _load_hf_dataset(self, split: str) -> Dataset:
        """Load dataset from HuggingFace."""
        kwargs = {}
        if self.source.config:
            kwargs["config"] = self.source.config
        if self.source.revision:
            kwargs["revision"] = self.source.revision
        
        dataset = load_dataset(
            self.source.repo,
            split=split,
            **kwargs
        )
        
        # Apply any necessary preprocessing
        return self._preprocess_dataset(dataset)
    
    def _load_script_dataset(self, split: str) -> Dataset:
        """Load dataset from a script."""
        # This would handle custom dataset loading scripts
        # For now, we'll raise an error
        raise NotImplementedError("Script-based datasets not yet implemented")
    
    def _load_local_dataset(self, split: str) -> Dataset:
        """Load dataset from local files."""
        if not self.source.path:
            raise ValueError("Local dataset requires a path")
        
        data_path = Path(self.source.path)
        if not data_path.exists():
            raise FileNotFoundError(f"Dataset path not found: {data_path}")
        
        # Load based on file extension
        if data_path.suffix == ".jsonl":
            return Dataset.from_json(str(data_path))
        elif data_path.suffix == ".csv":
            return Dataset.from_csv(str(data_path))
        elif data_path.suffix == ".parquet":
            return Dataset.from_parquet(str(data_path))
        else:
            raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Apply benchmark-specific preprocessing."""
        # This is a placeholder for dataset preprocessing
        # Different benchmarks may need different preprocessing steps
        return dataset
    
    def get_judge_configs(self) -> List[JudgeConfig]:
        """Get the judge configurations for this benchmark."""
        return self.judges
    
    def get_metrics(self) -> List[str]:
        """Get the metrics for this benchmark."""
        return self.metrics
    
    def get_taxonomy(self) -> List[str]:
        """Get the taxonomy categories for this benchmark."""
        return self.taxonomy


def load_benchmark(benchmark_id: str) -> Benchmark:
    """
    Load a benchmark from the registry.
    
    Args:
        benchmark_id: The ID of the benchmark to load
        
    Returns:
        Benchmark instance
        
    Raises:
        FileNotFoundError: If the benchmark is not found in the registry
    """
    return Benchmark.load(benchmark_id)


def list_benchmarks() -> List[str]:
    """List all available benchmarks in the registry."""
    registry_path = Path(__file__).parent.parent.parent.parent / "benchmarks"
    if not registry_path.exists():
        return []
    
    benchmarks = []
    for yaml_file in registry_path.glob("*.yaml"):
        benchmarks.append(yaml_file.stem)
    
    return sorted(benchmarks)


def filter_benchmarks(taxonomy: Optional[List[str]] = None, 
                     task: Optional[str] = None) -> List[str]:
    """
    Filter benchmarks by taxonomy and/or task.
    
    Args:
        taxonomy: List of taxonomy categories to filter by
        task: Task type to filter by
        
    Returns:
        List of benchmark IDs that match the filter
    """
    all_benchmarks = list_benchmarks()
    filtered = []
    
    for benchmark_id in all_benchmarks:
        try:
            benchmark = load_benchmark(benchmark_id)
            
            # Filter by taxonomy
            if taxonomy and not any(t in benchmark.taxonomy for t in taxonomy):
                continue
            
            # Filter by task
            if task and benchmark.task != task:
                continue
            
            filtered.append(benchmark_id)
            
        except Exception as e:
            logger.warning(f"Failed to load benchmark {benchmark_id}: {e}")
            continue
    
    return filtered

