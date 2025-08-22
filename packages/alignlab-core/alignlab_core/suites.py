"""
Benchmark suites for running multiple evaluations together.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import yaml

from .benchmark import Benchmark

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkSuite:
    """A suite of benchmarks to run together."""
    id: str
    name: str
    description: str
    benchmarks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    version: str = "1.0.0"
    
    @classmethod
    def load(cls, suite_id: str) -> "BenchmarkSuite":
        """Load a suite from the registry."""
        registry_path = Path(__file__).parent.parent.parent.parent / "benchmarks" / "suites"
        suite_file = registry_path / f"{suite_id}.yaml"
        
        if not suite_file.exists():
            raise FileNotFoundError(f"Suite {suite_id} not found in registry")
        
        with open(suite_file, "r") as f:
            data = yaml.safe_load(f)
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BenchmarkSuite":
        """Create a suite from a dictionary."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data["description"],
            benchmarks=data["benchmarks"],
            metadata=data.get("metadata", {}),
            version=data.get("version", "1.0.0")
        )
    
    def get_benchmarks(self) -> List[Benchmark]:
        """Load all benchmarks in the suite."""
        benchmarks = []
        for benchmark_id in self.benchmarks:
            try:
                benchmark = Benchmark.load(benchmark_id)
                benchmarks.append(benchmark)
            except Exception as e:
                logger.warning(f"Failed to load benchmark {benchmark_id}: {e}")
                continue
        return benchmarks
    
    def get_taxonomy_coverage(self) -> List[str]:
        """Get all taxonomy categories covered by this suite."""
        taxonomy_categories = set()
        
        for benchmark in self.get_benchmarks():
            taxonomy_categories.update(benchmark.taxonomy)
        
        return sorted(list(taxonomy_categories))


# Predefined suites
SAFETY_CORE_V1 = BenchmarkSuite(
    id="alignlab:safety_core_v1",
    name="Safety Core v1",
    description="Comprehensive safety evaluation suite covering harm, toxicity, truthfulness, and bias",
    benchmarks=[
        "harmbench",
        "jailbreakbench",
        "ultrasafety",
        "realtoxicityprompts",
        "polyglotoxicityprompts",
        "truthfulqa",
        "bbq",
        "crowspairs",
        "stereoset",
        "multilingual_holisticbias"
    ],
    metadata={
        "taxonomy": ["violence", "hate_speech", "sexual_content", "self_harm", "harassment", 
                    "privacy", "misinformation", "fraud", "disinformation"],
        "languages": ["en", "multilingual"],
        "severity_levels": ["low", "medium", "high", "critical"]
    },
    version="1.0.0"
)

AGENT_ROBUSTNESS_V1 = BenchmarkSuite(
    id="alignlab:agent_robustness_v1",
    name="Agent Robustness v1",
    description="Agentic evaluation suite for jailbreak robustness and prompt injection",
    benchmarks=[
        "jailbreakbench",
        "harmbench_agentic",
        "prompt_injection_battery"
    ],
    metadata={
        "taxonomy": ["violence", "hate_speech", "fraud", "privacy"],
        "agent_capabilities": ["tool_use", "web_browsing", "file_io"],
        "attack_types": ["jailbreak", "prompt_injection", "role_play"]
    },
    version="1.0.0"
)

TRUTHFULNESS_V1 = BenchmarkSuite(
    id="alignlab:truthfulness_v1",
    name="Truthfulness v1",
    description="Truthfulness and hallucination evaluation suite",
    benchmarks=[
        "truthfulqa",
        "truthfulqa_multilingual",
        "factual_consistency"
    ],
    metadata={
        "taxonomy": ["misinformation", "disinformation"],
        "evaluation_types": ["factual_accuracy", "hallucination_detection", "consistency"]
    },
    version="1.0.0"
)

BIAS_FAIRNESS_V1 = BenchmarkSuite(
    id="alignlab:bias_fairness_v1",
    name="Bias and Fairness v1",
    description="Bias and fairness evaluation suite",
    benchmarks=[
        "bbq",
        "crowspairs",
        "stereoset",
        "multilingual_holisticbias",
        "gender_bias",
        "racial_bias"
    ],
    metadata={
        "taxonomy": ["hate_speech", "harassment"],
        "bias_types": ["gender", "racial", "religious", "age", "disability"],
        "languages": ["en", "multilingual"]
    },
    version="1.0.0"
)

TOXICITY_V1 = BenchmarkSuite(
    id="alignlab:toxicity_v1",
    name="Toxicity v1",
    description="Toxicity evaluation suite",
    benchmarks=[
        "realtoxicityprompts",
        "polyglotoxicityprompts",
        "hate_speech_detection",
        "offensive_language"
    ],
    metadata={
        "taxonomy": ["hate_speech", "harassment"],
        "toxicity_types": ["hate_speech", "offensive_language", "harassment"],
        "languages": ["en", "multilingual"]
    },
    version="1.0.0"
)


# Registry of predefined suites
PREDEFINED_SUITES = {
    "alignlab:safety_core_v1": SAFETY_CORE_V1,
    "alignlab:agent_robustness_v1": AGENT_ROBUSTNESS_V1,
    "alignlab:truthfulness_v1": TRUTHFULNESS_V1,
    "alignlab:bias_fairness_v1": BIAS_FAIRNESS_V1,
    "alignlab:toxicity_v1": TOXICITY_V1,
}


def load_suite(suite_id: str) -> BenchmarkSuite:
    """
    Load a benchmark suite.
    
    Args:
        suite_id: ID of the suite to load
        
    Returns:
        BenchmarkSuite instance
        
    Raises:
        FileNotFoundError: If the suite is not found
    """
    # Check predefined suites first
    if suite_id in PREDEFINED_SUITES:
        return PREDEFINED_SUITES[suite_id]
    
    # Try to load from registry
    try:
        return BenchmarkSuite.load(suite_id)
    except FileNotFoundError:
        raise FileNotFoundError(f"Suite {suite_id} not found in predefined suites or registry")


def list_suites() -> List[str]:
    """List all available suites."""
    suites = list(PREDEFINED_SUITES.keys())
    
    # Add suites from registry
    registry_path = Path(__file__).parent.parent.parent.parent / "benchmarks" / "suites"
    if registry_path.exists():
        for yaml_file in registry_path.glob("*.yaml"):
            suite_id = yaml_file.stem
            if suite_id not in suites:
                suites.append(suite_id)
    
    return sorted(suites)


def filter_suites(taxonomy: Optional[List[str]] = None,
                  language: Optional[str] = None,
                  severity: Optional[str] = None) -> List[str]:
    """
    Filter suites by criteria.
    
    Args:
        taxonomy: List of taxonomy categories to filter by
        language: Language to filter by ("en", "multilingual")
        severity: Severity level to filter by
        
    Returns:
        List of suite IDs that match the filter
    """
    all_suites = list_suites()
    filtered = []
    
    for suite_id in all_suites:
        try:
            suite = load_suite(suite_id)
            
            # Filter by taxonomy
            if taxonomy:
                suite_taxonomy = suite.metadata.get("taxonomy", [])
                if not any(t in suite_taxonomy for t in taxonomy):
                    continue
            
            # Filter by language
            if language:
                suite_languages = suite.metadata.get("languages", [])
                if language not in suite_languages:
                    continue
            
            # Filter by severity
            if severity:
                suite_severities = suite.metadata.get("severity_levels", [])
                if severity not in suite_severities:
                    continue
            
            filtered.append(suite_id)
            
        except Exception as e:
            logger.warning(f"Failed to load suite {suite_id}: {e}")
            continue
    
    return filtered


def get_suite_summary(suite_id: str) -> Dict[str, Any]:
    """
    Get a summary of a suite.
    
    Args:
        suite_id: ID of the suite
        
    Returns:
        Dictionary with suite summary
    """
    suite = load_suite(suite_id)
    
    summary = {
        "id": suite.id,
        "name": suite.name,
        "description": suite.description,
        "version": suite.version,
        "num_benchmarks": len(suite.benchmarks),
        "benchmarks": suite.benchmarks,
        "metadata": suite.metadata,
        "taxonomy_coverage": suite.get_taxonomy_coverage()
    }
    
    return summary


def create_custom_suite(suite_id: str, name: str, description: str, 
                       benchmarks: List[str], metadata: Optional[Dict[str, Any]] = None) -> BenchmarkSuite:
    """
    Create a custom benchmark suite.
    
    Args:
        suite_id: Unique identifier for the suite
        name: Human-readable name
        description: Description of the suite
        benchmarks: List of benchmark IDs to include
        metadata: Optional metadata dictionary
        
    Returns:
        BenchmarkSuite instance
    """
    return BenchmarkSuite(
        id=suite_id,
        name=name,
        description=description,
        benchmarks=benchmarks,
        metadata=metadata or {},
        version="1.0.0"
    )


def save_suite(suite: BenchmarkSuite, output_path: Path):
    """
    Save a suite to YAML file.
    
    Args:
        suite: BenchmarkSuite to save
        output_path: Path to save the suite
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "id": suite.id,
        "name": suite.name,
        "description": suite.description,
        "benchmarks": suite.benchmarks,
        "metadata": suite.metadata,
        "version": suite.version
    }
    
    with open(output_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, indent=2)

