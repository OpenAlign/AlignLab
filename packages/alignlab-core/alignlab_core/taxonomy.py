"""
Taxonomy management for alignment evaluation categories.
"""

import logging
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


@dataclass
class TaxonomyCategory:
    """A taxonomy category definition."""
    id: str
    name: str
    description: str
    parent: Optional[str] = None
    children: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    severity: Optional[str] = None  # "low", "medium", "high", "critical"


class Taxonomy:
    """Manages taxonomy categories for alignment evaluation."""
    
    # MLCommons 13-category taxonomy
    MLC_TAXONOMY = {
        "violence": {
            "name": "Violence",
            "description": "Content promoting or describing violence, harm, or dangerous activities",
            "severity": "high",
            "examples": ["instructions for making weapons", "threats of violence", "glorification of harm"]
        },
        "hate_speech": {
            "name": "Hate Speech",
            "description": "Content that promotes hatred, discrimination, or violence against groups",
            "severity": "high",
            "examples": ["racial slurs", "discriminatory language", "hateful rhetoric"]
        },
        "sexual_content": {
            "name": "Sexual Content",
            "description": "Explicit sexual content, pornography, or inappropriate sexual material",
            "severity": "high",
            "examples": ["explicit sexual descriptions", "pornographic content", "sexual harassment"]
        },
        "self_harm": {
            "name": "Self-Harm",
            "description": "Content promoting self-harm, suicide, or dangerous behaviors",
            "severity": "critical",
            "examples": ["suicide instructions", "self-harm encouragement", "dangerous challenges"]
        },
        "harassment": {
            "name": "Harassment",
            "description": "Content that harasses, bullies, or intimidates individuals",
            "severity": "medium",
            "examples": ["cyberbullying", "personal attacks", "stalking behavior"]
        },
        "privacy": {
            "name": "Privacy",
            "description": "Content that violates privacy or contains personal information",
            "severity": "medium",
            "examples": ["personal data exposure", "privacy violations", "doxxing"]
        },
        "misinformation": {
            "name": "Misinformation",
            "description": "False or misleading information that could cause harm",
            "severity": "medium",
            "examples": ["conspiracy theories", "false medical advice", "fake news"]
        },
        "financial_advice": {
            "name": "Financial Advice",
            "description": "Potentially harmful financial advice or investment recommendations",
            "severity": "medium",
            "examples": ["investment scams", "financial fraud", "risky trading advice"]
        },
        "legal_advice": {
            "name": "Legal Advice",
            "description": "Legal advice that could be misleading or harmful",
            "severity": "medium",
            "examples": ["false legal information", "misleading legal advice", "legal fraud"]
        },
        "health_advice": {
            "name": "Health Advice",
            "description": "Medical or health advice that could be harmful",
            "severity": "high",
            "examples": ["false medical treatments", "dangerous health practices", "medical misinformation"]
        },
        "government_decision": {
            "name": "Government Decision",
            "description": "Content related to government decisions or political manipulation",
            "severity": "medium",
            "examples": ["election interference", "political manipulation", "government misinformation"]
        },
        "fraud": {
            "name": "Fraud",
            "description": "Content promoting fraud, scams, or deceptive practices",
            "severity": "high",
            "examples": ["scam instructions", "fraudulent schemes", "deceptive practices"]
        },
        "disinformation": {
            "name": "Disinformation",
            "description": "Deliberately false information spread to deceive",
            "severity": "high",
            "examples": ["propaganda", "coordinated disinformation", "fake news campaigns"]
        }
    }
    
    def __init__(self, taxonomy_file: Optional[Path] = None):
        """
        Initialize taxonomy.
        
        Args:
            taxonomy_file: Optional path to custom taxonomy YAML file
        """
        self.categories: Dict[str, TaxonomyCategory] = {}
        self.benchmark_mappings: Dict[str, List[str]] = {}
        
        if taxonomy_file and taxonomy_file.exists():
            self.load_from_file(taxonomy_file)
        else:
            self.load_mlc_taxonomy()
    
    def load_mlc_taxonomy(self):
        """Load the MLCommons 13-category taxonomy."""
        for category_id, category_data in self.MLC_TAXONOMY.items():
            self.categories[category_id] = TaxonomyCategory(
                id=category_id,
                name=category_data["name"],
                description=category_data["description"],
                severity=category_data["severity"],
                examples=category_data["examples"]
            )
    
    def load_from_file(self, taxonomy_file: Path):
        """Load taxonomy from YAML file."""
        with open(taxonomy_file, "r") as f:
            data = yaml.safe_load(f)
        
        # Load categories
        for category_data in data.get("categories", []):
            category = TaxonomyCategory(
                id=category_data["id"],
                name=category_data["name"],
                description=category_data["description"],
                parent=category_data.get("parent"),
                children=category_data.get("children", []),
                examples=category_data.get("examples", []),
                severity=category_data.get("severity")
            )
            self.categories[category.id] = category
        
        # Load benchmark mappings
        self.benchmark_mappings = data.get("benchmark_mappings", {})
    
    def get_category(self, category_id: str) -> Optional[TaxonomyCategory]:
        """Get a category by ID."""
        return self.categories.get(category_id)
    
    def get_categories_by_severity(self, severity: str) -> List[TaxonomyCategory]:
        """Get all categories with a specific severity level."""
        return [
            category for category in self.categories.values()
            if category.severity == severity
        ]
    
    def get_children(self, category_id: str) -> List[TaxonomyCategory]:
        """Get child categories of a given category."""
        category = self.get_category(category_id)
        if not category:
            return []
        
        return [
            self.categories[child_id] for child_id in category.children
            if child_id in self.categories
        ]
    
    def get_parent(self, category_id: str) -> Optional[TaxonomyCategory]:
        """Get parent category of a given category."""
        category = self.get_category(category_id)
        if not category or not category.parent:
            return None
        
        return self.categories.get(category.parent)
    
    def get_ancestors(self, category_id: str) -> List[TaxonomyCategory]:
        """Get all ancestor categories of a given category."""
        ancestors = []
        current = self.get_parent(category_id)
        
        while current:
            ancestors.append(current)
            current = self.get_parent(current.id)
        
        return ancestors
    
    def get_descendants(self, category_id: str) -> List[TaxonomyCategory]:
        """Get all descendant categories of a given category."""
        descendants = []
        category = self.get_category(category_id)
        
        if not category:
            return descendants
        
        def collect_descendants(cat_id: str):
            children = self.get_children(cat_id)
            for child in children:
                descendants.append(child)
                collect_descendants(child.id)
        
        collect_descendants(category_id)
        return descendants
    
    def map_benchmark_labels(self, benchmark_id: str, labels: List[str]) -> List[str]:
        """
        Map benchmark-specific labels to taxonomy categories.
        
        Args:
            benchmark_id: ID of the benchmark
            labels: List of benchmark-specific labels
            
        Returns:
            List of taxonomy category IDs
        """
        if benchmark_id not in self.benchmark_mappings:
            return labels  # Return original labels if no mapping exists
        
        mapping = self.benchmark_mappings[benchmark_id]
        mapped_categories = []
        
        for label in labels:
            if label in mapping:
                mapped_categories.append(mapping[label])
            else:
                # Try to find a match by name
                for category in self.categories.values():
                    if category.name.lower() == label.lower():
                        mapped_categories.append(category.id)
                        break
                else:
                    # If no match found, keep original label
                    mapped_categories.append(label)
        
        return mapped_categories
    
    def get_benchmark_coverage(self, benchmark_id: str) -> Dict[str, List[str]]:
        """
        Get taxonomy coverage for a benchmark.
        
        Args:
            benchmark_id: ID of the benchmark
            
        Returns:
            Dictionary mapping taxonomy categories to benchmark labels
        """
        if benchmark_id not in self.benchmark_mappings:
            return {}
        
        coverage = {}
        mapping = self.benchmark_mappings[benchmark_id]
        
        # Reverse the mapping
        for benchmark_label, taxonomy_category in mapping.items():
            if taxonomy_category not in coverage:
                coverage[taxonomy_category] = []
            coverage[taxonomy_category].append(benchmark_label)
        
        return coverage
    
    def get_severity_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Get distribution of severity levels in evaluation results.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary mapping severity levels to counts
        """
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        
        for result in results:
            # Extract taxonomy categories from result
            categories = result.get("metadata", {}).get("taxonomy", [])
            
            for category_id in categories:
                category = self.get_category(category_id)
                if category and category.severity:
                    severity_counts[category.severity] += 1
        
        return severity_counts
    
    def get_category_breakdown(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Get detailed breakdown by taxonomy category.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary mapping category IDs to metrics
        """
        category_metrics = {}
        
        for result in results:
            categories = result.get("metadata", {}).get("taxonomy", [])
            score = result.get("score", 0.0)
            
            for category_id in categories:
                if category_id not in category_metrics:
                    category_metrics[category_id] = {
                        "count": 0,
                        "scores": [],
                        "severity": self.get_category(category_id).severity if self.get_category(category_id) else None
                    }
                
                category_metrics[category_id]["count"] += 1
                category_metrics[category_id]["scores"].append(score)
        
        # Compute statistics for each category
        for category_id, metrics in category_metrics.items():
            scores = metrics["scores"]
            if scores:
                metrics["mean_score"] = sum(scores) / len(scores)
                metrics["min_score"] = min(scores)
                metrics["max_score"] = max(scores)
            else:
                metrics["mean_score"] = 0.0
                metrics["min_score"] = 0.0
                metrics["max_score"] = 0.0
        
        return category_metrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert taxonomy to dictionary representation."""
        return {
            "categories": {
                cat_id: {
                    "id": cat.id,
                    "name": cat.name,
                    "description": cat.description,
                    "parent": cat.parent,
                    "children": cat.children,
                    "examples": cat.examples,
                    "severity": cat.severity
                }
                for cat_id, cat in self.categories.items()
            },
            "benchmark_mappings": self.benchmark_mappings
        }
    
    def save(self, output_file: Path):
        """Save taxonomy to YAML file."""
        with open(output_file, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)


# Global taxonomy instance
_global_taxonomy = None


def get_taxonomy() -> Taxonomy:
    """Get the global taxonomy instance."""
    global _global_taxonomy
    if _global_taxonomy is None:
        _global_taxonomy = Taxonomy()
    return _global_taxonomy


def set_taxonomy(taxonomy: Taxonomy):
    """Set the global taxonomy instance."""
    global _global_taxonomy
    _global_taxonomy = taxonomy

