"""
Rule-based guard for hard filtering.
"""

import logging
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import yaml

from .llama_guard import GuardResult

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """A filtering rule."""
    name: str
    pattern: str
    category: str
    severity: str = "medium"
    description: Optional[str] = None
    enabled: bool = True


class RuleGuard:
    """Rule-based guard for hard filtering."""
    
    def __init__(self, rules: Optional[List[Rule]] = None):
        """
        Initialize RuleGuard.
        
        Args:
            rules: List of filtering rules
        """
        self.rules = rules or []
        self._compiled_patterns = {}
        self._compile_patterns()
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RuleGuard":
        """
        Create RuleGuard from YAML file.
        
        Args:
            yaml_path: Path to YAML file with rules
            
        Returns:
            RuleGuard instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Rules file not found: {yaml_path}")
        
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        
        rules = []
        for rule_data in data.get("rules", []):
            rule = Rule(
                name=rule_data["name"],
                pattern=rule_data["pattern"],
                category=rule_data["category"],
                severity=rule_data.get("severity", "medium"),
                description=rule_data.get("description"),
                enabled=rule_data.get("enabled", True)
            )
            rules.append(rule)
        
        return cls(rules)
    
    def add_rule(self, rule: Rule):
        """Add a rule to the guard."""
        self.rules.append(rule)
        self._compile_patterns()
    
    def remove_rule(self, rule_name: str):
        """Remove a rule by name."""
        self.rules = [r for r in self.rules if r.name != rule_name]
        self._compile_patterns()
    
    def enable_rule(self, rule_name: str):
        """Enable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = True
                break
    
    def disable_rule(self, rule_name: str):
        """Disable a rule by name."""
        for rule in self.rules:
            if rule.name == rule_name:
                rule.enabled = False
                break
    
    def check_prompt(self, prompt: str) -> GuardResult:
        """
        Check a prompt against all rules.
        
        Args:
            prompt: Input prompt to check
            
        Returns:
            GuardResult with rule violations
        """
        violations = []
        
        for rule in self.rules:
            if not rule.enabled:
                continue
            
            if rule.name in self._compiled_patterns:
                pattern = self._compiled_patterns[rule.name]
                if pattern.search(prompt):
                    violations.append({
                        "rule": rule.name,
                        "category": rule.category,
                        "severity": rule.severity,
                        "description": rule.description
                    })
        
        if violations:
            # Return the most severe violation
            severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            most_severe = max(violations, key=lambda v: severity_order.get(v["severity"], 0))
            
            return GuardResult(
                is_safe=False,
                category=most_severe["category"],
                confidence=1.0,  # Rule-based detection is deterministic
                rationale=f"Rule violation: {most_severe['rule']} - {most_severe['description'] or 'Pattern matched'}",
                metadata={"violations": violations}
            )
        
        return GuardResult(is_safe=True, confidence=1.0)
    
    def check_response(self, prompt: str, response: str) -> GuardResult:
        """
        Check a response against all rules.
        
        Args:
            prompt: Original prompt
            response: Model response to check
            
        Returns:
            GuardResult with rule violations
        """
        # Check both prompt and response
        prompt_result = self.check_prompt(prompt)
        if not prompt_result.is_safe:
            return prompt_result
        
        response_result = self.check_prompt(response)
        if not response_result.is_safe:
            return response_result
        
        return GuardResult(is_safe=True, confidence=1.0)
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self._compiled_patterns = {}
        
        for rule in self.rules:
            if rule.enabled:
                try:
                    self._compiled_patterns[rule.name] = re.compile(rule.pattern, re.IGNORECASE)
                except re.error as e:
                    logger.warning(f"Invalid regex pattern for rule {rule.name}: {e}")
    
    def to_yaml(self, output_path: str):
        """Save rules to YAML file."""
        data = {
            "rules": [
                {
                    "name": rule.name,
                    "pattern": rule.pattern,
                    "category": rule.category,
                    "severity": rule.severity,
                    "description": rule.description,
                    "enabled": rule.enabled
                }
                for rule in self.rules
            ]
        }
        
        with open(output_path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)


# Predefined rule sets
def get_mlc_taxonomy_rules() -> List[Rule]:
    """Get MLCommons taxonomy-based rules."""
    return [
        Rule(
            name="violence_keywords",
            pattern=r"\b(kill|murder|assassinate|bomb|explode|shoot|gun|weapon)\b",
            category="violence",
            severity="high",
            description="Violence-related keywords"
        ),
        Rule(
            name="hate_speech_keywords",
            pattern=r"\b(hate|racist|sexist|bigot|discriminate)\b",
            category="hate_speech",
            severity="high",
            description="Hate speech keywords"
        ),
        Rule(
            name="sexual_content_keywords",
            pattern=r"\b(sex|porn|nude|explicit|adult)\b",
            category="sexual_content",
            severity="high",
            description="Sexual content keywords"
        ),
        Rule(
            name="self_harm_keywords",
            pattern=r"\b(suicide|self-harm|kill myself|end my life)\b",
            category="self_harm",
            severity="critical",
            description="Self-harm keywords"
        ),
        Rule(
            name="financial_advice_keywords",
            pattern=r"\b(invest|stock|trading|financial advice|money advice)\b",
            category="financial_advice",
            severity="medium",
            description="Financial advice keywords"
        ),
        Rule(
            name="health_advice_keywords",
            pattern=r"\b(medical advice|health advice|diagnosis|treatment|medicine)\b",
            category="health_advice",
            severity="high",
            description="Health advice keywords"
        ),
        Rule(
            name="legal_advice_keywords",
            pattern=r"\b(legal advice|lawyer|attorney|legal opinion|court)\b",
            category="legal_advice",
            severity="medium",
            description="Legal advice keywords"
        ),
        Rule(
            name="privacy_keywords",
            pattern=r"\b(personal data|private information|ssn|credit card|password)\b",
            category="privacy",
            severity="medium",
            description="Privacy-related keywords"
        ),
    ]


def get_basic_safety_rules() -> List[Rule]:
    """Get basic safety rules."""
    return [
        Rule(
            name="profanity",
            pattern=r"\b(fuck|shit|bitch|asshole|damn)\b",
            category="harassment",
            severity="low",
            description="Profanity detection"
        ),
        Rule(
            name="threats",
            pattern=r"\b(threaten|harm you|hurt you|attack you)\b",
            category="violence",
            severity="high",
            description="Threat detection"
        ),
        Rule(
            name="personal_info_request",
            pattern=r"\b(your name|your address|your phone|your email|your location)\b",
            category="privacy",
            severity="medium",
            description="Personal information request"
        ),
    ]

