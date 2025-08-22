"""
Ensemble guard for combining multiple safety filters.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
import numpy as np
from collections import Counter

from .llama_guard import GuardResult, LlamaGuard
from .rule_guard import RuleGuard

logger = logging.getLogger(__name__)


@dataclass
class GuardVote:
    """A vote from a guard."""
    guard_name: str
    is_safe: bool
    confidence: float
    category: Optional[str] = None
    rationale: Optional[str] = None


class EnsembleGuard:
    """Ensemble of multiple guards with voting and calibration."""
    
    def __init__(self, guards: List[Union[LlamaGuard, RuleGuard]], 
                 voting_strategy: str = "majority",
                 confidence_threshold: float = 0.5):
        """
        Initialize EnsembleGuard.
        
        Args:
            guards: List of guard instances
            voting_strategy: Voting strategy ("majority", "weighted", "unanimous")
            confidence_threshold: Minimum confidence for safe classification
        """
        self.guards = guards
        self.voting_strategy = voting_strategy
        self.confidence_threshold = confidence_threshold
        self.guard_weights = {}  # For weighted voting
        self.calibration_data = []  # For calibration
        
        logger.info(f"Initialized ensemble with {len(guards)} guards using {voting_strategy} voting")
    
    def add_guard(self, guard: Union[LlamaGuard, RuleGuard], weight: float = 1.0):
        """Add a guard to the ensemble."""
        self.guards.append(guard)
        self.guard_weights[guard.__class__.__name__] = weight
    
    def remove_guard(self, guard: Union[LlamaGuard, RuleGuard]):
        """Remove a guard from the ensemble."""
        if guard in self.guards:
            self.guards.remove(guard)
            if guard.__class__.__name__ in self.guard_weights:
                del self.guard_weights[guard.__class__.__name__]
    
    def set_guard_weight(self, guard_name: str, weight: float):
        """Set weight for a guard type."""
        self.guard_weights[guard_name] = weight
    
    def classify_prompt(self, prompt: str) -> GuardResult:
        """
        Classify a prompt using all guards.
        
        Args:
            prompt: Input prompt to classify
            
        Returns:
            GuardResult with ensemble decision
        """
        votes = []
        
        for guard in self.guards:
            try:
                if isinstance(guard, LlamaGuard):
                    result = guard.classify_prompt(prompt)
                elif isinstance(guard, RuleGuard):
                    result = guard.check_prompt(prompt)
                else:
                    logger.warning(f"Unknown guard type: {type(guard)}")
                    continue
                
                votes.append(GuardVote(
                    guard_name=guard.__class__.__name__,
                    is_safe=result.is_safe,
                    confidence=result.confidence or 0.5,
                    category=result.category,
                    rationale=result.rationale
                ))
                
            except Exception as e:
                logger.error(f"Error in guard {guard.__class__.__name__}: {e}")
                continue
        
        return self._combine_votes(votes, "prompt")
    
    def classify_response(self, prompt: str, response: str) -> GuardResult:
        """
        Classify a response using all guards.
        
        Args:
            prompt: Original prompt
            response: Model response to classify
            
        Returns:
            GuardResult with ensemble decision
        """
        votes = []
        
        for guard in self.guards:
            try:
                if isinstance(guard, LlamaGuard):
                    result = guard.classify_response(prompt, response)
                elif isinstance(guard, RuleGuard):
                    result = guard.check_response(prompt, response)
                else:
                    logger.warning(f"Unknown guard type: {type(guard)}")
                    continue
                
                votes.append(GuardVote(
                    guard_name=guard.__class__.__name__,
                    is_safe=result.is_safe,
                    confidence=result.confidence or 0.5,
                    category=result.category,
                    rationale=result.rationale
                ))
                
            except Exception as e:
                logger.error(f"Error in guard {guard.__class__.__name__}: {e}")
                continue
        
        return self._combine_votes(votes, "response")
    
    def _combine_votes(self, votes: List[GuardVote], context: str) -> GuardResult:
        """Combine votes from all guards."""
        if not votes:
            return GuardResult(
                is_safe=True,
                confidence=0.0,
                rationale="No guards available"
            )
        
        if self.voting_strategy == "majority":
            return self._majority_vote(votes, context)
        elif self.voting_strategy == "weighted":
            return self._weighted_vote(votes, context)
        elif self.voting_strategy == "unanimous":
            return self._unanimous_vote(votes, context)
        else:
            raise ValueError(f"Unknown voting strategy: {self.voting_strategy}")
    
    def _majority_vote(self, votes: List[GuardVote], context: str) -> GuardResult:
        """Majority voting strategy."""
        safe_votes = sum(1 for vote in votes if vote.is_safe)
        total_votes = len(votes)
        
        is_safe = safe_votes > total_votes / 2
        
        # Calculate average confidence
        avg_confidence = np.mean([vote.confidence for vote in votes])
        
        # Get most common category among unsafe votes
        unsafe_categories = [vote.category for vote in votes if not vote.is_safe and vote.category]
        category = Counter(unsafe_categories).most_common(1)[0][0] if unsafe_categories else None
        
        # Combine rationales
        rationales = [vote.rationale for vote in votes if vote.rationale]
        combined_rationale = "; ".join(rationales) if rationales else None
        
        return GuardResult(
            is_safe=is_safe,
            category=category,
            confidence=avg_confidence,
            rationale=combined_rationale,
            metadata={
                "voting_strategy": "majority",
                "safe_votes": safe_votes,
                "total_votes": total_votes,
                "votes": [{"guard": v.guard_name, "safe": v.is_safe, "confidence": v.confidence} for v in votes]
            }
        )
    
    def _weighted_vote(self, votes: List[GuardVote], context: str) -> GuardResult:
        """Weighted voting strategy."""
        total_weight = 0
        weighted_safe_score = 0
        weighted_confidence = 0
        
        for vote in votes:
            weight = self.guard_weights.get(vote.guard_name, 1.0)
            total_weight += weight
            
            if vote.is_safe:
                weighted_safe_score += weight
            else:
                weighted_safe_score -= weight
            
            weighted_confidence += vote.confidence * weight
        
        is_safe = weighted_safe_score > 0
        avg_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.0
        
        # Get most common category among unsafe votes
        unsafe_categories = [vote.category for vote in votes if not vote.is_safe and vote.category]
        category = Counter(unsafe_categories).most_common(1)[0][0] if unsafe_categories else None
        
        # Combine rationales
        rationales = [vote.rationale for vote in votes if vote.rationale]
        combined_rationale = "; ".join(rationales) if rationales else None
        
        return GuardResult(
            is_safe=is_safe,
            category=category,
            confidence=avg_confidence,
            rationale=combined_rationale,
            metadata={
                "voting_strategy": "weighted",
                "weighted_safe_score": weighted_safe_score,
                "total_weight": total_weight,
                "votes": [{"guard": v.guard_name, "safe": v.is_safe, "confidence": v.confidence, "weight": self.guard_weights.get(v.guard_name, 1.0)} for v in votes]
            }
        )
    
    def _unanimous_vote(self, votes: List[GuardVote], context: str) -> GuardResult:
        """Unanimous voting strategy (all guards must agree)."""
        all_safe = all(vote.is_safe for vote in votes)
        all_unsafe = all(not vote.is_safe for vote in votes)
        
        if all_safe:
            is_safe = True
        elif all_unsafe:
            is_safe = False
        else:
            # Mixed votes - use majority as tiebreaker
            safe_votes = sum(1 for vote in votes if vote.is_safe)
            is_safe = safe_votes > len(votes) / 2
        
        # Calculate average confidence
        avg_confidence = np.mean([vote.confidence for vote in votes])
        
        # Get most common category among unsafe votes
        unsafe_categories = [vote.category for vote in votes if not vote.is_safe and vote.category]
        category = Counter(unsafe_categories).most_common(1)[0][0] if unsafe_categories else None
        
        # Combine rationales
        rationales = [vote.rationale for vote in votes if vote.rationale]
        combined_rationale = "; ".join(rationales) if rationales else None
        
        return GuardResult(
            is_safe=is_safe,
            category=category,
            confidence=avg_confidence,
            rationale=combined_rationale,
            metadata={
                "voting_strategy": "unanimous",
                "all_safe": all_safe,
                "all_unsafe": all_unsafe,
                "votes": [{"guard": v.guard_name, "safe": v.is_safe, "confidence": v.confidence} for v in votes]
            }
        )
    
    def calibrate(self, calibration_data: List[Dict[str, Any]]):
        """
        Calibrate the ensemble using labeled data.
        
        Args:
            calibration_data: List of dicts with keys: prompt, response, is_safe
        """
        self.calibration_data = calibration_data
        
        # Calculate guard performance
        guard_performance = {}
        
        for guard in self.guards:
            guard_name = guard.__class__.__name__
            correct = 0
            total = 0
            
            for item in calibration_data:
                try:
                    if "response" in item:
                        result = guard.classify_response(item["prompt"], item["response"])
                    else:
                        result = guard.classify_prompt(item["prompt"])
                    
                    if result.is_safe == item["is_safe"]:
                        correct += 1
                    total += 1
                    
                except Exception as e:
                    logger.warning(f"Error calibrating guard {guard_name}: {e}")
                    continue
            
            if total > 0:
                accuracy = correct / total
                guard_performance[guard_name] = accuracy
                logger.info(f"Guard {guard_name} accuracy: {accuracy:.3f}")
        
        # Update weights based on performance
        if guard_performance:
            max_accuracy = max(guard_performance.values())
            for guard_name, accuracy in guard_performance.items():
                # Weight proportional to accuracy
                weight = accuracy / max_accuracy if max_accuracy > 0 else 1.0
                self.guard_weights[guard_name] = weight
                logger.info(f"Updated weight for {guard_name}: {weight:.3f}")
    
    def get_guard_agreement(self, prompt: str, response: Optional[str] = None) -> Dict[str, Any]:
        """
        Get agreement statistics between guards.
        
        Args:
            prompt: Input prompt
            response: Optional response
            
        Returns:
            Dictionary with agreement statistics
        """
        votes = []
        
        for guard in self.guards:
            try:
                if response:
                    result = guard.classify_response(prompt, response)
                else:
                    result = guard.classify_prompt(prompt)
                
                votes.append(GuardVote(
                    guard_name=guard.__class__.__name__,
                    is_safe=result.is_safe,
                    confidence=result.confidence or 0.5,
                    category=result.category
                ))
                
            except Exception as e:
                logger.error(f"Error in guard {guard.__class__.__name__}: {e}")
                continue
        
        if not votes:
            return {"agreement": 0.0, "num_guards": 0}
        
        # Calculate agreement
        safe_votes = sum(1 for vote in votes if vote.is_safe)
        total_votes = len(votes)
        
        # Agreement is the proportion of votes that agree with the majority
        majority_safe = safe_votes > total_votes / 2
        agreeing_votes = sum(1 for vote in votes if vote.is_safe == majority_safe)
        agreement = agreeing_votes / total_votes
        
        return {
            "agreement": agreement,
            "num_guards": total_votes,
            "safe_votes": safe_votes,
            "unsafe_votes": total_votes - safe_votes,
            "majority_decision": "safe" if majority_safe else "unsafe",
            "votes": [{"guard": v.guard_name, "safe": v.is_safe, "confidence": v.confidence} for v in votes]
        }

