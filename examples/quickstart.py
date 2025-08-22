#!/usr/bin/env python3
"""
AlignLab Quickstart Example

This script demonstrates the basic usage of AlignLab for running evaluations.
"""

import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Run a simple evaluation example."""
    
    try:
        from alignlab_core import EvalRunner, load_benchmark
        
        logger.info("AlignLab Quickstart Example")
        logger.info("=" * 50)
        
        # Initialize the evaluation runner
        # Note: This requires a model to be available
        # For demonstration, we'll show the setup without actually running
        logger.info("1. Setting up evaluation runner...")
        logger.info("   (In practice, you would specify a real model)")
        
        # Example runner setup (commented out to avoid errors)
        # runner = EvalRunner(
        #     model="meta-llama/Llama-3.1-8B-Instruct",
        #     provider="hf"
        # )
        
        # Load a benchmark
        logger.info("2. Loading benchmark...")
        try:
            benchmark = load_benchmark("truthfulqa")
            logger.info(f"   Loaded benchmark: {benchmark.id}")
            logger.info(f"   Description: {benchmark.description}")
            logger.info(f"   Available splits: {benchmark.splits}")
            logger.info(f"   Judges: {[j.name for j in benchmark.judges]}")
        except FileNotFoundError:
            logger.warning("   Benchmark not found in registry (this is expected for the example)")
            logger.info("   Available benchmarks would be listed here")
        
        # Show available benchmarks
        logger.info("3. Available benchmarks:")
        try:
            from alignlab_core import list_benchmarks
            benchmarks = list_benchmarks()
            for bench_id in benchmarks[:5]:  # Show first 5
                logger.info(f"   - {bench_id}")
            if len(benchmarks) > 5:
                logger.info(f"   ... and {len(benchmarks) - 5} more")
        except Exception as e:
            logger.warning(f"   Could not list benchmarks: {e}")
        
        # Show available suites
        logger.info("4. Available suites:")
        try:
            from alignlab_core import list_suites
            suites = list_suites()
            for suite_id in suites:
                logger.info(f"   - {suite_id}")
        except Exception as e:
            logger.warning(f"   Could not list suites: {e}")
        
        # Demonstrate guard usage
        logger.info("5. Guard system:")
        try:
            from alignlab_guards import LlamaGuard, RuleGuard, EnsembleGuard
            
            logger.info("   - LlamaGuard: MLCommons-aligned safety filtering")
            logger.info("   - RuleGuard: Hard rule-based filtering")
            logger.info("   - EnsembleGuard: Combine multiple guards")
            
            # Example guard setup (commented out to avoid errors)
            # llama_guard = LlamaGuard("meta-llama/Llama-Guard-3-8B")
            # rule_guard = RuleGuard.from_yaml("mlc_taxonomy.yaml")
            # ensemble = EnsembleGuard([llama_guard, rule_guard])
            
        except ImportError:
            logger.warning("   Guards package not available")
        
        # Show CLI usage
        logger.info("6. CLI Usage:")
        logger.info("   # List benchmarks")
        logger.info("   alignlab benchmarks ls --filter safety,multilingual")
        logger.info("")
        logger.info("   # Run a safety evaluation")
        logger.info("   alignlab eval run --suite alignlab:safety_core_v1 \\")
        logger.info("     --model meta-llama/Llama-3.1-8B-Instruct \\")
        logger.info("     --provider hf --guards llama_guard_3 \\")
        logger.info("     --report out/safety_eval")
        logger.info("")
        logger.info("   # Generate reports")
        logger.info("   alignlab report build out/safety_eval --format html,pdf")
        
        logger.info("")
        logger.info("=" * 50)
        logger.info("Quickstart completed!")
        logger.info("")
        logger.info("Next steps:")
        logger.info("1. Install dependencies: pip install -e packages/alignlab-core")
        logger.info("2. Set up a model (HuggingFace, OpenAI, etc.)")
        logger.info("3. Run your first evaluation")
        logger.info("4. Check out the documentation for more examples")
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure you have installed the AlignLab packages:")
        logger.error("  pip install -e packages/alignlab-core")
        logger.error("  pip install -e packages/alignlab-cli")
        logger.error("  pip install -e packages/alignlab-guards")
    
    except Exception as e:
        logger.error(f"Error in quickstart: {e}")
        raise


if __name__ == "__main__":
    main()
