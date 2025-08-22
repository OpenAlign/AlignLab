"""
Main CLI entry point for AlignLab.
"""

import click
import logging
from pathlib import Path
from typing import Optional, List

from alignlab_core import EvalRunner, load_benchmark, list_benchmarks, filter_benchmarks
from alignlab_core import load_suite, list_suites, filter_suites
from alignlab_core import EvalResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.group()
@click.version_option()
def cli():
    """AlignLab - Comprehensive alignment evaluation framework."""
    pass


@cli.group()
def benchmarks():
    """Manage benchmarks."""
    pass


@benchmarks.command("ls")
@click.option("--filter", "filters", multiple=True, help="Filter by taxonomy categories")
@click.option("--task", help="Filter by task type")
def list_benchmarks_cmd(filters, task):
    """List available benchmarks."""
    if filters:
        benchmark_ids = filter_benchmarks(taxonomy=list(filters), task=task)
    else:
        benchmark_ids = list_benchmarks()
    
    if not benchmark_ids:
        click.echo("No benchmarks found.")
        return
    
    click.echo("Available benchmarks:")
    for benchmark_id in benchmark_ids:
        try:
            benchmark = load_benchmark(benchmark_id)
            click.echo(f"  {benchmark_id}: {benchmark.description or 'No description'}")
        except Exception as e:
            click.echo(f"  {benchmark_id}: Error loading - {e}")


@cli.group()
def models():
    """Manage models."""
    pass


@models.command("ls")
def list_models():
    """List available models (placeholder)."""
    click.echo("Model listing functionality coming soon.")
    click.echo("For now, you can use any HuggingFace model ID or OpenAI model name.")


@cli.group()
def eval():
    """Run evaluations."""
    pass


@eval.command("run")
@click.argument("benchmark_or_suite")
@click.option("--model", required=True, help="Model identifier")
@click.option("--provider", default="hf", help="Model provider (hf, openai, vertex, vllm)")
@click.option("--split", default="validation", help="Dataset split to use")
@click.option("--judge", help="Judge specification (exact_match, llm_rubric, etc.)")
@click.option("--guards", help="Guard models to use")
@click.option("--max-samples", type=int, help="Maximum number of samples to evaluate")
@click.option("--report", help="Output directory for results and reports")
@click.option("--seed", default=42, help="Random seed")
def run_eval(benchmark_or_suite, model, provider, split, judge, guards, max_samples, report, seed):
    """Run an evaluation on a benchmark or suite."""
    try:
        # Initialize runner
        runner = EvalRunner(model=model, provider=provider)
        
        # Check if it's a suite
        if benchmark_or_suite.startswith("alignlab:"):
            # Run suite
            results = runner.run_suite(
                benchmark_or_suite,
                split=split,
                judge=judge,
                max_samples=max_samples,
                output_dir=report,
                seed=seed
            )
            
            click.echo(f"Completed suite evaluation: {benchmark_or_suite}")
            for benchmark_id, result in results.items():
                click.echo(f"  {benchmark_id}: {len(result.results)} examples")
        
        else:
            # Run single benchmark
            benchmark = load_benchmark(benchmark_or_suite)
            result = runner.run(
                benchmark,
                split=split,
                judge=judge,
                max_samples=max_samples,
                output_dir=report,
                seed=seed
            )
            
            click.echo(f"Completed evaluation: {benchmark_or_suite}")
            click.echo(f"  Examples: {len(result.results)}")
            click.echo(f"  Results saved to: {report}")
    
    except Exception as e:
        click.echo(f"Error running evaluation: {e}", err=True)
        raise click.Abort()


@cli.group()
def jailbreak():
    """Jailbreak evaluation commands."""
    pass


@jailbreak.command("run")
@click.option("--suite", default="jb_default_v1", help="Jailbreak suite to run")
@click.option("--defense", default="none", help="Defense mechanism to test")
@click.option("--report", required=True, help="Output directory for results")
@click.option("--model", required=True, help="Model to test")
@click.option("--provider", default="hf", help="Model provider")
def run_jailbreak(suite, defense, report, model, provider):
    """Run jailbreak evaluation."""
    try:
        # This would integrate with JailbreakBench
        click.echo(f"Running jailbreak evaluation with {defense} defense")
        click.echo(f"Model: {model}")
        click.echo(f"Suite: {suite}")
        click.echo(f"Results will be saved to: {report}")
        
        # Placeholder for actual jailbreak evaluation
        click.echo("Jailbreak evaluation functionality coming soon.")
        
    except Exception as e:
        click.echo(f"Error running jailbreak evaluation: {e}", err=True)
        raise click.Abort()


@cli.group()
def report():
    """Generate reports."""
    pass


@report.command("build")
@click.argument("results_dir")
@click.option("--format", "formats", multiple=True, default=["html"], 
              help="Output formats (html, pdf, markdown)")
def build_report(results_dir, formats):
    """Build reports from evaluation results."""
    try:
        results_path = Path(results_dir)
        if not results_path.exists():
            click.echo(f"Results directory not found: {results_dir}", err=True)
            raise click.Abort()
        
        # Load results
        result = EvalResult.load(results_path)
        
        # Generate reports
        for fmt in formats:
            if fmt == "html":
                output_path = results_path / "report.html"
                result.to_report(str(output_path), format="html")
                click.echo(f"Generated HTML report: {output_path}")
            
            elif fmt == "pdf":
                output_path = results_path / "report.pdf"
                result.to_report(str(output_path), format="pdf")
                click.echo(f"Generated PDF report: {output_path}")
            
            elif fmt == "markdown":
                output_path = results_path / "report.md"
                result.to_report(str(output_path), format="markdown")
                click.echo(f"Generated Markdown report: {output_path}")
            
            else:
                click.echo(f"Unknown format: {fmt}", err=True)
    
    except Exception as e:
        click.echo(f"Error building report: {e}", err=True)
        raise click.Abort()


@cli.group()
def suites():
    """Manage benchmark suites."""
    pass


@suites.command("ls")
@click.option("--filter", "filters", multiple=True, help="Filter by taxonomy categories")
@click.option("--language", help="Filter by language")
@click.option("--severity", help="Filter by severity level")
def list_suites_cmd(filters, language, severity):
    """List available suites."""
    if filters or language or severity:
        suite_ids = filter_suites(
            taxonomy=list(filters) if filters else None,
            language=language,
            severity=severity
        )
    else:
        suite_ids = list_suites()
    
    if not suite_ids:
        click.echo("No suites found.")
        return
    
    click.echo("Available suites:")
    for suite_id in suite_ids:
        try:
            suite = load_suite(suite_id)
            click.echo(f"  {suite_id}: {suite.description}")
        except Exception as e:
            click.echo(f"  {suite_id}: Error loading - {e}")


@suites.command("info")
@click.argument("suite_id")
def suite_info(suite_id):
    """Show detailed information about a suite."""
    try:
        from alignlab_core import get_suite_summary
        summary = get_suite_summary(suite_id)
        
        click.echo(f"Suite: {summary['name']}")
        click.echo(f"ID: {summary['id']}")
        click.echo(f"Description: {summary['description']}")
        click.echo(f"Version: {summary['version']}")
        click.echo(f"Benchmarks: {summary['num_benchmarks']}")
        click.echo(f"Taxonomy coverage: {', '.join(summary['taxonomy_coverage'])}")
        
        click.echo("\nBenchmarks:")
        for benchmark_id in summary['benchmarks']:
            click.echo(f"  - {benchmark_id}")
        
        if summary['metadata']:
            click.echo("\nMetadata:")
            for key, value in summary['metadata'].items():
                click.echo(f"  {key}: {value}")
    
    except Exception as e:
        click.echo(f"Error getting suite info: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()

