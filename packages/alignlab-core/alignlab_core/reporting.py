"""
Report generation for evaluation results.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import numpy as np
from datetime import datetime

from .results import EvalResult
from .taxonomy import get_taxonomy

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates reports from evaluation results."""
    
    def __init__(self):
        self.taxonomy = get_taxonomy()
    
    def generate_html_report(self, result: EvalResult, output_path: str) -> str:
        """Generate an HTML report."""
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".html")
        
        html_content = self._generate_html_content(result)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        logger.info(f"Generated HTML report: {output_path}")
        return str(output_path)
    
    def generate_pdf_report(self, result: EvalResult, output_path: str) -> str:
        """Generate a PDF report."""
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".pdf")
        
        # First generate HTML
        html_path = output_path.with_suffix(".html")
        self.generate_html_report(result, str(html_path))
        
        # Convert to PDF using weasyprint
        try:
            from weasyprint import HTML
            HTML(filename=str(html_path)).write_pdf(str(output_path))
            html_path.unlink()  # Remove temporary HTML file
            logger.info(f"Generated PDF report: {output_path}")
        except ImportError:
            logger.warning("weasyprint not available, keeping HTML version")
            output_path = html_path
        
        return str(output_path)
    
    def generate_markdown_report(self, result: EvalResult, output_path: str) -> str:
        """Generate a Markdown report."""
        output_path = Path(output_path)
        if not output_path.suffix:
            output_path = output_path.with_suffix(".md")
        
        markdown_content = self._generate_markdown_content(result)
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        logger.info(f"Generated Markdown report: {output_path}")
        return str(output_path)
    
    def _generate_html_content(self, result: EvalResult) -> str:
        """Generate HTML content for the report."""
        summary = result.get_summary()
        metrics = result.get_metrics()
        
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlignLab Evaluation Report - {summary['benchmark_id']}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        h1 {{
            border-bottom: 3px solid #007acc;
            padding-bottom: 10px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .summary-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007acc;
        }}
        .metric-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .metric-table th, .metric-table td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        .metric-table th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        .score-high {{ color: #28a745; }}
        .score-medium {{ color: #ffc107; }}
        .score-low {{ color: #dc3545; }}
        .chart-container {{
            margin: 30px 0;
            height: 400px;
        }}
        .error-examples {{
            background: #fff3cd;
            border: 1px solid #ffeaa7;
            border-radius: 6px;
            padding: 15px;
            margin: 20px 0;
        }}
        .example {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 15px;
            margin: 10px 0;
        }}
        .prompt {{ font-weight: 600; color: #495057; }}
        .response {{ color: #6c757d; font-style: italic; }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <h1>AlignLab Evaluation Report</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>Benchmark</h3>
                <p><strong>{summary['benchmark_id']}</strong></p>
            </div>
            <div class="summary-card">
                <h3>Model</h3>
                <p><strong>{summary['model_id']}</strong></p>
                <p>Provider: {summary['provider']}</p>
            </div>
            <div class="summary-card">
                <h3>Dataset</h3>
                <p>Split: <strong>{summary['split']}</strong></p>
                <p>Examples: <strong>{summary['num_examples']}</strong></p>
            </div>
            <div class="summary-card">
                <h3>Evaluation</h3>
                <p>Date: <strong>{summary['timestamp'][:10]}</strong></p>
            </div>
        </div>
        
        <h2>Overall Metrics</h2>
        {self._generate_metrics_table(metrics)}
        
        <h2>Judge Results</h2>
        {self._generate_judge_results_html(metrics)}
        
        <h2>Taxonomy Breakdown</h2>
        {self._generate_taxonomy_breakdown_html(result)}
        
        <h2>Error Analysis</h2>
        {self._generate_error_analysis_html(result)}
        
        <div class="chart-container">
            <canvas id="metricsChart"></canvas>
        </div>
        
        <script>
            {self._generate_chart_js(metrics)}
        </script>
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_markdown_content(self, result: EvalResult) -> str:
        """Generate Markdown content for the report."""
        summary = result.get_summary()
        metrics = result.get_metrics()
        
        markdown = f"""# AlignLab Evaluation Report

## Summary

- **Benchmark**: {summary['benchmark_id']}
- **Model**: {summary['model_id']} ({summary['provider']})
- **Dataset**: {summary['split']} split
- **Examples**: {summary['num_examples']}
- **Date**: {summary['timestamp'][:10]}

## Overall Metrics

{self._generate_metrics_markdown(metrics)}

## Judge Results

{self._generate_judge_results_markdown(metrics)}

## Taxonomy Breakdown

{self._generate_taxonomy_breakdown_markdown(result)}

## Error Analysis

{self._generate_error_analysis_markdown(result)}

## Raw Results

The complete evaluation results are available in the following files:
- `results.jsonl`: Raw evaluation results
- `metadata.json`: Evaluation metadata
- `summary.json`: Summary statistics
"""
        
        return markdown
    
    def _generate_metrics_table(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML table for metrics."""
        if not metrics:
            return "<p>No metrics available.</p>"
        
        html = '<table class="metric-table">'
        html += '<tr><th>Metric</th><th>Value</th><th>CI (95%)</th></tr>'
        
        for judge_name, judge_metrics in metrics.items():
            if judge_name == "overall":
                continue
            
            if "mean" in judge_metrics:
                mean = judge_metrics["mean"]
                ci = judge_metrics.get("ci_95", [None, None])
                
                score_class = self._get_score_class(mean)
                
                html += f'<tr>'
                html += f'<td>{judge_name}</td>'
                html += f'<td class="{score_class}">{mean:.3f}</td>'
                if ci[0] is not None and ci[1] is not None:
                    html += f'<td>[{ci[0]:.3f}, {ci[1]:.3f}]</td>'
                else:
                    html += f'<td>-</td>'
                html += f'</tr>'
        
        html += '</table>'
        return html
    
    def _generate_metrics_markdown(self, metrics: Dict[str, Any]) -> str:
        """Generate Markdown table for metrics."""
        if not metrics:
            return "No metrics available."
        
        markdown = "| Metric | Value | CI (95%) |\n"
        markdown += "|--------|-------|----------|\n"
        
        for judge_name, judge_metrics in metrics.items():
            if judge_name == "overall":
                continue
            
            if "mean" in judge_metrics:
                mean = judge_metrics["mean"]
                ci = judge_metrics.get("ci_95", [None, None])
                
                if ci[0] is not None and ci[1] is not None:
                    ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                else:
                    ci_str = "-"
                
                markdown += f"| {judge_name} | {mean:.3f} | {ci_str} |\n"
        
        return markdown
    
    def _generate_judge_results_html(self, metrics: Dict[str, Any]) -> str:
        """Generate HTML for judge results."""
        html = ""
        
        for judge_name, judge_metrics in metrics.items():
            if judge_name == "overall":
                continue
            
            html += f'<h3>{judge_name.title()}</h3>'
            
            if "categories" in judge_metrics:
                html += '<table class="metric-table">'
                html += '<tr><th>Category</th><th>Count</th><th>Percentage</th></tr>'
                
                total = sum(judge_metrics["categories"].values())
                for category, count in judge_metrics["categories"].items():
                    percentage = (count / total) * 100
                    html += f'<tr><td>{category}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>'
                
                html += '</table>'
            
            html += '<br>'
        
        return html
    
    def _generate_judge_results_markdown(self, metrics: Dict[str, Any]) -> str:
        """Generate Markdown for judge results."""
        markdown = ""
        
        for judge_name, judge_metrics in metrics.items():
            if judge_name == "overall":
                continue
            
            markdown += f"### {judge_name.title()}\n\n"
            
            if "categories" in judge_metrics:
                markdown += "| Category | Count | Percentage |\n"
                markdown += "|----------|-------|------------|\n"
                
                total = sum(judge_metrics["categories"].values())
                for category, count in judge_metrics["categories"].items():
                    percentage = (count / total) * 100
                    markdown += f"| {category} | {count} | {percentage:.1f}% |\n"
                
                markdown += "\n"
        
        return markdown
    
    def _generate_taxonomy_breakdown_html(self, result: EvalResult) -> str:
        """Generate HTML for taxonomy breakdown."""
        category_breakdown = self.taxonomy.get_category_breakdown(result.results)
        
        if not category_breakdown:
            return "<p>No taxonomy breakdown available.</p>"
        
        html = '<table class="metric-table">'
        html += '<tr><th>Category</th><th>Count</th><th>Mean Score</th><th>Severity</th></tr>'
        
        for category_id, metrics in category_breakdown.items():
            category = self.taxonomy.get_category(category_id)
            category_name = category.name if category else category_id
            
            html += f'<tr>'
            html += f'<td>{category_name}</td>'
            html += f'<td>{metrics["count"]}</td>'
            html += f'<td>{metrics["mean_score"]:.3f}</td>'
            html += f'<td>{metrics["severity"] or "unknown"}</td>'
            html += f'</tr>'
        
        html += '</table>'
        return html
    
    def _generate_taxonomy_breakdown_markdown(self, result: EvalResult) -> str:
        """Generate Markdown for taxonomy breakdown."""
        category_breakdown = self.taxonomy.get_category_breakdown(result.results)
        
        if not category_breakdown:
            return "No taxonomy breakdown available."
        
        markdown = "| Category | Count | Mean Score | Severity |\n"
        markdown += "|----------|-------|------------|----------|\n"
        
        for category_id, metrics in category_breakdown.items():
            category = self.taxonomy.get_category(category_id)
            category_name = category.name if category else category_id
            
            markdown += f"| {category_name} | {metrics['count']} | {metrics['mean_score']:.3f} | {metrics['severity'] or 'unknown'} |\n"
        
        return markdown
    
    def _generate_error_analysis_html(self, result: EvalResult) -> str:
        """Generate HTML for error analysis."""
        html = ""
        
        # Get error examples for each judge
        for judge_name in result.get_metrics().keys():
            if judge_name == "overall":
                continue
            
            error_examples = result.get_error_examples(judge_name, max_examples=3)
            
            if error_examples:
                html += f'<h3>Error Examples - {judge_name.title()}</h3>'
                html += '<div class="error-examples">'
                
                for example in error_examples:
                    html += '<div class="example">'
                    html += f'<div class="prompt">Prompt: {example["prompt"][:200]}...</div>'
                    html += f'<div class="response">Response: {example["response"][:200]}...</div>'
                    html += '</div>'
                
                html += '</div>'
        
        return html
    
    def _generate_error_analysis_markdown(self, result: EvalResult) -> str:
        """Generate Markdown for error analysis."""
        markdown = ""
        
        # Get error examples for each judge
        for judge_name in result.get_metrics().keys():
            if judge_name == "overall":
                continue
            
            error_examples = result.get_error_examples(judge_name, max_examples=3)
            
            if error_examples:
                markdown += f"### Error Examples - {judge_name.title()}\n\n"
                
                for i, example in enumerate(error_examples, 1):
                    markdown += f"**Example {i}:**\n"
                    markdown += f"- **Prompt:** {example['prompt'][:200]}...\n"
                    markdown += f"- **Response:** {example['response'][:200]}...\n\n"
        
        return markdown
    
    def _generate_chart_js(self, metrics: Dict[str, Any]) -> str:
        """Generate JavaScript for charts."""
        js = """
        const ctx = document.getElementById('metricsChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ["""
        
        labels = []
        values = []
        
        for judge_name, judge_metrics in metrics.items():
            if judge_name == "overall":
                continue
            
            if "mean" in judge_metrics:
                labels.append(f"'{judge_name}'")
                values.append(judge_metrics["mean"])
        
        js += ", ".join(labels)
        js += "],"
        js += f"""
                datasets: [{
                    label: 'Mean Score',
                    data: [{", ".join(map(str, values))}],
                    backgroundColor: 'rgba(0, 122, 204, 0.8)',
                    borderColor: 'rgba(0, 122, 204, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Evaluation Metrics'
                    }
                }
            }
        });
        """
        
        return js
    
    def _get_score_class(self, score: float) -> str:
        """Get CSS class for score styling."""
        if score >= 0.7:
            return "score-high"
        elif score >= 0.4:
            return "score-medium"
        else:
            return "score-low"

