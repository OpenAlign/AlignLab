"""
Report generation for AlignLab evaluation results.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from alignlab_core import EvalResult, get_taxonomy

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generate comprehensive evaluation reports."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.taxonomy = get_taxonomy()
    
    def generate_html_report(self, result: EvalResult, output_path: str) -> str:
        """
        Generate an HTML report.
        
        Args:
            result: Evaluation result
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            html_content = self._generate_html_content(result)
            
            output_file = Path(output_path)
            if not output_file.suffix == ".html":
                output_file = output_file.with_suffix(".html")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
    
    def generate_pdf_report(self, result: EvalResult, output_path: str) -> str:
        """
        Generate a PDF report.
        
        Args:
            result: Evaluation result
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            # First generate HTML
            html_path = output_path.replace(".pdf", ".html")
            html_content = self._generate_html_content(result)
            
            # Convert to PDF
            try:
                import pdfkit
                
                output_file = Path(output_path)
                if not output_file.suffix == ".pdf":
                    output_file = output_file.with_suffix(".pdf")
                
                # Configure pdfkit options
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': "UTF-8",
                }
                
                pdfkit.from_string(html_content, str(output_file), options=options)
                logger.info(f"PDF report generated: {output_file}")
                return str(output_file)
                
            except ImportError:
                logger.warning("pdfkit not available. Install with: pip install pdfkit")
                # Fallback to HTML
                html_file = Path(html_path)
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html_content)
                logger.info(f"HTML report generated (PDF fallback): {html_file}")
                return str(html_file)
                
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            raise
    
    def generate_markdown_report(self, result: EvalResult, output_path: str) -> str:
        """
        Generate a Markdown report.
        
        Args:
            result: Evaluation result
            output_path: Path to save the report
            
        Returns:
            Path to the generated report
        """
        try:
            markdown_content = self._generate_markdown_content(result)
            
            output_file = Path(output_path)
            if not output_file.suffix == ".md":
                output_file = output_file.with_suffix(".md")
            
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(markdown_content)
            
            logger.info(f"Markdown report generated: {output_file}")
            return str(output_file)
            
        except Exception as e:
            logger.error(f"Error generating Markdown report: {e}")
            raise
    
    def _generate_html_content(self, result: EvalResult) -> str:
        """Generate HTML content for the report."""
        # Calculate metrics
        metrics = self._calculate_metrics(result)
        
        # Generate HTML
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AlignLab Evaluation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 2em;
            font-weight: bold;
            margin-bottom: 5px;
        }}
        .metric-label {{
            font-size: 0.9em;
            opacity: 0.9;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .error {{
            color: #e74c3c;
            font-weight: bold;
        }}
        .success {{
            color: #27ae60;
            font-weight: bold;
        }}
        .info {{
            background-color: #d1ecf1;
            border: 1px solid #bee5eb;
            color: #0c5460;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AlignLab Evaluation Report</h1>
        
        <div class="info">
            <strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
            <strong>Benchmark:</strong> {result.benchmark_id}<br>
            <strong>Model:</strong> {result.model_id}<br>
            <strong>Provider:</strong> {result.provider}<br>
            <strong>Samples:</strong> {len(result.results)}
        </div>
        
        <h2>Overall Metrics</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-value">{metrics['total_samples']}</div>
                <div class="metric-label">Total Samples</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['average_score']:.3f}</div>
                <div class="metric-label">Average Score</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['success_rate']:.1%}</div>
                <div class="metric-label">Success Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{metrics['error_rate']:.1%}</div>
                <div class="metric-label">Error Rate</div>
            </div>
        </div>
        
        <h2>Taxonomy Breakdown</h2>
        {self._generate_taxonomy_table_html(result)}
        
        <h2>Score Distribution</h2>
        {self._generate_score_distribution_html(result)}
        
        <h2>Error Analysis</h2>
        {self._generate_error_analysis_html(result)}
        
        <h2>Sample Results</h2>
        {self._generate_sample_results_html(result)}
    </div>
</body>
</html>
        """
        
        return html
    
    def _generate_markdown_content(self, result: EvalResult) -> str:
        """Generate Markdown content for the report."""
        metrics = self._calculate_metrics(result)
        
        markdown = f"""# AlignLab Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Benchmark:** {result.benchmark_id}  
**Model:** {result.model_id}  
**Provider:** {result.provider}  
**Samples:** {len(result.results)}

## Overall Metrics

| Metric | Value |
|--------|-------|
| Total Samples | {metrics['total_samples']} |
| Average Score | {metrics['average_score']:.3f} |
| Success Rate | {metrics['success_rate']:.1%} |
| Error Rate | {metrics['error_rate']:.1%} |

## Taxonomy Breakdown

{self._generate_taxonomy_table_markdown(result)}

## Score Distribution

{self._generate_score_distribution_markdown(result)}

## Error Analysis

{self._generate_error_analysis_markdown(result)}

## Sample Results

{self._generate_sample_results_markdown(result)}
        """
        
        return markdown
    
    def _calculate_metrics(self, result: EvalResult) -> Dict[str, Any]:
        """Calculate comprehensive metrics from results."""
        if not result.results:
            return {
                'total_samples': 0,
                'average_score': 0.0,
                'success_rate': 0.0,
                'error_rate': 0.0
            }
        
        # Extract scores
        scores = []
        errors = 0
        
        for sample in result.results:
            # Get score from judge results
            score = 0.0
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    score = judge_result["score"]
                    break
            
            scores.append(score)
            
            # Check for errors
            if "error" in sample.get("metadata", {}):
                errors += 1
        
        total_samples = len(result.results)
        average_score = np.mean(scores) if scores else 0.0
        success_rate = sum(1 for s in scores if s > 0.5) / total_samples if total_samples > 0 else 0.0
        error_rate = errors / total_samples if total_samples > 0 else 0.0
        
        return {
            'total_samples': total_samples,
            'average_score': average_score,
            'success_rate': success_rate,
            'error_rate': error_rate,
            'scores': scores
        }
    
    def _generate_taxonomy_table_html(self, result: EvalResult) -> str:
        """Generate taxonomy breakdown table in HTML."""
        # Group by taxonomy category
        taxonomy_scores = {}
        
        for sample in result.results:
            category = sample.get("metadata", {}).get("taxonomy_category", "unknown")
            if category not in taxonomy_scores:
                taxonomy_scores[category] = []
            
            # Get score
            score = 0.0
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    score = judge_result["score"]
                    break
            
            taxonomy_scores[category].append(score)
        
        # Generate table
        html = "<table><tr><th>Category</th><th>Count</th><th>Average Score</th><th>Success Rate</th></tr>"
        
        for category, scores in taxonomy_scores.items():
            count = len(scores)
            avg_score = np.mean(scores) if scores else 0.0
            success_rate = sum(1 for s in scores if s > 0.5) / count if count > 0 else 0.0
            
            html += f"<tr><td>{category}</td><td>{count}</td><td>{avg_score:.3f}</td><td>{success_rate:.1%}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_taxonomy_table_markdown(self, result: EvalResult) -> str:
        """Generate taxonomy breakdown table in Markdown."""
        # Similar logic to HTML version
        taxonomy_scores = {}
        
        for sample in result.results:
            category = sample.get("metadata", {}).get("taxonomy_category", "unknown")
            if category not in taxonomy_scores:
                taxonomy_scores[category] = []
            
            score = 0.0
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    score = judge_result["score"]
                    break
            
            taxonomy_scores[category].append(score)
        
        markdown = "| Category | Count | Average Score | Success Rate |\n"
        markdown += "|----------|-------|---------------|--------------|\n"
        
        for category, scores in taxonomy_scores.items():
            count = len(scores)
            avg_score = np.mean(scores) if scores else 0.0
            success_rate = sum(1 for s in scores if s > 0.5) / count if count > 0 else 0.0
            
            markdown += f"| {category} | {count} | {avg_score:.3f} | {success_rate:.1%} |\n"
        
        return markdown
    
    def _generate_score_distribution_html(self, result: EvalResult) -> str:
        """Generate score distribution visualization in HTML."""
        scores = []
        for sample in result.results:
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    scores.append(judge_result["score"])
                    break
        
        if not scores:
            return "<p>No scores available</p>"
        
        # Create histogram
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(scores, bins=bins)
        
        html = "<table><tr><th>Score Range</th><th>Count</th><th>Percentage</th></tr>"
        
        total = len(scores)
        for i in range(len(hist)):
            if hist[i] > 0:
                range_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                percentage = hist[i] / total * 100
                html += f"<tr><td>{range_label}</td><td>{hist[i]}</td><td>{percentage:.1f}%</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_score_distribution_markdown(self, result: EvalResult) -> str:
        """Generate score distribution in Markdown."""
        scores = []
        for sample in result.results:
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    scores.append(judge_result["score"])
                    break
        
        if not scores:
            return "No scores available"
        
        # Create histogram
        bins = np.linspace(0, 1, 11)
        hist, _ = np.histogram(scores, bins=bins)
        
        markdown = "| Score Range | Count | Percentage |\n"
        markdown += "|-------------|-------|------------|\n"
        
        total = len(scores)
        for i in range(len(hist)):
            if hist[i] > 0:
                range_label = f"{bins[i]:.1f}-{bins[i+1]:.1f}"
                percentage = hist[i] / total * 100
                markdown += f"| {range_label} | {hist[i]} | {percentage:.1f}% |\n"
        
        return markdown
    
    def _generate_error_analysis_html(self, result: EvalResult) -> str:
        """Generate error analysis in HTML."""
        error_types = {}
        
        for sample in result.results:
            if "error" in sample.get("metadata", {}):
                error_type = sample["metadata"]["error"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if not error_types:
            return "<p>No errors found</p>"
        
        html = "<table><tr><th>Error Type</th><th>Count</th><th>Percentage</th></tr>"
        
        total_errors = sum(error_types.values())
        for error_type, count in error_types.items():
            percentage = count / total_errors * 100
            html += f"<tr><td>{error_type}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_error_analysis_markdown(self, result: EvalResult) -> str:
        """Generate error analysis in Markdown."""
        error_types = {}
        
        for sample in result.results:
            if "error" in sample.get("metadata", {}):
                error_type = sample["metadata"]["error"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        if not error_types:
            return "No errors found"
        
        markdown = "| Error Type | Count | Percentage |\n"
        markdown += "|------------|-------|------------|\n"
        
        total_errors = sum(error_types.values())
        for error_type, count in error_types.items():
            percentage = count / total_errors * 100
            markdown += f"| {error_type} | {count} | {percentage:.1f}% |\n"
        
        return markdown
    
    def _generate_sample_results_html(self, result: EvalResult) -> str:
        """Generate sample results table in HTML."""
        html = "<table><tr><th>ID</th><th>Prompt</th><th>Response</th><th>Score</th><th>Status</th></tr>"
        
        for i, sample in enumerate(result.results[:10]):  # Show first 10
            # Get score
            score = 0.0
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    score = judge_result["score"]
                    break
            
            # Check for errors
            has_error = "error" in sample.get("metadata", {})
            status_class = "error" if has_error else "success"
            status_text = "Error" if has_error else "Success"
            
            prompt = sample.get("prompt", "")[:100] + "..." if len(sample.get("prompt", "")) > 100 else sample.get("prompt", "")
            response = sample.get("response", "")[:100] + "..." if len(sample.get("response", "")) > 100 else sample.get("response", "")
            
            html += f"<tr><td>{i+1}</td><td>{prompt}</td><td>{response}</td><td>{score:.3f}</td><td class='{status_class}'>{status_text}</td></tr>"
        
        html += "</table>"
        return html
    
    def _generate_sample_results_markdown(self, result: EvalResult) -> str:
        """Generate sample results table in Markdown."""
        markdown = "| ID | Prompt | Response | Score | Status |\n"
        markdown += "|----|--------|----------|-------|--------|\n"
        
        for i, sample in enumerate(result.results[:10]):  # Show first 10
            # Get score
            score = 0.0
            for judge_name, judge_result in sample.get("judge_results", {}).items():
                if isinstance(judge_result, dict) and "score" in judge_result:
                    score = judge_result["score"]
                    break
            
            # Check for errors
            has_error = "error" in sample.get("metadata", {})
            status_text = "Error" if has_error else "Success"
            
            prompt = sample.get("prompt", "")[:50] + "..." if len(sample.get("prompt", "")) > 50 else sample.get("prompt", "")
            response = sample.get("response", "")[:50] + "..." if len(sample.get("response", "")) > 50 else sample.get("response", "")
            
            markdown += f"| {i+1} | {prompt} | {response} | {score:.3f} | {status_text} |\n"
        
        return markdown

