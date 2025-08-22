"""
Chart and visualization generation for AlignLab evaluation results.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from alignlab_core import EvalResult, get_taxonomy

logger = logging.getLogger(__name__)


class ChartGenerator:
    """Generate charts and visualizations from evaluation results."""
    
    def __init__(self):
        """Initialize the chart generator."""
        self.taxonomy = get_taxonomy()
    
    def create_score_distribution_chart(self, result: EvalResult, **kwargs) -> Dict[str, Any]:
        """
        Create a score distribution histogram.
        
        Args:
            result: Evaluation result
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
            # Extract scores
            scores = []
            for sample in result.results:
                for judge_name, judge_result in sample.get("judge_results", {}).items():
                    if isinstance(judge_result, dict) and "score" in judge_result:
                        scores.append(judge_result["score"])
                        break
            
            if not scores:
                return {"error": "No scores available"}
            
            # Create histogram
            fig = go.Figure(data=[
                go.Histogram(
                    x=scores,
                    nbinsx=20,
                    name="Score Distribution",
                    marker_color='#3498db',
                    opacity=0.7
                )
            ])
            
            fig.update_layout(
                title="Score Distribution",
                xaxis_title="Score",
                yaxis_title="Count",
                showlegend=False,
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating score distribution chart: {e}")
            return {"error": str(e)}
    
    def create_taxonomy_breakdown_chart(self, result: EvalResult, **kwargs) -> Dict[str, Any]:
        """
        Create a taxonomy breakdown bar chart.
        
        Args:
            result: Evaluation result
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
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
            
            # Calculate averages
            categories = []
            avg_scores = []
            counts = []
            
            for category, scores in taxonomy_scores.items():
                categories.append(category)
                avg_scores.append(np.mean(scores) if scores else 0.0)
                counts.append(len(scores))
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=avg_scores,
                    text=[f"{score:.3f}<br>({count} samples)" for score, count in zip(avg_scores, counts)],
                    textposition='auto',
                    marker_color='#2ecc71',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title="Average Scores by Taxonomy Category",
                xaxis_title="Category",
                yaxis_title="Average Score",
                showlegend=False,
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating taxonomy breakdown chart: {e}")
            return {"error": str(e)}
    
    def create_error_analysis_chart(self, result: EvalResult, **kwargs) -> Dict[str, Any]:
        """
        Create an error analysis pie chart.
        
        Args:
            result: Evaluation result
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
            # Count error types
            error_types = {}
            
            for sample in result.results:
                if "error" in sample.get("metadata", {}):
                    error_type = sample["metadata"]["error"]
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            
            if not error_types:
                return {"error": "No errors found"}
            
            # Create pie chart
            fig = go.Figure(data=[
                go.Pie(
                    labels=list(error_types.keys()),
                    values=list(error_types.values()),
                    hole=0.3,
                    marker_colors=['#e74c3c', '#f39c12', '#9b59b6', '#34495e', '#1abc9c']
                )
            ])
            
            fig.update_layout(
                title="Error Analysis",
                showlegend=True,
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating error analysis chart: {e}")
            return {"error": str(e)}
    
    def create_model_comparison_chart(self, results: List[EvalResult], **kwargs) -> Dict[str, Any]:
        """
        Create a model comparison chart.
        
        Args:
            results: List of evaluation results
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
            # Prepare data
            models = []
            avg_scores = []
            success_rates = []
            
            for result in results:
                models.append(result.model_id)
                
                # Calculate average score
                scores = []
                for sample in result.results:
                    for judge_name, judge_result in sample.get("judge_results", {}).items():
                        if isinstance(judge_result, dict) and "score" in judge_result:
                            scores.append(judge_result["score"])
                            break
                
                avg_score = np.mean(scores) if scores else 0.0
                avg_scores.append(avg_score)
                
                # Calculate success rate
                success_rate = sum(1 for s in scores if s > 0.5) / len(scores) if scores else 0.0
                success_rates.append(success_rate)
            
            # Create grouped bar chart
            fig = go.Figure(data=[
                go.Bar(
                    name='Average Score',
                    x=models,
                    y=avg_scores,
                    marker_color='#3498db',
                    opacity=0.8
                ),
                go.Bar(
                    name='Success Rate',
                    x=models,
                    y=success_rates,
                    marker_color='#2ecc71',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title="Model Comparison",
                xaxis_title="Model",
                yaxis_title="Score",
                barmode='group',
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating model comparison chart: {e}")
            return {"error": str(e)}
    
    def create_benchmark_comparison_chart(self, results: List[EvalResult], **kwargs) -> Dict[str, Any]:
        """
        Create a benchmark comparison chart.
        
        Args:
            results: List of evaluation results
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
            # Prepare data
            benchmarks = []
            avg_scores = []
            
            for result in results:
                benchmarks.append(result.benchmark_id)
                
                # Calculate average score
                scores = []
                for sample in result.results:
                    for judge_name, judge_result in sample.get("judge_results", {}).items():
                        if isinstance(judge_result, dict) and "score" in judge_result:
                            scores.append(judge_result["score"])
                            break
                
                avg_score = np.mean(scores) if scores else 0.0
                avg_scores.append(avg_score)
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=benchmarks,
                    y=avg_scores,
                    marker_color='#9b59b6',
                    opacity=0.8
                )
            ])
            
            fig.update_layout(
                title="Benchmark Comparison",
                xaxis_title="Benchmark",
                yaxis_title="Average Score",
                showlegend=False,
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating benchmark comparison chart: {e}")
            return {"error": str(e)}
    
    def create_timeline_chart(self, results: List[EvalResult], **kwargs) -> Dict[str, Any]:
        """
        Create a timeline chart showing performance over time.
        
        Args:
            results: List of evaluation results
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            from datetime import datetime
            
            # Prepare data
            timestamps = []
            avg_scores = []
            
            for result in results:
                # Parse timestamp
                try:
                    timestamp = datetime.fromisoformat(result.timestamp.replace('Z', '+00:00'))
                    timestamps.append(timestamp)
                except:
                    # Use current time if parsing fails
                    timestamps.append(datetime.now())
                
                # Calculate average score
                scores = []
                for sample in result.results:
                    for judge_name, judge_result in sample.get("judge_results", {}).items():
                        if isinstance(judge_result, dict) and "score" in judge_result:
                            scores.append(judge_result["score"])
                            break
                
                avg_score = np.mean(scores) if scores else 0.0
                avg_scores.append(avg_score)
            
            # Create line chart
            fig = go.Figure(data=[
                go.Scatter(
                    x=timestamps,
                    y=avg_scores,
                    mode='lines+markers',
                    name='Average Score',
                    line=dict(color='#e74c3c', width=3),
                    marker=dict(size=8)
                )
            ])
            
            fig.update_layout(
                title="Performance Timeline",
                xaxis_title="Time",
                yaxis_title="Average Score",
                showlegend=False,
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating timeline chart: {e}")
            return {"error": str(e)}
    
    def create_heatmap_chart(self, results: List[EvalResult], **kwargs) -> Dict[str, Any]:
        """
        Create a heatmap showing performance across different dimensions.
        
        Args:
            results: List of evaluation results
            **kwargs: Additional chart options
            
        Returns:
            Chart data in Plotly format
        """
        try:
            import plotly.graph_objects as go
            
            # Prepare data matrix
            benchmarks = []
            models = []
            scores_matrix = []
            
            # Collect unique benchmarks and models
            for result in results:
                if result.benchmark_id not in benchmarks:
                    benchmarks.append(result.benchmark_id)
                if result.model_id not in models:
                    models.append(result.model_id)
            
            # Create score matrix
            for benchmark in benchmarks:
                row = []
                for model in models:
                    # Find matching result
                    score = 0.0
                    for result in results:
                        if result.benchmark_id == benchmark and result.model_id == model:
                            # Calculate average score
                            scores = []
                            for sample in result.results:
                                for judge_name, judge_result in sample.get("judge_results", {}).items():
                                    if isinstance(judge_result, dict) and "score" in judge_result:
                                        scores.append(judge_result["score"])
                                        break
                            score = np.mean(scores) if scores else 0.0
                            break
                    row.append(score)
                scores_matrix.append(row)
            
            # Create heatmap
            fig = go.Figure(data=[
                go.Heatmap(
                    z=scores_matrix,
                    x=models,
                    y=benchmarks,
                    colorscale='Viridis',
                    text=[[f"{score:.3f}" for score in row] for row in scores_matrix],
                    texttemplate="%{text}",
                    textfont={"size": 12},
                    colorbar=dict(title="Score")
                )
            ])
            
            fig.update_layout(
                title="Performance Heatmap",
                xaxis_title="Model",
                yaxis_title="Benchmark",
                **kwargs
            )
            
            return fig.to_dict()
            
        except ImportError:
            logger.warning("Plotly not available for chart generation")
            return {"error": "Plotly not available"}
        except Exception as e:
            logger.error(f"Error creating heatmap chart: {e}")
            return {"error": str(e)}
    
    def save_chart(self, chart_data: Dict[str, Any], output_path: str, format: str = "html"):
        """
        Save a chart to file.
        
        Args:
            chart_data: Chart data from chart generation methods
            output_path: Path to save the chart
            format: Output format (html, png, svg, pdf)
        """
        try:
            import plotly.graph_objects as go
            
            if "error" in chart_data:
                logger.error(f"Cannot save chart: {chart_data['error']}")
                return
            
            # Create figure from data
            fig = go.Figure(chart_data)
            
            # Save based on format
            if format == "html":
                fig.write_html(output_path)
            elif format == "png":
                fig.write_image(output_path)
            elif format == "svg":
                fig.write_image(output_path, format="svg")
            elif format == "pdf":
                fig.write_image(output_path, format="pdf")
            else:
                logger.warning(f"Unsupported format: {format}. Saving as HTML.")
                fig.write_html(output_path)
            
            logger.info(f"Chart saved to {output_path}")
            
        except ImportError:
            logger.warning("Plotly not available for chart saving")
        except Exception as e:
            logger.error(f"Error saving chart: {e}")
    
    def create_dashboard_charts(self, result: EvalResult) -> Dict[str, Any]:
        """
        Create a complete set of dashboard charts.
        
        Args:
            result: Evaluation result
            
        Returns:
            Dictionary containing all chart data
        """
        charts = {
            "score_distribution": self.create_score_distribution_chart(result),
            "taxonomy_breakdown": self.create_taxonomy_breakdown_chart(result),
            "error_analysis": self.create_error_analysis_chart(result)
        }
        
        return charts

