"""
Interactive dashboard for AlignLab evaluation results.
"""

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import webbrowser
import tempfile

from alignlab_core import EvalResult, get_taxonomy

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Configuration for the dashboard."""
    port: int = 8050
    host: str = "localhost"
    debug: bool = False
    theme: str = "light"  # light, dark
    auto_open: bool = True


class Dashboard:
    """Interactive dashboard for exploring evaluation results."""
    
    def __init__(self, config: Optional[DashboardConfig] = None):
        """
        Initialize the dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config or DashboardConfig()
        self._dash_available = self._check_dash_availability()
        self.results: List[EvalResult] = []
        self.taxonomy = get_taxonomy()
    
    def _check_dash_availability(self) -> bool:
        """Check if Dash is available."""
        try:
            import dash
            return True
        except ImportError:
            logger.warning("Dash not available. Install with: pip install dash plotly")
            return False
    
    def add_results(self, results: EvalResult):
        """
        Add evaluation results to the dashboard.
        
        Args:
            results: Evaluation results to add
        """
        self.results.append(results)
        logger.info(f"Added results for {results.benchmark_id} ({len(results.results)} samples)")
    
    def add_results_from_file(self, file_path: str):
        """
        Add results from a saved file.
        
        Args:
            file_path: Path to results file
        """
        try:
            results = EvalResult.load(file_path)
            self.add_results(results)
        except Exception as e:
            logger.error(f"Error loading results from {file_path}: {e}")
            raise
    
    def add_results_from_directory(self, directory_path: str):
        """
        Add all results from a directory.
        
        Args:
            directory_path: Path to directory containing results
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        for file_path in directory.glob("*.json"):
            try:
                self.add_results_from_file(str(file_path))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")
    
    def create_dashboard(self) -> str:
        """
        Create and start the interactive dashboard.
        
        Returns:
            URL of the dashboard
        """
        if not self._dash_available:
            raise ImportError("Dash not available")
        
        if not self.results:
            raise ValueError("No results to display")
        
        try:
            import dash
            from dash import Dash, html, dcc, Input, Output, callback
            import plotly.express as px
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            import pandas as pd
            
            # Create Dash app
            app = Dash(__name__, title="AlignLab Dashboard")
            
            # Prepare data
            df = self._prepare_dataframe()
            
            # Layout
            app.layout = html.Div([
                html.H1("AlignLab Evaluation Dashboard", 
                       style={'textAlign': 'center', 'marginBottom': 30}),
                
                # Filters
                html.Div([
                    html.Label("Benchmark:"),
                    dcc.Dropdown(
                        id='benchmark-filter',
                        options=[{'label': r.benchmark_id, 'value': r.benchmark_id} 
                                for r in self.results],
                        value=self.results[0].benchmark_id if self.results else None,
                        style={'width': '100%', 'marginBottom': 20}
                    ),
                    
                    html.Label("Model:"),
                    dcc.Dropdown(
                        id='model-filter',
                        options=[{'label': r.model_id, 'value': r.model_id} 
                                for r in self.results],
                        value=self.results[0].model_id if self.results else None,
                        style={'width': '100%', 'marginBottom': 20}
                    ),
                ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Charts
                html.Div([
                    # Overall metrics
                    html.Div([
                        html.H3("Overall Metrics"),
                        dcc.Graph(id='overall-metrics')
                    ], style={'width': '100%', 'marginBottom': 30}),
                    
                    # Taxonomy breakdown
                    html.Div([
                        html.H3("Taxonomy Breakdown"),
                        dcc.Graph(id='taxonomy-breakdown')
                    ], style={'width': '100%', 'marginBottom': 30}),
                    
                    # Score distribution
                    html.Div([
                        html.H3("Score Distribution"),
                        dcc.Graph(id='score-distribution')
                    ], style={'width': '100%', 'marginBottom': 30}),
                    
                    # Error analysis
                    html.Div([
                        html.H3("Error Analysis"),
                        dcc.Graph(id='error-analysis')
                    ], style={'width': '100%'}),
                ], style={'width': '70%', 'display': 'inline-block', 'verticalAlign': 'top'}),
                
                # Results table
                html.Div([
                    html.H3("Detailed Results"),
                    html.Div(id='results-table')
                ], style={'width': '100%', 'marginTop': 30}),
            ])
            
            # Callbacks
            @app.callback(
                Output('overall-metrics', 'figure'),
                [Input('benchmark-filter', 'value'),
                 Input('model-filter', 'value')]
            )
            def update_overall_metrics(benchmark_id, model_id):
                filtered_df = df[
                    (df['benchmark_id'] == benchmark_id) & 
                    (df['model_id'] == model_id)
                ]
                
                if filtered_df.empty:
                    return go.Figure()
                
                # Calculate overall metrics
                metrics = {
                    'Total Samples': len(filtered_df),
                    'Average Score': filtered_df['score'].mean(),
                    'Success Rate': (filtered_df['score'] > 0.5).mean(),
                    'Error Rate': filtered_df['error'].sum()
                }
                
                fig = go.Figure(data=[
                    go.Bar(x=list(metrics.keys()), y=list(metrics.values()))
                ])
                fig.update_layout(title="Overall Metrics", height=400)
                return fig
            
            @app.callback(
                Output('taxonomy-breakdown', 'figure'),
                [Input('benchmark-filter', 'value'),
                 Input('model-filter', 'value')]
            )
            def update_taxonomy_breakdown(benchmark_id, model_id):
                filtered_df = df[
                    (df['benchmark_id'] == benchmark_id) & 
                    (df['model_id'] == model_id)
                ]
                
                if filtered_df.empty:
                    return go.Figure()
                
                # Group by taxonomy category
                taxonomy_scores = filtered_df.groupby('taxonomy_category')['score'].mean()
                
                fig = go.Figure(data=[
                    go.Bar(x=taxonomy_scores.index, y=taxonomy_scores.values)
                ])
                fig.update_layout(title="Average Scores by Taxonomy Category", height=400)
                return fig
            
            @app.callback(
                Output('score-distribution', 'figure'),
                [Input('benchmark-filter', 'value'),
                 Input('model-filter', 'value')]
            )
            def update_score_distribution(benchmark_id, model_id):
                filtered_df = df[
                    (df['benchmark_id'] == benchmark_id) & 
                    (df['model_id'] == model_id)
                ]
                
                if filtered_df.empty:
                    return go.Figure()
                
                fig = px.histogram(filtered_df, x='score', nbins=20)
                fig.update_layout(title="Score Distribution", height=400)
                return fig
            
            @app.callback(
                Output('error-analysis', 'figure'),
                [Input('benchmark-filter', 'value'),
                 Input('model-filter', 'value')]
            )
            def update_error_analysis(benchmark_id, model_id):
                filtered_df = df[
                    (df['benchmark_id'] == benchmark_id) & 
                    (df['model_id'] == model_id)
                ]
                
                if filtered_df.empty:
                    return go.Figure()
                
                # Error categories
                error_categories = filtered_df[filtered_df['error'] == True]['error_type'].value_counts()
                
                fig = go.Figure(data=[
                    go.Pie(labels=error_categories.index, values=error_categories.values)
                ])
                fig.update_layout(title="Error Analysis", height=400)
                return fig
            
            @app.callback(
                Output('results-table', 'children'),
                [Input('benchmark-filter', 'value'),
                 Input('model-filter', 'value')]
            )
            def update_results_table(benchmark_id, model_id):
                filtered_df = df[
                    (df['benchmark_id'] == benchmark_id) & 
                    (df['model_id'] == model_id)
                ]
                
                if filtered_df.empty:
                    return html.P("No results to display")
                
                # Create table
                table_data = []
                for _, row in filtered_df.head(10).iterrows():  # Show first 10 results
                    table_data.append(html.Tr([
                        html.Td(row['example_id']),
                        html.Td(row['prompt'][:100] + "..." if len(row['prompt']) > 100 else row['prompt']),
                        html.Td(row['score']),
                        html.Td(row['taxonomy_category']),
                        html.Td("Error" if row['error'] else "Success")
                    ]))
                
                return html.Table([
                    html.Thead(html.Tr([
                        html.Th("ID"), html.Th("Prompt"), html.Th("Score"), 
                        html.Th("Category"), html.Th("Status")
                    ])),
                    html.Tbody(table_data)
                ])
            
            # Start the server
            url = f"http://{self.config.host}:{self.config.port}"
            
            if self.config.auto_open:
                webbrowser.open(url)
            
            logger.info(f"Dashboard available at: {url}")
            
            # Run the app
            app.run_server(
                host=self.config.host,
                port=self.config.port,
                debug=self.config.debug
            )
            
            return url
            
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise
    
    def _prepare_dataframe(self):
        """Prepare data for the dashboard."""
        import pandas as pd
        
        data = []
        for result in self.results:
            for i, sample in enumerate(result.results):
                # Extract score from judge results
                score = 0.0
                for judge_name, judge_result in sample.get("judge_results", {}).items():
                    if isinstance(judge_result, dict) and "score" in judge_result:
                        score = judge_result["score"]
                        break
                
                # Extract taxonomy category
                taxonomy_category = "unknown"
                if "taxonomy_category" in sample.get("metadata", {}):
                    taxonomy_category = sample["metadata"]["taxonomy_category"]
                
                # Check for errors
                error = False
                error_type = "none"
                if "error" in sample.get("metadata", {}):
                    error = True
                    error_type = sample["metadata"]["error"]
                
                data.append({
                    "benchmark_id": result.benchmark_id,
                    "model_id": result.model_id,
                    "provider": result.provider,
                    "example_id": sample.get("example_id", i),
                    "prompt": sample.get("prompt", ""),
                    "response": sample.get("response", ""),
                    "score": score,
                    "taxonomy_category": taxonomy_category,
                    "error": error,
                    "error_type": error_type,
                    "timestamp": result.timestamp
                })
        
        return pd.DataFrame(data)
    
    def export_dashboard_data(self, output_path: str):
        """
        Export dashboard data for external analysis.
        
        Args:
            output_path: Path to save the data
        """
        try:
            df = self._prepare_dataframe()
            
            output_file = Path(output_path)
            if output_file.suffix == ".csv":
                df.to_csv(output_file, index=False)
            elif output_file.suffix == ".json":
                df.to_json(output_file, orient="records", indent=2)
            else:
                df.to_csv(output_file.with_suffix(".csv"), index=False)
            
            logger.info(f"Dashboard data exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting dashboard data: {e}")
            raise
    
    def generate_summary_report(self, output_path: str):
        """
        Generate a summary report of all results.
        
        Args:
            output_path: Path to save the report
        """
        try:
            from .reports import ReportGenerator
            
            report_gen = ReportGenerator()
            
            # Create combined results
            combined_results = []
            for result in self.results:
                combined_results.extend(result.results)
            
            # Create a mock combined result for reporting
            if self.results:
                combined_result = EvalResult(
                    benchmark_id="combined",
                    model_id="multiple",
                    provider="multiple",
                    split="combined",
                    results=combined_results
                )
                
                report_gen.generate_html_report(combined_result, output_path)
                logger.info(f"Summary report generated at {output_path}")
            else:
                logger.warning("No results to generate report for")
                
        except Exception as e:
            logger.error(f"Error generating summary report: {e}")
            raise

