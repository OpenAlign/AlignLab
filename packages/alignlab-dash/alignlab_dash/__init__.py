"""
AlignLab Dashboard - Visualization and reporting components.
"""

from .dashboard import Dashboard
from .reports import ReportGenerator
from .visualizations import ChartGenerator

__version__ = "0.1.0"
__all__ = [
    "Dashboard",
    "ReportGenerator", 
    "ChartGenerator"
]

