"""
ConnMatrixHops — Visualize signal flow through N-hop subcircuits in connectivity matrices.
"""

__version__ = "0.1.0"

from connmatrixhops.analyzer import MatrixAnalyzer

try:
    from connmatrixhops.widgets import ConnMatrixHopsWidget
    __all__ = ["MatrixAnalyzer", "ConnMatrixHopsWidget"]
except ImportError:
    ConnMatrixHopsWidget = None
    __all__ = ["MatrixAnalyzer"]
