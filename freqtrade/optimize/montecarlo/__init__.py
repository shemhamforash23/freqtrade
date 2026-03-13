"""
Monte Carlo Permutation Tests (MCPT) module for Freqtrade.

Provides statistical validation of trading strategy performance through
permutation testing methods.
"""

from .mcpt_integration import MCPTIntegration
from .mcpt_runner import MCPTRunner
from .mcpt_types import MCPTResult, MCPTSummary, calculate_mcpt_metric, calculate_p_value
from .montecarlo import MonteCarlo


__all__ = [
    "MonteCarlo",
    "MCPTRunner",
    "MCPTIntegration",
    "MCPTResult",
    "MCPTSummary",
    "calculate_mcpt_metric",
    "calculate_p_value",
]
