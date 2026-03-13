from enum import Enum


class MCPTState(Enum):
    """
    MCPT application states for progress tracking.

    Mirrors BacktestState pattern for consistency.
    """

    STARTUP = 1
    DATALOAD = 2
    ORIGINAL_BACKTEST = 3
    PERMUTATION_TESTS = 4
    VALIDATION = 5
    REPORTING = 6

    def __str__(self):
        return f"{self.name.lower()}"
