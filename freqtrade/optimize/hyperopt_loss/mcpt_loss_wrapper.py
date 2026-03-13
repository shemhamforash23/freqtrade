import logging
from datetime import datetime
from typing import Any

from pandas import DataFrame

from freqtrade.constants import Config
from freqtrade.data.metrics import calculate_underwater
from freqtrade.optimize.hyperopt_loss.hyperopt_loss_interface import IHyperOptLoss
from freqtrade.optimize.montecarlo.mcpt_types import calculate_p_value
from freqtrade.optimize.montecarlo.permuters.trade_shuffle import TradeShufflePermuter


logger = logging.getLogger(__name__)


class MCPTLossWrapper(IHyperOptLoss):
    """
    Wrapper for any HyperOptLoss class that adds MCPT (Monte Carlo Permutation Test) penalty.

    It calculates the p-value of the strategy's profit against a random shuffle of its own trades.
    If the p-value is high (meaning the result is likely due to luck), the loss is penalized.
    """

    def __init__(self, original_loss: IHyperOptLoss, config: Config):
        self.original_loss = original_loss
        self.config = config

        # Load MCPT parameters from config
        mcpt_base_config = config.get("mcpt", {})
        mcpt_hyperopt_config = mcpt_base_config.get("hyperopt", {})
        self.mcpt_seed = mcpt_base_config.get("seed", 42)
        self.mcpt_runs = mcpt_hyperopt_config.get("runs", 100)
        self.mcpt_alpha = mcpt_hyperopt_config.get("alpha", 2.0)

        # Initialize Permuter once
        self.permuter = TradeShufflePermuter(scope="global")

        logger.info(
            f"MCPT: Wrapped {original_loss.__class__.__name__} with penalty "
            f"(runs={self.mcpt_runs}, alpha={self.mcpt_alpha})"
        )

    def hyperopt_loss_function(
        self,
        results: DataFrame,
        trade_count: int,
        min_date: datetime,
        max_date: datetime,
        config: Config,
        processed: dict[str, DataFrame],
        backtest_stats: dict[str, Any],
        starting_balance: float,
        **kwargs,
    ) -> float:
        # 1. Get the "pure" loss from the original loss function
        base_loss = self.original_loss.hyperopt_loss_function(
            results=results,
            trade_count=trade_count,
            min_date=min_date,
            max_date=max_date,
            config=config,
            processed=processed,
            backtest_stats=backtest_stats,
            starting_balance=starting_balance,
            **kwargs,
        )

        # Note: Minimum trade count is handled by hyperopt's --min-trades parameter
        # No need to duplicate that logic here

        # Calculate MCPT P-value
        p_value = self._calculate_p_value(results)

        # 3. Apply Penalty
        final_loss = self._apply_penalty(base_loss, p_value)

        # Log MCPT details for analysis
        logger.info(
            f"MCPT: trades={trade_count}, base_loss={base_loss:.5f}, "
            f"p_value={p_value:.4f}, final_loss={final_loss:.5f}"
        )

        return final_loss

    def _calculate_p_value(self, results: DataFrame) -> float:
        """
        Calculate p-value using Trade Shuffle.

        Tests Max Drawdown - a metric that DEPENDS on trade order.
        Null hypothesis: The strategy's drawdown is not significantly better than random trade ordering.

        Lower drawdown = better, so we test if real_dd <= permuted_dd (alternative="less")
        """
        # Calculate real max drawdown using freqtrade's metrics
        real_drawdown = self._get_max_drawdown(results)

        logger.debug(
            f"MCPT: Calculating p-value for max_drawdown={real_drawdown:.4f} "
            f"with {self.mcpt_runs} permutations"
        )

        # Collect permuted max drawdowns
        permuted_drawdowns: list[float] = [
            self._get_max_drawdown(permuted_df)
            for permuted_df in self.permuter.permute_batch(
                results, self.mcpt_runs, base_seed=self.mcpt_seed
            )
        ]

        # Lower drawdown is BETTER, so we test if real <= permuted
        # p_value = probability of getting drawdown as low as real by chance
        p_value, _ = calculate_p_value(real_drawdown, permuted_drawdowns, alternative="less")
        return p_value

    def _get_max_drawdown(self, trades: DataFrame) -> float:
        """
        Calculate max drawdown from trades DataFrame.
        Reuses freqtrade's calculate_underwater function.
        """
        try:
            drawdown_df = calculate_underwater(trades, value_col="profit_abs")
            return abs(min(drawdown_df["drawdown"]))
        except (ValueError, KeyError):
            # Empty trades or missing columns
            return 0.0

    def _apply_penalty(self, base_loss: float, p_value: float) -> float:
        """
        Apply penalty to the base loss based on p-value.

        Standard Hyperopt losses (Sharpe, Sortino, Profit) return NEGATIVE values (minimization).
        E.g. Sharpe = 3.0 -> Loss = -3.0.

        We want to INCREASE the loss (make it closer to 0 or positive) if p-value is high (bad).
        """
        if base_loss < 0:
            # Example:
            # base_loss = -3.0 (Good)
            # p_value = 0.0 (Good) -> penalty_mult = 1.0 -> final = -3.0
            # p_value = 0.5 (Bad, alpha=2) -> penalty_mult = 0.0 -> final = 0.0 (Bad)
            penalty_multiplier = 1.0 - (p_value * self.mcpt_alpha)

            # Safety floor to avoid flipping sign to positive aggressively
            # (unless we explicitly want to punish severely)
            if penalty_multiplier < 0.1:
                penalty_multiplier = 0.1

            return base_loss * penalty_multiplier
        else:
            # If base_loss is positive (already bad or different metric type), we increase it further
            return base_loss * (1.0 + p_value * self.mcpt_alpha)
