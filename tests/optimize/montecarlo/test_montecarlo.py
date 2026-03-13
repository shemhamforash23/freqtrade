import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.montecarlo.montecarlo import MonteCarlo


class TestMonteCarlo:
    """Test MonteCarlo driver functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration with MCPT enabled."""
        return {
            "mcpt": {
                "enabled": True,
                "seed": 777,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 100,
                        "metric": "profit_total",
                        "scope": "global",
                        "seed": 42,
                    }
                ],
            }
        }

    @pytest.fixture
    def sample_method(self):
        """Return default MCPT method configuration."""
        return {
            "name": "trade_shuffle",
            "runs": 100,
            "metric": "profit_total",
            "scope": "global",
            "seed": 42,
        }

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data for testing."""
        np.random.seed(42)
        n_trades = 50

        return pd.DataFrame(
            {
                "pair": np.random.choice(["BTC/USDT", "ETH/USDT"], n_trades),
                "direction": np.random.choice([1, -1], n_trades),
                "profit_abs": np.random.normal(0.02, 0.05, n_trades),
                "open_time": pd.date_range("2023-01-01", periods=n_trades, freq="1h"),
            }
        )

    @pytest.fixture
    def sample_stats(self, sample_trades):
        """Create sample backtest statistics."""
        profit_total = sample_trades["profit_abs"].sum()
        return {
            "profit_total": profit_total,
            "profit_total_abs": abs(profit_total),
            "sharpe": 1.5,
            "sortino": 2.0,
            "calmar": 1.2,
            "profit_factor": 1.8,
            "max_drawdown_abs": 0.15,
        }

    def test_get_enabled_methods_merges_seed(self):
        """Test that global seed propagates to enabled methods."""
        config = {
            "mcpt": {
                "enabled": True,
                "seed": 11,
                "methods": [
                    {"name": "trade_shuffle", "runs": 10},
                    {"name": "bar_permute", "runs": 5, "seed": 99},
                ],
            }
        }

        methods = MonteCarlo.get_enabled_methods(config)

        assert methods[0]["seed"] == 11
        assert methods[1]["seed"] == 99

    def test_validate_basic(self, sample_config, sample_method, sample_trades, sample_stats):
        """Test basic validation workflow."""
        mc = MonteCarlo(sample_config, sample_method)
        result = mc.validate(sample_trades, sample_stats)

        assert result.metric_name == "profit_total"
        assert result.metric_real == sample_stats["profit_total"]
        assert result.n_permutations == 100
        assert 0 <= result.p_value <= 1
        assert 0 <= result.count_better <= 100
        assert len(result.metric_permuted) == 100

    def test_validate_with_seed_reproducibility(
        self, sample_config, sample_method, sample_trades, sample_stats
    ):
        """Test that validation with same seed produces same results."""
        mc1 = MonteCarlo(sample_config, sample_method)
        mc2 = MonteCarlo(sample_config, sample_method)

        result1 = mc1.validate(sample_trades, sample_stats)
        result2 = mc2.validate(sample_trades, sample_stats)

        assert result1.p_value == result2.p_value
        assert result1.count_better == result2.count_better
        assert result1.metric_permuted == result2.metric_permuted

    def test_extract_metric(self, sample_config, sample_method, sample_stats):
        """Test metric extraction from backtest stats."""
        mc = MonteCarlo(sample_config, sample_method)

        assert mc._extract_metric(sample_stats, "profit_total") == sample_stats["profit_total"]
        assert mc._extract_metric(sample_stats, "sharpe") == sample_stats["sharpe"]
        assert mc._extract_metric(sample_stats, "max_drawdown") == sample_stats["max_drawdown_abs"]

    def test_extract_metric_missing(self, sample_config, sample_method):
        """Test metric extraction with missing stat."""
        mc = MonteCarlo(sample_config, sample_method)
        stats = {}

        assert mc._extract_metric(stats, "profit_total") == 0.0

    def test_extract_metric_none_value(self, sample_config, sample_method):
        """Test metric extraction with None value."""
        mc = MonteCarlo(sample_config, sample_method)
        stats = {"profit_total": None}

        assert mc._extract_metric(stats, "profit_total") == 0.0

    def test_calculate_trade_metric_profit(self, sample_config, sample_method, sample_trades):
        """Test trade metric calculation for profit."""
        mc = MonteCarlo(sample_config, sample_method)

        result = mc._calculate_trade_metric(sample_trades, "profit_total")
        expected = sample_trades["profit_abs"].sum()

        assert result == expected

    def test_calculate_trade_metric_profit_factor(self, sample_config, sample_method):
        """Test trade metric calculation for profit factor."""
        mc = MonteCarlo(sample_config, sample_method)
        trades = pd.DataFrame(
            {
                "profit_abs": [0.1, 0.2, -0.05, -0.05, 0.3],
            }
        )

        result = mc._calculate_trade_metric(trades, "profit_factor")
        expected = 0.6 / 0.1  # wins / losses

        assert abs(result - expected) < 1e-10  # Use tolerance for floating point

    def test_calculate_trade_metric_empty_trades(self, sample_config, sample_method):
        """Test trade metric calculation with empty trades."""
        mc = MonteCarlo(sample_config, sample_method)
        trades = pd.DataFrame({"profit_abs": []})

        result = mc._calculate_trade_metric(trades, "profit_total")

        assert result == 0.0

    def test_validate_different_metrics(self, sample_config, sample_trades, sample_stats):
        """Test validation with different metrics."""
        for metric in ["profit_total", "profit_total_abs", "profit_factor"]:
            method_cfg = {
                "name": "trade_shuffle",
                "runs": 50,
                "metric": metric,
                "seed": 42,
            }
            mc = MonteCarlo(sample_config, method_cfg)
            result = mc.validate(sample_trades, sample_stats)

            assert result.metric_name == metric
            assert result.n_permutations == 50

    def test_progress_tracking(self, sample_config, sample_method, sample_trades, sample_stats):
        """Test that progress is tracked during validation."""
        mc = MonteCarlo(sample_config, sample_method)

        # Initial state
        assert mc.progress.progress == 0

        # After validation
        mc.validate(sample_trades, sample_stats)

        # Progress should be complete
        assert mc.progress.progress == 1.0

    def test_trade_shuffle_preserves_total_profit(self, sample_config):
        """Test that Trade Shuffle preserves total profit.

        Trade Shuffle permutes the order of trades, but the sum of profit_abs
        remains the same. Therefore, for metric="profit_total", p-value should
        be ~1.0 (all permutations have the same total as the original).

        This is the correct behavior - Trade Shuffle tests sequence risk,
        not whether the total profit is statistically significant.
        """
        np.random.seed(42)
        trades = pd.DataFrame(
            {
                "pair": ["BTC/USDT"] * 50,
                "direction": [1] * 50,
                "profit_abs": np.random.normal(0.02, 0.05, 50),
            }
        )

        stats = {"profit_total": trades["profit_abs"].sum()}

        method_cfg = {
            "name": "trade_shuffle",
            "runs": 50,
            "metric": "profit_total",
            "seed": 42,
        }
        mc = MonteCarlo(sample_config, method_cfg)
        result = mc.validate(trades, stats)

        # All permuted metrics should be approximately equal to real metric
        # (Trade Shuffle preserves the sum, just changes order)
        for permuted_value in result.metric_permuted:
            assert abs(permuted_value - result.metric_real) < 1e-10

    def test_trade_shuffle_with_profit_factor(self, sample_config):
        """Test Trade Shuffle with profit_factor metric.

        Unlike profit_total, profit_factor can change when trades are shuffled
        if we calculate it based on running equity (not implemented yet).
        For now, profit_factor is calculated from totals, so it also stays same.
        """
        np.random.seed(42)
        trades = pd.DataFrame(
            {
                "pair": ["BTC/USDT"] * 50,
                "direction": [1] * 50,
                "profit_abs": np.concatenate(
                    [
                        np.random.uniform(0.01, 0.1, 30),  # wins
                        np.random.uniform(-0.05, -0.01, 20),  # losses
                    ]
                ),
            }
        )

        wins = trades[trades["profit_abs"] > 0]["profit_abs"].sum()
        losses = abs(trades[trades["profit_abs"] < 0]["profit_abs"].sum())
        stats = {"profit_factor": wins / losses}

        method_cfg = {
            "name": "trade_shuffle",
            "runs": 50,
            "metric": "profit_factor",
            "seed": 42,
        }
        mc = MonteCarlo(sample_config, method_cfg)
        result = mc.validate(trades, stats)

        # Profit factor should also be preserved with Trade Shuffle
        # (same wins and losses, just different order)
        for permuted_value in result.metric_permuted:
            assert abs(permuted_value - result.metric_real) < 1e-10
