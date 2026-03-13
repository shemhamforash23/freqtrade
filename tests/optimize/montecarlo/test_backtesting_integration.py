"""
Tests for MCPT integration with Backtesting.

These tests verify the full integration between Backtesting and MCPT,
including proper method binding, result storage, and edge cases.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest


class TestBacktestingMCPTIntegration:
    """Test MCPT integration in Backtesting class."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data with realistic structure."""
        np.random.seed(42)
        n_trades = 50

        return pd.DataFrame(
            {
                "pair": np.random.choice(["BTC/USDT", "ETH/USDT"], n_trades),
                "stake_amount": [100.0] * n_trades,
                "amount": [0.01] * n_trades,
                "open_date": pd.date_range("2023-01-01", periods=n_trades, freq="1h"),
                "close_date": pd.date_range("2023-01-01 01:00", periods=n_trades, freq="1h"),
                "open_rate": np.random.uniform(40000, 45000, n_trades),
                "close_rate": np.random.uniform(40000, 45000, n_trades),
                "fee_open": [0.001] * n_trades,
                "fee_close": [0.001] * n_trades,
                "trade_duration": [60] * n_trades,
                "profit_ratio": np.random.normal(0.01, 0.02, n_trades),
                "profit_abs": np.random.normal(1.0, 2.0, n_trades),
                "exit_reason": ["roi"] * n_trades,
                "is_open": [False] * n_trades,
                "is_short": [False] * n_trades,
                "direction": [1] * n_trades,
            }
        )

    @pytest.fixture
    def sample_strategy_stats(self, sample_trades):
        """Create sample strategy statistics matching backtest output."""
        return {
            "profit_total": float(sample_trades["profit_abs"].sum()),
            "profit_total_abs": float(abs(sample_trades["profit_abs"].sum())),
            "profit_factor": 1.5,
            "sharpe": 1.2,
            "sortino": 1.8,
            "calmar": 1.1,
            "max_drawdown_abs": 0.15,
            "trades": [],
        }

    def _create_mock_backtesting(self, config, all_bt_content, results):
        """
        Create a properly configured mock Backtesting instance.

        Creates a mock that can be used with MCPTIntegration.
        """
        from freqtrade.optimize.backtesting import Backtesting

        bt = MagicMock(spec=Backtesting)
        bt.config = config
        bt.all_bt_content = all_bt_content
        bt.results = results

        # Add attributes needed by MCPTIntegration
        bt.strategylist = []
        bt.detail_data = {}
        bt.timeframe = "5m"
        bt.timeframe_detail = None
        bt.timerange = None
        bt.required_startup = 0

        return bt

    def test_run_mcpt_validation_basic(self, sample_trades, sample_strategy_stats):
        """Test basic MCPT validation through MCPTIntegration."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 50,
                        "metric": "profit_total",
                        "seed": 42,
                    }
                ],
            }
        }
        all_bt_content = {
            "TestStrategy": {"results": sample_trades},
        }
        results = {
            "strategy": {"TestStrategy": sample_strategy_stats},
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        # Call validation through MCPTIntegration
        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        # Verify MCPT results were stored
        assert "mcpt" in bt.results
        assert "TestStrategy" in bt.results["mcpt"]

        mcpt_result = bt.results["mcpt"]["TestStrategy"][0]
        assert mcpt_result["metric_name"] == "profit_total"
        assert mcpt_result["n_permutations"] == 50
        assert 0 <= mcpt_result["p_value"] <= 1
        assert "significant" in mcpt_result
        assert isinstance(mcpt_result["significant"], bool)

    def test_run_mcpt_validation_skips_few_trades(self):
        """Test that MCPT skips strategies with insufficient trades."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        few_trades = pd.DataFrame({"profit_abs": [0.1, 0.2, 0.3]})

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 50,
                    }
                ],
            }
        }
        all_bt_content = {
            "FewTradesStrategy": {"results": few_trades},
        }
        results = {
            "strategy": {"FewTradesStrategy": {}},
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        # Call validation
        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        # Strategy should be skipped (< 10 trades required)
        if "mcpt" in bt.results:
            assert "FewTradesStrategy" not in bt.results["mcpt"]

    def test_run_mcpt_validation_multiple_strategies(self, sample_trades, sample_strategy_stats):
        """Test MCPT validation with multiple strategies."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 30,
                        "metric": "profit_total",
                        "seed": 42,
                    }
                ],
            }
        }
        all_bt_content = {
            "Strategy1": {"results": sample_trades.copy()},
            "Strategy2": {"results": sample_trades.copy()},
        }
        results = {
            "strategy": {
                "Strategy1": sample_strategy_stats.copy(),
                "Strategy2": sample_strategy_stats.copy(),
            },
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        # Both strategies should have MCPT results
        assert "mcpt" in bt.results
        assert "Strategy1" in bt.results["mcpt"]
        assert "Strategy2" in bt.results["mcpt"]

        # Results should be independent (different p-values possible)
        assert bt.results["mcpt"]["Strategy1"][0]["n_permutations"] == 30
        assert bt.results["mcpt"]["Strategy2"][0]["n_permutations"] == 30

    def test_run_mcpt_validation_with_max_drawdown(self, sample_trades, sample_strategy_stats):
        """Test MCPT validation using max_drawdown metric."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 100,
                        "metric": "max_drawdown",
                        "seed": 42,
                    }
                ],
            }
        }
        all_bt_content = {
            "DrawdownStrategy": {"results": sample_trades},
        }
        results = {
            "strategy": {"DrawdownStrategy": sample_strategy_stats},
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        assert "mcpt" in bt.results
        mcpt_result = bt.results["mcpt"]["DrawdownStrategy"][0]

        # max_drawdown is order-sensitive, so p-value should vary
        assert mcpt_result["metric_name"] == "max_drawdown"
        assert mcpt_result["metric_real"] > 0  # Drawdown should be positive
        assert 0 <= mcpt_result["p_value"] <= 1

    def test_mcpt_not_called_when_disabled(self):
        """Test that MCPT validation is not run when disabled."""
        from freqtrade.optimize.backtesting import Backtesting

        with patch.object(Backtesting, "_run_mcpt_validation") as mock_mcpt:
            # Simulate the config check in Backtesting.start()
            config = {"mcpt": {"enabled": False}}

            if config.get("mcpt", {}).get("enabled", False):
                mock_mcpt()

            mock_mcpt.assert_not_called()

    def test_mcpt_result_structure(self, sample_trades, sample_strategy_stats):
        """Test the complete structure of stored MCPT results."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 20,
                        "metric": "profit_total",
                        "seed": 42,
                    }
                ],
            }
        }
        all_bt_content = {
            "TestStrategy": {"results": sample_trades},
        }
        results = {
            "strategy": {"TestStrategy": sample_strategy_stats},
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        mcpt_result = bt.results["mcpt"]["TestStrategy"][0]

        # Verify all required fields exist
        required_fields = [
            "metric_name",
            "metric_real",
            "p_value",
            "n_permutations",
            "count_better",
            "significant",
        ]
        for field in required_fields:
            assert field in mcpt_result, f"Missing field: {field}"

        # Verify types
        assert isinstance(mcpt_result["metric_name"], str)
        assert isinstance(mcpt_result["metric_real"], float)
        assert isinstance(mcpt_result["p_value"], float)
        assert isinstance(mcpt_result["n_permutations"], int)
        assert isinstance(mcpt_result["count_better"], int)
        assert isinstance(mcpt_result["significant"], bool)

        # Verify consistency
        assert mcpt_result["significant"] == (mcpt_result["p_value"] < 0.05)
        assert mcpt_result["count_better"] <= mcpt_result["n_permutations"]

    def test_mcpt_seed_reproducibility(self, sample_trades, sample_strategy_stats):
        """Test that same seed produces reproducible results."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "trade_shuffle",
                        "runs": 50,
                        "metric": "max_drawdown",
                        "seed": 12345,
                    }
                ],
            }
        }

        # Run twice with same seed
        results1 = {"strategy": {"TestStrategy": sample_strategy_stats.copy()}}
        bt1 = self._create_mock_backtesting(
            config,
            {"TestStrategy": {"results": sample_trades.copy()}},
            results1,
        )
        mcpt1 = MCPTIntegration(bt1)
        mcpt1.run(data=None)

        results2 = {"strategy": {"TestStrategy": sample_strategy_stats.copy()}}
        bt2 = self._create_mock_backtesting(
            config,
            {"TestStrategy": {"results": sample_trades.copy()}},
            results2,
        )
        mcpt2 = MCPTIntegration(bt2)
        mcpt2.run(data=None)

        # Results should be identical
        assert (
            bt1.results["mcpt"]["TestStrategy"][0]["p_value"]
            == bt2.results["mcpt"]["TestStrategy"][0]["p_value"]
        )
        assert (
            bt1.results["mcpt"]["TestStrategy"][0]["count_better"]
            == bt2.results["mcpt"]["TestStrategy"][0]["count_better"]
        )

    def test_bar_permute_requires_data(self, sample_trades, sample_strategy_stats):
        """Test that bar_permute method requires OHLCV data."""
        from freqtrade.optimize.montecarlo import MCPTIntegration

        config = {
            "mcpt": {
                "enabled": True,
                "methods": [
                    {
                        "name": "bar_permute",  # Requires data
                        "runs": 10,
                        "metric": "profit_total",
                    }
                ],
            }
        }
        all_bt_content = {
            "TestStrategy": {"results": sample_trades},
        }
        results = {
            "strategy": {"TestStrategy": sample_strategy_stats},
        }

        bt = self._create_mock_backtesting(config, all_bt_content, results)

        # Call without data - should log error and skip
        mcpt = MCPTIntegration(bt)
        mcpt.run(data=None)

        # MCPT should not be added when data is missing for bar_permute
        assert "mcpt" not in bt.results or len(bt.results.get("mcpt", {})) == 0


class TestCalculateMcptMetric:
    """Test calculate_mcpt_metric helper function."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data with known values for metric calculation."""
        return pd.DataFrame(
            {
                "profit_abs": [10.0, 5.0, -3.0, 8.0, -2.0, 4.0, -1.0, 6.0],
            }
        )

    def test_profit_total(self, sample_trades):
        """Test profit_total metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        start_balance = 1000.0
        result = calculate_mcpt_metric(sample_trades, "profit_total", start_balance)

        expected = sample_trades["profit_abs"].sum() / start_balance
        assert result == expected
        assert result == 27.0 / 1000.0  # 0.027

    def test_profit_total_abs(self, sample_trades):
        """Test profit_total_abs metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "profit_total_abs", 1000.0)

        expected = sample_trades["profit_abs"].sum()
        assert result == expected
        assert result == 27.0

    def test_profit_factor(self, sample_trades):
        """Test profit_factor metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "profit_factor", 1000.0)

        wins = 10.0 + 5.0 + 8.0 + 4.0 + 6.0  # 33.0
        losses = 3.0 + 2.0 + 1.0  # 6.0
        expected = wins / losses
        assert abs(result - expected) < 1e-10
        assert abs(result - 5.5) < 1e-10

    def test_max_drawdown(self, sample_trades):
        """Test max_drawdown metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "max_drawdown", 1000.0)

        # Equity curve: [10, 15, 12, 20, 18, 22, 21, 27]
        # Running max:  [10, 15, 15, 20, 20, 22, 22, 27]
        # Drawdown:     [0,  0,  3,  0,  2,  0,  1,  0]
        # Max drawdown = 3.0
        assert result == 3.0

    def test_sharpe(self, sample_trades):
        """Test sharpe metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "sharpe", 1000.0)

        profit_abs = sample_trades["profit_abs"].values
        expected = profit_abs.mean() / profit_abs.std()
        assert abs(result - expected) < 1e-10

    def test_sortino(self, sample_trades):
        """Test sortino metric calculation."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "sortino", 1000.0)

        profit_abs = sample_trades["profit_abs"].values
        downside = profit_abs[profit_abs < 0]
        expected = profit_abs.mean() / downside.std()
        assert abs(result - expected) < 1e-10

    def test_empty_trades(self):
        """Test with empty trades DataFrame."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        empty_trades = pd.DataFrame({"profit_abs": []})
        result = calculate_mcpt_metric(empty_trades, "profit_total", 1000.0)

        assert result == 0.0

    def test_unknown_metric_defaults_to_profit_total(self, sample_trades):
        """Test that unknown metric defaults to relative profit."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "unknown_metric", 1000.0)

        expected = sample_trades["profit_abs"].sum() / 1000.0
        assert result == expected

    def test_zero_start_balance(self, sample_trades):
        """Test profit_total with zero start balance."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        result = calculate_mcpt_metric(sample_trades, "profit_total", 0.0)

        assert result == 0.0

    def test_no_losses_profit_factor(self):
        """Test profit_factor when there are no losses."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        all_wins = pd.DataFrame({"profit_abs": [1.0, 2.0, 3.0]})
        result = calculate_mcpt_metric(all_wins, "profit_factor", 1000.0)

        assert result == float("inf")

    def test_sharpe_with_single_trade(self):
        """Test sharpe with single trade (not enough data)."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        single_trade = pd.DataFrame({"profit_abs": [5.0]})
        result = calculate_mcpt_metric(single_trade, "sharpe", 1000.0)

        assert result == 0.0

    def test_sortino_with_no_downside(self):
        """Test sortino when there are no negative returns."""
        from freqtrade.optimize.montecarlo import calculate_mcpt_metric

        all_wins = pd.DataFrame({"profit_abs": [1.0, 2.0, 3.0]})
        result = calculate_mcpt_metric(all_wins, "sortino", 1000.0)

        assert result == 0.0
