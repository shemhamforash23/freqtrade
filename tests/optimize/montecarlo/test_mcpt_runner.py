"""
Tests for MCPTRunner class.

Tests the cross-platform parallel execution functionality.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.montecarlo.mcpt_runner import MCPTRunner


class TestMCPTRunner:
    """Test MCPTRunner functionality."""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration."""
        return {
            "stake_currency": "USDT",
            "stake_amount": 100,
            "dry_run_wallet": 1000,
            "timeframe": "5m",
            "strategy": "TestStrategy",
        }

    @pytest.fixture
    def sample_timerange(self):
        """Create sample timerange mock."""
        tr = MagicMock()
        tr.startts = 1609459200000  # 2021-01-01
        tr.stopts = 1609545600000  # 2021-01-02
        return tr

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_bars = 100

        return {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2021-01-01", periods=n_bars, freq="5min"),
                    "open": np.random.uniform(30000, 35000, n_bars),
                    "high": np.random.uniform(35000, 40000, n_bars),
                    "low": np.random.uniform(25000, 30000, n_bars),
                    "close": np.random.uniform(30000, 35000, n_bars),
                    "volume": np.random.uniform(100, 1000, n_bars),
                }
            )
        }

    def test_init(self, sample_config, sample_timerange):
        """Test MCPTRunner initialization."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
            timeframe_detail=None,
        )

        assert runner.config == sample_config
        assert runner.timerange == sample_timerange
        assert runner.required_startup == 50
        assert runner.timeframe == "5m"
        assert runner.timeframe_detail is None
        assert runner._backtesting is None  # Lazy init
        assert runner._temp_dir.exists()

    def test_init_with_detail(self, sample_config, sample_timerange):
        """Test MCPTRunner initialization with detail timeframe."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
            timeframe_detail="1m",
        )

        assert runner.timeframe_detail == "1m"

    def test_save_permuted_data(self, sample_config, sample_timerange, sample_ohlcv_data):
        """Test saving permuted data to pickle files."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
        )

        # Save data
        data_file, detail_file = runner.save_permuted_data(
            run_index=0,
            permuted_data=sample_ohlcv_data,
            permuted_detail=None,
        )

        # Verify file was created
        assert Path(data_file).exists()
        assert detail_file is None

        # Cleanup
        runner.cleanup_temp_files([0])
        assert not Path(data_file).exists()

    def test_save_permuted_data_with_detail(
        self, sample_config, sample_timerange, sample_ohlcv_data
    ):
        """Test saving permuted data with detail timeframe."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
            timeframe_detail="1m",
        )

        detail_data = {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2021-01-01", periods=500, freq="1min"),
                    "open": np.random.uniform(30000, 35000, 500),
                    "high": np.random.uniform(35000, 40000, 500),
                    "low": np.random.uniform(25000, 30000, 500),
                    "close": np.random.uniform(30000, 35000, 500),
                    "volume": np.random.uniform(100, 1000, 500),
                }
            )
        }

        # Save data
        data_file, detail_file = runner.save_permuted_data(
            run_index=1,
            permuted_data=sample_ohlcv_data,
            permuted_detail=detail_data,
        )

        # Verify files were created
        assert Path(data_file).exists()
        assert Path(detail_file).exists()

        # Cleanup
        runner.cleanup_temp_files([1])
        assert not Path(data_file).exists()
        assert not Path(detail_file).exists()

    def test_cleanup_multiple_files(self, sample_config, sample_timerange, sample_ohlcv_data):
        """Test cleanup of multiple temporary files."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
        )

        # Save multiple files
        files = []
        for i in range(5):
            data_file, _ = runner.save_permuted_data(i, sample_ohlcv_data, None)
            files.append(data_file)

        # Verify all files exist
        for f in files:
            assert Path(f).exists()

        # Cleanup all
        runner.cleanup_temp_files(list(range(5)))

        # Verify all files removed
        for f in files:
            assert not Path(f).exists()

    def test_temp_dir_creation(self, sample_config, sample_timerange):
        """Test that temp directory is created properly."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
        )

        assert runner._temp_dir.exists()
        assert runner._temp_dir.name == "freqtrade_mcpt"

    def test_run_permuted_backtest_is_delayed(self, sample_config, sample_timerange):
        """Test that run_permuted_backtest has @delayed decorator."""
        runner = MCPTRunner(
            config=sample_config,
            timerange=sample_timerange,
            required_startup=50,
            timeframe="5m",
        )

        # Check that the method returns a Delayed object when called
        # (This is a characteristic of @delayed decorator)
        result = runner.run_permuted_backtest(0, "/tmp/test.pkl", None, "profit_total")

        # The result should be a delayed callable, not an immediate result
        # joblib.delayed returns a tuple (func, args, kwargs)
        assert result is not None
        # Delayed objects are tuples
        assert isinstance(result, tuple)
