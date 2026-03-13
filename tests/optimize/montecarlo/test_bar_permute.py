"""Tests for BarPermutePermuter."""

import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.montecarlo.permuters.bar_permute import BarPermutePermuter


class TestBarPermutePermuter:
    """Test BarPermutePermuter functionality."""

    @pytest.fixture
    def sample_ohlcv(self):
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_bars = 100

        return pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n_bars, freq="1h"),
                "open": np.random.uniform(100, 110, n_bars),
                "high": np.random.uniform(110, 120, n_bars),
                "low": np.random.uniform(90, 100, n_bars),
                "close": np.random.uniform(100, 110, n_bars),
                "volume": np.random.uniform(1000, 5000, n_bars),
            }
        )

    @pytest.fixture
    def multi_pair_data(self, sample_ohlcv):
        """Create multi-pair OHLCV data."""
        return {
            "BTC/USDT": sample_ohlcv.copy(),
            "ETH/USDT": sample_ohlcv.copy(),
        }

    def test_init_default(self):
        """Test default initialization."""
        permuter = BarPermutePermuter()

        assert permuter.block_size == 20
        assert permuter.overlap is False

    def test_init_custom(self):
        """Test custom initialization."""
        permuter = BarPermutePermuter(block_size=50, overlap=True)

        assert permuter.block_size == 50
        assert permuter.overlap is True

    def test_permute_single_df(self, sample_ohlcv):
        """Test permutation of single DataFrame."""
        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(sample_ohlcv, seed=42)

        # Should have same shape
        assert len(permuted) == len(sample_ohlcv)
        assert list(permuted.columns) == list(sample_ohlcv.columns)

        # Should have different order (most likely)
        # Check that at least some values are in different positions
        original_close = sample_ohlcv["close"].values
        permuted_close = permuted["close"].values

        # Not all values should be in same position
        same_position = sum(original_close == permuted_close)
        assert same_position < len(original_close)

    def test_permute_multi_pair(self, multi_pair_data):
        """Test permutation of multi-pair data."""
        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(multi_pair_data, seed=42)

        # Should have same pairs
        assert set(permuted.keys()) == set(multi_pair_data.keys())

        # Each pair should have same length
        for pair in multi_pair_data:
            assert len(permuted[pair]) == len(multi_pair_data[pair])

    def test_permute_preserves_distribution(self, sample_ohlcv):
        """Returns-based permutation should preserve coarse statistics of closes."""
        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(sample_ohlcv, seed=42)

        orig_close = sample_ohlcv["close"].values
        perm_close = permuted["close"].values

        # Ensure all generated prices remain positive to avoid log artifacts
        assert np.all(orig_close > 0)
        assert np.all(perm_close > 0)

        # Log-return distribution (which is preserved by returns-based permutation)
        orig_returns = np.diff(np.log(orig_close))
        perm_returns = np.diff(np.log(perm_close))

        assert np.isclose(orig_returns.mean(), perm_returns.mean(), atol=1e-3)
        assert np.isclose(orig_returns.std(), perm_returns.std(), atol=5e-3)

    def test_permute_reproducibility(self, sample_ohlcv):
        """Test that same seed produces same result."""
        permuter = BarPermutePermuter(block_size=10)

        permuted1 = permuter.permute(sample_ohlcv, seed=42)
        permuted2 = permuter.permute(sample_ohlcv, seed=42)

        pd.testing.assert_frame_equal(permuted1, permuted2)

    def test_permute_different_seeds(self, sample_ohlcv):
        """Test that different seeds produce different results."""
        permuter = BarPermutePermuter(block_size=10)

        permuted1 = permuter.permute(sample_ohlcv, seed=42)
        permuted2 = permuter.permute(sample_ohlcv, seed=43)

        # Should be different
        assert not permuted1["close"].equals(permuted2["close"])

    def test_permute_batch(self, sample_ohlcv):
        """Test batch permutation generation."""
        permuter = BarPermutePermuter(block_size=10)
        n_permutations = 5

        permutations = list(permuter.permute_batch(sample_ohlcv, n_permutations, base_seed=42))

        assert len(permutations) == n_permutations

        # Each permutation should be different
        for i in range(len(permutations) - 1):
            assert not permutations[i]["close"].equals(permutations[i + 1]["close"])

    def test_permute_small_data(self):
        """Test permutation with data smaller than block size."""
        small_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=5, freq="1h"),
                "close": [1, 2, 3, 4, 5],
            }
        )

        permuter = BarPermutePermuter(block_size=20)  # Larger than data
        permuted = permuter.permute(small_data, seed=42)

        # Should still work, just shuffle rows
        assert len(permuted) == len(small_data)

    def test_permute_with_overlap(self, sample_ohlcv):
        """Test permutation with overlapping blocks."""
        permuter = BarPermutePermuter(block_size=10, overlap=True)
        permuted = permuter.permute(sample_ohlcv, seed=42)

        assert len(permuted) == len(sample_ohlcv)

    def test_properties(self):
        """Test properties lists."""
        permuter = BarPermutePermuter()

        # Should have non-empty lists
        assert len(permuter.preserves_properties) > 0
        assert len(permuter.breaks_properties) > 0

        # Should not contain None
        assert None not in permuter.preserves_properties
        assert None not in permuter.breaks_properties

        # Check expected properties
        assert "marginal_distribution" in permuter.preserves_properties
        assert "long_term_trends" in permuter.breaks_properties

    def test_date_column_preserved(self, sample_ohlcv):
        """Test that date column maintains original sequence."""
        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(sample_ohlcv, seed=42)

        # Date column should be same as original (monotonic)
        pd.testing.assert_series_equal(
            permuted["date"].reset_index(drop=True),
            sample_ohlcv["date"].reset_index(drop=True),
        )

    def test_permute_price_continuity(self, sample_ohlcv):
        """Test that permutation preserves price continuity (no jumps)."""
        # Create data with large jump between potential blocks
        # Block 1: 100-110, Block 2: 200-210
        # If we permute raw blocks, we might get 110 -> 200 jump
        # With returns permutation, we should get smooth transition

        # Create synthetic data with obvious trend
        n_bars = 40
        trend_data = pd.DataFrame(
            {
                "date": pd.date_range("2023-01-01", periods=n_bars, freq="1h"),
                "open": np.linspace(100, 200, n_bars),
                "high": np.linspace(105, 205, n_bars),
                "low": np.linspace(95, 195, n_bars),
                "close": np.linspace(102, 202, n_bars),
                "volume": np.ones(n_bars) * 1000,
            }
        )

        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(trend_data, seed=42)

        # Check inter-bar gaps
        # Gap = Open[t+1] - Close[t]
        # In original data, gaps are small (~0.5)
        # In permuted data, gaps should also be small (distribution preserved)

        perm_gaps = permuted["open"].iloc[1:].values - permuted["close"].iloc[:-1].values

        # Max gap in permuted data should be comparable to original
        # (not 100 like in raw block shuffle)
        assert np.max(np.abs(perm_gaps)) < 5.0  # Well below the 100 jump

    def test_permute_ohlc_geometry(self, sample_ohlcv):
        """Test that candle geometry is preserved."""
        permuter = BarPermutePermuter(block_size=10)
        permuted = permuter.permute(sample_ohlcv, seed=42)

        # Check High >= Low
        assert np.all(permuted["high"] >= permuted["low"])

        # Check High >= Open and High >= Close
        assert np.all(permuted["high"] >= permuted["open"])
        assert np.all(permuted["high"] >= permuted["close"])

        # Check Low <= Open and Low <= Close
        assert np.all(permuted["low"] <= permuted["open"])
        assert np.all(permuted["low"] <= permuted["close"])

    def test_block_structure(self, sample_ohlcv):
        """Test that blocks are properly formed."""
        block_size = 10
        permuter = BarPermutePermuter(block_size=block_size, overlap=False)

        # For returns-based permutation, we can't check values directly
        # as they are reconstructed. We verify by checking if segments of
        # returns/deltas match original segments.

        # Calculate intra-bar deltas (High-Open)
        orig_deltas = np.log(sample_ohlcv["high"]) - np.log(sample_ohlcv["open"])

        # Get permuted data
        permuted = permuter.permute(sample_ohlcv, seed=42)
        perm_deltas = np.log(permuted["high"]) - np.log(permuted["open"])

        # Find positions of permuted deltas in original
        # Rounding to avoid float precision issues
        orig_deltas = np.round(orig_deltas, 6)
        perm_deltas = np.round(perm_deltas, 6)

        positions = []
        for val in perm_deltas:
            pos = np.where(orig_deltas == val)[0]
            if len(pos) > 0:
                positions.append(pos[0])

        # Count how many consecutive pairs have consecutive positions
        consecutive_count = 0
        for i in range(len(positions) - 1):
            if positions[i + 1] == positions[i] + 1:
                consecutive_count += 1

        # With block bootstrap, should have many consecutive pairs
        # (at least within each block)
        expected_min = (len(positions) // block_size) * (block_size - 1) * 0.5
        assert consecutive_count >= expected_min

    def test_multi_timeframe_permutation(self):
        """Test synchronized multi-timeframe permutation."""
        np.random.seed(42)

        # Create 5m data (20 bars = 100 minutes)
        n_5m = 20
        main_data = {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=n_5m, freq="5min"),
                    "open": np.arange(n_5m) * 10 + 100,  # Ensure strictly positive prices
                    "high": np.arange(n_5m) * 10 + 105,
                    "low": np.arange(n_5m) * 10 + 95,
                    "close": np.arange(n_5m) * 10 + 101,
                    "volume": np.ones(n_5m) * 1000,
                }
            )
        }

        # Create 1m data (100 bars = same 100 minutes)
        n_1m = 100
        detail_data = {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=n_1m, freq="1min"),
                    "open": np.arange(n_1m) + 200,
                    "high": np.arange(n_1m) + 200.5,
                    "low": np.arange(n_1m) + 199.5,
                    "close": np.arange(n_1m) + 200.1,
                    "volume": np.ones(n_1m) * 100,
                }
            )
        }

        permuter = BarPermutePermuter(block_size=5)  # 5 bars of 5m = 25 minutes

        # Permute both timeframes
        perm_main, perm_detail = permuter.permute_multi_timeframe(
            main_data, detail_data, "5m", "1m", seed=42
        )

        # Check that both have same pairs
        assert set(perm_main.keys()) == set(main_data.keys())
        assert set(perm_detail.keys()) == set(detail_data.keys())

        # Check lengths preserved
        assert len(perm_main["BTC/USDT"]) == n_5m
        assert len(perm_detail["BTC/USDT"]) == n_1m

        # Check that block structure is synchronized
        # If 5m block 0 (bars 0-4) moved somewhere, then 1m block 0 (bars 0-24)
        # should move to corresponding position
        main_first_open = perm_main["BTC/USDT"]["open"].iloc[0]
        detail_first_open = perm_detail["BTC/USDT"]["open"].iloc[0]

        # First 5m bar's open value divided by 10 gives block index
        main_block_idx = int((main_first_open - 100) // 10)
        # First 1m bar's open value divided by 5 gives block index (5 1m bars per 5m block)
        detail_block_idx = int((detail_first_open - 200) // 5)

        # They should point to same time block
        assert main_block_idx == detail_block_idx

    def test_multi_timeframe_batch(self):
        """Test batch generation for multi-timeframe permutation."""
        np.random.seed(42)

        main_data = {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=20, freq="5min"),
                    "open": np.random.uniform(100, 110, 20),
                    "high": np.random.uniform(110, 120, 20),
                    "low": np.random.uniform(90, 100, 20),
                    "close": np.random.uniform(100, 110, 20),
                    "volume": np.random.uniform(1000, 5000, 20),
                }
            )
        }

        detail_data = {
            "BTC/USDT": pd.DataFrame(
                {
                    "date": pd.date_range("2023-01-01", periods=100, freq="1min"),
                    "open": np.random.uniform(100, 110, 100),
                    "high": np.random.uniform(110, 120, 100),
                    "low": np.random.uniform(90, 100, 100),
                    "close": np.random.uniform(100, 110, 100),
                    "volume": np.random.uniform(1000, 5000, 100),
                }
            )
        }

        permuter = BarPermutePermuter(block_size=5)

        # Generate 3 permutations
        results = list(
            permuter.permute_batch_multi_timeframe(
                main_data, detail_data, "5m", "1m", n_permutations=3, base_seed=42
            )
        )

        assert len(results) == 3

        # Each result should be a tuple of (main, detail)
        for perm_main, perm_detail in results:
            assert "BTC/USDT" in perm_main
            assert "BTC/USDT" in perm_detail
            assert len(perm_main["BTC/USDT"]) == 20
            assert len(perm_detail["BTC/USDT"]) == 100
