import numpy as np
import pandas as pd
import pytest

from freqtrade.optimize.montecarlo.permuters.trade_shuffle import TradeShufflePermuter


class TestTradeShufflePermuter:
    """Test TradeShufflePermuter functionality."""

    @pytest.fixture
    def sample_trades(self):
        """Create sample trade data for testing."""
        np.random.seed(42)
        n_trades = 100

        return pd.DataFrame(
            {
                "pair": np.random.choice(["BTC/USDT", "ETH/USDT"], n_trades),
                "direction": np.random.choice([1, -1], n_trades),
                "profit_abs": np.random.normal(0.01, 0.05, n_trades),
                "open_time": pd.date_range("2023-01-01", periods=n_trades, freq="1h"),
            }
        )

    def test_global_scope_permutation(self, sample_trades):
        """Test global scope permutation preserves distribution."""
        permuter = TradeShufflePermuter(scope="global")

        # Get original values
        original_pnl = sample_trades["profit_abs"].values.copy()

        # Permute with fixed seed
        permuted = permuter.permute(sample_trades, seed=123)
        permuted_pnl = permuted["profit_abs"].values

        # Check that values are shuffled (not in same order)
        assert not np.array_equal(original_pnl, permuted_pnl)

        # Check that distribution is preserved (same multiset)
        assert np.array_equal(np.sort(original_pnl), np.sort(permuted_pnl))

        # Check that other columns are unchanged
        assert np.array_equal(sample_trades["pair"].values, permuted["pair"].values)
        assert np.array_equal(sample_trades["direction"].values, permuted["direction"].values)
        assert np.array_equal(sample_trades["open_time"].values, permuted["open_time"].values)

    def test_by_pair_scope_permutation(self, sample_trades):
        """Test by_pair scope permutation preserves distribution within each pair."""
        permuter = TradeShufflePermuter(scope="by_pair")

        # Get original values by pair
        btc_original = sample_trades[sample_trades["pair"] == "BTC/USDT"]["profit_abs"].values
        eth_original = sample_trades[sample_trades["pair"] == "ETH/USDT"]["profit_abs"].values

        # Permute with fixed seed
        permuted = permuter.permute(sample_trades, seed=123)

        # Check that values are shuffled within each pair
        btc_permuted = permuted[permuted["pair"] == "BTC/USDT"]["profit_abs"].values
        eth_permuted = permuted[permuted["pair"] == "ETH/USDT"]["profit_abs"].values

        assert not np.array_equal(btc_original, btc_permuted)
        assert not np.array_equal(eth_original, eth_permuted)

        # Check that distributions are preserved within pairs
        assert np.array_equal(np.sort(btc_original), np.sort(btc_permuted))
        assert np.array_equal(np.sort(eth_original), np.sort(eth_permuted))

    def test_by_direction_scope_permutation(self, sample_trades):
        """Test by_direction scope permutation preserves distribution within directions."""
        permuter = TradeShufflePermuter(scope="by_direction")

        # Get original values by direction
        long_original = sample_trades[sample_trades["direction"] == 1]["profit_abs"].values
        short_original = sample_trades[sample_trades["direction"] == -1]["profit_abs"].values

        # Permute with fixed seed
        permuted = permuter.permute(sample_trades, seed=123)

        # Check that values are shuffled within each direction
        long_permuted = permuted[permuted["direction"] == 1]["profit_abs"].values
        short_permuted = permuted[permuted["direction"] == -1]["profit_abs"].values

        if len(long_original) > 0 and len(short_original) > 0:
            assert not np.array_equal(long_original, long_permuted)
            assert not np.array_equal(short_original, short_permuted)

            # Check that distributions are preserved within directions
            assert np.array_equal(np.sort(long_original), np.sort(long_permuted))
            assert np.array_equal(np.sort(short_original), np.sort(short_permuted))

    def test_permute_batch(self, sample_trades):
        """Test batch permutation generator."""
        permuter = TradeShufflePermuter(scope="global")
        n_permutations = 5

        permutations = list(permuter.permute_batch(sample_trades, n_permutations, base_seed=100))

        assert len(permutations) == n_permutations

        # Check that each permutation is different
        for i in range(len(permutations)):
            for j in range(i + 1, len(permutations)):
                assert not np.array_equal(
                    permutations[i]["profit_abs"].values, permutations[j]["profit_abs"].values
                )

        # Check that all preserve the distribution
        original_sorted = np.sort(sample_trades["profit_abs"].values)
        for perm in permutations:
            assert np.array_equal(original_sorted, np.sort(perm["profit_abs"].values))

    def test_invalid_scope(self):
        """Test that invalid scope raises ValueError."""
        with pytest.raises(ValueError, match="Invalid scope"):
            TradeShufflePermuter(scope="invalid")

    def test_properties(self):
        """Test preserved and broken properties."""
        permuter = TradeShufflePermuter(scope="global")

        preserved = permuter.preserves_properties
        broken = permuter.breaks_properties

        assert "marginal_distribution" in preserved
        assert "trade_count" in preserved
        assert "autocorrelation" in broken
        assert "temporal_sequence" in broken

        # Check that None values are filtered out for global scope
        assert None not in preserved
