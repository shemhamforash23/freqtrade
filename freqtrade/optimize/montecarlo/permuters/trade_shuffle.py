"""
Trade Shuffle permutation for MCPT with optional GPU acceleration.

Uses JAX for GPU acceleration on NVIDIA CUDA when available,
falls back to optimized NumPy otherwise.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Generator

import numpy as np
from pandas import DataFrame

from freqtrade.optimize.montecarlo.permuters.base_permuter import IPermuter


logger = logging.getLogger(__name__)

# JAX GPU support - only load on Linux (NVIDIA CUDA)
# Skip on macOS to avoid loading Metal backend which is broken on Python 3.13
JAX_AVAILABLE = False
JAX_USE_GPU = False
jax = None  # type: ignore
jnp = None  # type: ignore

if sys.platform == "linux":
    try:
        import jax as _jax
        import jax.numpy as _jnp

        _jax_devices = _jax.devices()
        _device_type = str(_jax_devices[0]).split("(")[0] if _jax_devices else "cpu"

        if _device_type in ("cuda", "gpu", "CUDA", "GPU"):
            jax = _jax
            jnp = _jnp
            JAX_AVAILABLE = True
            JAX_USE_GPU = True
            logger.info(f"JAX CUDA available for Trade Shuffle, devices: {_jax_devices}")
        else:
            logger.debug("JAX CPU detected, using NumPy for better performance")
    except ImportError:
        logger.debug("JAX not installed, using NumPy for Trade Shuffle")
else:
    logger.debug(f"Platform {sys.platform}: using NumPy for Trade Shuffle")


class TradeShufflePermuter(IPermuter):
    """
    Trade Shuffle permutation for MCPT.

    Uses JAX for GPU acceleration on NVIDIA CUDA when available,
    falls back to optimized NumPy otherwise.

    Shuffles the order of trade P&L values while preserving:
    - Individual trade P&L distribution
    - Total number of trades
    - Trade count per pair (if scope is not global)

    Breaks:
    - Temporal sequence of trades
    - Autocorrelation in returns
    - Volatility clustering patterns
    """

    def __init__(self, scope: str = "global") -> None:
        """
        Initialize TradeShufflePermuter.

        :param scope: Permutation scope - 'global', 'by_pair', or 'by_direction'
        """
        if scope not in ["global", "by_pair", "by_direction"]:
            raise ValueError(
                f"Invalid scope: {scope}. Must be 'global', 'by_pair', or 'by_direction'"
            )
        self.scope = scope
        self._use_jax = JAX_AVAILABLE

    def permute(self, data: DataFrame, seed: int | None = None) -> DataFrame:
        """
        Generate a single shuffled version of trade data.

        :param data: DataFrame with trades (must have 'profit_abs' column)
        :param seed: Random seed for reproducibility
        :return: DataFrame with shuffled profit_abs values
        """
        rng = np.random.default_rng(seed)
        profit_arr = data["profit_abs"].values.copy()

        if self.scope == "global":
            perm_indices = rng.permutation(len(profit_arr))
            if self._use_jax:
                # Use JAX for fast array indexing on GPU
                profit_jax = jnp.array(profit_arr)
                shuffled = np.asarray(profit_jax[perm_indices])
            else:
                shuffled = profit_arr[perm_indices]

        elif self.scope == "by_pair":
            shuffled = np.empty_like(profit_arr)
            for pair in data["pair"].unique():
                mask = data["pair"] == pair
                pair_profits = profit_arr[mask]
                perm_indices = rng.permutation(len(pair_profits))
                shuffled[mask] = pair_profits[perm_indices]

        else:  # by_direction
            shuffled = profit_arr.copy()
            for direction in [1, -1]:
                mask = data["direction"] == direction
                if mask.any():
                    dir_profits = profit_arr[mask]
                    perm_indices = rng.permutation(len(dir_profits))
                    shuffled[mask] = dir_profits[perm_indices]

        df = data.copy()
        df["profit_abs"] = shuffled
        return df

    def permute_batch(
        self,
        data: DataFrame,
        n_permutations: int,
        base_seed: int | None = None,
    ) -> Generator[DataFrame, None, None]:
        """
        Generator yielding n_permutations shuffled versions.

        For global scope uses a vectorized batch path: generates all N permutation
        index matrices in a single rng.permuted() call instead of N separate calls.

        :param data: Original DataFrame with trades
        :param n_permutations: Number of permutations to generate
        :param base_seed: Base seed for reproducibility
        :yields: DataFrame with shuffled profit_abs values
        """
        if self.scope == "global" and not self._use_jax and n_permutations > 1:
            yield from self._permute_batch_global_vectorized(data, n_permutations, base_seed)
        else:
            for i in range(n_permutations):
                seed = base_seed + i if base_seed is not None else None
                yield self.permute(data, seed)

    def _permute_batch_global_vectorized(
        self,
        data: DataFrame,
        n_permutations: int,
        base_seed: int | None = None,
    ) -> Generator[DataFrame, None, None]:
        """
        Vectorized batch permutation for global scope.

        Generates all N shuffled profit_abs arrays at once via a single
        rng.permuted() call on a (n_permutations, n_trades) matrix.
        Eliminates N separate RNG creations and N sequential rng.permutation() calls.

        :param data: Original DataFrame with trades
        :param n_permutations: Number of permutations to generate
        :param base_seed: Random seed for reproducibility
        :yields: DataFrame with shuffled profit_abs values
        """
        profit_arr = data["profit_abs"].values
        n = len(profit_arr)
        rng = np.random.default_rng(base_seed)

        # Generate all N permutation indices in one vectorized call: (n_permutations, n)
        indices_matrix = np.tile(np.arange(n), (n_permutations, 1))
        rng.permuted(indices_matrix, axis=1, out=indices_matrix)

        # Apply all permutations at once: (n_permutations, n)
        shuffled_batch = profit_arr[indices_matrix]

        template = data.copy()
        for i in range(n_permutations):
            df = template.copy()
            df["profit_abs"] = shuffled_batch[i]
            yield df

    @property
    def preserves_properties(self) -> list[str]:
        """Properties preserved by Trade Shuffle."""
        properties = [
            "marginal_distribution",
            "trade_count",
            "profit_distribution",
            "pair_distribution" if self.scope != "global" else None,
            "direction_distribution" if self.scope != "global" else None,
        ]
        return [p for p in properties if p is not None]

    @property
    def breaks_properties(self) -> list[str]:
        """Properties broken by Trade Shuffle."""
        return [
            "autocorrelation",
            "temporal_sequence",
            "volatility_clustering",
            "momentum_patterns",
            "mean_reversion_patterns",
        ]
