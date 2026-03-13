"""
Bar Permutation Permuter for MCPT.

Uses Returns-Based Block Bootstrap to permute OHLCV data while:
- Preserving realistic price continuity (no artificial jumps)
- Maintaining OHLC bar geometry (intra-bar relationships)
- Breaking temporal autocorrelation between blocks

This method requires full backtest re-run for each permutation.

Supports synchronized permutation across multiple timeframes to ensure
consistency between main timeframe (e.g., 5m) and detail timeframe (e.g., 1m).

Based on best practices from quantitative finance research:
- Log returns permutation preserves marginal distribution
- Intra-bar deltas preserve candle geometry
- Block bootstrap preserves short-term volatility clustering
"""

from collections.abc import Generator

import numpy as np
from pandas import DataFrame

from freqtrade.optimize.montecarlo.permuters.base_permuter import IPermuter


def timeframe_to_minutes(timeframe: str) -> int:
    """Convert timeframe string to minutes."""
    multipliers = {"m": 1, "h": 60, "d": 1440, "w": 10080}
    unit = timeframe[-1].lower()
    value = int(timeframe[:-1])
    return value * multipliers.get(unit, 1)


class BarPermutePermuter(IPermuter):
    """
    Permuter that shuffles OHLCV bars using Returns-Based Block Bootstrap.

    Instead of shuffling raw price blocks (which creates unrealistic jumps),
    this permuter works with log returns:

    1. Computes intra-bar deltas: ΔH, ΔL, ΔC relative to Open
    2. Computes inter-bar returns: log(O[t+1]/C[t])
    3. Shuffles blocks of returns (preserving local structure)
    4. Reconstructs prices from shuffled returns

    This approach:
    - Eliminates artificial price jumps between blocks
    - Preserves realistic OHLC candle geometry
    - Maintains marginal distribution of returns
    - Breaks long-term trends and autocorrelation

    Supports multi-timeframe synchronized permutation for consistency
    between main and detail timeframes.
    """

    def __init__(self, block_size: int = 20, overlap: bool = False):
        """
        Initialize BarPermutePermuter.

        :param block_size: Number of bars per block (default: 20)
        :param overlap: Whether to allow overlapping blocks (default: False)
        """
        self.block_size = block_size
        self.overlap = overlap

    def permute(
        self, data: dict[str, DataFrame] | DataFrame, seed: int | None = None
    ) -> dict[str, DataFrame] | DataFrame:
        """
        Permute OHLCV data using block bootstrap.

        :param data: Dict of pair -> DataFrame or single DataFrame with OHLCV data
        :param seed: Random seed for reproducibility
        :return: Permuted data in same format as input
        """
        rng = np.random.default_rng(seed)

        if isinstance(data, dict):
            # Multi-pair data: _permute_df_with_indices creates its own copy, so no need to copy here
            return {pair: self._permute_df(df, rng) for pair, df in data.items()}
        else:
            # Single DataFrame: same as above
            return self._permute_df(data, rng)

    def _permute_df(self, df: DataFrame, rng: np.random.Generator) -> DataFrame:
        """
        Permute a single DataFrame using Returns-Based Block Bootstrap.

        This method:
        1. Converts OHLCV to log-space
        2. Computes intra-bar deltas (preserving candle geometry)
        3. Computes inter-bar returns (close-to-open gaps)
        4. Shuffles blocks of these components
        5. Reconstructs prices without artificial jumps

        :param df: DataFrame with OHLCV data (must have open, high, low, close columns)
        :param rng: Random number generator
        :return: Permuted DataFrame with realistic price continuity
        """
        n_rows = len(df)
        if n_rows <= self.block_size:
            # Data too small for block bootstrap, return as-is
            return df.copy()

        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            # Fallback to simple index permutation if no OHLC
            return self._permute_df_simple(df, rng)

        block_indices = self._generate_block_indices(n_rows, rng)
        return self._permute_df_with_indices(df, block_indices)

    def _permute_df_simple(self, df: DataFrame, rng: np.random.Generator) -> DataFrame:
        """
        Simple index-based permutation fallback when OHLC not available.

        :param df: DataFrame to permute
        :param rng: Random number generator
        :return: Permuted DataFrame
        """
        block_indices = self._generate_block_indices(len(df), rng)
        result = df.iloc[block_indices].reset_index(drop=True)
        if "date" in df.columns:
            result["date"] = df["date"].reset_index(drop=True)
        return result

    def _generate_block_indices(self, n_rows: int, rng: np.random.Generator) -> np.ndarray:
        """
        Generate permuted indices using block bootstrap.

        :param n_rows: Number of rows in data
        :param rng: Random number generator
        :return: Array of permuted indices
        """
        if n_rows <= self.block_size:
            return rng.permutation(n_rows)

        # Calculate number of blocks needed
        n_blocks = (n_rows + self.block_size - 1) // self.block_size

        # Generate random block starting positions
        if self.overlap:
            max_start = n_rows - self.block_size
            block_starts = rng.integers(0, max_start + 1, size=n_blocks)
        else:
            available_starts = list(range(0, n_rows - self.block_size + 1, self.block_size))
            block_starts = rng.choice(available_starts, size=n_blocks, replace=True)

        # Build permuted indices from blocks via vectorized broadcasting
        # available_starts guarantees every start + block_size <= n_rows, so no clipping needed
        starts_arr = np.asarray(block_starts)[:, np.newaxis]  # (n_blocks, 1)
        offsets = np.arange(self.block_size)[np.newaxis, :]  # (1, block_size)
        permuted_indices = (starts_arr + offsets).ravel()  # (n_blocks * block_size,)
        return permuted_indices[:n_rows]

    def _generate_block_mapping(self, n_rows: int, rng: np.random.Generator) -> list[int]:
        """
        Generate block index mapping for permutation.

        :param n_rows: Number of rows in data
        :param rng: Random number generator
        :return: List of block start indices in permuted order
        """
        if n_rows <= self.block_size:
            return list(rng.permutation(n_rows))

        n_blocks = (n_rows + self.block_size - 1) // self.block_size

        if self.overlap:
            max_start = n_rows - self.block_size
            block_starts = rng.integers(0, max_start + 1, size=n_blocks)
        else:
            available_starts = list(range(0, n_rows - self.block_size + 1, self.block_size))
            block_starts = rng.choice(available_starts, size=n_blocks, replace=True)

        return list(block_starts)

    def _permute_df_with_indices(self, df: DataFrame, block_indices: np.ndarray) -> DataFrame:
        """
        Permute DataFrame using pre-generated block indices with returns-based method.

        :param df: DataFrame with OHLCV data
        :param block_indices: Pre-generated permutation indices
        :return: Permuted DataFrame with realistic price continuity
        """
        n_rows = len(df)

        # Ensure we have required columns
        required_cols = ["open", "high", "low", "close"]
        if not all(col in df.columns for col in required_cols):
            # Fallback to simple index permutation if no OHLC
            result = df.iloc[block_indices].reset_index(drop=True)
            if "date" in df.columns:
                result["date"] = df["date"].reset_index(drop=True)
            return result

        # Step 1: Compute intra-bar price ratios (no log/exp needed)
        # high_ratio = high/open, low_ratio = low/open, close_ratio = close/open
        eps = 1e-10
        open_vals = df["open"].values.astype(np.float64) + eps
        high_ratio = df["high"].values / open_vals
        low_ratio = df["low"].values / open_vals
        close_ratio = (df["close"].values + eps) / open_vals

        # Step 2: Inter-bar factors: open[i] / close[i-1]
        # Permuting whole blocks keeps candle geometry intact
        inter_factor = np.ones(n_rows, dtype=np.float64)
        inter_factor[1:] = open_vals[1:] / (df["close"].values[:-1] + eps)

        # Step 3: Permute all ratio arrays using provided indices
        perm_high_ratio = high_ratio[block_indices]
        perm_low_ratio = low_ratio[block_indices]
        perm_close_ratio = close_ratio[block_indices]
        perm_inter_factor = inter_factor[block_indices]

        # Step 4: Reconstruct open prices via cumprod (no exp needed)
        # new_open[i] = open[0] * prod_{j=0}^{i-1}(close_ratio[j] * inter_factor[j+1])
        combined_factor = perm_close_ratio[:-1] * perm_inter_factor[1:]
        new_open = np.empty(n_rows, dtype=np.float64)
        new_open[0] = df["open"].values[0]
        new_open[1:] = new_open[0] * np.cumprod(combined_factor)

        # Step 5: Derive remaining prices from open via permuted ratios
        result = df.copy()
        result["open"] = new_open
        result["high"] = new_open * perm_high_ratio
        result["low"] = new_open * perm_low_ratio
        result["close"] = new_open * perm_close_ratio

        if "volume" in df.columns:
            result["volume"] = df["volume"].values[block_indices]

        return result

    def _block_starts_to_indices(
        self, block_starts: list[int], block_size: int, n_rows: int
    ) -> np.ndarray:
        """
        Convert block start positions to full index array.

        Vectorized via NumPy broadcasting: builds (n_blocks, block_size) matrix,
        clips out-of-bounds indices, then pads with wraparound if needed.

        :param block_starts: List of block start indices
        :param block_size: Size of each block
        :param n_rows: Total number of rows
        :return: Array of permuted indices
        """
        starts_arr = np.asarray(block_starts)
        # Filter starts that are out of bounds
        valid_starts = starts_arr[starts_arr < n_rows]

        if len(valid_starts) == 0:
            return np.arange(n_rows)

        # Vectorized: (n_valid, 1) + (1, block_size) → (n_valid, block_size) → flatten
        raw_indices = (valid_starts[:, np.newaxis] + np.arange(block_size)[np.newaxis, :]).ravel()

        # Clip to [0, n_rows) — handles blocks that extend past end of data
        permuted_indices = raw_indices[raw_indices < n_rows]

        # Wraparound padding for detail timeframe that exceeds generated indices
        if len(permuted_indices) < n_rows:
            missing = n_rows - len(permuted_indices)
            n = len(permuted_indices)
            pad = permuted_indices[np.arange(missing) % n]
            permuted_indices = np.concatenate([permuted_indices, pad])

        return permuted_indices[:n_rows]

    def permute_multi_timeframe(
        self,
        main_data: dict[str, DataFrame],
        detail_data: dict[str, DataFrame],
        main_timeframe: str,
        detail_timeframe: str,
        seed: int | None = None,
    ) -> tuple[dict[str, DataFrame], dict[str, DataFrame]]:
        """
        Permute main and detail timeframe data synchronously using returns-based method.

        Ensures that blocks in main timeframe (e.g., 5m) correspond to
        the same time periods in detail timeframe (e.g., 1m).
        Both are permuted using returns-based approach for realistic price continuity.

        :param main_data: Dict of pair -> DataFrame for main timeframe
        :param detail_data: Dict of pair -> DataFrame for detail timeframe
        :param main_timeframe: Main timeframe string (e.g., "5m")
        :param detail_timeframe: Detail timeframe string (e.g., "1m")
        :param seed: Random seed for reproducibility
        :return: Tuple of (permuted_main_data, permuted_detail_data)
        """
        rng = np.random.default_rng(seed)

        # Calculate timeframe ratio
        main_minutes = timeframe_to_minutes(main_timeframe)
        detail_minutes = timeframe_to_minutes(detail_timeframe)
        tf_ratio = main_minutes // detail_minutes  # e.g., 5m/1m = 5

        permuted_main: dict[str, DataFrame] = {}
        permuted_detail: dict[str, DataFrame] = {}

        for pair, main_df in main_data.items():
            n_main = len(main_df)

            # Generate block starts for main timeframe
            block_starts = self._generate_block_mapping(n_main, rng)

            # Convert to full indices for main timeframe
            main_indices = self._block_starts_to_indices(block_starts, self.block_size, n_main)

            # Apply returns-based permutation to main data
            permuted_main[pair] = self._permute_df_with_indices(main_df, main_indices)

            # Apply scaled mapping to detail data if available
            if pair in detail_data:
                detail_df = detail_data[pair]
                n_detail = len(detail_df)

                # Scale block starts and size for detail timeframe
                detail_block_size = self.block_size * tf_ratio
                detail_block_starts = [start * tf_ratio for start in block_starts]

                # Convert to full indices for detail timeframe
                detail_indices = self._block_starts_to_indices(
                    detail_block_starts, detail_block_size, n_detail
                )

                # Apply returns-based permutation to detail data
                permuted_detail[pair] = self._permute_df_with_indices(detail_df, detail_indices)

        return permuted_main, permuted_detail

    def permute_batch(
        self,
        data: dict[str, DataFrame] | DataFrame,
        n_permutations: int,
        base_seed: int | None = None,
    ) -> Generator:
        """
        Generate multiple permutations of data.

        :param data: Original OHLCV data
        :param n_permutations: Number of permutations to generate
        :param base_seed: Base seed for reproducibility
        :yields: Permuted data for each iteration
        """
        for i in range(n_permutations):
            seed = base_seed + i if base_seed is not None else None
            yield self.permute(data, seed)

    def permute_batch_multi_timeframe(
        self,
        main_data: dict[str, DataFrame],
        detail_data: dict[str, DataFrame],
        main_timeframe: str,
        detail_timeframe: str,
        n_permutations: int,
        base_seed: int | None = None,
    ) -> Generator:
        """
        Generate multiple synchronized permutations of multi-timeframe data.

        :param main_data: Dict of pair -> DataFrame for main timeframe
        :param detail_data: Dict of pair -> DataFrame for detail timeframe
        :param main_timeframe: Main timeframe string (e.g., "5m")
        :param detail_timeframe: Detail timeframe string (e.g., "1m")
        :param n_permutations: Number of permutations to generate
        :param base_seed: Base seed for reproducibility
        :yields: Tuple of (permuted_main, permuted_detail) for each iteration
        """
        for i in range(n_permutations):
            seed = base_seed + i if base_seed is not None else None
            yield self.permute_multi_timeframe(
                main_data, detail_data, main_timeframe, detail_timeframe, seed
            )

    @property
    def preserves_properties(self) -> list[str]:
        """Properties preserved by Returns-Based Bar Permutation."""
        return [
            "marginal_distribution",  # Distribution of returns
            "ohlc_geometry",  # Candle structure (H >= O, L <= O, etc.)
            "price_continuity",  # No artificial jumps between bars
            "local_autocorrelation",  # Within blocks
            "volatility_clustering",  # Within blocks
        ]

    @property
    def breaks_properties(self) -> list[str]:
        """Properties broken by Returns-Based Bar Permutation."""
        return [
            "long_term_trends",  # Global drift is preserved but trend timing broken
            "seasonal_patterns",  # Intraday/weekly patterns broken
            "multi_day_momentum",  # Momentum beyond block size
            "cross_pair_correlation",  # Pairs permuted independently
        ]
