from abc import ABC, abstractmethod
from collections.abc import Generator

from pandas import DataFrame


class IPermuter(ABC):
    """
    Base interface for MCPT data permutation generators.

    Permuters generate modified versions of data (trades or OHLCV bars)
    for Monte Carlo hypothesis testing.
    """

    @abstractmethod
    def permute(
        self,
        data: dict[str, DataFrame] | DataFrame,
        seed: int | None = None,
    ) -> dict[str, DataFrame] | DataFrame:
        """
        Generate a single permuted version of the data.

        :param data: Original data (OHLCV dict for bars, DataFrame for trades)
        :param seed: Random seed for reproducibility
        :return: Permuted data in the same format
        """
        pass

    @abstractmethod
    def permute_batch(
        self,
        data: dict[str, DataFrame] | DataFrame,
        n_permutations: int,
        base_seed: int | None = None,
    ) -> Generator:
        """
        Generator yielding n_permutations permuted versions.

        :param data: Original data
        :param n_permutations: Number of permutations to generate
        :param base_seed: Base seed (each iteration uses base_seed + i)
        :yields: Permuted data
        """
        pass

    @property
    @abstractmethod
    def preserves_properties(self) -> list[str]:
        """
        List of statistical properties preserved by this permuter.
        E.g., ['marginal_distribution', 'cross_asset_correlation']
        """
        pass

    @property
    @abstractmethod
    def breaks_properties(self) -> list[str]:
        """
        List of statistical properties broken by this permuter.
        E.g., ['autocorrelation', 'volatility_clustering']
        """
        pass
