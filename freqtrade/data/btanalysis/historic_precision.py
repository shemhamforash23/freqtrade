import numpy as np
from pandas import DataFrame, Series


def _count_decimal_places_vec(values: np.ndarray) -> np.ndarray:
    """
    Vectorized count of significant decimal places for each float value.

    Uses np.char.mod("%.15f") — the same C-level printf as Python's "{:.15f}".format —
    so the output is bit-for-bit identical to the original .apply() approach.
    All string ops (rstrip, str_len) run at C level: no Python loop.

    Returns np.nan for integer-valued inputs (no non-zero fractional digits).
    """
    formatted = np.char.mod("%.15f", np.round(values, 14))
    # partition on "." gives shape (n, 3): [before, ".", after]
    dec_parts = np.char.partition(formatted, ".")[:, 2]
    stripped = np.char.rstrip(dec_parts, "0")
    lengths = np.char.str_len(stripped).astype(float)
    lengths[lengths == 0] = np.nan
    return lengths


def get_tick_size_over_time(candles: DataFrame) -> Series:
    """
    Calculate the number of significant digits for candles over time.
    It's using the Monthly maximum of the number of significant digits for each month.
    :param candles: DataFrame with OHLCV data
    :return: Series with the average number of significant digits for each month
    """
    counts = np.stack(
        [
            _count_decimal_places_vec(np.asarray(candles[col]))
            for col in ["open", "high", "low", "close"]
        ],
        axis=1,
    )
    max_count = np.nanmax(counts, axis=1)

    candles1 = candles.set_index("date", drop=True)
    # Group by month and take the max number of significant digits per month
    monthly_count_max = Series(max_count, index=candles1.index).resample("MS").max()
    # convert from 5.0 to 0.00001, 4.0 to 0.0001, ...
    monthly_open_count_avg = 1 / 10**monthly_count_max

    return monthly_open_count_avg
