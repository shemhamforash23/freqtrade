"""
FreqAI Data Prefetcher

This module provides parallel data loading for FreqAI backtesting.
It pre-loads OHLCV data and backtesting predictions before the main loop.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cloudpickle
import pandas as pd
import rapidjson
from pandas import DataFrame

from freqtrade.configuration import TimeRange
from freqtrade.constants import Config
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from freqtrade.exchange import timeframe_to_seconds


logger = logging.getLogger(__name__)

# Constants for metadata file names (must match data_drawer.py)
METADATA = "metadata"
FEATURE_PIPELINE = "feature_pipeline"
LABEL_PIPELINE = "label_pipeline"


class FreqAIPrefetcher:
    """
    Handles parallel prefetching of data for FreqAI backtesting.
    
    This class preloads:
    1. OHLCV data for all pairs and timeframes
    2. Backtesting prediction files (.feather)
    """
    
    def __init__(
        self,
        config: Config,
        pairs: list[str],
        timerange: TimeRange,
        max_workers: int = 8,
    ):
        self.config = config
        self.pairs = pairs
        self.timerange = timerange
        self.max_workers = max_workers
        
        # Cache storage
        self.ohlcv_cache: dict[tuple[str, str, CandleType], DataFrame] = {}
        self.predictions_cache: dict[str, DataFrame] = {}
        self.metadata_cache: dict[str, dict[str, Any]] = {}  # {model_path: {metadata, feature_pipeline, label_pipeline}}
        
        # FreqAI config
        self.freqai_config = config.get("freqai", {})
        self.identifier = self.freqai_config.get("identifier", "")
        
    def get_informative_timeframes(self) -> list[str]:
        """Get list of informative timeframes from FreqAI config."""
        feature_params = self.freqai_config.get("feature_parameters", {})
        return feature_params.get("include_timeframes", [])
    
    def get_models_path(self) -> Path:
        """Get path to FreqAI models directory."""
        return Path(
            self.config["user_data_dir"] / "models" / str(self.identifier)
        )
    
    def calculate_startup_candles(self, timeframe: str) -> int:
        """Calculate required startup candles for a timeframe."""
        if not self.freqai_config.get("enabled", False):
            return self.config.get("startup_candle_count", 0)
        
        startup_candles = self.config.get("startup_candle_count", 0)
        indicator_periods = self.freqai_config.get(
            "feature_parameters", {}
        ).get("indicator_periods_candles", [])
        
        if indicator_periods:
            startup_candles = max(startup_candles, max(indicator_periods))
        
        tf_seconds = timeframe_to_seconds(timeframe)
        train_candles = self.freqai_config.get("train_period_days", 0) * 86400 / tf_seconds
        
        return int(startup_candles + train_candles)
    
    def _load_ohlcv_single(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
    ) -> tuple[tuple[str, str, CandleType], DataFrame | None]:
        """Load OHLCV data for a single pair/timeframe combination."""
        key = (pair, timeframe, candle_type)
        
        try:
            startup_candles = self.calculate_startup_candles(timeframe)
            tf_seconds = timeframe_to_seconds(timeframe)
            
            # Create adjusted timerange
            tr = TimeRange(
                starttype=self.timerange.starttype,
                stoptype=self.timerange.stoptype,
                startts=self.timerange.startts,
                stopts=self.timerange.stopts,
            )
            tr.subtract_start(tf_seconds * startup_candles)
            
            df = load_pair_history(
                pair=pair,
                timeframe=timeframe,
                datadir=self.config["datadir"],
                timerange=tr,
                data_format=self.config["dataformat_ohlcv"],
                candle_type=candle_type,
            )
            
            logger.debug(f"Prefetched OHLCV: {pair} {timeframe} ({len(df)} rows)")
            return key, df
            
        except Exception as e:
            logger.warning(f"Failed to prefetch {pair} {timeframe}: {e}")
            return key, None
    
    def _load_prediction_single(
        self,
        prediction_path: Path,
    ) -> tuple[str, DataFrame | None]:
        """Load a single backtesting prediction file."""
        key = str(prediction_path)
        
        try:
            if prediction_path.is_file():
                df = pd.read_feather(prediction_path)
                logger.debug(f"Prefetched prediction: {prediction_path.name}")
                return key, df
            return key, None
            
        except Exception as e:
            logger.warning(f"Failed to prefetch prediction {prediction_path}: {e}")
            return key, None
    
    def _load_metadata_single(
        self,
        model_dir: Path,
    ) -> tuple[str, dict[str, Any] | None]:
        """Load metadata, feature_pipeline, and label_pipeline for a single model."""
        key = str(model_dir)
        
        try:
            # Find the model filename from directory
            metadata_files = list(model_dir.glob(f"*_{METADATA}.json"))
            if not metadata_files:
                return key, None
            
            model_filename = metadata_files[0].stem.replace(f"_{METADATA}", "")
            
            result: dict[str, Any] = {}
            
            # Load metadata JSON
            metadata_path = model_dir / f"{model_filename}_{METADATA}.json"
            if metadata_path.is_file():
                with metadata_path.open("r") as fp:
                    result[METADATA] = rapidjson.load(fp, number_mode=rapidjson.NM_NATIVE)
            
            # Load feature pipeline
            feature_path = model_dir / f"{model_filename}_{FEATURE_PIPELINE}.pkl"
            if feature_path.is_file():
                with feature_path.open("rb") as fp:
                    result[FEATURE_PIPELINE] = cloudpickle.load(fp)
            
            # Load label pipeline
            label_path = model_dir / f"{model_filename}_{LABEL_PIPELINE}.pkl"
            if label_path.is_file():
                with label_path.open("rb") as fp:
                    result[LABEL_PIPELINE] = cloudpickle.load(fp)
            
            if result:
                logger.debug(f"Prefetched metadata: {model_dir.name}")
                return key, result
            return key, None
            
        except Exception as e:
            logger.warning(f"Failed to prefetch metadata {model_dir}: {e}")
            return key, None
    
    def prefetch_ohlcv(self) -> dict[tuple[str, str, CandleType], DataFrame]:
        """
        Prefetch OHLCV data for all pairs and informative timeframes in parallel.
        
        Returns:
            Dictionary with (pair, timeframe, candle_type) as key and DataFrame as value.
        """
        timeframes = self.get_informative_timeframes()
        if not timeframes:
            logger.info("No informative timeframes configured, skipping OHLCV prefetch")
            return {}
        
        candle_type = self.config.get("candle_type_def", CandleType.SPOT)
        
        # Build list of tasks
        tasks = []
        for pair in self.pairs:
            for tf in timeframes:
                tasks.append((pair, tf, candle_type))
        
        logger.info(
            f"Prefetching OHLCV data: {len(self.pairs)} pairs × "
            f"{len(timeframes)} timeframes = {len(tasks)} tasks"
        )
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_ohlcv_single, pair, tf, ct): (pair, tf, ct)
                for pair, tf, ct in tasks
            }
            
            loaded = 0
            for future in as_completed(futures):
                key, df = future.result()
                if df is not None:
                    self.ohlcv_cache[key] = df
                    loaded += 1
        
        logger.info(f"Prefetched {loaded}/{len(tasks)} OHLCV datasets")
        return self.ohlcv_cache
    
    def prefetch_predictions(self) -> dict[str, DataFrame]:
        """
        Prefetch backtesting prediction files for all pairs in parallel.
        
        Returns:
            Dictionary with file path as key and DataFrame as value.
        """
        models_path = self.get_models_path()
        predictions_folder = models_path / "backtesting_predictions"
        
        if not predictions_folder.is_dir():
            logger.info("No backtesting predictions folder found, skipping prefetch")
            return {}
        
        # Find all prediction files
        prediction_files = list(predictions_folder.glob("*_prediction.feather"))
        
        if not prediction_files:
            logger.info("No prediction files found")
            return {}
        
        logger.info(f"Prefetching {len(prediction_files)} prediction files")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_prediction_single, path): path
                for path in prediction_files
            }
            
            loaded = 0
            for future in as_completed(futures):
                key, df = future.result()
                if df is not None:
                    self.predictions_cache[key] = df
                    loaded += 1
        
        logger.info(f"Prefetched {loaded}/{len(prediction_files)} prediction files")
        return self.predictions_cache
    
    def prefetch_metadata(self) -> dict[str, dict[str, Any]]:
        """
        Prefetch model metadata (JSON + pipelines) for all sub-train directories in parallel.
        
        Returns:
            Dictionary with model directory path as key and metadata dict as value.
        """
        models_path = self.get_models_path()
        
        if not models_path.is_dir():
            logger.info("No models folder found, skipping metadata prefetch")
            return {}
        
        # Find all sub-train directories
        model_dirs = [d for d in models_path.iterdir() if d.is_dir() and d.name.startswith("sub-train-")]
        
        if not model_dirs:
            logger.info("No model directories found")
            return {}
        
        logger.info(f"Prefetching metadata for {len(model_dirs)} models")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._load_metadata_single, model_dir): model_dir
                for model_dir in model_dirs
            }
            
            loaded = 0
            for future in as_completed(futures):
                key, metadata = future.result()
                if metadata is not None:
                    self.metadata_cache[key] = metadata
                    loaded += 1
        
        logger.info(f"Prefetched {loaded}/{len(model_dirs)} model metadata")
        return self.metadata_cache
    
    def prefetch_all(self) -> None:
        """Prefetch all data (OHLCV, predictions, and metadata) in parallel."""
        logger.info("Starting FreqAI data prefetch...")
        
        # Run all prefetches in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            ohlcv_future = executor.submit(self.prefetch_ohlcv)
            pred_future = executor.submit(self.prefetch_predictions)
            meta_future = executor.submit(self.prefetch_metadata)
            
            # Wait for all to complete
            ohlcv_future.result()
            pred_future.result()
            meta_future.result()
        
        logger.info(
            f"Prefetch complete: {len(self.ohlcv_cache)} OHLCV, "
            f"{len(self.predictions_cache)} predictions, "
            f"{len(self.metadata_cache)} metadata cached"
        )
    
    def get_cached_ohlcv(
        self,
        pair: str,
        timeframe: str,
        candle_type: CandleType,
    ) -> DataFrame | None:
        """Get cached OHLCV data if available."""
        key = (pair, timeframe, candle_type)
        df = self.ohlcv_cache.get(key)
        if df is not None:
            return df.copy()
        return None
    
    def get_cached_prediction(self, prediction_path: str | Path) -> DataFrame | None:
        """Get cached prediction if available."""
        key = str(prediction_path)
        df = self.predictions_cache.get(key)
        if df is not None:
            return df.copy()
        return None
    
    def get_cached_metadata(self, model_dir: str | Path) -> dict[str, Any] | None:
        """Get cached metadata if available."""
        key = str(model_dir)
        return self.metadata_cache.get(key)


# Global prefetcher instance (set during backtesting initialization)
_prefetcher: FreqAIPrefetcher | None = None


def init_prefetcher(
    config: Config,
    pairs: list[str],
    timerange: TimeRange,
    max_workers: int = 8,
) -> FreqAIPrefetcher:
    """Initialize and run the global prefetcher."""
    global _prefetcher
    _prefetcher = FreqAIPrefetcher(config, pairs, timerange, max_workers)
    _prefetcher.prefetch_all()
    return _prefetcher


def get_prefetcher() -> FreqAIPrefetcher | None:
    """Get the global prefetcher instance."""
    return _prefetcher


def clear_prefetcher() -> None:
    """Clear the global prefetcher and free memory."""
    global _prefetcher
    if _prefetcher:
        _prefetcher.ohlcv_cache.clear()
        _prefetcher.predictions_cache.clear()
        _prefetcher.metadata_cache.clear()
    _prefetcher = None
