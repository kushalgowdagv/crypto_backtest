"""
Cryptocurrency Backtesting Library
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

# Initialize logger
logger = logging.getLogger('backtest_lib')
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Set formatter to handlers
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(console_handler)

# Import main modules
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.engine.backtest import Backtest
from backtest_lib.analysis.analysis import Analysis
from backtest_lib.data_adapters.data_loader import DataLoader
from backtest_lib.data_adapters.data_processor import DataProcessor
from backtest_lib.indicators.indicators import Indicators

# Import strategies
from backtest_lib.strategies.trend_following import TrendFollowingStrategy
from backtest_lib.strategies.mean_reversion import MeanReversionStrategy
from backtest_lib.strategies.volatility_breakout import VolatilityBreakoutStrategy
from backtest_lib.strategies.volume_price_divergence import VolumePriceDivergenceStrategy
from backtest_lib.strategies.multi_timeframe import MultiTimeframeStrategy
from backtest_lib.strategies.pair_trading import PairTradingStrategy
from backtest_lib.strategies.machine_learning import MachineLearningStrategy

__version__ = "1.0.0"