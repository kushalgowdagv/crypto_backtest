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

# Version info
__version__ = "1.0.0"