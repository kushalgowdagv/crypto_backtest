import pandas as pd
import numpy as np
import logging
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.indicators.indicators import Indicators

# Get logger
logger = logging.getLogger('backtest_lib')


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy Implementation
    
    Enter positions when price breaks out of low-volatility consolidation periods
    """
    
    def __init__(self, name: str = "VolatilityBreakout", period: int = 20, std_dev: float = 2.0, atr_period: int = 14, atr_multiplier: float = 1.5):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        period : int
            Period for Bollinger Bands
        std_dev : float
            Standard deviation multiplier for Bollinger Bands
        atr_period : int
            Period for ATR calculation
        atr_multiplier : float
            ATR multiplier for breakout threshold
        """
        super().__init__(name)
        self.set_parameters(
            period=period,
            std_dev=std_dev,
            atr_period=atr_period,
            atr_multiplier=atr_multiplier
        )
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        df = data.copy()
        
        # Calculate Bollinger Bands
        period = self.parameters['period']
        std_dev = self.parameters['std_dev']
        df['upper_band'], df['middle_band'], df['lower_band'] = Indicators.bollinger_bands(
            df['close'], period, std_dev
        )
        
        # Calculate Bollinger Band width (measure of volatility)
        df['bb_width'] = (df['upper_band'] - df['lower_band']) / df['middle_band']
        
        # Calculate ATR
        atr_period = self.parameters['atr_period']
        atr_multiplier = self.parameters['atr_multiplier']
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], atr_period)
        
        # Calculate breakout thresholds
        df['upper_threshold'] = df['high'].shift(1) + (df['atr'].shift(1) * atr_multiplier)
        df['lower_threshold'] = df['low'].shift(1) - (df['atr'].shift(1) * atr_multiplier)
        
        # Calculate volatility contraction
        df['vol_contraction'] = df['bb_width'] < df['bb_width'].shift(1)
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signal: price breaks above upper threshold after volatility contraction
        long_condition = (
            (df['high'] > df['upper_threshold']) & 
            (df['bb_width'].shift(1) < df['bb_width'].shift(1).rolling(window=5).mean())
        )
        df.loc[long_condition, 'signal'] = 1
        
        # Sell signal: price breaks below lower threshold after volatility contraction
        short_condition = (
            (df['low'] < df['lower_threshold']) & 
            (df['bb_width'].shift(1) < df['bb_width'].shift(1).rolling(window=5).mean())
        )
        df.loc[short_condition, 'signal'] = -1
        
        # Calculate position
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df