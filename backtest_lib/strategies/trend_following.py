
import pandas as pd
import numpy as np
import logging
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.indicators.indicators import Indicators

# Get logger
logger = logging.getLogger('backtest_lib')


class TrendFollowingStrategy(Strategy):
    """
    Trend-Following Strategy Implementation
    
    Identify and follow established price trends using moving average crossovers
    """
    
    def __init__(self, name: str = "TrendFollowing", fast_period: int = 5, slow_period: int = 20, atr_period: int = 14):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        fast_period : int
            Period for fast moving average
        slow_period : int
            Period for slow moving average
        atr_period : int
            Period for ATR calculation
        """
        super().__init__(name)
        self.set_parameters(
            fast_period=fast_period,
            slow_period=slow_period,
            atr_period=atr_period
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
        
        # Calculate moving averages
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        
        df[f'sma_{fast_period}'] = Indicators.sma(df['close'], fast_period)
        df[f'sma_{slow_period}'] = Indicators.sma(df['close'], slow_period)
        
        # Calculate ATR for position sizing (optional)
        atr_period = self.parameters['atr_period']
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], atr_period)
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signal: fast MA crosses above slow MA
        crossover_up = (df[f'sma_{fast_period}'] > df[f'sma_{slow_period}']) & (df[f'sma_{fast_period}'].shift(1) <= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_up, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        crossover_down = (df[f'sma_{fast_period}'] < df[f'sma_{slow_period}']) & (df[f'sma_{fast_period}'].shift(1) >= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_down, 'signal'] = -1
        
        # Calculate position: 1 for long, -1 for short, 0 for flat
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df