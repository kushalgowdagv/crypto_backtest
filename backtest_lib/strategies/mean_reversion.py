import pandas as pd
import numpy as np
import logging
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.indicators.indicators import Indicators

# Get logger
logger = logging.getLogger('backtest_lib')


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy Implementation
    
    Capitalize on price returning to an average after deviation
    """
    
    def __init__(self, name: str = "MeanReversion", ma_period: int = 20, std_dev: float = 2.0, rsi_period: int = 14, 
                 oversold_threshold: int = 30, overbought_threshold: int = 70):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        ma_period : int
            Period for moving average
        std_dev : float
            Standard deviation multiplier for bands
        rsi_period : int
            Period for RSI calculation
        oversold_threshold : int
            RSI threshold for oversold condition
        overbought_threshold : int
            RSI threshold for overbought condition
        """
        super().__init__(name)
        self.set_parameters(
            ma_period=ma_period,
            std_dev=std_dev,
            rsi_period=rsi_period,
            oversold_threshold=oversold_threshold,
            overbought_threshold=overbought_threshold
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
        ma_period = self.parameters['ma_period']
        std_dev = self.parameters['std_dev']
        
        df['upper_band'], df['middle_band'], df['lower_band'] = Indicators.bollinger_bands(
            df['close'], ma_period, std_dev
        )
        
        # Calculate RSI
        rsi_period = self.parameters['rsi_period']
        df['rsi'] = Indicators.rsi(df['close'], rsi_period)
        
        # Calculate Z-score
        df['z_score'] = Indicators.z_score(df['close'], ma_period)
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signals (mean reversion up)
        # 1. Price touches or crosses below lower Bollinger Band
        # 2. RSI is oversold
        oversold_threshold = self.parameters['oversold_threshold']
        buy_signal = (
            ((df['close'] <= df['lower_band']) | (df['low'] <= df['lower_band'])) &
            (df['rsi'] < oversold_threshold) &
            (df['z_score'] < -2)
        )
        df.loc[buy_signal, 'signal'] = 1
        
        # Sell signals (mean reversion down)
        # 1. Price touches or crosses above upper Bollinger Band
        # 2. RSI is overbought
        overbought_threshold = self.parameters['overbought_threshold']
        sell_signal = (
            ((df['close'] >= df['upper_band']) | (df['high'] >= df['upper_band'])) &
            (df['rsi'] > overbought_threshold) &
            (df['z_score'] > 2)
        )
        df.loc[sell_signal, 'signal'] = -1
        
        # Exit signals
        # Exit long when price crosses above middle band
        exit_long = (df['position'].shift(1) == 1) & (df['close'] > df['middle_band']) & (df['z_score'] > 0)
        
        # Exit short when price crosses below middle band
        exit_short = (df['position'].shift(1) == -1) & (df['close'] < df['middle_band']) & (df['z_score'] < 0)
        
        df.loc[exit_long | exit_short, 'signal'] = 0
        
        # Calculate position
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df