import pandas as pd
import numpy as np
import logging
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.indicators.indicators import Indicators
from backtest_lib.data_adapters.data_loader import DataLoader

# Get logger
logger = logging.getLogger('backtest_lib')


class MultiTimeframeStrategy(Strategy):
    """
    Multi-Timeframe Analysis Strategy Implementation
    
    Combine signals from different timeframes for more robust entries
    """
    
    def __init__(self, name: str = "MultiTimeframe", primary_tf: str = '10m', secondary_tf: str = '1h', trend_period: int = 20, entry_period: int = 5):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        primary_tf : str
            Primary timeframe
        secondary_tf : str
            Secondary timeframe
        trend_period : int
            Period for trend MA
        entry_period : int
            Period for entry MA
        """
        super().__init__(name)
        self.set_parameters(
            primary_tf=primary_tf,
            secondary_tf=secondary_tf,
            trend_period=trend_period,
            entry_period=entry_period
        )
    
    def generate_signals(self, data: pd.DataFrame, data_loader: DataLoader = None) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
        data_loader : DataLoader, optional
            DataLoader instance for resampling
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        if data_loader is None:
            logger.warning("DataLoader not provided, using primary timeframe only")
            df_primary = data.copy()
            df_secondary = data.copy()
        else:
            # Get data for both timeframes
            df_primary = data.copy()
            df_secondary = data_loader.resample_timeframe(data, self.parameters['secondary_tf'])
        
        # Calculate indicators for primary timeframe
        entry_period = self.parameters['entry_period']
        df_primary[f'ema_{entry_period}'] = Indicators.ema(df_primary['close'], entry_period)
        df_primary['rsi'] = Indicators.rsi(df_primary['close'])
        
        # Calculate indicators for secondary timeframe
        trend_period = self.parameters['trend_period']
        df_secondary[f'ema_{trend_period}'] = Indicators.ema(df_secondary['close'], trend_period)
        
        # Align secondary timeframe data with primary
        trend_data = df_secondary[[f'ema_{trend_period}']].reindex(df_primary.index, method='ffill')
        
        # Merge data
        df = pd.concat([df_primary, trend_data], axis=1)
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signal: price above higher timeframe trend and RSI recovering
        buy_condition = (
            (df['close'] > df[f'ema_{trend_period}']) &
            (df['close'] > df[f'ema_{entry_period}']) &
            (df['rsi'] > 50) &
            (df['rsi'].shift(1) < 50)
        )
        df.loc[buy_condition, 'signal'] = 1
        
        # Sell signal: price below higher timeframe trend and RSI weakening
        sell_condition = (
            (df['close'] < df[f'ema_{trend_period}']) &
            (df['close'] < df[f'ema_{entry_period}']) &
            (df['rsi'] < 50) &
            (df['rsi'].shift(1) > 50)
        )
        df.loc[sell_condition, 'signal'] = -1
        
        # Calculate position
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df