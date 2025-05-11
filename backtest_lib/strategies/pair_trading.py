import pandas as pd
import numpy as np
import logging
from typing import List
from backtest_lib.strategies.strategy import Strategy
from backtest_lib.indicators.indicators import Indicators

# Get logger
logger = logging.getLogger('backtest_lib')



class PairTradingStrategy(Strategy):
    """
    Statistical Pair Trading Strategy Implementation
    
    Trade the relationship between two assets when they diverge from historical correlation
    """
    
    def __init__(self, name: str = "PairTrading", window: int = 60, z_score_threshold: float = 2.0, assets: List[str] = None):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        window : int
            Rolling window for z-score calculation
        z_score_threshold : float
            Z-score threshold for entry/exit
        assets : List[str], optional
            List of assets to trade as pairs
        """
        super().__init__(name)
        
        if assets is None:
            assets = ['BTCUSD', 'ETHUSD']
        
        self.set_parameters(
            window=window,
            z_score_threshold=z_score_threshold,
            assets=assets
        )
    
    def _calculate_spread(self, df: pd.DataFrame, asset1: str, asset2: str) -> pd.Series:
        """
        Calculate spread between two assets
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with price data for both assets
        asset1 : str
            First asset
        asset2 : str
            Second asset
            
        Returns:
        --------
        pd.Series
            Spread between the two assets
        """
        # Get close prices for both assets
        price1 = df[f'close_{asset1}']
        price2 = df[f'close_{asset2}']
        
        # Calculate log prices
        log_price1 = np.log(price1)
        log_price2 = np.log(price2)
        
        # Calculate hedge ratio using OLS regression
        window = self.parameters['window']
        hedge_ratio = pd.Series(index=df.index)
        
        for i in range(window, len(df)):
            x = log_price2.iloc[i-window:i].values.reshape(-1, 1)
            y = log_price1.iloc[i-window:i].values
            beta = np.linalg.lstsq(x, y, rcond=None)[0][0]
            hedge_ratio.iloc[i] = beta
        
        # Calculate spread
        spread = log_price1 - hedge_ratio * log_price2
        
        return spread, hedge_ratio
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV for both assets
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        df = data.copy()
        assets = self.parameters['assets']
        window = self.parameters['window']
        z_score_threshold = self.parameters['z_score_threshold']
        
        # Check if data contains both assets
        asset1, asset2 = assets[0], assets[1]
        required_cols = [f'close_{asset1}', f'close_{asset2}']
        
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"Missing required column: {col}")
                raise ValueError(f"Data missing required column: {col}")
        
        # Calculate spread and z-score
        spread, hedge_ratio = self._calculate_spread(df, asset1, asset2)
        
        # Calculate z-score of spread
        z_score = Indicators.z_score(spread, window)
        
        # Generate signals
        df['spread'] = spread
        df['hedge_ratio'] = hedge_ratio
        df['z_score'] = z_score
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signal: z-score below negative threshold
        df.loc[df['z_score'] < -z_score_threshold, 'signal'] = 1
        
        # Sell signal: z-score above positive threshold
        df.loc[df['z_score'] > z_score_threshold, 'signal'] = -1
        
        # Exit signal: z-score crosses back to mean
        exit_long = (df['z_score'] > 0) & (df['z_score'].shift(1) < 0)
        exit_short = (df['z_score'] < 0) & (df['z_score'].shift(1) > 0)
        
        df.loc[exit_long | exit_short, 'signal'] = 0
        
        # Calculate position
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df