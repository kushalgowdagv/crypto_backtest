import pandas as pd
import numpy as np
from typing import Tuple
import logging

# Get logger
logger = logging.getLogger('backtest_lib')


class Indicators:
    """Technical indicators for trading strategies"""
    
    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        """
        Simple Moving Average
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            SMA period
            
        Returns:
        --------
        pd.Series
            SMA values
        """
        return series.rolling(window=period).mean()
    
    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        """
        Exponential Moving Average
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            EMA period
            
        Returns:
        --------
        pd.Series
            EMA values
        """
        return series.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        """
        Relative Strength Index with improved stability
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            RSI period
            
        Returns:
        --------
        pd.Series
            RSI values
        """
        # Validate input
        if period <= 0:
            raise ValueError(f"Period must be positive, got {period}")
        
        # Handle edge cases
        if len(series) <= period:
            return pd.Series(index=series.index, dtype=float)
        
        # Remove NaN values for calculation
        series_clean = series.copy()
        series_clean.fillna(method='ffill', inplace=True)
        
        # Calculate difference with error handling
        delta = series_clean.diff()
        
        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # Calculate RS with safeguard against division by zero
        rs = pd.Series(index=series.index, dtype=float)
        valid_indices = ~(avg_loss.isna() | (avg_loss == 0))
        rs[valid_indices] = avg_gain[valid_indices] / avg_loss[valid_indices]
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        # Ensure values are within range
        rsi = rsi.clip(0, 100)
        
        return rsi
    
    @staticmethod
    def bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            Period for moving average
        std_dev : float
            Number of standard deviations for bands
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            Upper band, middle band, lower band
        """
        middle_band = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """
        Average True Range
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int
            ATR period
            
        Returns:
        --------
        pd.Series
            ATR values
        """
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    @staticmethod
    def macd(series: pd.Series, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Moving Average Convergence Divergence
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        fast_period : int
            Fast EMA period
        slow_period : int
            Slow EMA period
        signal_period : int
            Signal EMA period
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series]
            MACD line, signal line, histogram
        """
        fast_ema = Indicators.ema(series, fast_period)
        slow_ema = Indicators.ema(series, slow_period)
        
        macd_line = fast_ema - slow_ema
        signal_line = Indicators.ema(macd_line, signal_period)
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    

    @staticmethod
    def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
        """
        Average Directional Index
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        period : int
            ADX period
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
            ADX, +DI, -DI, DX values
        """
        try:
            import talib
            # Use TALib for ADX calculation
            adx = pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
            plus_di = pd.Series(talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
            minus_di = pd.Series(talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
            dx = pd.Series(talib.DX(high.values, low.values, close.values, timeperiod=period), index=close.index)
            
            return adx, plus_di, minus_di, dx
        except ImportError:
            logger.warning("TA-Lib not installed. Using simplified ADX calculation.")
            # Simplified ADX calculation without TA-Lib
            # Calculate True Range
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(period).mean()
            
            # Calculate +DM and -DM
            up_move = high - high.shift(1)
            down_move = low.shift(1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            plus_dm = pd.Series(plus_dm, index=close.index)
            
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            minus_dm = pd.Series(minus_dm, index=close.index)
            
            # Calculate smoothed +DM and -DM
            plus_dm_smooth = plus_dm.rolling(period).mean()
            minus_dm_smooth = minus_dm.rolling(period).mean()
            
            # Calculate +DI and -DI
            plus_di = 100 * (plus_dm_smooth / atr)
            minus_di = 100 * (minus_dm_smooth / atr)
            
            # Calculate DX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            
            # Calculate ADX
            adx = dx.rolling(period).mean()
            
            return adx, plus_di, minus_di, dx

    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        On-Balance Volume
        
        Parameters:
        -----------
        close : pd.Series
            Close prices
        volume : pd.Series
            Volume
            
        Returns:
        --------
        pd.Series
            OBV values
        """
        price_change = close.diff()
        obv = pd.Series(index=close.index, dtype=float)
        obv.iloc[0] = 0
        
        for i in range(1, len(close)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
        
        return obv
    
    @staticmethod
    def volatility(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Volatility (Standard Deviation)
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            Period for volatility calculation
            
        Returns:
        --------
        pd.Series
            Volatility values
        """
        return series.rolling(window=period).std()
    
    @staticmethod
    def z_score(series: pd.Series, period: int = 20) -> pd.Series:
        """
        Z-Score
        
        Parameters:
        -----------
        series : pd.Series
            Price series
        period : int
            Period for Z-Score calculation
            
        Returns:
        --------
        pd.Series
            Z-Score values
        """
        rolling_mean = series.rolling(window=period).mean()
        rolling_std = series.rolling(window=period).std()
        
        return (series - rolling_mean) / rolling_std
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator
        
        Parameters:
        -----------
        high : pd.Series
            High prices
        low : pd.Series
            Low prices
        close : pd.Series
            Close prices
        k_period : int
            %K period
        d_period : int
            %D period
            
        Returns:
        --------
        Tuple[pd.Series, pd.Series]
            %K, %D values
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d = k.rolling(window=d_period).mean()
        
        return k, d