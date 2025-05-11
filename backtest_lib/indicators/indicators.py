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
        Relative Strength Index
        
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
        delta = series.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
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
        # Use TALib for ADX calculation
        adx = pd.Series(talib.ADX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        plus_di = pd.Series(talib.PLUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        minus_di = pd.Series(talib.MINUS_DI(high.values, low.values, close.values, timeperiod=period), index=close.index)
        dx = pd.Series(talib.DX(high.values, low.values, close.values, timeperiod=period), index=close.index)
        
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