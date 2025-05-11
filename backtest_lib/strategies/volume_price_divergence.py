class VolumePriceDivergenceStrategy(Strategy):
    """
    Volume-Price Divergence Strategy Implementation
    
    Identify when volume doesn't confirm price movement
    """
    
    def __init__(self, name: str = "VolumePriceDivergence", price_period: int = 14, volume_period: int = 14):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        price_period : int
            Period for price MA
        volume_period : int
            Period for volume MA
        """
        super().__init__(name)
        self.set_parameters(price_period=price_period, volume_period=volume_period)
    
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
        
        # Calculate price and volume moving averages
        price_period = self.parameters['price_period']
        volume_period = self.parameters['volume_period']
        
        df['price_ma'] = Indicators.sma(df['close'], price_period)
        df['volume_ma'] = Indicators.sma(df['volume'], volume_period)
        
        # Calculate price and volume changes
        df['price_change'] = df['close'].pct_change(periods=1)
        df['volume_change'] = df['volume'].pct_change(periods=1)
        
        # Calculate OBV
        df['obv'] = Indicators.obv(df['close'], df['volume'])
        df['obv_ma'] = Indicators.sma(df['obv'], volume_period)
        
        # Generate signals based on price-volume divergence
        df['signal'] = 0
        df['position'] = 0
        
        # Bearish divergence: price making higher highs but volume/OBV making lower highs
        bearish_div = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2)) &
            (df['obv'] < df['obv'].shift(1)) &
            (df['obv_ma'] < df['obv_ma'].shift(1))
        )
        df.loc[bearish_div, 'signal'] = -1
        
        # Bullish divergence: price making lower lows but volume/OBV making higher lows
        bullish_div = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2)) &
            (df['obv'] > df['obv'].shift(1)) &
            (df['obv_ma'] > df['obv_ma'].shift(1))
        )
        df.loc[bullish_div, 'signal'] = 1
        
        # Calculate position
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df