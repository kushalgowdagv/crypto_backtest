import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
import logging

# Get logger
logger = logging.getLogger('backtest_lib')

class DataProcessor:
    """Process and prepare data for backtesting"""
    
    def __init__(self):
        """Initialize the data processor"""
        pass
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and preprocess data
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw DataFrame
            
        Returns:
        --------
        pd.DataFrame
            Cleaned DataFrame
        """
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Check for and handle missing values
        if cleaned_df.isnull().any().any():
            logger.warning(f"Found {cleaned_df.isnull().sum().sum()} missing values")
            # Forward fill missing values
            cleaned_df.fillna(method='ffill', inplace=True)
            # Backward fill any remaining missing values
            cleaned_df.fillna(method='bfill', inplace=True)
        
        # Check for and remove duplicates
        duplicated_rows = cleaned_df.index.duplicated()
        if duplicated_rows.any():
            logger.warning(f"Found {duplicated_rows.sum()} duplicated rows")
            cleaned_df = cleaned_df[~duplicated_rows]
        
        # Ensure index is sorted
        if not cleaned_df.index.is_monotonic_increasing:
            logger.warning("Index is not monotonically increasing, sorting...")
            cleaned_df.sort_index(inplace=True)
        
        # Convert columns to correct data types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            if col in cleaned_df.columns:
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
        
        # Remove any remaining rows with NaN values
        cleaned_df.dropna(inplace=True)
        
        return cleaned_df
    
    def add_features(self, df: pd.DataFrame, features: List[str] = None) -> pd.DataFrame:
        """
        Add technical indicators and other features
        
        Parameters:
        -----------
        df : pd.DataFrame
            Market data with OHLCV
        features : List[str], optional
            List of features to add
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added features
        """
        from backtest_lib.indicators.indicators import Indicators
        
        df_features = df.copy()
        
        if features is None:
            features = [
                'sma_5', 'sma_20', 'ema_5', 'ema_20', 
                'rsi', 'macd', 'bollinger_bands', 
                'atr', 'obv', 'returns', 'volatility'
            ]
        
        # Calculate requested features
        for feature in features:
            if feature.startswith('sma_'):
                period = int(feature.split('_')[1])
                df_features[feature] = Indicators.sma(df_features['close'], period)
                
            elif feature.startswith('ema_'):
                period = int(feature.split('_')[1])
                df_features[feature] = Indicators.ema(df_features['close'], period)
                
            elif feature == 'rsi' or feature.startswith('rsi_'):
                if feature == 'rsi':
                    period = 14
                else:
                    period = int(feature.split('_')[1])
                df_features['rsi'] = Indicators.rsi(df_features['close'], period)
                
            elif feature == 'macd':
                df_features['macd'], df_features['macd_signal'], df_features['macd_hist'] = Indicators.macd(df_features['close'])
                
            elif feature == 'bollinger_bands':
                df_features['upper_band'], df_features['middle_band'], df_features['lower_band'] = Indicators.bollinger_bands(df_features['close'])
                df_features['bb_width'] = (df_features['upper_band'] - df_features['lower_band']) / df_features['middle_band']
                
            elif feature == 'atr' or feature.startswith('atr_'):
                if feature == 'atr':
                    period = 14
                else:
                    period = int(feature.split('_')[1])
                df_features['atr'] = Indicators.atr(df_features['high'], df_features['low'], df_features['close'], period)
                
            elif feature == 'obv':
                df_features['obv'] = Indicators.obv(df_features['close'], df_features['volume'])
                
            elif feature == 'returns':
                df_features['returns'] = df_features['close'].pct_change()
                df_features['log_returns'] = np.log(df_features['close'] / df_features['close'].shift(1))
                
            elif feature == 'volatility' or feature.startswith('volatility_'):
                if feature == 'volatility':
                    period = 20
                else:
                    period = int(feature.split('_')[1])
                df_features['volatility'] = Indicators.volatility(df_features['close'], period)
                
            elif feature == 'z_score' or feature.startswith('z_score_'):
                if feature == 'z_score':
                    period = 20
                else:
                    period = int(feature.split('_')[1])
                df_features['z_score'] = Indicators.z_score(df_features['close'], period)
                
            elif feature == 'stochastic':
                df_features['stoch_k'], df_features['stoch_d'] = Indicators.stochastic(df_features['high'], df_features['low'], df_features['close'])
        
        return df_features
    
    def normalize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Normalize data using min-max scaling
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to normalize
        columns : List[str], optional
            List of columns to normalize
            
        Returns:
        --------
        pd.DataFrame
            Normalized DataFrame
        """
        from sklearn.preprocessing import MinMaxScaler
        
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        df_normalized = df.copy()
        scaler = MinMaxScaler()
        
        df_normalized[columns] = scaler.fit_transform(df_normalized[columns])
        
        return df_normalized
    
    def standardize_data(self, df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
        """
        Standardize data using z-score
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to standardize
        columns : List[str], optional
            List of columns to standardize
            
        Returns:
        --------
        pd.DataFrame
            Standardized DataFrame
        """
        from sklearn.preprocessing import StandardScaler
        
        if columns is None:
            columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
        
        df_standardized = df.copy()
        scaler = StandardScaler()
        
        df_standardized[columns] = scaler.fit_transform(df_standardized[columns])
        
        return df_standardized
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, validation_size: float = 0.0) -> tuple:
        """
        Split data into train, test, and optionally validation sets
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame to split
        test_size : float
            Proportion of data to use for testing
        validation_size : float
            Proportion of data to use for validation
            
        Returns:
        --------
        tuple
            (train_df, test_df) or (train_df, validation_df, test_df)
        """
        # Ensure chronological ordering
        df_sorted = df.sort_index()
        
        # Calculate split indices
        total_size = len(df_sorted)
        test_index = int(total_size * (1 - test_size))
        
        if validation_size > 0:
            validation_index = int(total_size * (1 - test_size - validation_size))
            train_df = df_sorted.iloc[:validation_index]
            validation_df = df_sorted.iloc[validation_index:test_index]
            test_df = df_sorted.iloc[test_index:]
            return train_df, validation_df, test_df
        else:
            train_df = df_sorted.iloc[:test_index]
            test_df = df_sorted.iloc[test_index:]
            return train_df, test_df
    
    def prepare_data_for_ml(self, df: pd.DataFrame, features: List[str], target: str, lookback: int = 1) -> tuple:
        """
        Prepare data for machine learning by creating sequences
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        features : List[str]
            List of feature columns
        target : str
            Target column
        lookback : int
            Number of previous time steps to use
            
        Returns:
        --------
        tuple
            (X, y) where X is a 3D array of shape (samples, lookback, features)
            and y is the target values
        """
        from sklearn.preprocessing import StandardScaler
        
        # Extract features and target
        X_raw = df[features].values
        y_raw = df[target].values
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_raw)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(X_scaled)):
            X.append(X_scaled[i-lookback:i])
            y.append(y_raw[i])
        
        return np.array(X), np.array(y), scaler