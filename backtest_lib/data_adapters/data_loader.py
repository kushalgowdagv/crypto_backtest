import os
import pandas as pd
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict

# Get logger
logger = logging.getLogger('backtest_lib')


class DataLoader:
    """Load and preprocess cryptocurrency data from CSV files"""
    
    def __init__(self, data_dir: str = 'CoinLion_strat'):
        """
        Initialize the DataLoader
        
        Parameters:
        -----------
        data_dir : str
            Directory containing CSV data files
        """
        self.data_dir = data_dir
        self.data_cache = {}
        logger.info(f"DataLoader initialized with data directory: {data_dir}")
    
    def load_csv(self, filename: str, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file with improved error handling and validation
        
        Parameters:
        -----------
        filename : str
            Name of the CSV file
        symbol : str, optional
            Symbol to filter by
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the loaded data
        """
        file_path = os.path.join(self.data_dir, filename)
        
        if file_path in self.data_cache:
            logger.info(f"Using cached data for {file_path}")
            df = self.data_cache[file_path].copy()
        else:
            logger.info(f"Loading data from {file_path}")
            try:
                # Read CSV with explicit data types and NaN handling
                df = pd.read_csv(file_path, na_values=['nan', 'NaN', 'None', ''])
                
                # Verify required columns exist
                required_cols = ['time_utc', 'symbol', 'o', 'h', 'l', 'c', 'v']
                missing_cols = [col for col in required_cols if col not in df.columns]
                
                if missing_cols:
                    raise ValueError(f"Missing required columns in {filename}: {missing_cols}")
                
                # Convert time columns to datetime with proper error handling
                if 'time_utc' in df.columns:
                    df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce')
                    
                    # Check for and remove rows with invalid dates
                    invalid_dates = df['time_utc'].isna()
                    if invalid_dates.any():
                        logger.warning(f"Removed {invalid_dates.sum()} rows with invalid dates from {filename}")
                        df = df[~invalid_dates]
                    
                if 'time_est' in df.columns:
                    df['time_est'] = pd.to_datetime(df['time_est'], errors='coerce')
                
                # Convert numeric columns to correct data types
                numeric_cols = ['o', 'h', 'l', 'c', 'v']
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Set time_utc as index
                if 'time_utc' in df.columns:
                    df.set_index('time_utc', inplace=True)
                    
                    # Check for and remove duplicate timestamps
                    if df.index.duplicated().any():
                        dup_count = df.index.duplicated().sum()
                        logger.warning(f"Found {dup_count} duplicate timestamps in {filename}")
                        df = df[~df.index.duplicated(keep='first')]
                    
                    # Sort by timestamp
                    if not df.index.is_monotonic_increasing:
                        logger.warning(f"Timestamps not in order in {filename}. Sorting by timestamp.")
                        df.sort_index(inplace=True)
                
                # Replace inf values with NaN
                df.replace([np.inf, -np.inf], np.nan, inplace=True)
                
                # Check for NaN values
                nan_count = df.isna().sum().sum()
                if nan_count > 0:
                    logger.warning(f"Found {nan_count} NaN values in {filename}")
                
                # Cache the data
                self.data_cache[file_path] = df.copy()
                
            except Exception as e:
                logger.error(f"Error loading data from {file_path}: {e}")
                raise
        
        # Filter by symbol if provided
        if symbol and 'symbol' in df.columns:
            symbol_counts = df['symbol'].value_counts()
            logger.info(f"Symbols in {filename}: {dict(symbol_counts)}")
            
            filtered_df = df[df['symbol'] == symbol]
            if len(filtered_df) == 0:
                logger.warning(f"No data found for symbol {symbol} in {filename}")
            df = filtered_df
        
        # Standardize column names
        rename_dict = {
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        }
        
        # Only rename columns that exist
        rename_cols = {k: v for k, v in rename_dict.items() if k in df.columns}
        if rename_cols:
            df = df.rename(columns=rename_cols)
        
        # Final validation of OHLCV data
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                # Check for negative prices (shouldn't exist in most financial data)
                neg_count = (df[col] < 0).sum()
                if neg_count > 0 and col != 'volume':  # Volume can sometimes be negative
                    logger.warning(f"Found {neg_count} negative values in {col} column of {filename}")
                
                # Check for suspicious zeros in price columns
                if col in ['open', 'high', 'low', 'close']:
                    zero_count = (df[col] == 0).sum()
                    if zero_count > 0:
                        logger.warning(f"Found {zero_count} zero values in {col} column of {filename}")
        
        return df
    
    def get_ohlcv(self, symbol: str, timeframe: str = '10m') -> pd.DataFrame:
        """
        Get OHLCV data for a specific symbol
        
        Parameters:
        -----------
        symbol : str
            Trading symbol (e.g., 'BTCUSD')
        timeframe : str
            Timeframe of the data
            
        Returns:
        --------
        pd.DataFrame
            OHLCV data for the specified symbol
        """
        # Determine filename based on symbol and timeframe
        symbol_lower = symbol.lower().replace('usd', '')
        filename = f"{symbol_lower}usd_{timeframe}.csv"
        
        # Load data
        df = self.load_csv(filename, symbol)
        
        # Ensure OHLCV columns are present
        required_cols = ['o', 'h', 'l', 'c', 'v']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            raise ValueError(f"Data missing required columns: {missing_cols}")
        
        # Standardize column names
        df = df.rename(columns={
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume'
        })
        
        return df
    
    def resample_timeframe(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to a different timeframe
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with OHLCV data
        timeframe : str
            Target timeframe (e.g., '1h', '4h', '1d')
            
        Returns:
        --------
        pd.DataFrame
            Resampled DataFrame
        """
        # Ensure df has datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.error("DataFrame index must be DatetimeIndex for resampling")
            raise ValueError("DataFrame index must be DatetimeIndex for resampling")
        
        # Define resampling rules
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        })
        
        # Drop rows with NaN values
        resampled.dropna(inplace=True)
        
        return resampled
    
    def merge_symbols(self, symbols: List[str], timeframe: str = '10m') -> pd.DataFrame:
        """
        Merge data for multiple symbols
        
        Parameters:
        -----------
        symbols : List[str]
            List of symbols to merge
        timeframe : str
            Timeframe of the data
            
        Returns:
        --------
        pd.DataFrame
            Merged DataFrame containing data for all symbols
        """
        all_dfs = {}
        
        for symbol in symbols:
            df = self.get_ohlcv(symbol, timeframe)
            
            # Add suffix to columns to distinguish between symbols
            df_cols = {
                col: f"{col}_{symbol}" for col in df.columns 
                if col not in ['time_est', 'symbol']
            }
            df = df.rename(columns=df_cols)
            
            all_dfs[symbol] = df
        
        # Find common date range
        common_dates = None
        for df in all_dfs.values():
            if common_dates is None:
                common_dates = set(df.index)
            else:
                common_dates = common_dates.intersection(set(df.index))
        
        common_dates = sorted(list(common_dates))
        
        # Filter all dataframes to common date range
        for symbol in all_dfs:
            all_dfs[symbol] = all_dfs[symbol].loc[common_dates]
        
        # Merge all dataframes
        merged_df = pd.concat(all_dfs.values(), axis=1)
        
        return merged_df