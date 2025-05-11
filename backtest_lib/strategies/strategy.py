from abc import ABC, abstractmethod
import os
import pickle
import logging
import pandas as pd
from typing import Dict

# Get logger
logger = logging.getLogger('backtest_lib')


class Strategy(ABC):
    """Base class for all trading strategies"""
    
    def __init__(self, name: str):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Strategy name
        """
        self.name = name
        self.signals = None
        self.parameters = {}
        logger.info(f"Strategy '{name}' initialized")
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with signals
        """
        pass
    
    def set_parameters(self, **kwargs):
        """
        Set strategy parameters
        
        Parameters:
        -----------
        **kwargs : dict
            Strategy parameters
        """
        self.parameters.update(kwargs)
        logger.info(f"Updated parameters for strategy '{self.name}': {kwargs}")
    
    def get_parameters(self) -> Dict:
        """
        Get strategy parameters
        
        Returns:
        --------
        Dict
            Strategy parameters
        """
        return self.parameters
    
    def save(self, directory: str = 'saved_strategies'):
        """
        Save strategy to file
        
        Parameters:
        -----------
        directory : str
            Directory to save the strategy
        """
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, f"{self.name}.pkl")
        
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)
        
        logger.info(f"Strategy '{self.name}' saved to {file_path}")
    
    @classmethod
    def load(cls, file_path: str) -> 'Strategy':
        """
        Load strategy from file
        
        Parameters:
        -----------
        file_path : str
            Path to the strategy file
            
        Returns:
        --------
        Strategy
            Loaded strategy
        """
        with open(file_path, 'rb') as f:
            strategy = pickle.load(f)
        
        logger.info(f"Strategy '{strategy.name}' loaded from {file_path}")
        return strategy
