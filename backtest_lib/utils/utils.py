import logging
import pandas as pd
import numpy as np
from typing import Dict, List
import os




def setup_logger(name: str = 'backtest_lib', log_file: str = 'backtest.log', level: int = logging.INFO) -> logging.Logger:
    """
    Set up logger
    
    Parameters:
    -----------
    name : str
        Logger name
    log_file : str
        Log file path
    level : int
        Logging level
        
    Returns:
    --------
    logging.Logger
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    
    # Set formatters
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_cagr(initial_value: float, final_value: float, years: float) -> float:
    """
    Calculate Compound Annual Growth Rate
    
    Parameters:
    -----------
    initial_value : float
        Initial investment value
    final_value : float
        Final investment value
    years : float
        Number of years
        
    Returns:
    --------
    float
        CAGR as decimal
    """
    return (final_value / initial_value) ** (1 / years) - 1

def calculate_drawdowns(returns: pd.Series) -> pd.Series:
    """
    Calculate drawdowns
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
        
    Returns:
    --------
    pd.Series
        Drawdowns
    """
    cumulative_returns = (1 + returns).cumprod()
    max_returns = cumulative_returns.expanding().max()
    drawdowns = (cumulative_returns / max_returns) - 1
    return drawdowns

def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sharpe Ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns:
    --------
    float
        Sharpe Ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252) -> float:
    """
    Calculate Sortino Ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    risk_free_rate : float
        Risk-free rate
    periods_per_year : int
        Number of periods per year
        
    Returns:
    --------
    float
        Sortino Ratio
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = returns[returns < 0]
    downside_deviation = downside_returns.std() * np.sqrt(periods_per_year)
    
    if downside_deviation == 0:
        return float('inf')
    
    return excess_returns.mean() * periods_per_year / downside_deviation

def calculate_calmar_ratio(returns: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculate Calmar Ratio
    
    Parameters:
    -----------
    returns : pd.Series
        Series of returns
    periods_per_year : int
        Number of periods per year
        
    Returns:
    --------
    float
        Calmar Ratio
    """
    drawdowns = calculate_drawdowns(returns)
    max_drawdown = abs(drawdowns.min())
    annual_return = (1 + returns.mean()) ** periods_per_year - 1
    
    if max_drawdown == 0:
        return float('inf')
    
    return annual_return / max_drawdown

def export_to_excel(data: Dict, output_file: str) -> None:
    """
    Export data to Excel
    
    Parameters:
    -----------
    data : Dict
        Dictionary with data
    output_file : str
        Output file path
    """
    try:
        import pandas as pd
        
        # Create Excel writer
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Write each table to a different worksheet
            for sheet_name, df in data.items():
                if isinstance(df, pd.DataFrame):
                    df.to_excel(writer, sheet_name=sheet_name)
                elif isinstance(df, dict):
                    pd.DataFrame.from_dict(df, orient='index', columns=['Value']).to_excel(writer, sheet_name=sheet_name)
        
        return output_file
        
    except ImportError:
        logger.error("openpyxl not installed. Please install it to export to Excel.")
        raise ImportError("openpyxl not installed. Please install it to export to Excel.")
    
    except Exception as e:
        logger.error(f"Error exporting to Excel: {e}")
        raise