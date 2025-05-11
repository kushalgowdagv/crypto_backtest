import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Import necessary modules
from backtest_lib.strategies.trend_following import TrendFollowingStrategy
from backtest_lib.data_adapters.data_loader import DataLoader
from backtest_lib.engine.backtest import Backtest
from backtest_lib.analysis.analysis import Analysis
from backtest_lib.utils.utils import setup_logger

# Set up logging
logger = setup_logger(name='run_backtest', log_file='run_backtest.log')

def main():
    """
    Main function to run a backtest with the trend following strategy
    """
    # Create mock data since we don't have actual CSV files
    logger.info("Creating mock data...")
    
    # Create date range
    dates = pd.date_range(start='2024-01-01', end='2024-04-30', freq='10min')
    
    # Create sample price data
    np.random.seed(42)  # For reproducibility
    
    # Initial price
    initial_price = 50000
    
    # Generate random returns
    returns = np.random.normal(0.0001, 0.002, len(dates))
    
    # Calculate price series
    prices = initial_price * (1 + returns).cumprod()
    
    # Create mock OHLCV data
    df = pd.DataFrame(index=dates)
    df['open'] = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
    df['high'] = df['open'] * (1 + abs(np.random.normal(0, 0.001, len(dates))))
    df['low'] = df['open'] * (1 - abs(np.random.normal(0, 0.001, len(dates))))
    df['close'] = prices
    df['volume'] = np.random.lognormal(10, 1, len(dates))
    
    # Create strategy
    logger.info("Creating strategy...")
    strategy = TrendFollowingStrategy(
        name="BTC_Trend_Following",
        fast_period=5,
        slow_period=20
    )
    
    # Create backtest
    logger.info("Setting up backtest...")
    backtest = Backtest(
        data=df,
        strategy=strategy,
        initial_capital=10000.0,
        commission=0.001
    )
    
    # Set risk management parameters
    backtest.set_position_sizing(0.9)  # 90% of capital per trade
    backtest.set_stop_loss(2.0)  # 2% stop loss
    backtest.set_take_profit(6.0)  # 6% take profit (3:1 reward-risk ratio)
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtest.run()
    
    # Create output directory
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # Save results
    logger.info("Saving backtest results...")
    backtest.save_results(output_dir)
    
    # Analyze results
    logger.info("Analyzing results...")
    analysis = Analysis(backtest=backtest)
    report = analysis.generate_performance_report(output_dir)
    
    # Display key metrics
    metrics = backtest.get_performance_metrics()
    print("\n=== Performance Metrics ===")
    print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
    print(f"Annualized Return: {metrics['annual_return'] * 100:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
    print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
    print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")
    print(f"Number of Trades: {metrics['num_trades']}")
    print("===========================\n")
    
    logger.info("Backtest completed successfully!")
    
    # Ask if user wants to view plots
    view_plots = input("Do you want to view performance plots? (y/n): ")
    if view_plots.lower() == 'y':
        # Plot results
        logger.info("Plotting results...")
        
        # Plot equity curve
        analysis.plot_equity_curve()
        
        # Plot drawdowns
        analysis.plot_drawdowns()
        
        # Plot returns distribution
        analysis.plot_returns_distribution()
        
        # Plot trades
        analysis.plot_trades()
        
        plt.show()
    else:
        logger.info("Skipping plots as requested.")
        print("Plots were not displayed. You can find the saved charts in the output directory.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error running backtest: {e}", exc_info=True)
        print(f"Error: {e}")