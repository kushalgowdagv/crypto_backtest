import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Add parent directory to path to import the library
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import necessary modules
from backtest_lib.strategies.volatility_breakout import VolatilityBreakoutStrategy
from backtest_lib.data_adapters.data_loader import DataLoader
from backtest_lib.engine.backtest import Backtest
from backtest_lib.analysis.analysis import Analysis
from backtest_lib.utils.utils import setup_logger

# Set up logging
logger = setup_logger(name='example_volatility_breakout', log_file='example_volatility_breakout.log')

def main():
    # Paths
    data_dir = os.path.join('..', 'data')
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
    # Data loader
    logger.info("Loading data...")
    data_loader = DataLoader(data_dir)
    
    # Load BTC data
    btc_data = data_loader.get_ohlcv('BTCUSD', '10m')
    
    # Create strategy
    logger.info("Creating strategy...")
    strategy = VolatilityBreakoutStrategy(
        name="BTC_Volatility_Breakout",
        period=20,
        std_dev=2.0,
        atr_period=14,
        atr_multiplier=1.5
    )
    
    # Create backtest
    logger.info("Setting up backtest...")
    backtest = Backtest(
        data=btc_data,
        strategy=strategy,
        initial_capital=10000.0,
        commission=0.001
    )
    
    # Set risk management parameters
    backtest.set_position_sizing(0.75)  # 75% of capital per trade
    backtest.set_stop_loss(2.5)  # 2.5% stop loss
    backtest.set_take_profit(5.0)  # 5% take profit (2:1 reward-risk ratio)
    
    # Run backtest
    logger.info("Running backtest...")
    results = backtest.run()
    
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

if __name__ == "__main__":
    main()