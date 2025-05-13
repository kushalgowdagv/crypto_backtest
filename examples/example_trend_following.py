


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
from backtest_lib.strategies.trend_following import TrendFollowingStrategy
from backtest_lib.data_adapters.data_loader import DataLoader
from backtest_lib.engine.backtest import Backtest
from backtest_lib.analysis.analysis import Analysis
from backtest_lib.utils.utils import setup_logger

# Set up logging
logger = setup_logger(name='example_trend_following', log_file='example_trend_following.log')

def main():
    # Paths
    data_dir = os.path.join(os.getcwd(), 'data')
    output_dir = os.path.join(os.getcwd(), 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nOutput directory created at: {output_dir}")
    
    # Data loader
    logger.info("Loading data...")
    data_loader = DataLoader(data_dir)
    
    # Load BTC data
    btc_data = data_loader.get_ohlcv('BTCUSD', '10m')
    
    # Create strategy
    logger.info("Creating trend following strategy...")
    strategy = TrendFollowingStrategy(
        name="BTC_Trend_Following",
        fast_period=5,
        slow_period=20
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
    backtest.set_position_sizing(0.9)  # 90% of capital per trade
    backtest.set_stop_loss(2.0)  # 2% stop loss
    backtest.set_take_profit(6.0)  # 6% take profit (3:1 reward-risk ratio)
    
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
    print(f"Average Trade PnL: {metrics['avg_pnl'] * 100:.2f}%")
    print("===========================\n")
    
    # Plot results
    logger.info("Plotting results...")
    
    plt.savefig(os.path.join(output_dir, f"{strategy.name}_summary.png"))
    
    # Also create and save the detailed analysis plots
    analysis.plot_equity_curve()
    plt.savefig(os.path.join(output_dir, f"{strategy.name}_equity.png"))
    
    analysis.plot_drawdowns()
    plt.savefig(os.path.join(output_dir, f"{strategy.name}_drawdowns.png"))
    
    analysis.plot_returns_distribution()
    plt.savefig(os.path.join(output_dir, f"{strategy.name}_returns_dist.png"))
    
    analysis.plot_trades()
    plt.savefig(os.path.join(output_dir, f"{strategy.name}_trades.png"))
    
    # Display the plots
    plt.show()

    print("Plots saved in output directory.")
    print("=="*30)
    
    # Return key summary for easy reference
    return {
        'total_return': metrics['total_return'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'num_trades': metrics['num_trades']
    }

if __name__ == "__main__":
    main()