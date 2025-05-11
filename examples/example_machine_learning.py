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
from backtest_lib.strategies.machine_learning import MachineLearningStrategy
from backtest_lib.data_adapters.data_loader import DataLoader
from backtest_lib.engine.backtest import Backtest
from backtest_lib.analysis.analysis import Analysis
from backtest_lib.utils.utils import setup_logger

# Set up logging
logger = setup_logger(name='example_machine_learning', log_file='example_machine_learning.log')

def main():
    # Paths
    data_dir = os.path.join('..', 'data')
    output_dir = os.path.join('..', 'output')
    models_dir = os.path.join('..', 'saved_models')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    # Data loader
    logger.info("Loading data...")
    data_loader = DataLoader(data_dir)
    
    # Load BTC data
    btc_data = data_loader.get_ohlcv('BTCUSD', '10m')
    
    # Create strategy
    logger.info("Creating strategy...")
    strategy = MachineLearningStrategy(
        name="BTC_ML_Strategy",
        model_type="xgboost",  # Using XGBoost as it's faster to train than LSTM/GRU
        lookback_period=20,
        prediction_horizon=1,
        features=['close', 'volume', 'rsi', 'macd', 'bb_width']
    )
    
    # Split data for training and testing
    split_idx = int(len(btc_data) * 0.8)
    train_data = btc_data.iloc[:split_idx]
    test_data = btc_data.iloc[split_idx:]
    
    # Train the model
    logger.info("Training model...")
    try:
        # Check if model exists
        strategy.load_model(models_dir)
        logger.info("Model loaded from disk")
    except Exception as e:
        logger.info(f"Training new model: {e}")
        strategy.train_model(train_data, epochs=50, batch_size=32, early_stopping=True)
        strategy.save_model(models_dir)
    
    # Create backtest on test data
    logger.info("Setting up backtest...")
    backtest = Backtest(
        data=test_data,
        strategy=strategy,
        initial_capital=10000.0,
        commission=0.001
    )
    
    # Set risk management parameters
    backtest.set_position_sizing(0.7)  # 70% of capital per trade
    backtest.set_stop_loss(2.0)  # 2% stop loss
    backtest.set_take_profit(4.0)  # 4% take profit (2:1 reward-risk ratio)
    
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