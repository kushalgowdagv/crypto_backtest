

# import os
# import sys
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import logging
# from datetime import datetime

# # Add parent directory to path to import the library
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # Import necessary modules
# from backtest_lib.strategies.strategy import Strategy
# from backtest_lib.strategies.trend_following import TrendFollowingStrategy
# from backtest_lib.data_adapters.data_loader import DataLoader
# from backtest_lib.engine.backtest import Backtest
# from backtest_lib.analysis.analysis import Analysis
# from backtest_lib.utils.utils import setup_logger

# # Set up logging
# logger = setup_logger(name='example_trend_following', log_file='example_trend_following.log')

# def main():
#     # Paths
#     data_dir = os.path.join('..', 'data')
#     output_dir = os.path.join('..', 'output')
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Data loader
#     logger.info("Loading data...")
#     data_loader = DataLoader(data_dir)
    
#     # Load BTC data
#     btc_data = data_loader.get_ohlcv('BTCUSD', '10m')
    
#     # Create strategy
#     logger.info("Creating strategy...")
#     strategy = TrendFollowingStrategy(
#         name="BTC_Trend_Following",
#         fast_period=5,
#         slow_period=20
#     )
    
#     # Create backtest
#     logger.info("Setting up backtest...")
#     backtest = Backtest(
#         data=btc_data,
#         strategy=strategy,
#         initial_capital=10000.0,
#         commission=0.001
#     )
    
#     # Set risk management parameters
#     backtest.set_position_sizing(0.9)  # 90% of capital per trade
#     backtest.set_stop_loss(2.0)  # 2% stop loss
#     backtest.set_take_profit(6.0)  # 6% take profit (3:1 reward-risk ratio)
    
#     # Run backtest
#     logger.info("Running backtest...")
#     results = backtest.run()
    
#     # Save results
#     logger.info("Saving backtest results...")
#     backtest.save_results(output_dir)
    
#     # Analyze results
#     logger.info("Analyzing results...")
#     analysis = Analysis(backtest=backtest)
#     report = analysis.generate_performance_report(output_dir)
    
#     # Display key metrics
#     metrics = backtest.get_performance_metrics()
#     print("\n=== Performance Metrics ===")
#     print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
#     print(f"Annualized Return: {metrics['annual_return'] * 100:.2f}%")
#     print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
#     print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
#     print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
#     print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")
#     print(f"Number of Trades: {metrics['num_trades']}")
#     print("===========================\n")
    
#     # Plot results
#     logger.info("Plotting results...")
    
#     # Plot equity curve
#     analysis.plot_equity_curve()
    
#     # Plot drawdowns
#     analysis.plot_drawdowns()
    
#     # Plot returns distribution
#     analysis.plot_returns_distribution()
    
#     # Plot trades
#     analysis.plot_trades()
    
#     plt.show()

# if __name__ == "__main__":
#     main()

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
    # Create mock data since actual data files don't exist
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
    btc_data = pd.DataFrame(index=dates)
    btc_data['open'] = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
    btc_data['high'] = btc_data['open'] * (1 + abs(np.random.normal(0, 0.001, len(dates))))
    btc_data['low'] = btc_data['open'] * (1 - abs(np.random.normal(0, 0.001, len(dates))))
    btc_data['close'] = prices
    btc_data['volume'] = np.random.lognormal(10, 1, len(dates))
    
    # Create output directory
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    
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
    
    # First patch the trend_following.py issue with method=None
    # We're fixing it dynamically for this execution
    from types import MethodType
    
    def patched_generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals - patched version
        
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
        
        # Calculate moving averages
        fast_period = self.parameters['fast_period']
        slow_period = self.parameters['slow_period']
        
        df[f'sma_{fast_period}'] = df['close'].rolling(window=fast_period).mean()
        df[f'sma_{slow_period}'] = df['close'].rolling(window=slow_period).mean()
        
        # Calculate ATR for position sizing (optional)
        atr_period = self.parameters['atr_period']
        
        # Calculate ATR manually instead of using the Indicators class
        high = df['high']
        low = df['low']
        close = df['close']
        
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=atr_period).mean()
        
        # Generate signals
        df['signal'] = 0
        df['position'] = 0
        
        # Buy signal: fast MA crosses above slow MA
        crossover_up = (df[f'sma_{fast_period}'] > df[f'sma_{slow_period}']) & \
                     (df[f'sma_{fast_period}'].shift(1) <= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_up, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        crossover_down = (df[f'sma_{fast_period}'] < df[f'sma_{slow_period}']) & \
                       (df[f'sma_{fast_period}'].shift(1) >= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_down, 'signal'] = -1
        
        # Calculate position: 1 for long, -1 for short, 0 for flat
        # Use ffill method instead of the problematic method=None
        df['position'] = df['signal'].replace(to_replace=0, method='ffill')
        
        # Remove NaN values
        df.dropna(inplace=True)
        
        self.signals = df
        return df
    
    # Apply the patched method to the strategy
    strategy.generate_signals = MethodType(patched_generate_signals, strategy)
    
    # Run the backtest with patched strategy
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