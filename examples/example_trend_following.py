

# # import os
# # import sys
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import logging
# # from datetime import datetime

# # # Add parent directory to path to import the library
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # # Import necessary modules
# # from backtest_lib.strategies.strategy import Strategy
# # from backtest_lib.strategies.trend_following import TrendFollowingStrategy
# # from backtest_lib.data_adapters.data_loader import DataLoader
# # from backtest_lib.engine.backtest import Backtest
# # from backtest_lib.analysis.analysis import Analysis
# # from backtest_lib.utils.utils import setup_logger

# # # Set up logging
# # logger = setup_logger(name='example_trend_following', log_file='example_trend_following.log')

# # def main():
# #     # Paths
# #     data_dir = os.path.join('..', 'data')
# #     output_dir = os.path.join('..', 'output')
# #     os.makedirs(output_dir, exist_ok=True)
    
# #     # Data loader
# #     logger.info("Loading data...")
# #     data_loader = DataLoader(data_dir)
    
# #     # Load BTC data
# #     btc_data = data_loader.get_ohlcv('BTCUSD', '10m')
    
# #     # Create strategy
# #     logger.info("Creating strategy...")
# #     strategy = TrendFollowingStrategy(
# #         name="BTC_Trend_Following",
# #         fast_period=5,
# #         slow_period=20
# #     )
    
# #     # Create backtest
# #     logger.info("Setting up backtest...")
# #     backtest = Backtest(
# #         data=btc_data,
# #         strategy=strategy,
# #         initial_capital=10000.0,
# #         commission=0.001
# #     )
    
# #     # Set risk management parameters
# #     backtest.set_position_sizing(0.9)  # 90% of capital per trade
# #     backtest.set_stop_loss(2.0)  # 2% stop loss
# #     backtest.set_take_profit(6.0)  # 6% take profit (3:1 reward-risk ratio)
    
# #     # Run backtest
# #     logger.info("Running backtest...")
# #     results = backtest.run()
    
# #     # Save results
# #     logger.info("Saving backtest results...")
# #     backtest.save_results(output_dir)
    
# #     # Analyze results
# #     logger.info("Analyzing results...")
# #     analysis = Analysis(backtest=backtest)
# #     report = analysis.generate_performance_report(output_dir)
    
# #     # Display key metrics
# #     metrics = backtest.get_performance_metrics()
# #     print("\n=== Performance Metrics ===")
# #     print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
# #     print(f"Annualized Return: {metrics['annual_return'] * 100:.2f}%")
# #     print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
# #     print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
# #     print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
# #     print(f"Win Rate: {metrics['win_rate'] * 100:.2f}%")
# #     print(f"Number of Trades: {metrics['num_trades']}")
# #     print("===========================\n")
    
# #     # Plot results
# #     logger.info("Plotting results...")
    
# #     # Plot equity curve
# #     analysis.plot_equity_curve()
    
# #     # Plot drawdowns
# #     analysis.plot_drawdowns()
    
# #     # Plot returns distribution
# #     analysis.plot_returns_distribution()
    
# #     # Plot trades
# #     analysis.plot_trades()
    
# #     plt.show()

# # if __name__ == "__main__":
# #     main()

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
# from backtest_lib.strategies.trend_following import TrendFollowingStrategy
# from backtest_lib.data_adapters.data_loader import DataLoader
# from backtest_lib.engine.backtest import Backtest
# from backtest_lib.analysis.analysis import Analysis
# from backtest_lib.utils.utils import setup_logger

# # Set up logging
# logger = setup_logger(name='example_trend_following', log_file='example_trend_following.log')

# def main():
#     # Create mock data since actual data files don't exist
#     logger.info("Creating mock data...")
    
#     # Create date range
#     dates = pd.date_range(start='2024-01-01', end='2024-04-30', freq='10min')
    
#     # Create sample price data
#     np.random.seed(42)  # For reproducibility
    
#     # Initial price
#     initial_price = 50000
    
#     # Generate random returns
#     returns = np.random.normal(0.0001, 0.002, len(dates))
    
#     # Calculate price series
#     prices = initial_price * (1 + returns).cumprod()
    
#     # Create mock OHLCV data
#     btc_data = pd.DataFrame(index=dates)
#     btc_data['open'] = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
#     btc_data['high'] = btc_data['open'] * (1 + abs(np.random.normal(0, 0.001, len(dates))))
#     btc_data['low'] = btc_data['open'] * (1 - abs(np.random.normal(0, 0.001, len(dates))))
#     btc_data['close'] = prices
#     btc_data['volume'] = np.random.lognormal(10, 1, len(dates))
    
#     # Create output directory
#     output_dir = os.path.join('..', 'output')
#     os.makedirs(output_dir, exist_ok=True)
    
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
    
#     # First patch the trend_following.py issue with method=None
#     # We're fixing it dynamically for this execution
#     from types import MethodType
    
#     def patched_generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
#         """
#         Generate trading signals - patched version
        
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             Market data with OHLCV
            
#         Returns:
#         --------
#         pd.DataFrame
#             DataFrame with signals
#         """
#         df = data.copy()
        
#         # Calculate moving averages
#         fast_period = self.parameters['fast_period']
#         slow_period = self.parameters['slow_period']
        
#         df[f'sma_{fast_period}'] = df['close'].rolling(window=fast_period).mean()
#         df[f'sma_{slow_period}'] = df['close'].rolling(window=slow_period).mean()
        
#         # Calculate ATR for position sizing (optional)
#         atr_period = self.parameters['atr_period']
        
#         # Calculate ATR manually instead of using the Indicators class
#         high = df['high']
#         low = df['low']
#         close = df['close']
        
#         prev_close = close.shift(1)
#         tr1 = high - low
#         tr2 = (high - prev_close).abs()
#         tr3 = (low - prev_close).abs()
        
#         tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
#         df['atr'] = tr.rolling(window=atr_period).mean()
        
#         # Generate signals
#         df['signal'] = 0
#         df['position'] = 0
        
#         # Buy signal: fast MA crosses above slow MA
#         crossover_up = (df[f'sma_{fast_period}'] > df[f'sma_{slow_period}']) & \
#                      (df[f'sma_{fast_period}'].shift(1) <= df[f'sma_{slow_period}'].shift(1))
#         df.loc[crossover_up, 'signal'] = 1
        
#         # Sell signal: fast MA crosses below slow MA
#         crossover_down = (df[f'sma_{fast_period}'] < df[f'sma_{slow_period}']) & \
#                        (df[f'sma_{fast_period}'].shift(1) >= df[f'sma_{slow_period}'].shift(1))
#         df.loc[crossover_down, 'signal'] = -1
        
#         # Calculate position: 1 for long, -1 for short, 0 for flat
#         df['position'] = df['signal'].copy()
#         df['position'] = df['position'].replace(to_replace=0, method=None)
#         # Now use ffill() instead of fillna(method='ffill')
#         df['position'] = df['position'].ffill()
        
#         # Remove NaN values
#         df.dropna(inplace=True)
        
#         self.signals = df
#         return df
    
#     # Apply the patched method to the strategy
#     strategy.generate_signals = MethodType(patched_generate_signals, strategy)
    
#     # Run the backtest with patched strategy
#     results = backtest.run()
    
#     # Save results
#     logger.info("Saving backtest results...")
#     backtest.save_results(output_dir)
    
#     # Analyze results
#     logger.info("Analyzing results...")
#     analysis = Analysis(backtest=backtest)
#     report = analysis.generate_performance_report(output_dir)

#     # analysis.export_results_to_csv(results, results_dir)
#     # analysis.analyze_results(results_dir)
    
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
from backtest_lib.indicators.indicators import Indicators
from backtest_lib.utils.utils import setup_logger

# Set up logging
logger = setup_logger(name='example_trend_following', log_file='example_trend_following.log')

class ImprovedTrendFollowingStrategy(TrendFollowingStrategy):
    """
    Improved Trend-Following Strategy with fixed signal generation
    """
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals with correct position calculation
        
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
        
        # Use proper indicator calculations 
        df[f'sma_{fast_period}'] = Indicators.sma(df['close'], fast_period)
        df[f'sma_{slow_period}'] = Indicators.sma(df['close'], slow_period)
        
        # Calculate ATR for position sizing (optional)
        atr_period = self.parameters['atr_period']
        df['atr'] = Indicators.atr(df['high'], df['low'], df['close'], atr_period)
        
        # Generate signals
        df['signal'] = 0
        
        # Buy signal: fast MA crosses above slow MA
        crossover_up = (df[f'sma_{fast_period}'] > df[f'sma_{slow_period}']) & \
                       (df[f'sma_{fast_period}'].shift(1) <= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_up, 'signal'] = 1
        
        # Sell signal: fast MA crosses below slow MA
        crossover_down = (df[f'sma_{fast_period}'] < df[f'sma_{slow_period}']) & \
                         (df[f'sma_{fast_period}'].shift(1) >= df[f'sma_{slow_period}'].shift(1))
        df.loc[crossover_down, 'signal'] = -1
        
        # Calculate position: 1 for long, -1 for short, 0 for flat
        # Use a proper method to handle the forward filling without deprecation warnings
        df['position'] = df['signal'].replace(0, np.nan).ffill().fillna(0)
        
        # Remove NaN values
        df = df.dropna(subset=[f'sma_{fast_period}', f'sma_{slow_period}', 'atr'])
        
        self.signals = df
        return df

def generate_sample_data(start_date='2024-01-01', end_date='2024-04-30', freq='10min', seed=42):
    """
    Generate realistic sample price data for backtesting
    
    Parameters:
    -----------
    start_date : str
        Start date for the data
    end_date : str
        End date for the data
    freq : str
        Frequency of data points
    seed : int
        Random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with OHLCV data
    """
    # Create date range
    dates = pd.date_range(start=start_date, end=end_date, freq=freq)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Initial price and create price trend with momentum
    initial_price = 50000
    
    # Create a more realistic price series with trends and reversals
    # Start with random walk
    random_returns = np.random.normal(0.0001, 0.002, len(dates))
    
    # Add some trending behavior 
    trend_component = np.sin(np.linspace(0, 6*np.pi, len(dates))) * 0.001
    
    # Combine random and trend components
    returns = random_returns + trend_component
    
    # Calculate cumulative returns and price series
    cum_returns = np.cumprod(1 + returns)
    prices = initial_price * cum_returns
    
    # Create realistic OHLCV data
    btc_data = pd.DataFrame(index=dates)
    btc_data['open'] = prices * (1 + np.random.normal(0, 0.0005, len(dates)))
    btc_data['high'] = np.maximum(
        btc_data['open'] * (1 + abs(np.random.normal(0, 0.001, len(dates)))),
        prices * (1 + abs(np.random.normal(0, 0.0015, len(dates))))
    )
    btc_data['low'] = np.minimum(
        btc_data['open'] * (1 - abs(np.random.normal(0, 0.001, len(dates)))),
        prices * (1 - abs(np.random.normal(0, 0.0015, len(dates))))
    )
    btc_data['close'] = prices
    
    # Create volume with correlation to price changes
    price_changes = np.abs(np.diff(prices, prepend=prices[0]))
    volume_base = np.random.lognormal(10, 1, len(dates))
    volume_trend = 0.5 * volume_base + 0.5 * price_changes / price_changes.mean() * volume_base.mean()
    btc_data['volume'] = volume_trend
    
    return btc_data

def main():
    # Create output directory
    output_dir = os.path.join('..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    print(f"/n Output directory created at: {output_dir}")
    # Generate sample data
    logger.info("Generating sample data...")
    btc_data = generate_sample_data()
    
    # Create strategy with fixed implementation
    logger.info("Creating improved trend following strategy...")
    strategy = ImprovedTrendFollowingStrategy(
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
    
    # Create a figure with 4 subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Price and moving averages
    ax1 = plt.subplot(2, 2, 1)
    ax1.plot(results.index, results['close'], label='Price', alpha=0.5)
    ax1.plot(results.index, results[f'sma_{strategy.parameters["fast_period"]}'], label=f'SMA {strategy.parameters["fast_period"]}')
    ax1.plot(results.index, results[f'sma_{strategy.parameters["slow_period"]}'], label=f'SMA {strategy.parameters["slow_period"]}')
    
    # Mark buy and sell points
    buys = results[results['signal'] == 1]
    sells = results[results['signal'] == -1]
    ax1.scatter(buys.index, buys['close'], color='green', marker='^', s=100, label='Buy Signal')
    ax1.scatter(sells.index, sells['close'], color='red', marker='v', s=100, label='Sell Signal')
    
    ax1.set_title('Price Chart and Moving Averages')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Equity curve
    ax2 = plt.subplot(2, 2, 2)
    ax2.plot(results.index, results['portfolio_value'])
    ax2.set_title('Equity Curve')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Portfolio Value')
    ax2.grid(True)
    
    # Plot 3: Drawdowns
    ax3 = plt.subplot(2, 2, 3)
    returns = results['returns'].dropna()
    if len(returns) > 0:
        cumulative_returns = (1 + returns).cumprod()
        max_return = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns / max_return) - 1
        ax3.plot(drawdowns.index, drawdowns * 100)
        ax3.set_title('Drawdowns (%)')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Drawdown %')
        ax3.grid(True)
    
    # Plot 4: Position over time
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(results.index, results['position'])
    ax4.set_title('Position (1=Long, -1=Short, 0=Flat)')
    ax4.set_xlabel('Date')
    ax4.set_ylabel('Position')
    ax4.grid(True)
    
    plt.tight_layout()
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
    
    # Return key summary for easy reference
    return {
        'total_return': metrics['total_return'],
        'sharpe_ratio': metrics['sharpe_ratio'],
        'max_drawdown': metrics['max_drawdown'],
        'num_trades': metrics['num_trades']
    }

if __name__ == "__main__":
    main()