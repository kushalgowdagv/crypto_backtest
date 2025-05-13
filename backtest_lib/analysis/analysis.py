# # import os
# # import json
# # import pandas as pd
# # import numpy as np
# # import matplotlib.pyplot as plt
# # import seaborn as sns
# # from datetime import datetime
# # from typing import Dict, List
# # import logging

# # # Get logger
# # logger = logging.getLogger('backtest_lib')


# # # Custom JSON encoder to handle pandas Timestamp objects
# # class CustomJSONEncoder(json.JSONEncoder):
# #     def default(self, obj):
# #         if isinstance(obj, pd.Timestamp):
# #             return obj.strftime('%Y-%m-%d %H:%M:%S')
# #         return super().default(obj)


# # class Analysis:
# #     """
# #     Performance analysis and reporting tools
# #     """
    
# #     def __init__(self, backtest=None, results: pd.DataFrame = None):
# #         """
# #         Initialize the analysis module
        
# #         Parameters:
# #         -----------
# #         backtest : Backtest, optional
# #             Backtest instance
# #         results : pd.DataFrame, optional
# #             Backtest results
# #         """
# #         if backtest is not None:
# #             self.backtest = backtest
# #             self.results = backtest.results
# #             self.strategy_name = backtest.strategy.name
# #         elif results is not None:
# #             self.backtest = None
# #             self.results = results
# #             self.strategy_name = "Unknown"
# #         else:
# #             logger.error("Either backtest or results must be provided")
# #             raise ValueError("Either backtest or results must be provided")
    
# #     def generate_performance_report(self, directory: str = 'reports', save_fig: bool = True) -> Dict:
# #         """
# #         Generate performance report
        
# #         Parameters:
# #         -----------
# #         directory : str
# #             Directory to save report
# #         save_fig : bool
# #             Whether to save figures
            
# #         Returns:
# #         --------
# #         Dict
# #             Dictionary with report details
# #         """
# #         # Create directory
# #         os.makedirs(directory, exist_ok=True)
        
# #         # Create timestamp
# #         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
# #         # Get performance metrics
# #         if self.backtest is not None:
# #             metrics = self.backtest.get_performance_metrics()
# #             trades = self.backtest.get_trades()
# #         else:
# #             # Calculate metrics from results
# #             metrics = self._calculate_metrics()
# #             trades = self._extract_trades()
        
# #         # Convert Timestamp objects to strings in trades dataframe
# #         trades_dict = []
# #         for _, trade in trades.iterrows():
# #             trade_dict = trade.to_dict()
# #             # Convert all timestamp objects to strings
# #             for key, value in trade_dict.items():
# #                 if isinstance(value, pd.Timestamp):
# #                     trade_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
# #             trades_dict.append(trade_dict)
        
# #         # Create report content
# #         report = {
# #             'strategy_name': self.strategy_name,
# #             'timestamp': timestamp,
# #             'metrics': metrics,
# #             'trades': trades_dict,
# #             'figures': []
# #         }
        
# #         # Generate figures
# #         if save_fig:
# #             report['figures'] = self._generate_figures(directory, timestamp)
        
# #         # Save report to JSON using custom encoder
# #         report_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_report.json")
# #         with open(report_file, 'w') as f:
# #             json.dump(report, f, indent=4, cls=CustomJSONEncoder)
        
# #         logger.info(f"Performance report for strategy '{self.strategy_name}' saved to {report_file}")
        
# #         return report
    
# #     def _calculate_metrics(self) -> Dict:
# #         """
# #         Calculate performance metrics from results
        
# #         Returns:
# #         --------
# #         Dict
# #             Dictionary with performance metrics
# #         """
# #         # Extract returns
# #         returns = self.results['returns'].dropna()
        
# #         if len(returns) == 0:
# #             logger.warning("No returns data available")
# #             return {}
        
# #         # Calculate metrics
# #         initial_capital = self.results['portfolio_value'].iloc[0]
# #         total_return = self.results['portfolio_value'].iloc[-1] / initial_capital - 1
        
# #         # Annualized return
# #         trading_days = len(returns)
# #         annual_factor = 252 / trading_days
# #         annual_return = (1 + total_return) ** annual_factor - 1
        
# #         # Volatility
# #         volatility = returns.std() * np.sqrt(252)
        
# #         # Sharpe ratio
# #         risk_free_rate = 0.02  # Assuming 2% risk-free rate
# #         sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
# #         # Sortino ratio
# #         negative_returns = returns[returns < 0]
# #         downside_deviation = negative_returns.std() * np.sqrt(252)
# #         sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
# #         # Maximum drawdown
# #         cumulative_returns = (1 + returns).cumprod()
# #         max_return = cumulative_returns.expanding().max()
# #         drawdowns = (cumulative_returns / max_return) - 1
# #         max_drawdown = drawdowns.min()
        
# #         metrics = {
# #             'total_return': total_return,
# #             'annual_return': annual_return,
# #             'volatility': volatility,
# #             'sharpe_ratio': sharpe_ratio,
# #             'sortino_ratio': sortino_ratio,
# #             'max_drawdown': max_drawdown
# #         }
        
# #         return metrics
    
# #     def _extract_trades(self) -> pd.DataFrame:
# #         """
# #         Extract trades from results
        
# #         Returns:
# #         --------
# #         pd.DataFrame
# #             DataFrame with trade details
# #         """
# #         # Placeholder for trade extraction from results
# #         # This is a simplified version and may not capture all trades accurately
        
# #         # Detect position changes
# #         position_changes = self.results['position'].diff() != 0
        
# #         # Get potential trade entries and exits
# #         trades = []
        
# #         # Find trade entries
# #         for i in range(1, len(self.results)):
# #             if self.results['position'].iloc[i] != 0 and self.results['position'].iloc[i-1] == 0:
# #                 # Entry
# #                 entry_date = self.results.index[i]
# #                 entry_price = self.results['close'].iloc[i]
# #                 position_type = 'LONG' if self.results['position'].iloc[i] > 0 else 'SHORT'
                
# #                 # Find corresponding exit
# #                 exit_found = False
# #                 for j in range(i+1, len(self.results)):
# #                     if self.results['position'].iloc[j] == 0 and self.results['position'].iloc[j-1] != 0:
# #                         # Exit
# #                         exit_date = self.results.index[j]
# #                         exit_price = self.results['close'].iloc[j]
                        
# #                         # Calculate PnL
# #                         if position_type == 'LONG':
# #                             pnl = (exit_price - entry_price) / entry_price
# #                         else:
# #                             pnl = (entry_price - exit_price) / entry_price
                        
# #                         # Add trade
# #                         trades.append({
# #                             'entry_date': entry_date,
# #                             'exit_date': exit_date,
# #                             'position_type': position_type,
# #                             'entry_price': entry_price,
# #                             'exit_price': exit_price,
# #                             'pnl': pnl
# #                         })
                        
# #                         exit_found = True
# #                         break
                
# #                 # If no exit found, trade is still open
# #                 if not exit_found:
# #                     trades.append({
# #                         'entry_date': entry_date,
# #                         'exit_date': None,
# #                         'position_type': position_type,
# #                         'entry_price': entry_price,
# #                         'exit_price': None,
# #                         'pnl': None
# #                     })
        
# #         return pd.DataFrame(trades)
    
# #     def _generate_figures(self, directory: str, timestamp: str) -> List[str]:
# #         """
# #         Generate performance figures
        
# #         Parameters:
# #         -----------
# #         directory : str
# #             Directory to save figures
# #         timestamp : str
# #             Timestamp for file names
            
# #         Returns:
# #         --------
# #         List[str]
# #             List of figure file paths
# #         """
# #         figure_files = []
        
# #         # Set Seaborn style
# #         sns.set_style('whitegrid')
# #         plt.figure(figsize=(12, 8))
        
# #         # 1. Portfolio value
# #         fig1 = plt.figure(figsize=(12, 6))
# #         plt.plot(self.results.index, self.results['portfolio_value'])
# #         plt.title(f"Portfolio Value - {self.strategy_name}")
# #         plt.xlabel('Date')
# #         plt.ylabel('Value')
# #         plt.grid(True)
# #         fig1_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_portfolio_value.png")
# #         plt.savefig(fig1_file)
# #         plt.close(fig1)
# #         figure_files.append(fig1_file)
        
# #         # 2. Cumulative returns
# #         fig2 = plt.figure(figsize=(12, 6))
# #         plt.plot(self.results.index, self.results['cumulative_returns'])
# #         plt.title(f"Cumulative Returns - {self.strategy_name}")
# #         plt.xlabel('Date')
# #         plt.ylabel('Returns')
# #         plt.grid(True)
# #         fig2_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_cumulative_returns.png")
# #         plt.savefig(fig2_file)
# #         plt.close(fig2)
# #         figure_files.append(fig2_file)
        
# #         # 3. Drawdowns
# #         fig3 = plt.figure(figsize=(12, 6))
# #         returns = self.results['returns'].dropna()
# #         if len(returns) > 0:
# #             cumulative_returns = (1 + returns).cumprod()
# #             max_return = cumulative_returns.expanding().max()
# #             drawdowns = (cumulative_returns / max_return) - 1
# #             plt.plot(drawdowns.index, drawdowns)
# #             plt.title(f"Drawdowns - {self.strategy_name}")
# #             plt.xlabel('Date')
# #             plt.ylabel('Drawdown')
# #             plt.grid(True)
# #             fig3_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_drawdowns.png")
# #             plt.savefig(fig3_file)
# #             plt.close(fig3)
# #             figure_files.append(fig3_file)
        
# #         # 4. Position over time
# #         fig4 = plt.figure(figsize=(12, 6))
# #         plt.plot(self.results.index, self.results['position'])
# #         plt.title(f"Position - {self.strategy_name}")
# #         plt.xlabel('Date')
# #         plt.ylabel('Position')
# #         plt.grid(True)
# #         fig4_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_position.png")
# #         plt.savefig(fig4_file)
# #         plt.close(fig4)
# #         figure_files.append(fig4_file)
        
# #         # 5. Returns distribution
# #         fig5 = plt.figure(figsize=(12, 6))
# #         returns = self.results['returns'].dropna()
# #         if len(returns) > 0:
# #             sns.histplot(returns, kde=True)
# #             plt.title(f"Returns Distribution - {self.strategy_name}")
# #             plt.xlabel('Returns')
# #             plt.ylabel('Frequency')
# #             plt.grid(True)
# #             fig5_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_returns_dist.png")
# #             plt.savefig(fig5_file)
# #             plt.close(fig5)
# #             figure_files.append(fig5_file)
        
# #         return figure_files
    
# #     def plot_equity_curve(self) -> None:
# #         """
# #         Plot equity curve
# #         """
# #         plt.figure(figsize=(12, 6))
# #         plt.plot(self.results.index, self.results['portfolio_value'])
# #         plt.title(f"Equity Curve - {self.strategy_name}")
# #         plt.xlabel('Date')
# #         plt.ylabel('Portfolio Value')
# #         plt.grid(True)
# #         plt.show()
    
# #     def plot_drawdowns(self) -> None:
# #         """
# #         Plot drawdowns
# #         """
# #         plt.figure(figsize=(12, 6))
# #         returns = self.results['returns'].dropna()
        
# #         if len(returns) > 0:
# #             cumulative_returns = (1 + returns).cumprod()
# #             max_return = cumulative_returns.expanding().max()
# #             drawdowns = (cumulative_returns / max_return) - 1
            
# #             plt.plot(drawdowns.index, drawdowns)
# #             plt.title(f"Drawdowns - {self.strategy_name}")
# #             plt.xlabel('Date')
# #             plt.ylabel('Drawdown')
# #             plt.grid(True)
# #             plt.show()
    
# #     def plot_returns_distribution(self) -> None:
# #         """
# #         Plot returns distribution
# #         """
# #         plt.figure(figsize=(12, 6))
# #         returns = self.results['returns'].dropna()
        
# #         if len(returns) > 0:
# #             sns.histplot(returns, kde=True)
# #             plt.title(f"Returns Distribution - {self.strategy_name}")
# #             plt.xlabel('Returns')
# #             plt.ylabel('Frequency')
# #             plt.grid(True)
# #             plt.show()
    
# #     def plot_trades(self) -> None:
# #         """
# #         Plot trades on price chart
# #         """
# #         plt.figure(figsize=(12, 6))
        
# #         # Plot price
# #         plt.plot(self.results.index, self.results['close'])
        
# #         # Get trades
# #         if self.backtest is not None:
# #             trades = self.backtest.get_trades()
# #         else:
# #             trades = self._extract_trades()
        
# #         # Plot entry and exit points
# #         for _, trade in trades.iterrows():
# #             if trade['position_type'] == 'LONG':
# #                 # Entry
# #                 plt.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
                
# #                 # Exit
# #                 if trade['exit_date'] is not None:
# #                     plt.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100)
# #             else:
# #                 # Entry
# #                 plt.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
                
# #                 # Exit
# #                 if trade['exit_date'] is not None:
# #                     plt.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
        
# #         plt.title(f"Trades - {self.strategy_name}")
# #         plt.xlabel('Date')
# #         plt.ylabel('Price')
# #         plt.grid(True)
# #         plt.show()

# # # # #!/usr/bin/env python
# # # # # -*- coding: utf-8 -*-

# # # # import os
# # # # import logging
# # # # import pandas as pd
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt
# # # # from datetime import datetime

# # # # # Set up logging
# # # # logger = logging.getLogger(__name__)


# # # # def generate_performance_metrics(backtest_results):
# # # #     """
# # # #     Generate performance metrics from backtest results.
    
# # # #     Args:
# # # #         backtest_results: The results from a backtest run
        
# # # #     Returns:
# # # #         dict: A dictionary of performance metrics
# # # #     """
# # # #     # Initialize the report dictionary
# # # #     report = {}
    
# # # #     # Extract equity curve and trades from backtest results
# # # #     equity_curve = backtest_results.equity_curve if hasattr(backtest_results, 'equity_curve') else None
# # # #     trades = backtest_results.trades if hasattr(backtest_results, 'trades') else None
    
# # # #     if equity_curve is not None and not equity_curve.empty:
# # # #         # Calculate returns
# # # #         initial_equity = equity_curve['equity'].iloc[0]
# # # #         final_equity = equity_curve['equity'].iloc[-1]
# # # #         total_return = (final_equity / initial_equity) - 1
        
# # # #         # Calculate annualized return
# # # #         days = (equity_curve.index[-1] - equity_curve.index[0]).days
# # # #         if days > 0:
# # # #             annual_return = (1 + total_return) ** (365 / days) - 1
# # # #         else:
# # # #             annual_return = 0
        
# # # #         # Calculate drawdown
# # # #         equity_curve['high_water_mark'] = equity_curve['equity'].cummax()
# # # #         equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['high_water_mark']) - 1
# # # #         max_drawdown = equity_curve['drawdown'].min()
        
# # # #         # Calculate daily returns
# # # #         equity_curve['daily_return'] = equity_curve['equity'].pct_change()
        
# # # #         # Calculate Sharpe ratio (assuming risk-free rate of 0)
# # # #         if len(equity_curve) > 1:
# # # #             sharpe_ratio = equity_curve['daily_return'].mean() / equity_curve['daily_return'].std() * (252 ** 0.5)
# # # #         else:
# # # #             sharpe_ratio = 0
        
# # # #         # Add metrics to report
# # # #         report['total_return'] = total_return
# # # #         report['annual_return'] = annual_return
# # # #         report['max_drawdown'] = max_drawdown
# # # #         report['sharpe_ratio'] = sharpe_ratio
        
# # # #         # Calculate volatility (annualized)
# # # #         if len(equity_curve) > 1:
# # # #             volatility = equity_curve['daily_return'].std() * (252 ** 0.5)
# # # #         else:
# # # #             volatility = 0
# # # #         report['volatility'] = volatility
        
# # # #         # Calculate Sortino ratio (downside risk only)
# # # #         downside_returns = equity_curve['daily_return'][equity_curve['daily_return'] < 0]
# # # #         if len(downside_returns) > 0:
# # # #             downside_deviation = downside_returns.std() * (252 ** 0.5)
# # # #             sortino_ratio = equity_curve['daily_return'].mean() * 252 / downside_deviation if downside_deviation != 0 else 0
# # # #         else:
# # # #             sortino_ratio = float('inf')  # No negative returns
# # # #         report['sortino_ratio'] = sortino_ratio
        
# # # #         # Calculate maximum consecutive wins and losses
# # # #         if trades is not None and not trades.empty:
# # # #             trades_with_results = trades.copy()
# # # #             trades_with_results['is_win'] = trades_with_results['profit_pct'] > 0
            
# # # #             # Calculate streaks
# # # #             trades_with_results['streak_group'] = (trades_with_results['is_win'] != trades_with_results['is_win'].shift(1)).cumsum()
# # # #             streak_groups = trades_with_results.groupby(['streak_group', 'is_win']).size().reset_index(name='count')
            
# # # #             max_wins = streak_groups[streak_groups['is_win']]['count'].max() if not streak_groups[streak_groups['is_win']].empty else 0
# # # #             max_losses = streak_groups[~streak_groups['is_win']]['count'].max() if not streak_groups[~streak_groups['is_win']].empty else 0
            
# # # #             report['max_consecutive_wins'] = max_wins
# # # #             report['max_consecutive_losses'] = max_losses
    
# # # #     if trades is not None and not trades.empty:
# # # #         # Calculate trade metrics
# # # #         winning_trades = trades[trades['profit_pct'] > 0]
# # # #         losing_trades = trades[trades['profit_pct'] <= 0]
        
# # # #         win_rate = len(winning_trades) / len(trades) if len(trades) > 0 else 0
# # # #         avg_win = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0
# # # #         avg_loss = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
        
# # # #         # Calculate profit factor
# # # #         total_win = winning_trades['profit_pct'].sum() if not winning_trades.empty else 0
# # # #         total_loss = abs(losing_trades['profit_pct'].sum()) if not losing_trades.empty else 0
# # # #         profit_factor = total_win / total_loss if total_loss != 0 else float('inf')
        
# # # #         # Calculate average holding period
# # # #         if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
# # # #             # Handle NaT values by filtering them out
# # # #             valid_duration_trades = trades.dropna(subset=['entry_time', 'exit_time'])
# # # #             if not valid_duration_trades.empty:
# # # #                 avg_holding_period = (valid_duration_trades['exit_time'] - valid_duration_trades['entry_time']).mean()
# # # #                 report['avg_holding_period_seconds'] = avg_holding_period.total_seconds()
        
# # # #         # Add metrics to report
# # # #         report['total_trades'] = len(trades)
# # # #         report['winning_trades'] = len(winning_trades)
# # # #         report['losing_trades'] = len(losing_trades)
# # # #         report['win_rate'] = win_rate
# # # #         report['avg_win_pct'] = avg_win
# # # #         report['avg_loss_pct'] = avg_loss
# # # #         report['profit_factor'] = profit_factor
        
# # # #         # Calculate expectancy
# # # #         expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
# # # #         report['expectancy'] = expectancy
    
# # # #     return report


# # # # def generate_performance_report(output_dir):
# # # #     """
# # # #     Generate a performance report from the backtest results.
    
# # # #     Args:
# # # #         output_dir (str): The directory containing the backtest results.
        
# # # #     Returns:
# # # #         dict: A dictionary of performance metrics
# # # #     """
# # # #     logger.info(f"Generating performance report for {output_dir}")
    
# # # #     # Load equity curve
# # # #     equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
# # # #     if not os.path.exists(equity_curve_path):
# # # #         logger.warning(f"Equity curve file not found at {equity_curve_path}")
# # # #         return None
    
# # # #     equity_curve = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
    
# # # #     # Load trades
# # # #     trades_path = os.path.join(output_dir, 'trades.csv')
# # # #     trades = None
# # # #     if os.path.exists(trades_path):
# # # #         trades = pd.read_csv(trades_path, index_col=0, parse_dates=['entry_time', 'exit_time'])
    
# # # #     # Create a BacktestResults-like object with the loaded data
# # # #     class BacktestResults:
# # # #         pass
    
# # # #     results = BacktestResults()
# # # #     results.equity_curve = equity_curve
# # # #     results.trades = trades
    
# # # #     # Generate metrics
# # # #     metrics = generate_performance_metrics(results)
    
# # # #     # Save metrics as CSV
# # # #     metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
# # # #     metrics_df.index.name = 'Metric'
    
# # # #     metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
# # # #     metrics_df.to_csv(metrics_path)
# # # #     logger.info(f"Performance metrics saved to {metrics_path}")
    
# # # #     return metrics


# # # # def plot_equity_curve(equity_curve, output_path=None):
# # # #     """
# # # #     Plot the equity curve and drawdown.
    
# # # #     Args:
# # # #         equity_curve (pd.DataFrame): The equity curve data.
# # # #         output_path (str, optional): The path to save the plot. Defaults to None.
        
# # # #     Returns:
# # # #         matplotlib.figure.Figure: The figure object.
# # # #     """
# # # #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
# # # #     # Plot equity curve
# # # #     ax1.plot(equity_curve.index, equity_curve['equity'], label='Equity', color='blue')
# # # #     ax1.set_title('Equity Curve')
# # # #     ax1.set_ylabel('Equity')
# # # #     ax1.grid(True)
# # # #     ax1.legend()
    
# # # #     # Calculate and plot drawdown
# # # #     if 'drawdown' not in equity_curve.columns:
# # # #         equity_curve['high_water_mark'] = equity_curve['equity'].cummax()
# # # #         equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['high_water_mark']) - 1
    
# # # #     ax2.fill_between(equity_curve.index, 0, equity_curve['drawdown'] * 100, color='red', alpha=0.3)
# # # #     ax2.set_title('Drawdown (%)')
# # # #     ax2.set_ylabel('Drawdown (%)')
# # # #     ax2.set_ylim(equity_curve['drawdown'].min() * 100 * 1.1, 0)  # Add 10% margin
# # # #     ax2.grid(True)
    
# # # #     plt.tight_layout()
    
# # # #     # Save the plot if output_path is provided
# # # #     if output_path:
# # # #         plt.savefig(output_path)
# # # #         logger.info(f"Equity curve plot saved to {output_path}")
    
# # # #     return fig


# # # # def plot_trade_analysis(trades, output_path=None):
# # # #     """
# # # #     Plot trade analysis charts.
    
# # # #     Args:
# # # #         trades (pd.DataFrame): The trades data.
# # # #         output_path (str, optional): The path to save the plot. Defaults to None.
        
# # # #     Returns:
# # # #         matplotlib.figure.Figure: The figure object.
# # # #     """
# # # #     if trades is None or trades.empty:
# # # #         logger.warning("No trade data available for plotting")
# # # #         return None
    
# # # #     fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    
# # # #     # Calculate cumulative profits
# # # #     trades['cumulative_profit'] = trades['profit_pct'].cumsum()
    
# # # #     # Plot 1: Cumulative profit over time
# # # #     axs[0, 0].plot(trades.index, trades['cumulative_profit'] * 100, color='blue')
# # # #     axs[0, 0].set_title('Cumulative Profit (%)')
# # # #     axs[0, 0].set_ylabel('Profit (%)')
# # # #     axs[0, 0].grid(True)
    
# # # #     # Plot 2: Profit distribution
# # # #     axs[0, 1].hist(trades['profit_pct'] * 100, bins=50, color='green', alpha=0.7)
# # # #     axs[0, 1].axvline(x=0, color='red', linestyle='--')
# # # #     axs[0, 1].set_title('Profit Distribution (%)')
# # # #     axs[0, 1].set_xlabel('Profit (%)')
# # # #     axs[0, 1].set_ylabel('Frequency')
# # # #     axs[0, 1].grid(True)
    
# # # #     # Plot 3: Win/Loss ratio by month
# # # #     if 'entry_time' in trades.columns:
# # # #         # Add month column
# # # #         trades['month'] = trades['entry_time'].dt.to_period('M')
        
# # # #         # Group by month
# # # #         monthly_stats = trades.groupby('month').agg(
# # # #             win_count=('profit_pct', lambda x: (x > 0).sum()),
# # # #             loss_count=('profit_pct', lambda x: (x <= 0).sum()),
# # # #             total_profit=('profit_pct', 'sum')
# # # #         )
        
# # # #         monthly_stats['win_rate'] = monthly_stats['win_count'] / (monthly_stats['win_count'] + monthly_stats['loss_count'])
        
# # # #         # Plot win rate by month
# # # #         axs[1, 0].bar(monthly_stats.index.astype(str), monthly_stats['win_rate'], color='blue', alpha=0.7)
# # # #         axs[1, 0].set_title('Win Rate by Month')
# # # #         axs[1, 0].set_xlabel('Month')
# # # #         axs[1, 0].set_ylabel('Win Rate')
# # # #         axs[1, 0].set_ylim(0, 1)
# # # #         for tick in axs[1, 0].get_xticklabels():
# # # #             tick.set_rotation(45)
# # # #         axs[1, 0].grid(True)
        
# # # #         # Plot total profit by month
# # # #         axs[1, 1].bar(monthly_stats.index.astype(str), monthly_stats['total_profit'] * 100, color='green', alpha=0.7)
# # # #         axs[1, 1].set_title('Total Profit by Month (%)')
# # # #         axs[1, 1].set_xlabel('Month')
# # # #         axs[1, 1].set_ylabel('Profit (%)')
# # # #         for tick in axs[1, 1].get_xticklabels():
# # # #             tick.set_rotation(45)
# # # #         axs[1, 1].grid(True)
# # # #     else:
# # # #         axs[1, 0].text(0.5, 0.5, 'No entry time data available', ha='center', va='center')
# # # #         axs[1, 1].text(0.5, 0.5, 'No entry time data available', ha='center', va='center')
    
# # # #     plt.tight_layout()
    
# # # #     # Save the plot if output_path is provided
# # # #     if output_path:
# # # #         plt.savefig(output_path)
# # # #         logger.info(f"Trade analysis plot saved to {output_path}")
    
# # # #     return fig


# # # # def analyze_results(output_dir):
# # # #     """
# # # #     Analyze the backtest results and generate visualizations.
    
# # # #     Args:
# # # #         output_dir (str): The directory containing the backtest results.
        
# # # #     Returns:
# # # #         dict: A dictionary of performance metrics
# # # #     """
# # # #     logger.info(f"Analyzing results in {output_dir}")
    
# # # #     # Generate performance report
# # # #     report = generate_performance_report(output_dir)
    
# # # #     # Load equity curve
# # # #     equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
# # # #     if os.path.exists(equity_curve_path):
# # # #         equity_curve = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
        
# # # #         # Plot equity curve
# # # #         plot_path = os.path.join(output_dir, 'equity_curve.png')
# # # #         plot_equity_curve(equity_curve, plot_path)
    
# # # #     # Load trades
# # # #     trades_path = os.path.join(output_dir, 'trades.csv')
# # # #     if os.path.exists(trades_path):
# # # #         trades = pd.read_csv(trades_path, index_col=0, parse_dates=['entry_time', 'exit_time'])
        
# # # #         # Clean NaT values
# # # #         if 'entry_time' in trades.columns:
# # # #             trades = trades.dropna(subset=['entry_time'])
        
# # # #         # Plot trade analysis
# # # #         plot_path = os.path.join(output_dir, 'trade_analysis.png')
# # # #         plot_trade_analysis(trades, plot_path)
    
# # # #     return report


# # # # def export_results_to_csv(results, output_dir):
# # # #     """
# # # #     Export backtest results to CSV files.
    
# # # #     Args:
# # # #         results: The results from a backtest run
# # # #         output_dir (str): The directory to save the CSV files
        
# # # #     Returns:
# # # #         dict: Paths to the saved CSV files
# # # #     """
# # # #     saved_files = {}
    
# # # #     # Create directory if it doesn't exist
# # # #     os.makedirs(output_dir, exist_ok=True)
    
# # # #     # Save equity curve
# # # #     if hasattr(results, 'equity_curve') and not results.equity_curve.empty:
# # # #         equity_curve_path = os.path.join(output_dir, 'equity_curve.csv')
# # # #         results.equity_curve.to_csv(equity_curve_path)
# # # #         saved_files['equity_curve'] = equity_curve_path
# # # #         logger.info(f"Equity curve saved to {equity_curve_path}")
    
# # # #     # Save trades
# # # #     if hasattr(results, 'trades') and not results.trades.empty:
# # # #         # Handle NaT values before saving
# # # #         trades_df = results.trades.copy()
        
# # # #         # Convert timestamp columns to string if they contain NaT
# # # #         for col in trades_df.select_dtypes(include=['datetime']).columns:
# # # #             # Check if column contains NaT values
# # # #             if trades_df[col].isna().any():
# # # #                 # Convert to string with NaN for NaT values
# # # #                 trades_df[col] = trades_df[col].astype(str).replace('NaT', np.nan)
        
# # # #         trades_path = os.path.join(output_dir, 'trades.csv')
# # # #         trades_df.to_csv(trades_path)
# # # #         saved_files['trades'] = trades_path
# # # #         logger.info(f"Trades saved to {trades_path}")
    
# # # #     # Save signals
# # # #     if hasattr(results, 'signals') and not results.signals.empty:
# # # #         signals_path = os.path.join(output_dir, 'signals.csv')
# # # #         results.signals.to_csv(signals_path)
# # # #         saved_files['signals'] = signals_path
# # # #         logger.info(f"Signals saved to {signals_path}")
    
# # # #     # Save positions
# # # #     if hasattr(results, 'positions') and not results.positions.empty:
# # # #         positions_path = os.path.join(output_dir, 'positions.csv')
# # # #         results.positions.to_csv(positions_path)
# # # #         saved_files['positions'] = positions_path
# # # #         logger.info(f"Positions saved to {positions_path}")
    
# # # #     # Generate and save performance metrics
# # # #     metrics = generate_performance_metrics(results)
# # # #     if metrics:
# # # #         metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
# # # #         metrics_df.index.name = 'Metric'
        
# # # #         metrics_path = os.path.join(output_dir, 'performance_metrics.csv')
# # # #         metrics_df.to_csv(metrics_path)
# # # #         saved_files['metrics'] = metrics_path
# # # #         logger.info(f"Performance metrics saved to {metrics_path}")
    
# # # #     return saved_files


# # # # def compare_strategies(strategy_dirs, output_dir=None):
# # # #     """
# # # #     Compare the performance of multiple strategies.
    
# # # #     Args:
# # # #         strategy_dirs (list): List of directories containing backtest results for different strategies.
# # # #         output_dir (str, optional): Directory to save the comparison results. Defaults to None.
        
# # # #     Returns:
# # # #         pd.DataFrame: A DataFrame containing the comparison results.
# # # #     """
# # # #     # Initialize a list to store the results
# # # #     comparison_data = []
    
# # # #     # Loop through each strategy directory
# # # #     for strategy_dir in strategy_dirs:
# # # #         # Get the strategy name from the directory name
# # # #         strategy_name = os.path.basename(strategy_dir)
        
# # # #         # Load the performance metrics
# # # #         metrics_path = os.path.join(strategy_dir, 'performance_metrics.csv')
# # # #         if not os.path.exists(metrics_path):
# # # #             logger.warning(f"Performance metrics file not found for strategy {strategy_name}")
# # # #             continue
        
# # # #         # Load the metrics
# # # #         metrics_df = pd.read_csv(metrics_path, index_col=0)
        
# # # #         # Convert to dictionary
# # # #         metrics_dict = metrics_df['Value'].to_dict()
        
# # # #         # Add strategy name
# # # #         metrics_dict['strategy'] = strategy_name
        
# # # #         # Add to comparison data
# # # #         comparison_data.append(metrics_dict)
    
# # # #     # Create a DataFrame from the comparison data
# # # #     if comparison_data:
# # # #         comparison_df = pd.DataFrame(comparison_data)
        
# # # #         # Set the strategy column as the index
# # # #         comparison_df.set_index('strategy', inplace=True)
        
# # # #         # Save the comparison results if output_dir is provided
# # # #         if output_dir:
# # # #             os.makedirs(output_dir, exist_ok=True)
# # # #             comparison_path = os.path.join(output_dir, 'strategy_comparison.csv')
# # # #             comparison_df.to_csv(comparison_path)
# # # #             logger.info(f"Strategy comparison saved to {comparison_path}")
            
# # # #             # Create comparison charts
# # # #             fig, axs = plt.subplots(2, 2, figsize=(14, 10))
            
# # # #             # Plot 1: Total Return
# # # #             if 'total_return' in comparison_df.columns:
# # # #                 comparison_df['total_return'].sort_values().plot(kind='barh', ax=axs[0, 0], color='blue')
# # # #                 axs[0, 0].set_title('Total Return')
# # # #                 axs[0, 0].set_xlabel('Return')
# # # #                 axs[0, 0].grid(True)
            
# # # #             # Plot 2: Sharpe Ratio
# # # #             if 'sharpe_ratio' in comparison_df.columns:
# # # #                 comparison_df['sharpe_ratio'].sort_values().plot(kind='barh', ax=axs[0, 1], color='green')
# # # #                 axs[0, 1].set_title('Sharpe Ratio')
# # # #                 axs[0, 1].set_xlabel('Ratio')
# # # #                 axs[0, 1].grid(True)
            
# # # #             # Plot 3: Max Drawdown
# # # #             if 'max_drawdown' in comparison_df.columns:
# # # #                 comparison_df['max_drawdown'].sort_values(ascending=False).plot(kind='barh', ax=axs[1, 0], color='red')
# # # #                 axs[1, 0].set_title('Max Drawdown')
# # # #                 axs[1, 0].set_xlabel('Drawdown')
# # # #                 axs[1, 0].grid(True)
            
# # # #             # Plot 4: Win Rate
# # # #             if 'win_rate' in comparison_df.columns:
# # # #                 comparison_df['win_rate'].sort_values().plot(kind='barh', ax=axs[1, 1], color='purple')
# # # #                 axs[1, 1].set_title('Win Rate')
# # # #                 axs[1, 1].set_xlabel('Rate')
# # # #                 axs[1, 1].grid(True)
            
# # # #             plt.tight_layout()
# # # #             chart_path = os.path.join(output_dir, 'strategy_comparison.png')
# # # #             plt.savefig(chart_path)
# # # #             logger.info(f"Strategy comparison chart saved to {chart_path}")
        
# # # #         return comparison_df
    
# # # #     return None


# # # # if __name__ == "__main__":
# # # #     # This module is not meant to be run directly
# # # #     print("This module is not meant to be run directly. Import and use the functions in your code.")


# # # #!/usr/bin/env python
# # # # -*- coding: utf-8 -*-

# # # import os
# # # import pandas as pd
# # # import numpy as np
# # # import matplotlib.pyplot as plt
# # # from datetime import datetime
# # # import logging

# # # logger = logging.getLogger(__name__)

# # # class PerformanceMetrics:
# # #     """
# # #     Class for calculating and storing performance metrics.
# # #     """
# # #     def __init__(self):
# # #         self.metrics = {}

# # #     def add_metric(self, name, value):
# # #         """Add a metric to the metrics dictionary."""
# # #         self.metrics[name] = value

# # #     def get_metric(self, name):
# # #         """Get a metric by name."""
# # #         return self.metrics.get(name)

# # #     def get_all_metrics(self):
# # #         """Get all metrics."""
# # #         return self.metrics

# # #     def to_dataframe(self):
# # #         """Convert metrics to a DataFrame."""
# # #         return pd.DataFrame.from_dict(self.metrics, orient='index', columns=['Value'])


# # # def calculate_returns(equity_curve):
# # #     """
# # #     Calculate various return metrics from an equity curve.
    
# # #     Args:
# # #         equity_curve (pd.DataFrame): DataFrame containing 'equity' column
        
# # #     Returns:
# # #         dict: Dictionary of return metrics
# # #     """
# # #     metrics = {}
    
# # #     # Calculate total return
# # #     initial_equity = equity_curve['equity'].iloc[0]
# # #     final_equity = equity_curve['equity'].iloc[-1]
# # #     metrics['total_return'] = (final_equity / initial_equity) - 1
    
# # #     # Calculate annualized return
# # #     days = (equity_curve.index[-1] - equity_curve.index[0]).days
# # #     if days > 0:
# # #         metrics['annual_return'] = (1 + metrics['total_return']) ** (365 / days) - 1
# # #     else:
# # #         metrics['annual_return'] = 0
    
# # #     # Calculate daily returns
# # #     equity_curve['daily_return'] = equity_curve['equity'].pct_change()
    
# # #     # Calculate volatility
# # #     metrics['volatility'] = equity_curve['daily_return'].std() * (252 ** 0.5)  # Annualized
    
# # #     # Calculate Sharpe ratio (assuming risk-free rate of 0)
# # #     if len(equity_curve) > 1 and metrics['volatility'] != 0:
# # #         metrics['sharpe_ratio'] = metrics['annual_return'] / metrics['volatility']
# # #     else:
# # #         metrics['sharpe_ratio'] = 0
    
# # #     return metrics


# # # def calculate_drawdowns(equity_curve):
# # #     """
# # #     Calculate drawdown metrics from an equity curve.
    
# # #     Args:
# # #         equity_curve (pd.DataFrame): DataFrame containing 'equity' column
        
# # #     Returns:
# # #         dict: Dictionary of drawdown metrics
# # #     """
# # #     metrics = {}
    
# # #     # Calculate drawdown
# # #     equity_curve['high_water_mark'] = equity_curve['equity'].cummax()
# # #     equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['high_water_mark']) - 1
    
# # #     # Get max drawdown
# # #     metrics['max_drawdown'] = equity_curve['drawdown'].min()
    
# # #     # Get max drawdown duration
# # #     if metrics['max_drawdown'] < 0:
# # #         # Find the peak before the max drawdown
# # #         max_dd_idx = equity_curve['drawdown'].idxmin()
# # #         peak_idx = equity_curve.loc[:max_dd_idx, 'high_water_mark'].idxmax()
        
# # #         # Find when we recover from the drawdown
# # #         if max_dd_idx == equity_curve.index[-1]:
# # #             # We haven't recovered yet
# # #             recovery_idx = equity_curve.index[-1]
# # #         else:
# # #             recovery_mask = (equity_curve.index > max_dd_idx) & (equity_curve['drawdown'] >= 0)
# # #             if any(recovery_mask):
# # #                 recovery_idx = equity_curve.index[recovery_mask][0]
# # #             else:
# # #                 recovery_idx = equity_curve.index[-1]
        
# # #         # Calculate duration
# # #         metrics['max_drawdown_duration'] = (recovery_idx - peak_idx).days
# # #     else:
# # #         metrics['max_drawdown_duration'] = 0
    
# # #     return metrics


# # # def calculate_trade_metrics(trades):
# # #     """
# # #     Calculate trade metrics from a DataFrame of trades.
    
# # #     Args:
# # #         trades (pd.DataFrame): DataFrame containing trade information
        
# # #     Returns:
# # #         dict: Dictionary of trade metrics
# # #     """
# # #     if trades.empty:
# # #         return {}
    
# # #     metrics = {}
    
# # #     # Basic trade metrics
# # #     metrics['total_trades'] = len(trades)
    
# # #     # Win/loss metrics
# # #     winning_trades = trades[trades['profit_pct'] > 0]
# # #     losing_trades = trades[trades['profit_pct'] <= 0]
    
# # #     metrics['winning_trades'] = len(winning_trades)
# # #     metrics['losing_trades'] = metrics['total_trades'] - metrics['winning_trades']
    
# # #     metrics['win_rate'] = metrics['winning_trades'] / metrics['total_trades'] if metrics['total_trades'] > 0 else 0
    
# # #     # Average profit/loss
# # #     metrics['avg_profit_pct'] = trades['profit_pct'].mean() if metrics['total_trades'] > 0 else 0
# # #     metrics['avg_win_pct'] = winning_trades['profit_pct'].mean() if not winning_trades.empty else 0
# # #     metrics['avg_loss_pct'] = losing_trades['profit_pct'].mean() if not losing_trades.empty else 0
    
# # #     # Calculate profit factor
# # #     total_win = winning_trades['profit_pct'].sum() if not winning_trades.empty else 0
# # #     total_loss = abs(losing_trades['profit_pct'].sum()) if not losing_trades.empty else 0
# # #     metrics['profit_factor'] = total_win / total_loss if total_loss != 0 else float('inf')
    
# # #     # Calculate expectancy
# # #     metrics['expectancy'] = metrics['win_rate'] * metrics['avg_win_pct'] + (1 - metrics['win_rate']) * metrics['avg_loss_pct']
    
# # #     # Calculate average holding period
# # #     if 'exit_time' in trades.columns and 'entry_time' in trades.columns:
# # #         # Clean NaT values for calculation
# # #         valid_periods = trades.dropna(subset=['exit_time', 'entry_time'])
# # #         if not valid_periods.empty:
# # #             holding_periods = (valid_periods['exit_time'] - valid_periods['entry_time']).dt.total_seconds() / 3600  # in hours
# # #             metrics['avg_holding_period_hours'] = holding_periods.mean()
# # #         else:
# # #             metrics['avg_holding_period_hours'] = np.nan
    
# # #     return metrics


# # # def generate_performance_report(results_dir):
# # #     """
# # #     Generate a performance report from backtest results.
    
# # #     Args:
# # #         results_dir (str): Directory containing backtest results
        
# # #     Returns:
# # #         PerformanceMetrics: Object containing performance metrics
# # #     """
# # #     # Initialize performance metrics
# # #     performance = PerformanceMetrics()
    
# # #     # Load equity curve
# # #     equity_curve_path = os.path.join(results_dir, 'equity_curve.csv')
# # #     if os.path.exists(equity_curve_path):
# # #         try:
# # #             equity_curve = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
            
# # #             # Calculate return metrics
# # #             return_metrics = calculate_returns(equity_curve)
# # #             for name, value in return_metrics.items():
# # #                 performance.add_metric(name, value)
            
# # #             # Calculate drawdown metrics
# # #             drawdown_metrics = calculate_drawdowns(equity_curve)
# # #             for name, value in drawdown_metrics.items():
# # #                 performance.add_metric(name, value)
            
# # #         except Exception as e:
# # #             logger.error(f"Error calculating equity curve metrics: {e}")
    
# # #     # Load trades
# # #     trades_path = os.path.join(results_dir, 'trades.csv')
# # #     if os.path.exists(trades_path):
# # #         try:
# # #             trades = pd.read_csv(trades_path, index_col=0, parse_dates=['entry_time', 'exit_time'])
            
# # #             # Calculate trade metrics
# # #             trade_metrics = calculate_trade_metrics(trades)
# # #             for name, value in trade_metrics.items():
# # #                 performance.add_metric(name, value)
            
# # #         except Exception as e:
# # #             logger.error(f"Error calculating trade metrics: {e}")
    
# # #     # Add timestamp
# # #     performance.add_metric('report_generated', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
# # #     # Save the report as CSV
# # #     save_performance_report(performance, results_dir)
    
# # #     return performance


# # # def save_performance_report(performance, results_dir):
# # #     """
# # #     Save performance report to a CSV file.
    
# # #     Args:
# # #         performance (PerformanceMetrics): Performance metrics object
# # #         results_dir (str): Directory to save the report
# # #     """
# # #     try:
# # #         # Convert metrics to DataFrame
# # #         report_df = performance.to_dataframe()
# # #         report_df.index.name = 'Metric'
        
# # #         # Save as CSV
# # #         report_path = os.path.join(results_dir, 'performance_report.csv')
# # #         report_df.to_csv(report_path)
# # #         logger.info(f"Performance report saved to {report_path}")
        
# # #     except Exception as e:
# # #         logger.error(f"Error saving performance report: {e}")


# # # def plot_equity_curve(equity_curve, results_dir):
# # #     """
# # #     Plot the equity curve and save to results directory.
    
# # #     Args:
# # #         equity_curve (pd.DataFrame): DataFrame containing 'equity' column
# # #         results_dir (str): Directory to save the plot
# # #     """
# # #     try:
# # #         plt.figure(figsize=(12, 6))
# # #         plt.plot(equity_curve.index, equity_curve['equity'])
# # #         plt.title('Equity Curve')
# # #         plt.xlabel('Date')
# # #         plt.ylabel('Equity')
# # #         plt.grid(True)
        
# # #         # Save the plot
# # #         plot_path = os.path.join(results_dir, 'equity_curve.png')
# # #         plt.savefig(plot_path)
# # #         plt.close()
        
# # #         logger.info(f"Equity curve plot saved to {plot_path}")
        
# # #     except Exception as e:
# # #         logger.error(f"Error plotting equity curve: {e}")


# # # def plot_drawdown(equity_curve, results_dir):
# # #     """
# # #     Plot the drawdown and save to results directory.
    
# # #     Args:
# # #         equity_curve (pd.DataFrame): DataFrame containing 'drawdown' column
# # #         results_dir (str): Directory to save the plot
# # #     """
# # #     try:
# # #         if 'drawdown' not in equity_curve.columns:
# # #             equity_curve['high_water_mark'] = equity_curve['equity'].cummax()
# # #             equity_curve['drawdown'] = (equity_curve['equity'] / equity_curve['high_water_mark']) - 1
        
# # #         plt.figure(figsize=(12, 6))
# # #         plt.plot(equity_curve.index, equity_curve['drawdown'] * 100)  # Convert to percentage
# # #         plt.title('Drawdown')
# # #         plt.xlabel('Date')
# # #         plt.ylabel('Drawdown (%)')
# # #         plt.grid(True)
# # #         plt.fill_between(equity_curve.index, equity_curve['drawdown'] * 100, 0, alpha=0.3, color='red')
        
# # #         # Save the plot
# # #         plot_path = os.path.join(results_dir, 'drawdown.png')
# # #         plt.savefig(plot_path)
# # #         plt.close()
        
# # #         logger.info(f"Drawdown plot saved to {plot_path}")
        
# # #     except Exception as e:
# # #         logger.error(f"Error plotting drawdown: {e}")


# # # def analyze_results(results_dir):
# # #     """
# # #     Analyze backtest results and generate plots and reports.
    
# # #     Args:
# # #         results_dir (str): Directory containing backtest results
        
# # #     Returns:
# # #         PerformanceMetrics: Object containing performance metrics
# # #     """
# # #     # Load equity curve
# # #     equity_curve_path = os.path.join(results_dir, 'equity_curve.csv')
# # #     if os.path.exists(equity_curve_path):
# # #         try:
# # #             equity_curve = pd.read_csv(equity_curve_path, index_col=0, parse_dates=True)
            
# # #             # Plot equity curve
# # #             plot_equity_curve(equity_curve, results_dir)
            
# # #             # Plot drawdown
# # #             plot_drawdown(equity_curve, results_dir)
            
# # #         except Exception as e:
# # #             logger.error(f"Error analyzing equity curve: {e}")
    
# # #     # Generate performance report
# # #     performance = generate_performance_report(results_dir)
    
# # #     return performance


# # # def print_summary(performance):
# # #     """
# # #     Print a summary of the performance metrics.
    
# # #     Args:
# # #         performance (PerformanceMetrics): Performance metrics object
# # #     """
# # #     metrics = performance.get_all_metrics()
    
# # #     print("\n=== Performance Summary ===")
    
# # #     # Return metrics
# # #     if 'total_return' in metrics:
# # #         print(f"Total Return: {metrics['total_return'] * 100:.2f}%")
# # #     if 'annual_return' in metrics:
# # #         print(f"Annual Return: {metrics['annual_return'] * 100:.2f}%")
# # #     if 'sharpe_ratio' in metrics:
# # #         print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    
# # #     # Drawdown metrics
# # #     if 'max_drawdown' in metrics:
# # #         print(f"Max Drawdown: {metrics['max_drawdown'] * 100:.2f}%")
# # #     if 'max_drawdown_duration' in metrics:
# # #         print(f"Max Drawdown Duration: {metrics['max_drawdown_duration']} days")
    
# # #     # Trade metrics
# # #     if 'total_trades' in metrics:
# # #         print(f"\nTotal Trades: {metrics['total_trades']}")
# # #     if 'winning_trades' in metrics and 'total_trades' in metrics:
# # #         print(f"Win Rate: {metrics['win_rate'] * 100:.2f}% ({metrics['winning_trades']}/{metrics['total_trades']})")
# # #     if 'avg_profit_pct' in metrics:
# # #         print(f"Average Profit: {metrics['avg_profit_pct'] * 100:.2f}%")
# # #     if 'profit_factor' in metrics:
# # #         print(f"Profit Factor: {metrics['profit_factor']:.2f}")
# # #     if 'expectancy' in metrics:
# # #         print(f"Expectancy: {metrics['expectancy'] * 100:.2f}%")
    
# # #     print("=========================")


# import os
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# from typing import Dict, List
# import logging

# # Get logger
# logger = logging.getLogger('backtest_lib')

# class Analysis:
#     """
#     Performance analysis and reporting tools
#     """
    
#     def __init__(self, backtest=None, results: pd.DataFrame = None):
#         """
#         Initialize the analysis module
        
#         Parameters:
#         -----------
#         backtest : Backtest, optional
#             Backtest instance
#         results : pd.DataFrame, optional
#             Backtest results
#         """
#         if backtest is not None:
#             self.backtest = backtest
#             self.results = backtest.results
#             self.strategy_name = backtest.strategy.name
#         elif results is not None:
#             self.backtest = None
#             self.results = results
#             self.strategy_name = "Unknown"
#         else:
#             logger.error("Either backtest or results must be provided")
#             raise ValueError("Either backtest or results must be provided")
    
#     def generate_performance_report(self, directory: str = 'reports', save_fig: bool = True) -> Dict:
#         """
#         Generate performance report
        
#         Parameters:
#         -----------
#         directory : str
#             Directory to save report
#         save_fig : bool
#             Whether to save figures
            
#         Returns:
#         --------
#         Dict
#             Dictionary with report details
#         """
#         # Create directory
#         os.makedirs(directory, exist_ok=True)
        
#         # Create timestamp
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         # Get performance metrics
#         if self.backtest is not None:
#             metrics = self.backtest.get_performance_metrics()
#             trades = self.backtest.get_trades()
#         else:
#             # Calculate metrics from results
#             metrics = self._calculate_metrics()
#             trades = self._extract_trades()
        
#         # Convert metrics to DataFrame for easier analysis
#         metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
#         metrics_df.index.name = 'Metric'
        
#         # Ensure trades is a proper DataFrame if it's empty
#         if len(trades) == 0:
#             trades = pd.DataFrame(columns=['trade_id', 'entry_date', 'exit_date', 'position_type', 'entry_price', 'exit_price', 'pnl'])
        
#         # Create report content
#         report = {
#             'strategy_name': self.strategy_name,
#             'timestamp': timestamp,
#             'metrics': metrics,
#             'metrics_df': metrics_df,
#             'trades': trades,
#             'figures': []
#         }
        
#         # Generate figures
#         if save_fig:
#             report['figures'] = self._generate_figures(directory, timestamp)
        
#         # Save report components to CSV files for easier analysis
#         metrics_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_metrics.csv")
#         trades_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_trades.csv")
        
#         metrics_df.to_csv(metrics_file)
#         trades.to_csv(trades_file, index=False)
        
#         # Save a copy of the results dataframe with key columns
#         if self.results is not None:
#             # Select key columns to save
#             key_columns = ['close', 'portfolio_value', 'position', 'returns', 'cumulative_returns']
#             results_cols = [col for col in key_columns if col in self.results.columns]
            
#             if results_cols:
#                 results_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_results.csv")
#                 self.results[results_cols].to_csv(results_file)
#                 report['results_file'] = results_file
        
#         logger.info(f"Performance report for strategy '{self.strategy_name}' saved to {directory}")
        
#         return report
    
#     def _calculate_metrics(self) -> Dict:
#         """
#         Calculate performance metrics from results
        
#         Returns:
#         --------
#         Dict
#             Dictionary with performance metrics
#         """
#         # Extract returns
#         returns = self.results['returns'].dropna()
        
#         if len(returns) == 0:
#             logger.warning("No returns data available")
#             return {}
        
#         # Calculate metrics
#         initial_capital = self.results['portfolio_value'].iloc[0]
#         total_return = self.results['portfolio_value'].iloc[-1] / initial_capital - 1
        
#         # Annualized return
#         trading_days = len(returns)
#         annual_factor = 252 / trading_days
#         annual_return = (1 + total_return) ** annual_factor - 1
        
#         # Volatility
#         volatility = returns.std() * np.sqrt(252)
        
#         # Sharpe ratio
#         risk_free_rate = 0.02  # Assuming 2% risk-free rate
#         sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
#         # Sortino ratio
#         negative_returns = returns[returns < 0]
#         downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
#         sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
#         # Maximum drawdown
#         cumulative_returns = (1 + returns).cumprod()
#         max_return = cumulative_returns.expanding().max()
#         drawdowns = (cumulative_returns / max_return) - 1
#         max_drawdown = drawdowns.min()
        
#         metrics = {
#             'total_return': total_return,
#             'annual_return': annual_return,
#             'volatility': volatility,
#             'sharpe_ratio': sharpe_ratio,
#             'sortino_ratio': sortino_ratio,
#             'max_drawdown': max_drawdown
#         }
        
#         return metrics
    
#     def _extract_trades(self) -> pd.DataFrame:
#         """
#         Extract trades from results
        
#         Returns:
#         --------
#         pd.DataFrame
#             DataFrame with trade details
#         """
#         # Placeholder for trade extraction from results
#         # This is a simplified version and may not capture all trades accurately
        
#         # Detect position changes
#         position_changes = self.results['position'].diff() != 0
        
#         # Get potential trade entries and exits
#         trades = []
        
#         # Find trade entries
#         for i in range(1, len(self.results)):
#             if self.results['position'].iloc[i] != 0 and self.results['position'].iloc[i-1] == 0:
#                 # Entry
#                 entry_date = self.results.index[i]
#                 entry_price = self.results['close'].iloc[i]
#                 position_type = 'LONG' if self.results['position'].iloc[i] > 0 else 'SHORT'
                
#                 # Find corresponding exit
#                 exit_found = False
#                 for j in range(i+1, len(self.results)):
#                     if self.results['position'].iloc[j] == 0 and self.results['position'].iloc[j-1] != 0:
#                         # Exit
#                         exit_date = self.results.index[j]
#                         exit_price = self.results['close'].iloc[j]
                        
#                         # Calculate PnL
#                         if position_type == 'LONG':
#                             pnl = (exit_price - entry_price) / entry_price
#                         else:
#                             pnl = (entry_price - exit_price) / entry_price
                        
#                         # Add trade
#                         trades.append({
#                             'entry_date': entry_date,
#                             'exit_date': exit_date,
#                             'position_type': position_type,
#                             'entry_price': entry_price,
#                             'exit_price': exit_price,
#                             'pnl': pnl
#                         })
                        
#                         exit_found = True
#                         break
                
#                 # If no exit found, trade is still open
#                 if not exit_found:
#                     trades.append({
#                         'entry_date': entry_date,
#                         'exit_date': None,
#                         'position_type': position_type,
#                         'entry_price': entry_price,
#                         'exit_price': None,
#                         'pnl': None
#                     })
        
#         return pd.DataFrame(trades)
    
#     def _generate_figures(self, directory: str, timestamp: str) -> List[str]:
#         """
#         Generate performance figures
        
#         Parameters:
#         -----------
#         directory : str
#             Directory to save figures
#         timestamp : str
#             Timestamp for file names
                
#         Returns:
#         --------
#         List[str]
#             List of figure file paths
#         """
#         figure_files = []
        
#         # Set Seaborn style
#         try:
#             import seaborn as sns
#             sns.set_style('whitegrid')
#         except ImportError:
#             logger.warning("Seaborn not installed. Using default matplotlib style.")
        
#         plt.figure(figsize=(12, 8))
        
#         # 1. Portfolio value
#         fig1 = plt.figure(figsize=(12, 6))
#         plt.plot(self.results.index, self.results['portfolio_value'])
#         plt.title(f"Portfolio Value - {self.strategy_name}")
#         plt.xlabel('Date')
#         plt.ylabel('Value')
#         plt.grid(True)
#         fig1_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_portfolio_value.png")
#         plt.savefig(fig1_file)
#         plt.close(fig1)
#         figure_files.append(fig1_file)
        
#         # 2. Cumulative returns
#         if 'cumulative_returns' in self.results.columns:
#             fig2 = plt.figure(figsize=(12, 6))
#             plt.plot(self.results.index, self.results['cumulative_returns'])
#             plt.title(f"Cumulative Returns - {self.strategy_name}")
#             plt.xlabel('Date')
#             plt.ylabel('Returns')
#             plt.grid(True)
#             fig2_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_cumulative_returns.png")
#             plt.savefig(fig2_file)
#             plt.close(fig2)
#             figure_files.append(fig2_file)
        
#         # 3. Drawdowns
#         fig3 = plt.figure(figsize=(12, 6))
#         returns = self.results['returns'].dropna()
#         if len(returns) > 0:
#             cumulative_returns = (1 + returns).cumprod()
#             max_return = cumulative_returns.expanding().max()
#             drawdowns = (cumulative_returns / max_return) - 1
#             plt.plot(drawdowns.index, drawdowns)
#             plt.title(f"Drawdowns - {self.strategy_name}")
#             plt.xlabel('Date')
#             plt.ylabel('Drawdown')
#             plt.grid(True)
#             fig3_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_drawdowns.png")
#             plt.savefig(fig3_file)
#             plt.close(fig3)
#             figure_files.append(fig3_file)
        
#         # 4. Position over time
#         if 'position' in self.results.columns:
#             fig4 = plt.figure(figsize=(12, 6))
#             plt.plot(self.results.index, self.results['position'])
#             plt.title(f"Position - {self.strategy_name}")
#             plt.xlabel('Date')
#             plt.ylabel('Position')
#             plt.grid(True)
#             fig4_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_position.png")
#             plt.savefig(fig4_file)
#             plt.close(fig4)
#             figure_files.append(fig4_file)
        
#         # 5. Returns distribution
#         fig5 = plt.figure(figsize=(12, 6))
#         returns = self.results['returns'].dropna()
#         if len(returns) > 0:
#             try:
#                 import seaborn as sns
#                 sns.histplot(returns, kde=True)
#             except ImportError:
#                 plt.hist(returns, bins=50)
#             plt.title(f"Returns Distribution - {self.strategy_name}")
#             plt.xlabel('Returns')
#             plt.ylabel('Frequency')
#             plt.grid(True)
#             fig5_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_returns_dist.png")
#             plt.savefig(fig5_file)
#             plt.close(fig5)
#             figure_files.append(fig5_file)
        
#         return figure_files
    
#     def plot_equity_curve(self) -> None:
#         """
#         Plot equity curve
#         """
#         plt.figure(figsize=(12, 6))
#         plt.plot(self.results.index, self.results['portfolio_value'])
#         plt.title(f"Equity Curve - {self.strategy_name}")
#         plt.xlabel('Date')
#         plt.ylabel('Portfolio Value')
#         plt.grid(True)
#         plt.show()
    
#     def plot_drawdowns(self) -> None:
#         """
#         Plot drawdowns
#         """
#         plt.figure(figsize=(12, 6))
#         returns = self.results['returns'].dropna()
        
#         if len(returns) > 0:
#             cumulative_returns = (1 + returns).cumprod()
#             max_return = cumulative_returns.expanding().max()
#             drawdowns = (cumulative_returns / max_return) - 1
            
#             plt.plot(drawdowns.index, drawdowns)
#             plt.title(f"Drawdowns - {self.strategy_name}")
#             plt.xlabel('Date')
#             plt.ylabel('Drawdown')
#             plt.grid(True)
#             plt.show()
    
#     def plot_returns_distribution(self) -> None:
#         """
#         Plot returns distribution
#         """
#         plt.figure(figsize=(12, 6))
#         returns = self.results['returns'].dropna()
        
#         if len(returns) > 0:
#             sns.histplot(returns, kde=True)
#             plt.title(f"Returns Distribution - {self.strategy_name}")
#             plt.xlabel('Returns')
#             plt.ylabel('Frequency')
#             plt.grid(True)
#             plt.show()
    
#     def plot_trades(self) -> None:
#         """
#         Plot trades on price chart
#         """
#         plt.figure(figsize=(12, 6))
        
#         # Plot price
#         plt.plot(self.results.index, self.results['close'])
        
#         # Get trades
#         if self.backtest is not None:
#             trades = self.backtest.get_trades()
#         else:
#             trades = self._extract_trades()
        
#         # Plot entry and exit points
#         for _, trade in trades.iterrows():
#             if trade['position_type'] == 'LONG':
#                 # Entry
#                 plt.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
                
#                 # Exit
#                 if trade['exit_date'] is not None:
#                     plt.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100)
#             else:
#                 # Entry
#                 plt.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
                
#                 # Exit
#                 if trade['exit_date'] is not None:
#                     plt.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
        
#         plt.title(f"Trades - {self.strategy_name}")
#         plt.xlabel('Date')
#         plt.ylabel('Price')
#         plt.grid(True)
#         plt.show()
    
#     def export_analysis_to_csv(self, directory: str = 'analysis') -> Dict[str, str]:
#         """
#         Export analysis results to CSV files for further analysis
        
#         Parameters:
#         -----------
#         directory : str
#             Directory to save analysis files
            
#         Returns:
#         --------
#         Dict[str, str]
#             Dictionary with file paths
#         """
#         os.makedirs(directory, exist_ok=True)
#         timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
#         # Export results
#         results_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_results.csv")
#         self.results.to_csv(results_file)
        
#         # Export metrics
#         if self.backtest is not None:
#             metrics = self.backtest.get_performance_metrics()
#         else:
#             metrics = self._calculate_metrics()
        
#         metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
#         metrics_df.index.name = 'Metric'
#         metrics_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_metrics.csv")
#         metrics_df.to_csv(metrics_file)
        
#         # Export trades
#         if self.backtest is not None:
#             trades = self.backtest.get_trades()
#         else:
#             trades = self._extract_trades()
        
#         trades_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_trades.csv")
#         trades.to_csv(trades_file, index=False)
        
#         logger.info(f"Analysis data for strategy '{self.strategy_name}' exported to {directory}")
        
#         return {
#             'results': results_file,
#             'metrics': metrics_file,
#             'trades': trades_file
#         }


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import logging

# Get logger
logger = logging.getLogger('backtest_lib')

class Analysis:
    """
    Performance analysis and reporting tools
    """
    
    def __init__(self, backtest=None, results: pd.DataFrame = None):
        """
        Initialize the analysis module
        
        Parameters:
        -----------
        backtest : Backtest, optional
            Backtest instance
        results : pd.DataFrame, optional
            Backtest results
        """
        if backtest is not None:
            self.backtest = backtest
            self.results = backtest.results
            self.strategy_name = backtest.strategy.name
        elif results is not None:
            self.backtest = None
            self.results = results
            self.strategy_name = "Unknown"
        else:
            logger.error("Either backtest or results must be provided")
            raise ValueError("Either backtest or results must be provided")
        
        # Replace inf values with NaN in results
        if self.results is not None:
            self.results = self.results.replace([np.inf, -np.inf], np.nan)
    
    def generate_performance_report(self, directory: str = 'reports', save_fig: bool = True) -> Dict:
        """
        Generate performance report
        
        Parameters:
        -----------
        directory : str
            Directory to save report
        save_fig : bool
            Whether to save figures
            
        Returns:
        --------
        Dict
            Dictionary with report details
        """
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Get performance metrics
        if self.backtest is not None:
            metrics = self.backtest.get_performance_metrics()
            trades = self.backtest.get_trades()
        else:
            # Calculate metrics from results
            metrics = self._calculate_metrics()
            trades = self._extract_trades()
        
        # Convert metrics to DataFrame for easier analysis
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df.index.name = 'Metric'
        
        # Ensure trades is a proper DataFrame if it's empty
        if len(trades) == 0:
            trades = pd.DataFrame(columns=['trade_id', 'entry_date', 'exit_date', 'position_type', 'entry_price', 'exit_price', 'pnl'])
        
        # Create report content
        report = {
            'strategy_name': self.strategy_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'metrics_df': metrics_df,
            'trades': trades,
            'figures': []
        }
        
        # Generate figures
        if save_fig:
            report['figures'] = self._generate_figures(directory, timestamp)
        
        # Save report components to CSV files for easier analysis
        # Check if files exist to avoid duplicates
        metrics_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_metrics.csv")
        metrics_df.to_csv(metrics_file)
        
        # Only save trades if it's not already in the directory
        trades_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_trades.csv")
        trades.to_csv(trades_file, index=False)
        
        # Save a copy of the results dataframe with key columns if not already saved
        if self.results is not None:
            # Select key columns to save
            key_columns = ['close', 'portfolio_value', 'position', 'returns', 'cumulative_returns']
            results_cols = [col for col in key_columns if col in self.results.columns]
            
            if results_cols:
                results_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_results.csv")
                self.results[results_cols].to_csv(results_file)
                report['results_file'] = results_file
        
        logger.info(f"Performance report for strategy '{self.strategy_name}' saved to {directory}")
        
        return report
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from results
        
        Returns:
        --------
        Dict
            Dictionary with performance metrics
        """
        # Extract returns and replace inf values
        returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) == 0:
            logger.warning("No returns data available")
            return {}
        
        # Calculate metrics
        try:
            # Filter out NaN and inf values
            portfolio_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(portfolio_values) < 2:
                logger.warning("Not enough valid portfolio values to calculate returns")
                return {}
            
            initial_capital = portfolio_values.iloc[0]
            total_return = portfolio_values.iloc[-1] / initial_capital - 1
            
            # Annualized return
            trading_days = len(returns)
            annual_factor = 252 / trading_days
            annual_return = (1 + total_return) ** annual_factor - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe ratio
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 0 else 0
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns.fillna(0)).cumprod()
            max_return = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / max_return) - 1
            max_drawdown = drawdowns.min()
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def _extract_trades(self) -> pd.DataFrame:
            """
            Extract trades from results
            
            Returns:
            --------
            pd.DataFrame
                DataFrame with trade details
            """
            # Placeholder for trade extraction from results
            # This is a simplified version and may not capture all trades accurately
            
            # Detect position changes
            position_changes = self.results['position'].diff() != 0
            
            # Get potential trade entries and exits
            trades = []
            
            # Find trade entries
            for i in range(1, len(self.results)):
                if self.results['position'].iloc[i] != 0 and self.results['position'].iloc[i-1] == 0:
                    # Entry
                    entry_date = self.results.index[i]
                    entry_price = self.results['close'].iloc[i]
                    position_type = 'LONG' if self.results['position'].iloc[i] > 0 else 'SHORT'
                    
                    # Find corresponding exit
                    exit_found = False
                    for j in range(i+1, len(self.results)):
                        if self.results['position'].iloc[j] == 0 and self.results['position'].iloc[j-1] != 0:
                            # Exit
                            exit_date = self.results.index[j]
                            exit_price = self.results['close'].iloc[j]
                            
                            # Calculate PnL
                            if position_type == 'LONG':
                                pnl = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
                            else:
                                pnl = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
                            
                            # Add trade
                            trades.append({
                                'entry_date': entry_date,
                                'exit_date': exit_date,
                                'position_type': position_type,
                                'entry_price': entry_price,
                                'exit_price': exit_price,
                                'pnl': pnl
                            })
                            
                            exit_found = True
                            break
                    
                    # If no exit found, trade is still open
                    if not exit_found:
                        trades.append({
                            'entry_date': entry_date,
                            'exit_date': None,
                            'position_type': position_type,
                            'entry_price': entry_price,
                            'exit_price': None,
                            'pnl': None
                        })
            
            return pd.DataFrame(trades)
    
    def _generate_figures(self, directory: str, timestamp: str) -> List[str]:
        """
        Generate performance figures
        
        Parameters:
        -----------
        directory : str
            Directory to save figures
        timestamp : str
            Timestamp for file names
            
        Returns:
        --------
        List[str]
            List of figure file paths
        """
        figure_files = []
        
        # Set Seaborn style if available
        try:
            import seaborn as sns
            sns.set_style('whitegrid')
        except ImportError:
            logger.warning("Seaborn not installed. Using default matplotlib style.")
        
        try:
            # 1. Portfolio value
            fig1 = plt.figure(figsize=(12, 6))
            portfolio_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan)
            plt.plot(self.results.index, portfolio_values)
            plt.title(f"Portfolio Value - {self.strategy_name}")
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.grid(True)
            fig1_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_portfolio_value.png")
            plt.savefig(fig1_file)
            plt.close(fig1)
            figure_files.append(fig1_file)
            
            # 2. Cumulative returns
            if 'cumulative_returns' in self.results.columns:
                fig2 = plt.figure(figsize=(12, 6))
                cum_returns = self.results['cumulative_returns'].replace([np.inf, -np.inf], np.nan)
                plt.plot(self.results.index, cum_returns)
                plt.title(f"Cumulative Returns - {self.strategy_name}")
                plt.xlabel('Date')
                plt.ylabel('Returns')
                plt.grid(True)
                fig2_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_cumulative_returns.png")
                plt.savefig(fig2_file)
                plt.close(fig2)
                figure_files.append(fig2_file)
            
            # 3. Drawdowns
            fig3 = plt.figure(figsize=(12, 6))
            returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) > 0:
                cumulative_returns = (1 + returns.fillna(0)).cumprod()
                max_return = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns / max_return) - 1
                plt.plot(drawdowns.index, drawdowns)
                plt.title(f"Drawdowns - {self.strategy_name}")
                plt.xlabel('Date')
                plt.ylabel('Drawdown')
                plt.grid(True)
                fig3_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_drawdowns.png")
                plt.savefig(fig3_file)
                plt.close(fig3)
                figure_files.append(fig3_file)
            
            # 4. Position over time
            if 'position' in self.results.columns:
                fig4 = plt.figure(figsize=(12, 6))
                plt.plot(self.results.index, self.results['position'])
                plt.title(f"Position - {self.strategy_name}")
                plt.xlabel('Date')
                plt.ylabel('Position')
                plt.grid(True)
                fig4_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_position.png")
                plt.savefig(fig4_file)
                plt.close(fig4)
                figure_files.append(fig4_file)
            
            # 5. Returns distribution
            fig5 = plt.figure(figsize=(12, 6))
            returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(returns) > 0:
                try:
                    import seaborn as sns
                    sns.histplot(returns, kde=True)
                except (ImportError, ValueError):
                    # In case of error, use simple histogram
                    plt.hist(returns, bins=50)
                plt.title(f"Returns Distribution - {self.strategy_name}")
                plt.xlabel('Returns')
                plt.ylabel('Frequency')
                plt.grid(True)
                fig5_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_returns_dist.png")
                plt.savefig(fig5_file)
                plt.close(fig5)
                figure_files.append(fig5_file)
        
        except Exception as e:
            logger.error(f"Error generating figures: {e}")
        
        return figure_files
    
    def plot_equity_curve(self) -> None:
        """
        Plot equity curve
        """
        try:
            plt.figure(figsize=(12, 6))
            portfolio_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan)
            plt.plot(self.results.index, portfolio_values)
            plt.title(f"Equity Curve - {self.strategy_name}")
            plt.xlabel('Date')
            plt.ylabel('Portfolio Value')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting equity curve: {e}")
    
    def plot_drawdowns(self) -> None:
        """
        Plot drawdowns
        """
        try:
            plt.figure(figsize=(12, 6))
            returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) > 0:
                cumulative_returns = (1 + returns.fillna(0)).cumprod()
                max_return = cumulative_returns.expanding().max()
                drawdowns = (cumulative_returns / max_return) - 1
                
                plt.plot(drawdowns.index, drawdowns)
                plt.title(f"Drawdowns - {self.strategy_name}")
                plt.xlabel('Date')
                plt.ylabel('Drawdown')
                plt.grid(True)
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting drawdowns: {e}")
    
    def plot_returns_distribution(self) -> None:
        """
        Plot returns distribution
        """
        try:
            plt.figure(figsize=(12, 6))
            returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(returns) > 0:
                try:
                    import seaborn as sns
                    sns.histplot(returns, kde=True)
                except (ImportError, ValueError):
                    # In case of error, use simple histogram
                    plt.hist(returns, bins=50)
                plt.title(f"Returns Distribution - {self.strategy_name}")
                plt.xlabel('Returns')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()
        except Exception as e:
            logger.error(f"Error plotting returns distribution: {e}")
    
    def plot_trades(self) -> None:
        """
        Plot trades on price chart
        """
        try:
            plt.figure(figsize=(12, 6))
            
            # Plot price
            plt.plot(self.results.index, self.results['close'])
            
            # Get trades
            if self.backtest is not None:
                trades = self.backtest.get_trades()
            else:
                trades = self._extract_trades()
            
            # Plot entry and exit points
            for _, trade in trades.iterrows():
                if trade['position_type'] == 'LONG':
                    # Entry
                    plt.scatter(trade['entry_date'], trade['entry_price'], marker='^', color='green', s=100)
                    
                    # Exit
                    if trade['exit_date'] is not None:
                        plt.scatter(trade['exit_date'], trade['exit_price'], marker='v', color='red', s=100)
                else:
                    # Entry
                    plt.scatter(trade['entry_date'], trade['entry_price'], marker='v', color='red', s=100)
                    
                    # Exit
                    if trade['exit_date'] is not None:
                        plt.scatter(trade['exit_date'], trade['exit_price'], marker='^', color='green', s=100)
            
            plt.title(f"Trades - {self.strategy_name}")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.show()
        except Exception as e:
            logger.error(f"Error plotting trades: {e}")
    
    def export_analysis_to_csv(self, directory: str = 'analysis') -> Dict[str, str]:
        """
        Export analysis results to CSV files for further analysis
        
        Parameters:
        -----------
        directory : str
            Directory to save analysis files
            
        Returns:
        --------
        Dict[str, str]
            Dictionary with file paths
        """
        os.makedirs(directory, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Export results
        results_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_results.csv")
        self.results.to_csv(results_file)
        
        # Export metrics
        if self.backtest is not None:
            metrics = self.backtest.get_performance_metrics()
        else:
            metrics = self._calculate_metrics()
        
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_df.index.name = 'Metric'
        metrics_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_metrics.csv")
        metrics_df.to_csv(metrics_file)
        
        # Export trades
        if self.backtest is not None:
            trades = self.backtest.get_trades()
        else:
            trades = self._extract_trades()
        
        trades_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_trades.csv")
        trades.to_csv(trades_file, index=False)
        
        logger.info(f"Analysis data for strategy '{self.strategy_name}' exported to {directory}")
        
        return {
            'results': results_file,
            'metrics': metrics_file,
            'trades': trades_file
        }