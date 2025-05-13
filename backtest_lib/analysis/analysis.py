

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
    

    # def _calculate_metrics(self) -> Dict:
    #     """
    #     Calculate performance metrics from results with improved numerical stability
    #     """
    #     # Extract returns and replace inf values
    #     returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        
    #     if len(returns) == 0:
    #         logger.warning("No returns data available")
    #         return {}
        
    #     try:
    #         # Filter out NaN and inf values from portfolio values
    #         portfolio_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan).dropna()
            
    #         if len(portfolio_values) < 2:
    #             logger.warning("Not enough valid portfolio values to calculate returns")
    #             return {}
            
    #         # Calculate total return based on first and last valid portfolio values
    #         initial_capital = portfolio_values.iloc[0]
    #         final_capital = portfolio_values.iloc[-1]
    #         total_return = (final_capital / initial_capital) - 1
            
    #         # Annualized return calculation with safeguards
    #         trading_days = len(returns)
    #         if trading_days < 2:
    #             annual_return = total_return
    #         else:
    #             # Calculate trading days per year based on data frequency
    #             days_diff = (self.results.index[-1] - self.results.index[0]).total_seconds() / 86400
    #             if days_diff > 0:
    #                 annual_factor = min(252 / (trading_days / (days_diff / 365.25)), 252)
    #             else:
    #                 annual_factor = 1
                
    #             # Prevent extreme annualization
    #             annual_return = ((1 + total_return) ** (annual_factor / trading_days)) - 1
            
    #         # Volatility - with minimum value to prevent division by zero
    #         volatility = max(returns.std() * np.sqrt(252), 1e-6)
            
    #         # Sharpe ratio with safeguards
    #         risk_free_rate = 0.02  # Assuming 2% risk-free rate
    #         sharpe_ratio = (annual_return - risk_free_rate) / volatility
            
    #         # Sortino ratio with safeguards
    #         negative_returns = returns[returns < 0]
    #         if len(negative_returns) > 0:
    #             downside_deviation = negative_returns.std() * np.sqrt(252)
    #             # Prevent division by zero
    #             downside_deviation = max(downside_deviation, 1e-6)
    #         else:
    #             downside_deviation = 1e-6
            
    #         sortino_ratio = (annual_return - risk_free_rate) / downside_deviation
            
    #         # Maximum drawdown calculation with proper handling of NaN values
    #         returns_no_nan = returns.fillna(0)
    #         # Limit extreme returns to prevent overflow
    #         returns_no_nan = returns_no_nan.clip(-0.5, 0.5)
    #         cumulative_returns = (1 + returns_no_nan).cumprod()
    #         running_max = cumulative_returns.expanding().max()
    #         drawdowns = (cumulative_returns / running_max) - 1
    #         max_drawdown = drawdowns.min()
            
    #         # Calculate date ranges for context
    #         start_date = self.results.index[0].strftime('%Y-%m-%d')
    #         end_date = self.results.index[-1].strftime('%Y-%m-%d')
    #         period_days = (self.results.index[-1] - self.results.index[0]).days
            
    #         metrics = {
    #             'start_date': start_date,
    #             'end_date': end_date,
    #             'period_days': period_days,
    #             'start_value': initial_capital,
    #             'end_value': final_capital,
    #             'total_return': total_return,
    #             'annual_return': annual_return,
    #             'volatility': volatility,
    #             'sharpe_ratio': sharpe_ratio,
    #             'sortino_ratio': sortino_ratio,
    #             'max_drawdown': max_drawdown,
    #         }
            
    #         return metrics
            
    #     except Exception as e:
    #         logger.error(f"Error calculating metrics: {e}")
    #         return {
    #             'total_return': 0,
    #             'annual_return': 0,
    #             'volatility': 0,
    #             'sharpe_ratio': 0,
    #             'sortino_ratio': 0,
    #             'max_drawdown': 0,
    #         }

    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics with improved accuracy for cryptocurrency markets
        """
        try:
            # Filter out NaN and inf values from portfolio values and returns
            portfolio_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan).dropna()
            
            if len(portfolio_values) < 2:
                logger.warning("Not enough valid portfolio values to calculate metrics")
                return {}
            
            # Extract clean returns
            returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
            log_returns = self.results['log_returns'].replace([np.inf, -np.inf], np.nan).dropna() if 'log_returns' in self.results.columns else np.log(1 + returns)
            
            # Calculate date ranges for context
            start_date = self.results.index[0].strftime('%Y-%m-%d')
            end_date = self.results.index[-1].strftime('%Y-%m-%d')
            
            # Calculate time period - account for crypto trading 365 days a year
            period_days = (self.results.index[-1] - self.results.index[0]).days
            if period_days == 0:
                period_days = (self.results.index[-1] - self.results.index[0]).total_seconds() / 86400
            
            # Calculate initial and final capital
            initial_capital = portfolio_values.iloc[0]
            final_capital = portfolio_values.iloc[-1]
            
            # Calculate total return
            total_return = (final_capital / initial_capital) - 1
            
            # Annualized return calculation - properly annualized for crypto (365 days)
            # Use the standard formula: (1 + total_return)^(365/days) - 1
            if period_days > 0:
                annual_return = (1 + total_return) ** (365 / period_days) - 1
            else:
                annual_return = 0
            
            # Calculate trading frequency
            minutes_per_candle = 10  # Assuming 10-minute candlesticks
            annual_factor = (365 * 24 * 60) / minutes_per_candle  # Number of candles per year
            
            # Volatility calculation using log returns
            if len(log_returns) > 1:
                volatility = log_returns.std() * np.sqrt(annual_factor)
            else:
                volatility = 0
            
            # Sharpe ratio with industry standard risk-free rate (adjusted for crypto)
            risk_free_rate = 0.02  # 2% annual risk-free rate
            daily_risk_free = (1 + risk_free_rate) ** (1/365) - 1
            excess_return = annual_return - risk_free_rate
            sharpe_ratio = excess_return / volatility if volatility > 0 else 0
            
            # Sortino ratio - only downside volatility
            negative_returns = log_returns[log_returns < 0]
            if len(negative_returns) > 0 and negative_returns.std() > 0:
                downside_deviation = negative_returns.std() * np.sqrt(annual_factor)
                sortino_ratio = excess_return / downside_deviation
            else:
                sortino_ratio = 0
            
            # Maximum drawdown calculation
            if 'drawdown' in self.results.columns:
                max_drawdown = self.results['drawdown'].min()
            else:
                # Calculate drawdown series
                peak = portfolio_values.expanding().max()
                drawdown = (portfolio_values / peak) - 1
                max_drawdown = drawdown.min()
            
            # Calculate Calmar ratio
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown < 0 else float('inf')
            
            # Extract trade statistics
            if self.backtest is not None:
                trades = self.backtest.get_trades()
            else:
                trades = self._extract_trades()
            
            # Calculate trade metrics
            if len(trades) > 0:
                # Filter out trades without PnL
                trades_with_pnl = trades.dropna(subset=['pnl'])
                
                # Number of trades
                total_trades = len(trades)
                total_closed_trades = len(trades_with_pnl)
                total_open_trades = total_trades - total_closed_trades
                
                # Win/loss stats
                if len(trades_with_pnl) > 0:
                    winning_trades = len(trades_with_pnl[trades_with_pnl['pnl'] > 0])
                    losing_trades = len(trades_with_pnl[trades_with_pnl['pnl'] <= 0])
                    win_rate = winning_trades / len(trades_with_pnl) if len(trades_with_pnl) > 0 else 0
                    
                    # Trade performance
                    avg_trade = trades_with_pnl['pnl'].mean() if len(trades_with_pnl) > 0 else 0
                    avg_winning_trade = trades_with_pnl[trades_with_pnl['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
                    avg_losing_trade = trades_with_pnl[trades_with_pnl['pnl'] <= 0]['pnl'].mean() if losing_trades > 0 else 0
                    
                    # Best and worst trades
                    best_trade = trades_with_pnl['pnl'].max() if len(trades_with_pnl) > 0 else 0
                    worst_trade = trades_with_pnl['pnl'].min() if len(trades_with_pnl) > 0 else 0
                    
                    # Profit factor
                    gross_profit = trades_with_pnl[trades_with_pnl['pnl'] > 0]['pnl'].sum() if winning_trades > 0 else 0
                    gross_loss = abs(trades_with_pnl[trades_with_pnl['pnl'] <= 0]['pnl'].sum()) if losing_trades > 0 else 0
                    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                    
                    # Expectancy
                    expectancy = avg_trade
                else:
                    winning_trades = 0
                    losing_trades = 0
                    win_rate = 0
                    avg_trade = 0
                    avg_winning_trade = 0
                    avg_losing_trade = 0
                    best_trade = 0
                    worst_trade = 0
                    profit_factor = 0
                    expectancy = 0
            else:
                total_trades = 0
                total_closed_trades = 0
                total_open_trades = 0
                winning_trades = 0
                losing_trades = 0
                win_rate = 0
                avg_trade = 0
                avg_winning_trade = 0
                avg_losing_trade = 0
                best_trade = 0
                worst_trade = 0
                profit_factor = 0
                expectancy = 0
            
            # Calculate commission impact
            if 'commission_paid' in self.results.columns:
                total_fees_paid = self.results['commission_paid'].iloc[-1]
            else:
                total_fees_paid = 0
            
            # Compile all metrics
            metrics = {
                'start_date': start_date,
                'end_date': end_date,
                'period_days': period_days,
                'start_value': initial_capital,
                'end_value': final_capital,
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'total_trades': total_trades,
                'total_closed_trades': total_closed_trades,
                'total_open_trades': total_open_trades,
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': win_rate,
                'avg_trade': avg_trade,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'profit_factor': profit_factor,
                'expectancy': expectancy,
                'commission': self.backtest.commission if self.backtest is not None else 0,
                'total_fees_paid': total_fees_paid
            }
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            traceback.print_exc()
            return {
                'total_return': 0,
                'annual_return': 0,
                'volatility': 0,
                'sharpe_ratio': 0,
                'sortino_ratio': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'num_trades': 0
            }
    
    # def _extract_trades(self) -> pd.DataFrame:
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
    #                             pnl = (exit_price - entry_price) / entry_price if entry_price != 0 else 0
    #                         else:
    #                             pnl = (entry_price - exit_price) / entry_price if entry_price != 0 else 0
                            
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
    def _extract_trades(self) -> pd.DataFrame:
        """
        Extract trades from results with improved accuracy
        """
        # Placeholder for trade extraction from results
        # This is a more robust version
        
        trades = []
        
        # Find position changes
        position = self.results['position']
        entry_idx = []
        exit_idx = []
        
        # Identify entry points (position changes from 0 to non-zero)
        for i in range(1, len(position)):
            if position.iloc[i-1] == 0 and position.iloc[i] != 0:
                entry_idx.append(i)
            elif position.iloc[i-1] != 0 and (position.iloc[i] == 0 or position.iloc[i] * position.iloc[i-1] < 0):
                exit_idx.append(i)
        
        # Add final exit if position is still open at the end
        if len(entry_idx) > len(exit_idx) and position.iloc[-1] != 0:
            exit_idx.append(len(position) - 1)
        
        # Match entries with exits
        for i in range(len(entry_idx)):
            if i < len(exit_idx):
                # Entry details
                entry_date = self.results.index[entry_idx[i]]
                entry_price = self.results['close'].iloc[entry_idx[i]]
                position_type = 'LONG' if position.iloc[entry_idx[i]] > 0 else 'SHORT'
                
                # Exit details
                exit_date = self.results.index[exit_idx[i]]
                exit_price = self.results['close'].iloc[exit_idx[i]]
                
                # Calculate duration
                duration = exit_date - entry_date
                
                # Calculate PnL
                if position_type == 'LONG':
                    pnl = (exit_price - entry_price) / entry_price
                else:  # SHORT
                    pnl = (entry_price - exit_price) / entry_price
                
                # Add trade
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'duration': duration
                })
            else:
                # Open position without exit
                entry_date = self.results.index[entry_idx[i]]
                entry_price = self.results['close'].iloc[entry_idx[i]]
                position_type = 'LONG' if position.iloc[entry_idx[i]] > 0 else 'SHORT'
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': None,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': None,
                    'pnl': None,
                    'duration': None
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