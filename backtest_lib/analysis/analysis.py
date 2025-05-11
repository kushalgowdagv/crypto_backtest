import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List
import logging

# Get logger
logger = logging.getLogger('backtest_lib')


# Custom JSON encoder to handle pandas Timestamp objects
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        return super().default(obj)


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
        
        # Convert Timestamp objects to strings in trades dataframe
        trades_dict = []
        for _, trade in trades.iterrows():
            trade_dict = trade.to_dict()
            # Convert all timestamp objects to strings
            for key, value in trade_dict.items():
                if isinstance(value, pd.Timestamp):
                    trade_dict[key] = value.strftime('%Y-%m-%d %H:%M:%S')
            trades_dict.append(trade_dict)
        
        # Create report content
        report = {
            'strategy_name': self.strategy_name,
            'timestamp': timestamp,
            'metrics': metrics,
            'trades': trades_dict,
            'figures': []
        }
        
        # Generate figures
        if save_fig:
            report['figures'] = self._generate_figures(directory, timestamp)
        
        # Save report to JSON using custom encoder
        report_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_report.json")
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=4, cls=CustomJSONEncoder)
        
        logger.info(f"Performance report for strategy '{self.strategy_name}' saved to {report_file}")
        
        return report
    
    def _calculate_metrics(self) -> Dict:
        """
        Calculate performance metrics from results
        
        Returns:
        --------
        Dict
            Dictionary with performance metrics
        """
        # Extract returns
        returns = self.results['returns'].dropna()
        
        if len(returns) == 0:
            logger.warning("No returns data available")
            return {}
        
        # Calculate metrics
        initial_capital = self.results['portfolio_value'].iloc[0]
        total_return = self.results['portfolio_value'].iloc[-1] / initial_capital - 1
        
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
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
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
                            pnl = (exit_price - entry_price) / entry_price
                        else:
                            pnl = (entry_price - exit_price) / entry_price
                        
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
        
        # Set Seaborn style
        sns.set_style('whitegrid')
        plt.figure(figsize=(12, 8))
        
        # 1. Portfolio value
        fig1 = plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['portfolio_value'])
        plt.title(f"Portfolio Value - {self.strategy_name}")
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.grid(True)
        fig1_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_portfolio_value.png")
        plt.savefig(fig1_file)
        plt.close(fig1)
        figure_files.append(fig1_file)
        
        # 2. Cumulative returns
        fig2 = plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['cumulative_returns'])
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
        returns = self.results['returns'].dropna()
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod()
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
        returns = self.results['returns'].dropna()
        if len(returns) > 0:
            sns.histplot(returns, kde=True)
            plt.title(f"Returns Distribution - {self.strategy_name}")
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(True)
            fig5_file = os.path.join(directory, f"{self.strategy_name}_{timestamp}_returns_dist.png")
            plt.savefig(fig5_file)
            plt.close(fig5)
            figure_files.append(fig5_file)
        
        return figure_files
    
    def plot_equity_curve(self) -> None:
        """
        Plot equity curve
        """
        plt.figure(figsize=(12, 6))
        plt.plot(self.results.index, self.results['portfolio_value'])
        plt.title(f"Equity Curve - {self.strategy_name}")
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value')
        plt.grid(True)
        plt.show()
    
    def plot_drawdowns(self) -> None:
        """
        Plot drawdowns
        """
        plt.figure(figsize=(12, 6))
        returns = self.results['returns'].dropna()
        
        if len(returns) > 0:
            cumulative_returns = (1 + returns).cumprod()
            max_return = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / max_return) - 1
            
            plt.plot(drawdowns.index, drawdowns)
            plt.title(f"Drawdowns - {self.strategy_name}")
            plt.xlabel('Date')
            plt.ylabel('Drawdown')
            plt.grid(True)
            plt.show()
    
    def plot_returns_distribution(self) -> None:
        """
        Plot returns distribution
        """
        plt.figure(figsize=(12, 6))
        returns = self.results['returns'].dropna()
        
        if len(returns) > 0:
            sns.histplot(returns, kde=True)
            plt.title(f"Returns Distribution - {self.strategy_name}")
            plt.xlabel('Returns')
            plt.ylabel('Frequency')
            plt.grid(True)
            plt.show()
    
    def plot_trades(self) -> None:
        """
        Plot trades on price chart
        """
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

