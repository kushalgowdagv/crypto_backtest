

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
from typing import Dict
from backtest_lib.strategies.strategy import Strategy

# Get logger
logger = logging.getLogger('backtest_lib')


class Backtest:
    """
    Backtesting engine for evaluating trading strategies
    """
    
    def __init__(self, data: pd.DataFrame, strategy: Strategy, initial_capital: float = 10000.0, commission: float = 0.001):
        """
        Initialize the backtesting engine
        
        Parameters:
        -----------
        data : pd.DataFrame
            Market data with OHLCV
        strategy : Strategy
            Trading strategy
        initial_capital : float
            Initial capital
        commission : float
            Commission rate
        """
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.results = None
        self.position_size = 1.0  # Full position size by default
        self.leverage = 1.0  # No leverage by default
        self.stop_loss = None  # No stop-loss by default
        self.take_profit = None  # No take-profit by default
        
        logger.info(f"Backtest initialized for strategy '{strategy.name}' with {len(data)} data points")
    
    def set_position_sizing(self, position_size: float) -> None:
        """
        Set position sizing
        
        Parameters:
        -----------
        position_size : float
            Position size as a fraction of capital (0-1)
        """
        if position_size <= 0 or position_size > 1:
            logger.warning(f"Invalid position size: {position_size}. Using default value of 1.0")
            self.position_size = 1.0
        else:
            self.position_size = position_size
            logger.info(f"Position size set to {position_size}")
    
    def set_leverage(self, leverage: float) -> None:
        """
        Set leverage
        
        Parameters:
        -----------
        leverage : float
            Leverage multiplier
        """
        if leverage < 1:
            logger.warning(f"Invalid leverage: {leverage}. Using default value of 1.0")
            self.leverage = 1.0
        else:
            self.leverage = leverage
            logger.info(f"Leverage set to {leverage}")
    
    def set_stop_loss(self, stop_loss: float) -> None:
        """
        Set stop-loss
        
        Parameters:
        -----------
        stop_loss : float
            Stop-loss as a percentage
        """
        if stop_loss is not None and stop_loss <= 0:
            logger.warning(f"Invalid stop-loss: {stop_loss}. Stop-loss disabled")
            self.stop_loss = None
        else:
            self.stop_loss = stop_loss
            logger.info(f"Stop-loss set to {stop_loss}%")
    
    def set_take_profit(self, take_profit: float) -> None:
        """
        Set take-profit
        
        Parameters:
        -----------
        take_profit : float
            Take-profit as a percentage
        """
        if take_profit is not None and take_profit <= 0:
            logger.warning(f"Invalid take-profit: {take_profit}. Take-profit disabled")
            self.take_profit = None
        else:
            self.take_profit = take_profit
            logger.info(f"Take-profit set to {take_profit}%")
    


    # def run(self) -> pd.DataFrame:
    #     """
    #     Run backtest with improved error handling and numerical stability
    #     """
    #     # Generate signals
    #     signals = self.strategy.generate_signals(self.data)
        
    #     # Create a copy for results
    #     results = signals.copy()
        
    #     # Initialize portfolio metrics with proper data types
    #     results['holdings'] = 0.0
    #     results['cash'] = self.initial_capital
    #     results['portfolio_value'] = self.initial_capital
    #     results['entry_price'] = 0.0
    #     results['stop_loss_price'] = 0.0
    #     results['take_profit_price'] = 0.0
    #     results['trade_id'] = 0
    #     results['commission_paid'] = 0.0  # Track commissions separately
        
    #     # Portfolio tracking variables
    #     trade_id = 0
        
    #     # Process each row with error handling
    #     for i in range(1, len(results)):
    #         try:
    #             # Get current and previous values
    #             prev_position = results.iloc[i-1]['position']
    #             curr_position = results.iloc[i]['position']
    #             prev_cash = results.iloc[i-1]['cash']
    #             prev_holdings = results.iloc[i-1]['holdings']
    #             price = results.iloc[i]['close']
                
    #             # Safety check for price
    #             if np.isnan(price) or price <= 0:
    #                 logger.warning(f"Invalid price at index {i}: {price}. Using previous price.")
    #                 price = results.iloc[i-1]['close']
    #                 if np.isnan(price) or price <= 0:
    #                     price = 1.0  # Fallback to a safe value
                
    #             # Check for position changes
    #             if curr_position != prev_position:
    #                 # Close previous position if any
    #                 if prev_position != 0:
    #                     # Calculate PnL with safeguards
    #                     position_value = prev_holdings * price
    #                     commission_amount = position_value * self.commission
                        
    #                     # Ensure commission doesn't exceed position value
    #                     commission_amount = min(commission_amount, position_value * 0.99)
    #                     position_value = position_value - commission_amount
                        
    #                     # Update cash and tracking
    #                     results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value
    #                     results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                     results.iloc[i, results.columns.get_loc('commission_paid')] = \
    #                         results.iloc[i-1]['commission_paid'] + commission_amount
                    
    #                 # Open new position if any
    #                 if curr_position != 0:
    #                     # Calculate position size safely
    #                     trade_capital = max(results.iloc[i]['cash'] * self.position_size, 0)
    #                     position_size = (trade_capital * self.leverage) / max(price, 0.001)
                        
    #                     # Account for transaction costs
    #                     position_cost = position_size * price
    #                     commission_cost = position_cost * self.commission
                        
    #                     # Update cash, holdings, and entry parameters
    #                     if curr_position > 0:  # Long position
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash - position_cost - commission_cost
    #                         results.iloc[i, results.columns.get_loc('holdings')] = position_size
    #                         results.iloc[i, results.columns.get_loc('commission_paid')] = \
    #                             results.iloc[i-1]['commission_paid'] + commission_cost
                            
    #                         # Set entry price and risk management
    #                         entry_price = price
                            
    #                         if self.stop_loss:
    #                             stop_loss_price = entry_price * (1 - self.stop_loss / 100)
    #                         else:
    #                             stop_loss_price = 0
                            
    #                         if self.take_profit:
    #                             take_profit_price = entry_price * (1 + self.take_profit / 100)
    #                         else:
    #                             take_profit_price = float('inf')
                            
    #                     else:  # Short position
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_cost - commission_cost
    #                         results.iloc[i, results.columns.get_loc('holdings')] = -position_size
    #                         results.iloc[i, results.columns.get_loc('commission_paid')] = \
    #                             results.iloc[i-1]['commission_paid'] + commission_cost
                            
    #                         # Set entry price and risk management
    #                         entry_price = price
                            
    #                         if self.stop_loss:
    #                             stop_loss_price = entry_price * (1 + self.stop_loss / 100)
    #                         else:
    #                             stop_loss_price = float('inf')
                            
    #                         if self.take_profit:
    #                             take_profit_price = entry_price * (1 - self.take_profit / 100)
    #                         else:
    #                             take_profit_price = 0
                        
    #                     # Increment trade ID
    #                     trade_id += 1
                        
    #                     # Update trade tracking variables
    #                     results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
    #                     results.iloc[i, results.columns.get_loc('stop_loss_price')] = stop_loss_price
    #                     results.iloc[i, results.columns.get_loc('take_profit_price')] = take_profit_price
    #                     results.iloc[i, results.columns.get_loc('trade_id')] = trade_id
                    
    #             else:
    #                 # Check for stop-loss and take-profit
    #                 if prev_position > 0:  # Long position
    #                     # Check stop-loss
    #                     if self.stop_loss and price <= stop_loss_price:
    #                         # Close position
    #                         position_value = prev_holdings * price
    #                         commission_amount = position_value * self.commission
    #                         position_value = position_value - commission_amount
                            
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value
    #                         results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                         results.iloc[i, results.columns.get_loc('position')] = 0
                            
    #                         # Reset entry price and risk management
    #                         entry_price = 0
    #                         stop_loss_price = 0
    #                         take_profit_price = 0
                            
    #                     # Check take-profit
    #                     elif self.take_profit and price >= take_profit_price:
    #                         # Close position
    #                         position_value = prev_holdings * price
    #                         commission_amount = position_value * self.commission
    #                         position_value = position_value - commission_amount
                            
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value
    #                         results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                         results.iloc[i, results.columns.get_loc('position')] = 0
                            
    #                         # Reset entry price and risk management
    #                         entry_price = 0
    #                         stop_loss_price = 0
    #                         take_profit_price = 0
                            
    #                     else:
    #                         # Position unchanged
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash
    #                         results.iloc[i, results.columns.get_loc('holdings')] = prev_holdings
    #                         results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
    #                         results.iloc[i, results.columns.get_loc('stop_loss_price')] = stop_loss_price
    #                         results.iloc[i, results.columns.get_loc('take_profit_price')] = take_profit_price
    #                         results.iloc[i, results.columns.get_loc('trade_id')] = trade_id
                            
    #                 elif prev_position < 0:  # Short position
    #                     # Check stop-loss
    #                     if self.stop_loss and price >= stop_loss_price:
    #                         # Close position
    #                         position_value = abs(prev_holdings) * (2 * entry_price - price)
    #                         commission_amount = position_value * self.commission
    #                         position_value = position_value - commission_amount
                            
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value
    #                         results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                         results.iloc[i, results.columns.get_loc('position')] = 0
                            
    #                         # Reset entry price and risk management
    #                         entry_price = 0
    #                         stop_loss_price = 0
    #                         take_profit_price = 0
                            
    #                     # Check take-profit
    #                     elif self.take_profit and price <= take_profit_price:
    #                         # Close position
    #                         position_value = abs(prev_holdings) * (2 * entry_price - price)
    #                         commission_amount = position_value * self.commission
    #                         position_value = position_value - commission_amount
                            
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value
    #                         results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                         results.iloc[i, results.columns.get_loc('position')] = 0
                            
    #                         # Reset entry price and risk management
    #                         entry_price = 0
    #                         stop_loss_price = 0
    #                         take_profit_price = 0
                            
    #                     else:
    #                         # Position unchanged
    #                         results.iloc[i, results.columns.get_loc('cash')] = prev_cash
    #                         results.iloc[i, results.columns.get_loc('holdings')] = prev_holdings
    #                         results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
    #                         results.iloc[i, results.columns.get_loc('stop_loss_price')] = stop_loss_price
    #                         results.iloc[i, results.columns.get_loc('take_profit_price')] = take_profit_price
    #                         results.iloc[i, results.columns.get_loc('trade_id')] = trade_id
                            
    #                 else:  # No position
    #                     # Maintain current state
    #                     results.iloc[i, results.columns.get_loc('cash')] = prev_cash
    #                     results.iloc[i, results.columns.get_loc('holdings')] = 0
    #                     results.iloc[i, results.columns.get_loc('entry_price')] = 0
    #                     results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
    #                     results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
    #                     results.iloc[i, results.columns.get_loc('trade_id')] = 0                
    #             # Update portfolio value safely
    #             if results.iloc[i]['holdings'] > 0:  # Long position
    #                 holdings_value = results.iloc[i]['holdings'] * price
    #                 portfolio_value = results.iloc[i]['cash'] + holdings_value
    #                 # Clip to prevent unrealistic values
    #                 portfolio_value = min(max(portfolio_value, 0), 1e15)
    #                 results.iloc[i, results.columns.get_loc('portfolio_value')] = portfolio_value
    #             elif results.iloc[i]['holdings'] < 0:  # Short position
    #                 holdings_value = abs(results.iloc[i]['holdings']) * (2 * entry_price - price)
    #                 # Ensure holdings value can't be negative
    #                 holdings_value = max(holdings_value, 0)
    #                 portfolio_value = results.iloc[i]['cash'] + holdings_value
    #                 # Clip to prevent unrealistic values
    #                 portfolio_value = min(max(portfolio_value, 0), 1e15)
    #                 results.iloc[i, results.columns.get_loc('portfolio_value')] = portfolio_value
    #             else:  # No position
    #                 results.iloc[i, results.columns.get_loc('portfolio_value')] = results.iloc[i]['cash']
                    
    #         except Exception as e:
    #             logger.error(f"Error processing row {i}: {e}")
    #             # In case of error, copy the previous row values
    #             for col in results.columns:
    #                 results.iloc[i, results.columns.get_loc(col)] = results.iloc[i-1][col]
        
    #     # Calculate returns safely
    #     results['returns'] = results['portfolio_value'].pct_change()
        
    #     # Replace inf and -inf with NaN
    #     results.replace([np.inf, -np.inf], np.nan, inplace=True)
        
    #     # Calculate cumulative returns safely
    #     # First handle any NaN values in returns
    #     returns_no_nan = results['returns'].fillna(0)
    #     # Limit extreme returns that might cause overflow
    #     returns_no_nan = returns_no_nan.clip(-0.5, 0.5)  
    #     results['cumulative_returns'] = (1 + returns_no_nan).cumprod() - 1
        
    #     self.results = results
    #     return results

    def run(self) -> pd.DataFrame:
        """
        Run backtest with improved numerical stability and accuracy
        """
        # Generate signals
        signals = self.strategy.generate_signals(self.data)
        
        # Create a copy for results
        results = signals.copy()
        
        # Initialize portfolio metrics with proper data types
        results['holdings'] = 0.0
        results['cash'] = self.initial_capital
        results['portfolio_value'] = self.initial_capital
        results['entry_price'] = 0.0
        results['stop_loss_price'] = 0.0
        results['take_profit_price'] = 0.0
        results['trade_id'] = 0
        results['commission_paid'] = 0.0
        results['unrealized_pnl'] = 0.0  # Add unrealized PnL tracking
        
        # Track trade info
        trade_id = 0
        
        # Process each row with error handling
        for i in range(1, len(results)):
            try:
                # Get current and previous values
                prev_position = results.iloc[i-1]['position']
                curr_position = results.iloc[i]['position']
                prev_cash = results.iloc[i-1]['cash']
                prev_holdings = results.iloc[i-1]['holdings']
                entry_price = results.iloc[i-1]['entry_price']
                price = results.iloc[i]['close']
                
                # Safety check for price
                if np.isnan(price) or price <= 0:
                    logger.warning(f"Invalid price at index {i}: {price}. Using previous price.")
                    price = results.iloc[i-1]['close']
                    if np.isnan(price) or price <= 0:
                        price = 1.0  # Fallback to a safe value
                
                # Check for position changes
                if curr_position != prev_position:
                    # Close previous position if any
                    if prev_position != 0:
                        # Calculate PnL correctly for both long and short
                        if prev_position > 0:  # Long position
                            # Calculate position value before commission
                            position_value = prev_holdings * price
                            # Apply commission on exit (% of transaction value)
                            commission_amount = position_value * self.commission
                            # Update cash with position value minus commission
                            results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value - commission_amount
                        else:  # Short position
                            # For short positions, calculate profit/loss correctly
                            position_value = abs(prev_holdings) * entry_price  # Initial position value
                            price_diff = entry_price - price  # Profit if positive, loss if negative
                            pnl = abs(prev_holdings) * price_diff  # Actual P&L
                            
                            # Apply commission on exit (% of transaction value at current price)
                            exit_value = abs(prev_holdings) * price
                            commission_amount = exit_value * self.commission
                            
                            # Update cash: previous cash + initial position value + P&L - commission
                            results.iloc[i, results.columns.get_loc('cash')] = prev_cash + pnl - commission_amount
                        
                        # Reset holdings and update commission tracking
                        results.iloc[i, results.columns.get_loc('holdings')] = 0
                        results.iloc[i, results.columns.get_loc('commission_paid')] = \
                            results.iloc[i-1]['commission_paid'] + commission_amount
                        results.iloc[i, results.columns.get_loc('unrealized_pnl')] = 0
                    
                    # Open new position if any
                    if curr_position != 0:
                        # Calculate position size safely (using available cash)
                        available_cash = results.iloc[i]['cash']
                        position_size_cash = available_cash * self.position_size
                        
                        # Apply leverage and calculate actual position size
                        leveraged_cash = position_size_cash * self.leverage
                        position_size = leveraged_cash / max(price, 0.001)  # Shares/coins to buy/short
                        
                        # Calculate commission on entry (% of transaction value)
                        transaction_value = position_size * price
                        commission_amount = transaction_value * self.commission
                        
                        # Ensure we have enough cash after commission
                        if curr_position > 0:  # Long position
                            # Check if we have enough cash
                            if transaction_value + commission_amount > available_cash:
                                # Reduce position size to fit available cash including commission
                                position_size = (available_cash / (1 + self.commission)) / price
                                transaction_value = position_size * price
                                commission_amount = transaction_value * self.commission
                            
                            # Update cash and holdings
                            results.iloc[i, results.columns.get_loc('cash')] = available_cash - transaction_value - commission_amount
                            results.iloc[i, results.columns.get_loc('holdings')] = position_size
                            results.iloc[i, results.columns.get_loc('entry_price')] = price
                            
                            # Set stop-loss and take-profit
                            if self.stop_loss:
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = price * (1 - self.stop_loss / 100)
                            else:
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
                                
                            if self.take_profit:
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = price * (1 + self.take_profit / 100)
                            else:
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = float('inf')
                        
                        else:  # Short position
                            # For short positions, we borrow assets and sell them
                            # Cash increases by transaction value minus commission
                            results.iloc[i, results.columns.get_loc('cash')] = available_cash + transaction_value - commission_amount
                            results.iloc[i, results.columns.get_loc('holdings')] = -position_size  # Negative for short
                            results.iloc[i, results.columns.get_loc('entry_price')] = price
                            
                            # Set stop-loss and take-profit for short positions
                            if self.stop_loss:
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = price * (1 + self.stop_loss / 100)
                            else:
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = float('inf')
                                
                            if self.take_profit:
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = price * (1 - self.take_profit / 100)
                            else:
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
                        
                        # Increment trade ID and update commission tracking
                        trade_id += 1
                        results.iloc[i, results.columns.get_loc('trade_id')] = trade_id
                        results.iloc[i, results.columns.get_loc('commission_paid')] = \
                            results.iloc[i-1]['commission_paid'] + commission_amount
                    
                else:
                    # Position unchanged
                    # Update cash (remains the same)
                    results.iloc[i, results.columns.get_loc('cash')] = prev_cash
                    results.iloc[i, results.columns.get_loc('holdings')] = prev_holdings
                    results.iloc[i, results.columns.get_loc('entry_price')] = entry_price
                    results.iloc[i, results.columns.get_loc('stop_loss_price')] = results.iloc[i-1]['stop_loss_price']
                    results.iloc[i, results.columns.get_loc('take_profit_price')] = results.iloc[i-1]['take_profit_price']
                    results.iloc[i, results.columns.get_loc('commission_paid')] = results.iloc[i-1]['commission_paid']
                    results.iloc[i, results.columns.get_loc('trade_id')] = results.iloc[i-1]['trade_id']
                    
                    # Check for stop-loss and take-profit
                    if prev_position > 0:  # Long position
                        # Check stop-loss - use low price to simulate intrabar movement
                        low_price = results.iloc[i]['low'] if 'low' in results.columns else price
                        stop_loss_price = results.iloc[i-1]['stop_loss_price']
                        
                        if self.stop_loss and stop_loss_price > 0 and low_price <= stop_loss_price:
                            # Close position at stop-loss price with possible slippage
                            exit_price = max(low_price, stop_loss_price * 0.99)  # Add 1% slippage
                            position_value = prev_holdings * exit_price
                            commission_amount = position_value * self.commission
                            
                            results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value - commission_amount
                            results.iloc[i, results.columns.get_loc('holdings')] = 0
                            results.iloc[i, results.columns.get_loc('position')] = 0
                            results.iloc[i, results.columns.get_loc('entry_price')] = 0
                            results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
                            results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
                            results.iloc[i, results.columns.get_loc('commission_paid')] += commission_amount
                            results.iloc[i, results.columns.get_loc('unrealized_pnl')] = 0
                            
                        # Check take-profit - use high price to simulate intrabar movement
                        elif self.take_profit:
                            high_price = results.iloc[i]['high'] if 'high' in results.columns else price
                            take_profit_price = results.iloc[i-1]['take_profit_price']
                            
                            if take_profit_price > 0 and high_price >= take_profit_price:
                                # Close position at take-profit price with possible slippage
                                exit_price = min(high_price, take_profit_price * 1.01)  # Add 1% slippage
                                position_value = prev_holdings * exit_price
                                commission_amount = position_value * self.commission
                                
                                results.iloc[i, results.columns.get_loc('cash')] = prev_cash + position_value - commission_amount
                                results.iloc[i, results.columns.get_loc('holdings')] = 0
                                results.iloc[i, results.columns.get_loc('position')] = 0
                                results.iloc[i, results.columns.get_loc('entry_price')] = 0
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
                                results.iloc[i, results.columns.get_loc('commission_paid')] += commission_amount
                                results.iloc[i, results.columns.get_loc('unrealized_pnl')] = 0
                                
                            else:
                                # Update unrealized P&L for long position
                                unrealized_pnl = prev_holdings * (price - entry_price)
                                results.iloc[i, results.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                        
                        else:
                            # Update unrealized P&L for long position
                            unrealized_pnl = prev_holdings * (price - entry_price)
                            results.iloc[i, results.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                    
                    elif prev_position < 0:  # Short position
                        # Check stop-loss - use high price to simulate intrabar movement
                        high_price = results.iloc[i]['high'] if 'high' in results.columns else price
                        stop_loss_price = results.iloc[i-1]['stop_loss_price']
                        
                        if self.stop_loss and stop_loss_price > 0 and high_price >= stop_loss_price:
                            # Close position at stop-loss price with possible slippage
                            exit_price = min(high_price, stop_loss_price * 1.01)  # Add 1% slippage
                            
                            # Calculate P&L for short position (entry_price - exit_price)
                            pnl = abs(prev_holdings) * (entry_price - exit_price)
                            commission_amount = (abs(prev_holdings) * exit_price) * self.commission
                            
                            results.iloc[i, results.columns.get_loc('cash')] = prev_cash + pnl - commission_amount
                            results.iloc[i, results.columns.get_loc('holdings')] = 0
                            results.iloc[i, results.columns.get_loc('position')] = 0
                            results.iloc[i, results.columns.get_loc('entry_price')] = 0
                            results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
                            results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
                            results.iloc[i, results.columns.get_loc('commission_paid')] += commission_amount
                            results.iloc[i, results.columns.get_loc('unrealized_pnl')] = 0
                            
                        # Check take-profit - use low price to simulate intrabar movement
                        elif self.take_profit:
                            low_price = results.iloc[i]['low'] if 'low' in results.columns else price
                            take_profit_price = results.iloc[i-1]['take_profit_price']
                            
                            if take_profit_price > 0 and low_price <= take_profit_price:
                                # Close position at take-profit price with possible slippage
                                exit_price = max(low_price, take_profit_price * 0.99)  # Add 1% slippage
                                
                                # Calculate P&L for short position
                                pnl = abs(prev_holdings) * (entry_price - exit_price)
                                commission_amount = (abs(prev_holdings) * exit_price) * self.commission
                                
                                results.iloc[i, results.columns.get_loc('cash')] = prev_cash + pnl - commission_amount
                                results.iloc[i, results.columns.get_loc('holdings')] = 0
                                results.iloc[i, results.columns.get_loc('position')] = 0
                                results.iloc[i, results.columns.get_loc('entry_price')] = 0
                                results.iloc[i, results.columns.get_loc('stop_loss_price')] = 0
                                results.iloc[i, results.columns.get_loc('take_profit_price')] = 0
                                results.iloc[i, results.columns.get_loc('commission_paid')] += commission_amount
                                results.iloc[i, results.columns.get_loc('unrealized_pnl')] = 0
                                
                            else:
                                # Update unrealized P&L for short position
                                unrealized_pnl = abs(prev_holdings) * (entry_price - price)
                                results.iloc[i, results.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                        
                        else:
                            # Update unrealized P&L for short position
                            unrealized_pnl = abs(prev_holdings) * (entry_price - price)
                            results.iloc[i, results.columns.get_loc('unrealized_pnl')] = unrealized_pnl
                
                # Calculate portfolio value correctly
                if results.iloc[i]['holdings'] > 0:  # Long position
                    # Portfolio value = cash + current holdings value
                    holdings_value = results.iloc[i]['holdings'] * price
                    results.iloc[i, results.columns.get_loc('portfolio_value')] = results.iloc[i]['cash'] + holdings_value
                
                elif results.iloc[i]['holdings'] < 0:  # Short position
                    # For short position, use unrealized P&L to calculate portfolio value
                    results.iloc[i, results.columns.get_loc('portfolio_value')] = results.iloc[i]['cash'] + results.iloc[i]['unrealized_pnl']
                
                else:  # No position
                    results.iloc[i, results.columns.get_loc('portfolio_value')] = results.iloc[i]['cash']
            
            except Exception as e:
                logger.error(f"Error processing row {i}: {e}")
                # In case of error, copy the previous row values
                for col in results.columns:
                    results.iloc[i, results.columns.get_loc(col)] = results.iloc[i-1][col]
        
        # Calculate returns properly using log returns
        results['log_returns'] = np.log(results['portfolio_value'] / results['portfolio_value'].shift(1))
        results['returns'] = results['portfolio_value'].pct_change()
        
        # Calculate cumulative returns correctly
        results['cumulative_returns'] = (1 + results['returns'].fillna(0)).cumprod() - 1
        
        # Add drawdown calculation
        peak = results['portfolio_value'].expanding().max()
        results['drawdown'] = (results['portfolio_value'] / peak) - 1
        
        self.results = results
        return results
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get list of trades with improved accuracy and error handling
        """
        if self.results is None:
            logger.error("Backtest not run yet")
            raise ValueError("Backtest not run yet")
        
        trades = []
        
        try:
            # Identify all trade IDs greater than 0
            trade_ids = sorted(self.results['trade_id'].unique())
            trade_ids = [tid for tid in trade_ids if tid > 0]
            
            for trade_id in trade_ids:
                # Get all rows for this trade
                trade_data = self.results[self.results['trade_id'] == trade_id]
                
                if len(trade_data) == 0:
                    continue
                
                # Entry details
                entry_row = trade_data.iloc[0]
                entry_date = entry_row.name
                entry_price = entry_row['entry_price']
                
                # Validate entry price
                if np.isnan(entry_price) or entry_price <= 0:
                    logger.warning(f"Invalid entry price for trade {trade_id}: {entry_price}")
                    continue
                
                position_type = 'LONG' if entry_row['position'] > 0 else 'SHORT'
                
                # Check if trade was closed within the results data
                exit_found = False
                exit_row = None
                exit_date = None
                exit_price = None
                
                # Look for the next row after this trade where trade_id changes or becomes 0
                next_rows = self.results[self.results.index > trade_data.index[-1]]
                
                if len(next_rows) > 0:
                    for idx, row in next_rows.iterrows():
                        if row['trade_id'] != trade_id or row['position'] == 0:
                            exit_row = row
                            exit_date = idx
                            exit_price = row['close']
                            exit_found = True
                            break
                
                # Calculate PnL safely
                pnl = None
                if exit_found and exit_price is not None:
                    if position_type == 'LONG':
                        if entry_price > 0:  # Avoid division by zero
                            pnl = (exit_price - entry_price) / entry_price
                    else:  # SHORT
                        if entry_price > 0:  # Avoid division by zero
                            pnl = (entry_price - exit_price) / entry_price
                
                # Calculate duration if possible
                duration = None
                if exit_date is not None:
                    try:
                        duration = exit_date - entry_date
                    except:
                        duration = None
                
                # Add trade to list with all available data
                trade_info = {
                    'trade_id': trade_id,
                    'entry_date': entry_date,
                    'exit_date': exit_date,
                    'position_type': position_type,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'duration': duration
                }
                
                trades.append(trade_info)
            
            # Convert to DataFrame and handle potential conversion issues
            trades_df = pd.DataFrame(trades)
            if len(trades_df) > 0:
                # Convert potential object or string columns to proper types
                if 'pnl' in trades_df.columns:
                    trades_df['pnl'] = pd.to_numeric(trades_df['pnl'], errors='coerce')
                    
                # Set index if needed
                if 'trade_id' in trades_df.columns:
                    trades_df = trades_df.set_index('trade_id')
            
            return trades_df
            
        except Exception as e:
            logger.error(f"Error extracting trades: {e}")
            return pd.DataFrame(columns=['trade_id', 'entry_date', 'exit_date', 'position_type', 'entry_price', 'exit_price', 'pnl'])
    
    def get_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics
        
        Returns:
        --------
        Dict
            Dictionary with performance metrics
        """
        if self.results is None:
            logger.error("Backtest not run yet")
            raise ValueError("Backtest not run yet")
        
        # Extract returns, replacing inf and NaN with 0
        returns = self.results['returns'].replace([np.inf, -np.inf], np.nan).dropna()
        
        if len(returns) == 0:
            logger.warning("No valid returns data available")
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
        
        # Calculate metrics safely
        try:
            # Total return based on first and last valid portfolio value
            valid_values = self.results['portfolio_value'].replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid_values) >= 2:
                total_return = valid_values.iloc[-1] / valid_values.iloc[0] - 1
            else:
                total_return = 0
            
            # Annualized return
            trading_days = len(returns)
            annual_factor = 252 / max(trading_days, 1)
            annual_return = (1 + total_return) ** annual_factor - 1
            
            # Volatility
            volatility = returns.std() * np.sqrt(252) if len(returns) > 1 else 0
            
            # Sharpe ratio
            risk_free_rate = 0.02  # Assuming 2% risk-free rate
            sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
            
            # Sortino ratio
            negative_returns = returns[returns < 0]
            downside_deviation = negative_returns.std() * np.sqrt(252) if len(negative_returns) > 1 else 1e-6
            sortino_ratio = (annual_return - risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + returns.fillna(0)).cumprod()
            max_return = cumulative_returns.expanding().max()
            drawdowns = (cumulative_returns / max_return) - 1
            max_drawdown = drawdowns.min() if not drawdowns.empty else 0
            
            # Win rate and average PnL
            trades = self.get_trades()
            trades_with_pnl = trades.dropna(subset=['pnl'])
            
            wins = trades_with_pnl[trades_with_pnl['pnl'] > 0]
            win_rate = len(wins) / len(trades_with_pnl) if len(trades_with_pnl) > 0 else 0
            
            avg_pnl = trades_with_pnl['pnl'].mean() if len(trades_with_pnl) > 0 else 0
            
            metrics = {
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'num_trades': len(trades)
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
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
    
    def save_results(self, directory: str = 'backtest_results') -> None:
        """
        Save backtest results
        
        Parameters:
        -----------
        directory : str
            Directory to save results
        """
        if self.results is None:
            logger.error("Backtest not run yet")
            raise ValueError("Backtest not run yet")
        
        # Create directory
        os.makedirs(directory, exist_ok=True)
        
        # Create timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save results to CSV
        results_file = os.path.join(directory, f"{self.strategy.name}_{timestamp}_results.csv")
        self.results.to_csv(results_file)
        
        # Save trades to CSV
        trades_file = os.path.join(directory, f"{self.strategy.name}_{timestamp}_trades.csv")
        self.get_trades().to_csv(trades_file)
        
        # Save metrics to CSV
        metrics = self.get_performance_metrics()
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
        metrics_file = os.path.join(directory, f"{self.strategy.name}_{timestamp}_metrics.csv")
        metrics_df.to_csv(metrics_file)
        
        logger.info(f"Backtest results for strategy '{self.strategy.name}' saved to {directory}")
        
        return {
            'results_file': results_file,
            'trades_file': trades_file,
            'metrics_file': metrics_file
        }