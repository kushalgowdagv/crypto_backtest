#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
CryptoBacktest - Main entry point

This script demonstrates the usage of the CryptoBacktest framework
for backtesting cryptocurrency trading strategies.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Add parent directory to path to import the framework
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the framework
from cryptobacktest.backtest_lib.data.data_loader import DataLoader
from cryptobacktest.backtest_lib.strategies.trend_following import TrendFollowingStrategy
from cryptobacktest.backtest_lib.strategies.volatility_breakout import VolatilityBreakoutStrategy
from cryptobacktest.backtest_lib.strategies.volume_price_divergence import VolumePriceDivergenceStrategy
from cryptobacktest.backtest_lib.strategies.mean_reversion import MeanReversionStrategy
from cryptobacktest.backtest_lib.strategies.machine_learning import MachineLearningStrategy
from cryptobacktest.backtest_lib.strategies.multi_timeframe import MultiTimeframeStrategy
from cryptobacktest.backtest_lib.strategies.pair_trading import PairTradingStrategy
from cryptobacktest.backtest_lib.engine.backtest import Backtest
from cryptobacktest.backtest_lib.analysis.analysis import Analysis
from cryptobacktest.cryptobacktest import CryptoBacktest

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("cryptobacktest.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('CryptoBacktest')


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='CryptoBacktest - Cryptocurrency Trading Strategy Backtesting Framework')
    
    # Data options
    parser.add_argument('--data_dir', type=str, default='CoinLion_strat', help='Directory containing data files')
    parser.add_argument('--symbol', type=str, default='BTCUSD', help='Trading symbol (e.g., BTCUSD)')
    parser.add_argument('--timeframe', type=str, default='10m', help='Timeframe of the data')
    parser.add_argument('--start_date', type=str, default=None, help='Start date for backtest (format: YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None, help='End date for backtest (format: YYYY-MM-DD)')
    
    # Strategy options
    parser.add_argument('--strategy', type=str, default='trend_following', 
                       choices=['trend_following', 'volatility_breakout', 'volume_price_divergence', 
                               'mean_reversion', 'machine_learning', 'multi_timeframe', 'pair_trading'],
                       help='Strategy type')
    
    # Backtest options
    parser.add_argument('--initial_capital', type=float, default=10000.0, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.001, help='Commission rate')
    parser.add_argument('--position_size', type=float, default=1.0, help='Position size as a fraction of capital (0-1)')
    parser.add_argument('--leverage', type=float, default=1.0, help='Leverage multiplier')
    parser.add_argument('--stop_loss', type=float, default=None, help='Stop-loss as a percentage')
    parser.add_argument('--take_profit', type=float, default=None, help='Take-profit as a percentage')
    
    # Output options
    parser.add_argument('--output_dir', type=str, default='backtest_results', help='Directory to save results')
    parser.add_argument('--generate_report', action='store_true', help='Generate performance report')
    parser.add_argument('--optimize', action='store_true', help='Optimize strategy parameters')
    parser.add_argument('--monte_carlo', action='store_true', help='Run Monte Carlo simulation')
    parser.add_argument('--compare', action='store_true', help='Compare with other strategies')
    
    return parser.parse_args()


def create_strategy(strategy_type, **kwargs):
    """Create a strategy based on type"""
    if strategy_type == 'trend_following':
        return TrendFollowingStrategy(**kwargs)
    elif strategy_type == 'volatility_breakout':
        return VolatilityBreakoutStrategy(**kwargs)
    elif strategy_type == 'volume_price_divergence':
        return VolumePriceDivergenceStrategy(**kwargs)
    elif strategy_type == 'mean_reversion':
        return MeanReversionStrategy(**kwargs)
    elif strategy_type == 'machine_learning':
        return MachineLearningStrategy(**kwargs)
    elif strategy_type == 'multi_timeframe':
        return MultiTimeframeStrategy(**kwargs)
    elif strategy_type == 'pair_trading':
        return PairTradingStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")


def main():
    """Main entry point"""
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the framework
    cryptobacktest = CryptoBacktest(args.data_dir)
    
    # Create strategy
    strategy = create_strategy(args.strategy)
    
    # Add strategy to the framework
    cryptobacktest.add_strategy(strategy)
    
    # Run backtest
    backtest = cryptobacktest.run_backtest(
        strategy_name=strategy.name,
        symbol=args.symbol,
        timeframe=args.timeframe,
        start_date=args.start_date,
        end_date=args.end_date,
        initial_capital=args.initial_capital,
        commission=args.commission,
        position_size=args.position_size,
        leverage=args.leverage,
        stop_loss=args.stop_loss,
        take_profit=args.take_profit
    )
    
    # Save results
    results_files = backtest.save_results(args.output_dir)
    logger.info(f"Backtest results saved to: {results_files}")
    
    # Generate performance report
    if args.generate_report:
        report = cryptobacktest.generate_report(
            strategy_name=strategy.name,
            symbol=args.symbol,
            timeframe=args.timeframe,
            directory=args.output_dir
        )
        logger.info(f"Performance report generated: {report}")
    
    # Optimize strategy parameters
    if args.optimize:
        # Define parameter grid based on strategy type
        if args.strategy == 'trend_following':
            params_grid = {
                'fast_period': [3, 5, 10],
                'slow_period': [15, 20, 30]
            }
        elif args.strategy == 'volatility_breakout':
            params_grid = {
                'period': [10, 20, 30],
                'std_dev': [1.5, 2.0, 2.5]
            }
        elif args.strategy == 'mean_reversion':
            params_grid = {
                'rsi_period': [7, 14, 21],
                'overbought': [70, 75, 80],
                'oversold': [20, 25, 30]
            }
        else:
            params_grid = {}
        
        if params_grid:
            # Load data
            data = cryptobacktest.load_data(args.symbol, args.timeframe)
            
            # Filter by date range
            if args.start_date is not None:
                data = data[data.index >= args.start_date]
            
            if args.end_date is not None:
                data = data[data.index <= args.end_date]
            
            # Optimize strategy
            from cryptobacktest.backtest_lib.utils.utils import optimize_strategy
            
            best_params, optimization_results = optimize_strategy(
                strategy=strategy,
                data=data,
                params_grid=params_grid
            )
            
            logger.info(f"Best parameters: {best_params}")
            
            # Save optimization results
            import json
            optimization_file = os.path.join(args.output_dir, f"{strategy.name}_optimization.json")
            
            with open(optimization_file, 'w') as f:
                json.dump({
                    'best_params': best_params,
                    'metric_value': optimization_results['best']['metrics'].get('sharpe_ratio', 0)
                }, f, indent=4)
            
            logger.info(f"Optimization results saved to: {optimization_file}")
            
            # Run backtest with best parameters
            strategy.set_parameters(**best_params)
            
            backtest = cryptobacktest.run_backtest(
                strategy_name=strategy.name,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                commission=args.commission,
                position_size=args.position_size,
                leverage=args.leverage,
                stop_loss=args.stop_loss,
                take_profit=args.take_profit
            )
            
            # Save results with optimized parameters
            results_files = backtest.save_results(args.output_dir)
            logger.info(f"Backtest results with optimized parameters saved to: {results_files}")
    
    # Run Monte Carlo simulation
    if args.monte_carlo:
        from cryptobacktest.backtest_lib.utils.utils import run_monte_carlo
        
        monte_carlo_results = run_monte_carlo(backtest)
        
        # Save Monte Carlo results
        import json
        monte_carlo_file = os.path.join(args.output_dir, f"{strategy.name}_monte_carlo.json")
        
        with open(monte_carlo_file, 'w') as f:
            json.dump({
                'mean': float(monte_carlo_results['mean']),
                'median': float(monte_carlo_results['median']),
                'std': float(monte_carlo_results['std']),
                'percentiles': {k: float(v) for k, v in monte_carlo_results['percentiles'].items()}
            }, f, indent=4)
        
        logger.info(f"Monte Carlo results saved to: {monte_carlo_file}")
    
    # Compare with other strategies
    if args.compare:
        # Define strategies to compare
        strategies_to_compare = []
        
        if args.strategy != 'trend_following':
            strategies_to_compare.append('TrendFollowing')
        
        if args.strategy != 'volatility_breakout':
            strategies_to_compare.append('VolatilityBreakout')
        
        if args.strategy != 'mean_reversion':
            strategies_to_compare.append('MeanReversion')
        
        if strategies_to_compare:
            # Create and add strategies
            for strat_name in strategies_to_compare:
                if strat_name == 'TrendFollowing':
                    cryptobacktest.add_strategy(TrendFollowingStrategy())
                elif strat_name == 'VolatilityBreakout':
                    cryptobacktest.add_strategy(VolatilityBreakoutStrategy())
                elif strat_name == 'MeanReversion':
                    cryptobacktest.add_strategy(MeanReversionStrategy())
            
            # Add current strategy to comparison
            strategies_to_compare.append(strategy.name)
            
            # Compare strategies
            comparison = cryptobacktest.compare_strategies(
                strategies=strategies_to_compare,
                symbol=args.symbol,
                timeframe=args.timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                initial_capital=args.initial_capital,
                commission=args.commission
            )
            
            # Save comparison results
            import json
            comparison_file = os.path.join(args.output_dir, f"strategy_comparison.json")
            
            comparison_data = {}
            for strat_name, result in comparison.items():
                comparison_data[strat_name] = result['metrics']
            
            with open(comparison_file, 'w') as f:
                json.dump(comparison_data, f, indent=4)
            
            logger.info(f"Strategy comparison results saved to: {comparison_file}")
            
            # Plot comparison
            try:
                cryptobacktest.plot_strategy_comparison(comparison)
                plt.savefig(os.path.join(args.output_dir, f"strategy_comparison.png"))
                logger.info(f"Strategy comparison plot saved to: {os.path.join(args.output_dir, 'strategy_comparison.png')}")
            except Exception as e:
                logger.error(f"Error plotting strategy comparison: {e}")
    
    logger.info("Backtest completed successfully")


if __name__ == "__main__":
    main()