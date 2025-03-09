"""
Script for running backtests and comparing different portfolio strategies.
"""
import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import json

from portflo.config.settings import (
    DATA_DIR, MODELS_DIR, INITIAL_PORTFOLIO_VALUE, TRANSACTION_FEE_RATE
)
from portflo.data.processors.data_processor import DataProcessor
from portflo.strategies.traditional.equal_weight import EqualWeightStrategy
from portflo.strategies.traditional.mean_variance import MeanVarianceOptimizer
from portflo.strategies.rl_strategies.rl_strategy import RLStrategy
from portflo.backtesting.backtest import Backtest


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_price_data(asset_types, timeframe='1d', start_date=None, end_date=None):
    """
    Load price data for the given asset types.
    
    Parameters:
    -----------
    asset_types : list
        List of asset types to load ('stocks', 'etfs', 'crypto')
    timeframe : str, default '1d'
        Timeframe of the data
    start_date : str, optional
        Start date (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date (format: 'YYYY-MM-DD')
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with price data
    """
    all_dfs = []
    
    for asset_type in asset_types:
        # Path to processed data
        data_path = os.path.join(DATA_DIR, f"{asset_type}/processed/{timeframe}")
        
        if not os.path.exists(data_path):
            logger.warning(f"No processed data found for {asset_type} with timeframe {timeframe}")
            continue
        
        # Load all CSV files in the directory
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                # Load data
                file_path = os.path.join(data_path, file)
                df = pd.read_csv(file_path)
                
                # Add symbol column if not present
                if 'symbol' not in df.columns:
                    symbol = file.replace('.csv', '').replace('_', '/')
                    df['symbol'] = symbol
                
                # Convert date to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Append to list
                all_dfs.append(df)
    
    if not all_dfs:
        raise ValueError("No data found. Please run data collection first.")
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Pivot to get price data in the format: date x asset
    price_data = combined_df.pivot_table(
        index='date', columns='symbol', values='close'
    )
    
    # Filter by date range
    if start_date is not None:
        price_data = price_data[price_data.index >= start_date]
    
    if end_date is not None:
        price_data = price_data[price_data.index <= end_date]
    
    # Forward fill missing values
    price_data = price_data.ffill()
    
    return price_data


def run_backtest(
    price_data, strategy, strategy_name, benchmark=None, 
    rebalance_freq='M', start_date=None, end_date=None
):
    """
    Run a backtest for the given strategy.
    
    Parameters:
    -----------
    price_data : pandas.DataFrame
        DataFrame with price data
    strategy : object
        Strategy object with allocate and rebalance methods
    strategy_name : str
        Name of the strategy
    benchmark : str, optional
        Benchmark asset (e.g., 'SPY')
    rebalance_freq : str, default 'M'
        Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
    start_date : str, optional
        Start date for the backtest (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for the backtest (format: 'YYYY-MM-DD')
        
    Returns:
    --------
    BacktestResult
        Backtest result object
    """
    # Create backtest
    backtest = Backtest(
        price_data=price_data,
        strategy=strategy,
        strategy_name=strategy_name,
        initial_cash=INITIAL_PORTFOLIO_VALUE,
        transaction_fee_rate=TRANSACTION_FEE_RATE,
        benchmark=benchmark,
        rebalance_freq=rebalance_freq,
        start_date=start_date,
        end_date=end_date
    )
    
    # Run backtest
    logger.info(f"Running backtest for {strategy_name}")
    result = backtest.run()
    
    return result


def compare_strategies(results, benchmark_name=None, save_path=None):
    """
    Compare different strategies.
    
    Parameters:
    -----------
    results : list
        List of BacktestResult objects
    benchmark_name : str, optional
        Name of the benchmark
    save_path : str, optional
        Path to save the comparison results
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with comparison results
    """
    # Extract metrics from each result
    metrics_list = []
    for result in results:
        metrics = result.metrics.copy()
        metrics['strategy'] = result.strategy_name
        metrics_list.append(metrics)
    
    # Create comparison DataFrame
    comparison = pd.DataFrame(metrics_list)
    
    # Set strategy as index
    comparison = comparison.set_index('strategy')
    
    # Plot portfolio values
    plt.figure(figsize=(12, 6))
    
    for result in results:
        plt.plot(
            result.portfolio_values.index, 
            result.portfolio_values / result.portfolio_values.iloc[0], 
            label=result.strategy_name
        )
    
    # Plot benchmark if provided
    if benchmark_name is not None and results[0].benchmark_values is not None:
        benchmark_values = results[0].benchmark_values
        plt.plot(
            benchmark_values.index, 
            benchmark_values / benchmark_values.iloc[0], 
            label=benchmark_name,
            linestyle='--'
        )
    
    plt.title('Portfolio Value Comparison (Normalized)')
    plt.xlabel('Date')
    plt.ylabel('Value (Normalized)')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'portfolio_value_comparison.png'))
    
    plt.show()
    
    # Plot drawdowns
    plt.figure(figsize=(12, 6))
    
    for result in results:
        rolling_max = result.portfolio_values.cummax()
        drawdown = (result.portfolio_values - rolling_max) / rolling_max
        plt.plot(drawdown.index, drawdown, label=result.strategy_name)
    
    plt.title('Drawdown Comparison')
    plt.xlabel('Date')
    plt.ylabel('Drawdown')
    plt.legend()
    plt.grid(True)
    
    if save_path is not None:
        plt.savefig(os.path.join(save_path, 'drawdown_comparison.png'))
    
    plt.show()
    
    # Save comparison to CSV if path is provided
    if save_path is not None:
        comparison.to_csv(os.path.join(save_path, 'strategy_comparison.csv'))
    
    return comparison


def main():
    """
    Main function to run backtests and compare strategies.
    """
    parser = argparse.ArgumentParser(description='Run backtests for portfolio strategies')
    parser.add_argument('--strategies', type=str, nargs='+', 
                        choices=['equal_weight', 'mean_variance', 'rl_ppo', 'all'], 
                        default=['all'],
                        help='Strategies to backtest')
    parser.add_argument('--asset-types', type=str, nargs='+', 
                        choices=['stocks', 'etfs', 'crypto', 'all'], 
                        default=['all'],
                        help='Asset types to include')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe of the data')
    parser.add_argument('--start-date', type=str, default=None,
                        help='Start date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None,
                        help='End date for the backtest (YYYY-MM-DD)')
    parser.add_argument('--benchmark', type=str, default='SPY',
                        help='Benchmark asset')
    parser.add_argument('--rebalance-freq', type=str, default='M',
                        choices=['D', 'W', 'M', 'Q'],
                        help='Rebalancing frequency')
    parser.add_argument('--risk-aversion', type=float, default=1.0,
                        help='Risk aversion parameter for mean-variance optimization')
    parser.add_argument('--cash-weight', type=float, default=0.05,
                        help='Cash weight for equal-weight strategy')
    parser.add_argument('--rl-model-path', type=str, default=None,
                        help='Path to RL model')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory to save results')
    
    args = parser.parse_args()
    
    # Determine asset types
    if 'all' in args.asset_types:
        asset_types = ['stocks', 'etfs', 'crypto']
    else:
        asset_types = args.asset_types
    
    # Load price data
    logger.info(f"Loading price data for {asset_types}")
    price_data = load_price_data(
        asset_types, args.timeframe, args.start_date, args.end_date
    )
    logger.info(f"Loaded price data with {len(price_data.columns)} assets")
    
    # Create output directory if provided
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine strategies to backtest
    if 'all' in args.strategies:
        strategies = ['equal_weight', 'mean_variance', 'rl_ppo']
    else:
        strategies = args.strategies
    
    # Initialize strategies
    strategy_objects = []
    
    if 'equal_weight' in strategies:
        logger.info("Initializing Equal-Weight strategy")
        equal_weight = EqualWeightStrategy(
            include_cash=True,
            cash_weight=args.cash_weight
        )
        strategy_objects.append((equal_weight, 'Equal-Weight'))
    
    if 'mean_variance' in strategies:
        logger.info("Initializing Mean-Variance strategy")
        mean_variance = MeanVarianceOptimizer(
            risk_aversion=args.risk_aversion,
            min_weight=0.0,
            max_weight=0.3  # Limit maximum weight to 30% for diversification
        )
        strategy_objects.append((mean_variance, 'Mean-Variance'))
    
    if 'rl_ppo' in strategies:
        if args.rl_model_path is None:
            logger.warning("RL model path not provided, skipping RL strategy")
        else:
            logger.info("Initializing RL strategy")
            rl_strategy = RLStrategy(
                model_path=args.rl_model_path
            )
            strategy_objects.append((rl_strategy, 'RL-PPO'))
    
    # Run backtests
    results = []
    
    for strategy, strategy_name in strategy_objects:
        result = run_backtest(
            price_data=price_data,
            strategy=strategy,
            strategy_name=strategy_name,
            benchmark=args.benchmark,
            rebalance_freq=args.rebalance_freq,
            start_date=args.start_date,
            end_date=args.end_date
        )
        
        # Print metrics
        result.print_metrics()
        
        # Plot portfolio value
        result.plot_portfolio_value(
            benchmark_name=args.benchmark,
            save_path=args.output_dir
        )
        
        # Plot asset weights
        result.plot_asset_weights(
            save_path=args.output_dir
        )
        
        # Plot drawdown
        result.plot_drawdown(
            save_path=args.output_dir
        )
        
        # Append to results list
        results.append(result)
    
    # Compare strategies
    if len(results) > 1:
        logger.info("Comparing strategies")
        comparison = compare_strategies(
            results=results,
            benchmark_name=args.benchmark,
            save_path=args.output_dir
        )
        
        # Print comparison
        print("\nStrategy Comparison:")
        print(comparison)
    
    logger.info("Backtesting completed")


if __name__ == '__main__':
    main() 