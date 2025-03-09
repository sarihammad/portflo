"""
Main script to run the portfolio optimization project.
"""
import os
import argparse
import logging
import sys

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_data_collection(args):
    """
    Run data collection.
    """
    from portflo.data.collect import main as collect_main
    logger.info("Running data collection")
    sys.argv = [sys.argv[0]] + args
    collect_main()


def run_training(args):
    """
    Run RL agent training.
    """
    from portflo.models.train import main as train_main
    logger.info("Running RL agent training")
    sys.argv = [sys.argv[0]] + args
    train_main()


def run_backtesting(args):
    """
    Run backtesting.
    """
    from portflo.backtesting.run import main as backtest_main
    logger.info("Running backtesting")
    sys.argv = [sys.argv[0]] + args
    backtest_main()


def run_dashboard(args):
    """
    Run dashboard.
    """
    from portflo.dashboard.app import main as dashboard_main
    logger.info("Running dashboard")
    sys.argv = [sys.argv[0]] + args
    dashboard_main()


def main():
    """
    Main function to run the portfolio optimization project.
    """
    parser = argparse.ArgumentParser(description='Portfolio Optimization using Reinforcement Learning')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Data collection command
    collect_parser = subparsers.add_parser('collect', help='Collect market data')
    collect_parser.add_argument('--source', type=str, choices=['alpaca', 'yahoo', 'binance', 'all'], 
                               default='all', help='Data source')
    collect_parser.add_argument('--start-date', type=str, default=None, 
                               help='Start date (YYYY-MM-DD)')
    collect_parser.add_argument('--end-date', type=str, default=None, 
                               help='End date (YYYY-MM-DD)')
    collect_parser.add_argument('--timeframe', type=str, default='1d', 
                               choices=['1d', '1h', '15m', '5m', '1m'], 
                               help='Data timeframe')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train RL agent')
    train_parser.add_argument('--model', type=str, choices=['ppo'], default='ppo',
                             help='RL model to use')
    train_parser.add_argument('--asset-types', type=str, nargs='+', 
                             choices=['stocks', 'etfs', 'crypto', 'all'], default=['all'],
                             help='Asset types to include')
    train_parser.add_argument('--risk-level', type=str, 
                             choices=['conservative', 'balanced', 'aggressive'], 
                             default='balanced',
                             help='Risk level for the agent')
    train_parser.add_argument('--train-episodes', type=int, default=None,
                             help='Number of training episodes')
    
    # Backtesting command
    backtest_parser = subparsers.add_parser('backtest', help='Run backtests')
    backtest_parser.add_argument('--strategies', type=str, nargs='+', 
                                choices=['equal_weight', 'mean_variance', 'rl_ppo', 'all'], 
                                default=['all'],
                                help='Strategies to backtest')
    backtest_parser.add_argument('--asset-types', type=str, nargs='+', 
                                choices=['stocks', 'etfs', 'crypto', 'all'], 
                                default=['all'],
                                help='Asset types to include')
    backtest_parser.add_argument('--start-date', type=str, default=None,
                                help='Start date for the backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', type=str, default=None,
                                help='End date for the backtest (YYYY-MM-DD)')
    backtest_parser.add_argument('--rl-model-path', type=str, default=None,
                                help='Path to RL model')
    
    # Dashboard command
    dashboard_parser = subparsers.add_parser('dashboard', help='Run dashboard')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Run the specified command
    if args.command == 'collect':
        run_data_collection(sys.argv[2:])
    elif args.command == 'train':
        run_training(sys.argv[2:])
    elif args.command == 'backtest':
        run_backtesting(sys.argv[2:])
    elif args.command == 'dashboard':
        run_dashboard(sys.argv[2:])
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 