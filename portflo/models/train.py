"""
Script for training the RL agent for portfolio optimization.
"""
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import logging
from tqdm import tqdm
import json

from portflo.config.settings import (
    DATA_DIR, MODELS_DIR, LOOKBACK_WINDOW, TRAINING_EPISODES,
    VALIDATION_EPISODES, TEST_EPISODES, DEFAULT_STOCKS, DEFAULT_ETFS, DEFAULT_CRYPTO
)
from portflo.data.processors.data_processor import DataProcessor
from portflo.models.environments.portfolio_env import PortfolioEnv
from portflo.models.agents.ppo_agent import PPOAgent


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_data(asset_types, timeframe='1d'):
    """
    Load processed data for the given asset types.
    
    Parameters:
    -----------
    asset_types : list
        List of asset types to load ('stocks', 'etfs', 'crypto')
    timeframe : str, default '1d'
        Timeframe of the data
        
    Returns:
    --------
    pandas.DataFrame
        Combined DataFrame with all assets
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
    
    # Set MultiIndex
    combined_df = combined_df.set_index(['symbol', 'date'])
    
    return combined_df


def split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split data into training, validation, and test sets.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with price data
    train_ratio : float, default 0.7
        Ratio of data to use for training
    val_ratio : float, default 0.15
        Ratio of data to use for validation
    test_ratio : float, default 0.15
        Ratio of data to use for testing
        
    Returns:
    --------
    tuple
        (train_data, val_data, test_data)
    """
    # Get unique dates
    dates = data.index.get_level_values('date').unique()
    dates = sorted(dates)
    
    # Calculate split indices
    train_idx = int(len(dates) * train_ratio)
    val_idx = int(len(dates) * (train_ratio + val_ratio))
    
    # Split dates
    train_dates = dates[:train_idx]
    val_dates = dates[train_idx:val_idx]
    test_dates = dates[val_idx:]
    
    # Split data
    train_data = data.loc[data.index.get_level_values('date').isin(train_dates)]
    val_data = data.loc[data.index.get_level_values('date').isin(val_dates)]
    test_data = data.loc[data.index.get_level_values('date').isin(test_dates)]
    
    return train_data, val_data, test_data


def train_agent(
    env, agent, n_episodes, batch_size=64, update_freq=10, 
    save_freq=100, save_path=None, verbose=True
):
    """
    Train the RL agent.
    
    Parameters:
    -----------
    env : PortfolioEnv
        Portfolio environment
    agent : PPOAgent
        RL agent
    n_episodes : int
        Number of episodes to train
    batch_size : int, default 64
        Batch size for updates
    update_freq : int, default 10
        Frequency of updates (in episodes)
    save_freq : int, default 100
        Frequency of saving the agent (in episodes)
    save_path : str, optional
        Path to save the agent. If None, uses the default path.
    verbose : bool, default True
        Whether to print progress
        
    Returns:
    --------
    dict
        Dictionary with training metrics
    """
    # Initialize metrics
    metrics = {
        'episode_returns': [],
        'episode_lengths': [],
        'portfolio_values': [],
        'sharpe_ratios': [],
        'max_drawdowns': [],
        'actor_losses': [],
        'critic_losses': [],
        'entropies': [],
        'kls': [],
        'clip_fractions': []
    }
    
    # Initialize replay buffer
    buffer = {
        'market_data': [],
        'portfolios': [],
        'actions': [],
        'rewards': [],
        'next_market_data': [],
        'next_portfolios': [],
        'dones': []
    }
    
    # Training loop
    progress_bar = tqdm(range(n_episodes)) if verbose else range(n_episodes)
    for episode in progress_bar:
        # Reset environment
        obs, info = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        # Episode loop
        while not done:
            # Select action
            market_data = obs['market_data']
            portfolio = obs['portfolio']
            action = agent.select_action(market_data, portfolio)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Store in buffer
            buffer['market_data'].append(market_data)
            buffer['portfolios'].append(portfolio)
            buffer['actions'].append(action)
            buffer['rewards'].append(reward)
            buffer['next_market_data'].append(next_obs['market_data'])
            buffer['next_portfolios'].append(next_obs['portfolio'])
            buffer['dones'].append(done)
            
            # Update metrics
            episode_return += reward
            episode_length += 1
            
            # Update observation
            obs = next_obs
        
        # Store episode metrics
        metrics['episode_returns'].append(episode_return)
        metrics['episode_lengths'].append(episode_length)
        metrics['portfolio_values'].append(info['portfolio_value'])
        metrics['sharpe_ratios'].append(info['sharpe_ratio'])
        metrics['max_drawdowns'].append(info['max_drawdown'])
        
        # Update progress bar
        if verbose:
            progress_bar.set_description(
                f"Episode {episode+1}/{n_episodes} | "
                f"Return: {episode_return:.4f} | "
                f"Value: {info['portfolio_value']:.2f} | "
                f"Sharpe: {info['sharpe_ratio']:.4f}"
            )
        
        # Update agent
        if (episode + 1) % update_freq == 0 and len(buffer['rewards']) >= batch_size:
            # Convert buffer to numpy arrays
            market_data = np.array(buffer['market_data'])
            portfolios = np.array(buffer['portfolios'])
            actions = np.array(buffer['actions'])
            rewards = np.array(buffer['rewards'])
            next_market_data = np.array(buffer['next_market_data'])
            next_portfolios = np.array(buffer['next_portfolios'])
            dones = np.array(buffer['dones'])
            
            # Update agent
            update_metrics = agent.update(
                market_data, portfolios, actions, rewards,
                next_market_data, next_portfolios, dones,
                batch_size=batch_size
            )
            
            # Store update metrics
            for key, value in update_metrics.items():
                metrics[key].append(value)
            
            # Clear buffer
            for key in buffer:
                buffer[key] = []
        
        # Save agent
        if (episode + 1) % save_freq == 0:
            if save_path is None:
                save_path = os.path.join(MODELS_DIR, f"ppo_agent_episode_{episode+1}")
            
            agent.save(save_path)
            logger.info(f"Saved agent to {save_path}")
    
    return metrics


def evaluate_agent(env, agent, n_episodes, verbose=True):
    """
    Evaluate the RL agent.
    
    Parameters:
    -----------
    env : PortfolioEnv
        Portfolio environment
    agent : PPOAgent
        RL agent
    n_episodes : int
        Number of episodes to evaluate
    verbose : bool, default True
        Whether to print progress
        
    Returns:
    --------
    dict
        Dictionary with evaluation metrics
    """
    # Initialize metrics
    metrics = {
        'episode_returns': [],
        'episode_lengths': [],
        'portfolio_values': [],
        'sharpe_ratios': [],
        'max_drawdowns': []
    }
    
    # Evaluation loop
    progress_bar = tqdm(range(n_episodes)) if verbose else range(n_episodes)
    for episode in progress_bar:
        # Reset environment
        obs, info = env.reset()
        done = False
        episode_return = 0
        episode_length = 0
        
        # Episode loop
        while not done:
            # Select action (deterministic)
            market_data = obs['market_data']
            portfolio = obs['portfolio']
            action = agent.select_action(market_data, portfolio, deterministic=True)
            
            # Take step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Update metrics
            episode_return += reward
            episode_length += 1
            
            # Update observation
            obs = next_obs
        
        # Store episode metrics
        metrics['episode_returns'].append(episode_return)
        metrics['episode_lengths'].append(episode_length)
        metrics['portfolio_values'].append(info['portfolio_value'])
        metrics['sharpe_ratios'].append(info['sharpe_ratio'])
        metrics['max_drawdowns'].append(info['max_drawdown'])
        
        # Update progress bar
        if verbose:
            progress_bar.set_description(
                f"Episode {episode+1}/{n_episodes} | "
                f"Return: {episode_return:.4f} | "
                f"Value: {info['portfolio_value']:.2f} | "
                f"Sharpe: {info['sharpe_ratio']:.4f}"
            )
    
    return metrics


def plot_metrics(metrics, title, save_path=None):
    """
    Plot training or evaluation metrics.
    
    Parameters:
    -----------
    metrics : dict
        Dictionary with metrics
    title : str
        Title for the plot
    save_path : str, optional
        Path to save the plot. If None, the plot is not saved.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16)
    
    # Plot episode returns
    axes[0, 0].plot(metrics['episode_returns'])
    axes[0, 0].set_title('Episode Returns')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Return')
    axes[0, 0].grid(True)
    
    # Plot portfolio values
    axes[0, 1].plot(metrics['portfolio_values'])
    axes[0, 1].set_title('Portfolio Values')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True)
    
    # Plot Sharpe ratios
    axes[1, 0].plot(metrics['sharpe_ratios'])
    axes[1, 0].set_title('Sharpe Ratios')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Sharpe Ratio')
    axes[1, 0].grid(True)
    
    # Plot max drawdowns
    axes[1, 1].plot(metrics['max_drawdowns'])
    axes[1, 1].set_title('Max Drawdowns')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Max Drawdown')
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path)
    
    plt.show()


def main():
    """
    Main function to train and evaluate the RL agent.
    """
    parser = argparse.ArgumentParser(description='Train RL agent for portfolio optimization')
    parser.add_argument('--model', type=str, choices=['ppo'], default='ppo',
                        help='RL model to use')
    parser.add_argument('--asset-types', type=str, nargs='+', 
                        choices=['stocks', 'etfs', 'crypto', 'all'], default=['all'],
                        help='Asset types to include')
    parser.add_argument('--timeframe', type=str, default='1d',
                        help='Timeframe of the data')
    parser.add_argument('--risk-level', type=str, 
                        choices=['conservative', 'balanced', 'aggressive'], 
                        default='balanced',
                        help='Risk level for the agent')
    parser.add_argument('--train-episodes', type=int, default=TRAINING_EPISODES,
                        help='Number of training episodes')
    parser.add_argument('--val-episodes', type=int, default=VALIDATION_EPISODES,
                        help='Number of validation episodes')
    parser.add_argument('--test-episodes', type=int, default=TEST_EPISODES,
                        help='Number of test episodes')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for updates')
    parser.add_argument('--update-freq', type=int, default=10,
                        help='Frequency of updates (in episodes)')
    parser.add_argument('--save-freq', type=int, default=100,
                        help='Frequency of saving the agent (in episodes)')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension of the networks')
    parser.add_argument('--lr-actor', type=float, default=3e-4,
                        help='Learning rate for the actor network')
    parser.add_argument('--lr-critic', type=float, default=1e-3,
                        help='Learning rate for the critic network')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Determine asset types
    if 'all' in args.asset_types:
        asset_types = ['stocks', 'etfs', 'crypto']
    else:
        asset_types = args.asset_types
    
    # Load data
    logger.info(f"Loading data for {asset_types}")
    data = load_data(asset_types, args.timeframe)
    logger.info(f"Loaded data with {len(data.index.get_level_values('symbol').unique())} assets")
    
    # Split data
    logger.info("Splitting data into train, validation, and test sets")
    train_data, val_data, test_data = split_data(data)
    logger.info(f"Train: {len(train_data)} samples, "
               f"Validation: {len(val_data)} samples, "
               f"Test: {len(test_data)} samples")
    
    # Create environments
    logger.info("Creating environments")
    train_env = PortfolioEnv(
        price_data=train_data,
        lookback_window=LOOKBACK_WINDOW,
        risk_level=args.risk_level,
        random_start=True
    )
    val_env = PortfolioEnv(
        price_data=val_data,
        lookback_window=LOOKBACK_WINDOW,
        risk_level=args.risk_level,
        random_start=True
    )
    test_env = PortfolioEnv(
        price_data=test_data,
        lookback_window=LOOKBACK_WINDOW,
        risk_level=args.risk_level,
        random_start=False
    )
    
    # Determine device
    device = 'cpu' if args.no_cuda else None
    
    # Create agent
    logger.info(f"Creating {args.model.upper()} agent")
    if args.model == 'ppo':
        agent = PPOAgent(
            lookback_window=LOOKBACK_WINDOW,
            n_assets=train_env.n_assets,
            n_features=len(train_env.feature_columns),
            hidden_dim=args.hidden_dim,
            lr_actor=args.lr_actor,
            lr_critic=args.lr_critic,
            gamma=args.gamma,
            device=device
        )
    else:
        raise ValueError(f"Unsupported model: {args.model}")
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(
        MODELS_DIR, 
        f"{args.model}_{args.risk_level}_{timestamp}"
    )
    os.makedirs(save_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    config['n_assets'] = train_env.n_assets
    config['n_features'] = len(train_env.feature_columns)
    config['symbols'] = train_env.symbols
    config['feature_columns'] = train_env.feature_columns
    
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    
    # Train agent
    logger.info(f"Training agent for {args.train_episodes} episodes")
    train_metrics = train_agent(
        env=train_env,
        agent=agent,
        n_episodes=args.train_episodes,
        batch_size=args.batch_size,
        update_freq=args.update_freq,
        save_freq=args.save_freq,
        save_path=os.path.join(save_dir, 'agent')
    )
    
    # Save training metrics
    with open(os.path.join(save_dir, 'train_metrics.json'), 'w') as f:
        json.dump(train_metrics, f, indent=4)
    
    # Plot training metrics
    plot_metrics(
        metrics=train_metrics,
        title=f"Training Metrics ({args.model.upper()}, {args.risk_level})",
        save_path=os.path.join(save_dir, 'train_metrics.png')
    )
    
    # Evaluate on validation set
    logger.info(f"Evaluating agent on validation set for {args.val_episodes} episodes")
    val_metrics = evaluate_agent(
        env=val_env,
        agent=agent,
        n_episodes=args.val_episodes
    )
    
    # Save validation metrics
    with open(os.path.join(save_dir, 'val_metrics.json'), 'w') as f:
        json.dump(val_metrics, f, indent=4)
    
    # Plot validation metrics
    plot_metrics(
        metrics=val_metrics,
        title=f"Validation Metrics ({args.model.upper()}, {args.risk_level})",
        save_path=os.path.join(save_dir, 'val_metrics.png')
    )
    
    # Evaluate on test set
    logger.info(f"Evaluating agent on test set for {args.test_episodes} episodes")
    test_metrics = evaluate_agent(
        env=test_env,
        agent=agent,
        n_episodes=args.test_episodes
    )
    
    # Save test metrics
    with open(os.path.join(save_dir, 'test_metrics.json'), 'w') as f:
        json.dump(test_metrics, f, indent=4)
    
    # Plot test metrics
    plot_metrics(
        metrics=test_metrics,
        title=f"Test Metrics ({args.model.upper()}, {args.risk_level})",
        save_path=os.path.join(save_dir, 'test_metrics.png')
    )
    
    logger.info(f"Training and evaluation completed. Results saved to {save_dir}")


if __name__ == '__main__':
    main() 