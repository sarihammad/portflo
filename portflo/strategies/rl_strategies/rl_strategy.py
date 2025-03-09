"""
RL strategy wrapper for backtesting.
"""
import os
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any

from portflo.models.agents.ppo_agent import PPOAgent
from portflo.data.processors.data_processor import DataProcessor
from portflo.config.settings import LOOKBACK_WINDOW


class RLStrategy:
    """
    RL strategy wrapper for backtesting.
    
    This class wraps a trained RL agent to be used in the backtesting framework.
    It provides the allocate and rebalance methods required by the Backtest class.
    """
    
    def __init__(
        self,
        model_path: str,
        lookback_window: int = LOOKBACK_WINDOW,
        feature_columns: Optional[List[str]] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the RL strategy.
        
        Parameters:
        -----------
        model_path : str
            Path to the trained RL model
        lookback_window : int, default from settings
            Number of days to look back for state representation
        feature_columns : list, optional
            List of columns to use as features. If None, uses a default set of features.
        device : str, optional
            Device to use for computation ('cpu' or 'cuda')
        """
        self.model_path = model_path
        self.lookback_window = lookback_window
        self.feature_columns = feature_columns or [
            'close', 'daily_return', 'macd', 'rsi', 
            'bb_width', 'volatility_20', 'rolling_sharpe_20'
        ]
        self.device = device
        
        # Load model configuration
        self.config = self._load_config()
        
        # Initialize agent
        self.agent = self._initialize_agent()
        
        # Initialize data processor
        self.processor = DataProcessor(lookback_window=lookback_window)
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load model configuration.
        
        Returns:
        --------
        dict
            Model configuration
        """
        # Check if model path is a directory or a file
        if os.path.isdir(self.model_path):
            # Look for config.json in the directory
            config_path = os.path.join(self.model_path, 'config.json')
            if os.path.exists(config_path):
                import json
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return config
        
        # If no config file found, return default configuration
        return {
            'n_assets': None,  # Will be determined dynamically
            'n_features': len(self.feature_columns),
            'hidden_dim': 128,
            'lookback_window': self.lookback_window
        }
    
    def _initialize_agent(self) -> PPOAgent:
        """
        Initialize the RL agent.
        
        Returns:
        --------
        PPOAgent
            Initialized RL agent
        """
        # Determine model file path
        if os.path.isdir(self.model_path):
            # Look for agent file in the directory
            for file in os.listdir(self.model_path):
                if file.startswith('agent') and not file.endswith('.json'):
                    model_file = os.path.join(self.model_path, file)
                    break
            else:
                # If no agent file found, use the directory itself
                model_file = self.model_path
        else:
            # Use the provided path directly
            model_file = self.model_path
        
        # Create agent
        agent = PPOAgent(
            lookback_window=self.config.get('lookback_window', self.lookback_window),
            n_assets=self.config.get('n_assets', 10),  # Placeholder, will be updated
            n_features=self.config.get('n_features', len(self.feature_columns)),
            hidden_dim=self.config.get('hidden_dim', 128),
            device=self.device
        )
        
        # Load trained weights
        agent.load(model_file)
        
        return agent
    
    def _prepare_state(
        self,
        returns: pd.DataFrame,
        current_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare state for the RL agent.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with asset returns
        current_weights : dict, optional
            Current asset weights
            
        Returns:
        --------
        tuple
            (market_data, portfolio) where market_data is a numpy array of shape
            (lookback_window, n_assets, n_features) and portfolio is a numpy array
            of shape (n_assets + 1,)
        """
        # Get asset names
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Update agent if number of assets changed
        if self.agent.actor.mean_layer.out_features != n_assets + 1:
            # Reinitialize agent with correct number of assets
            self.config['n_assets'] = n_assets
            self.agent = self._initialize_agent()
        
        # Process returns to get features
        # First, convert returns to price data (starting with 100)
        price_data = (1 + returns).cumprod() * 100
        
        # Add required columns for feature engineering
        price_data['close'] = price_data.iloc[:, 0]  # Use first asset as close price
        price_data['volume'] = 1000000  # Dummy volume
        
        # Process data
        processed_data = self.processor.process_data(price_data)
        
        # Extract features for each asset
        market_data = np.zeros((self.lookback_window, n_assets, len(self.feature_columns)))
        
        for i, asset in enumerate(assets):
            # Get asset data
            asset_data = processed_data[self.feature_columns].values
            
            # Fill market data array
            if len(asset_data) >= self.lookback_window:
                market_data[:, i, :] = asset_data[-self.lookback_window:]
            else:
                # Pad with zeros if not enough data
                market_data[-len(asset_data):, i, :] = asset_data
        
        # Create portfolio array
        if current_weights is None:
            # Initialize with equal weights
            portfolio = np.ones(n_assets + 1) / (n_assets + 1)
        else:
            # Use current weights
            portfolio = np.zeros(n_assets + 1)
            for i, asset in enumerate(assets):
                portfolio[i] = current_weights.get(asset, 0.0)
            portfolio[-1] = current_weights.get('cash', 0.0)
        
        return market_data, portfolio
    
    def allocate(
        self,
        assets: List[str],
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Allocate portfolio weights using the RL agent.
        
        Parameters:
        -----------
        assets : list
            List of asset names
        returns : pandas.DataFrame, optional
            DataFrame with asset returns
            
        Returns:
        --------
        dict
            Dictionary with allocation results:
            - 'weights': Asset weights
        """
        if returns is None or len(returns) < self.lookback_window:
            # Not enough data, use equal weights
            weights = {asset: 1.0 / (len(assets) + 1) for asset in assets}
            weights['cash'] = 1.0 / (len(assets) + 1)
            return {'weights': weights}
        
        # Prepare state
        market_data, portfolio = self._prepare_state(returns)
        
        # Get action from agent (deterministic)
        action = self.agent.select_action(market_data, portfolio, deterministic=True)
        
        # Convert action to weights
        weights = {asset: action[i] for i, asset in enumerate(assets)}
        weights['cash'] = action[-1]
        
        return {'weights': weights}
    
    def rebalance(
        self,
        current_weights: Dict[str, float],
        assets: List[str],
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Rebalance portfolio weights using the RL agent.
        
        Parameters:
        -----------
        current_weights : dict
            Current asset weights
        assets : list
            List of asset names
        returns : pandas.DataFrame, optional
            DataFrame with asset returns
            
        Returns:
        --------
        dict
            Dictionary with allocation results (same as allocate method)
        """
        if returns is None or len(returns) < self.lookback_window:
            # Not enough data, keep current weights
            return {'weights': current_weights}
        
        # Prepare state
        market_data, portfolio = self._prepare_state(returns, current_weights)
        
        # Get action from agent (deterministic)
        action = self.agent.select_action(market_data, portfolio, deterministic=True)
        
        # Convert action to weights
        weights = {asset: action[i] for i, asset in enumerate(assets)}
        weights['cash'] = action[-1]
        
        return {'weights': weights} 