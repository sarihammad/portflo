"""
Custom Gym environment for portfolio optimization.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, List, Tuple, Optional, Union

from portflo.config.settings import RISK_LEVELS, INITIAL_PORTFOLIO_VALUE, TRANSACTION_FEE_RATE


class PortfolioEnv(gym.Env):
    """
    A custom Gym environment for portfolio optimization using reinforcement learning.
    
    This environment simulates a portfolio of assets where an agent can reallocate
    funds between different assets to maximize returns while managing risk.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        lookback_window: int = 30,
        max_steps: int = None,
        initial_cash: float = INITIAL_PORTFOLIO_VALUE,
        transaction_fee_rate: float = TRANSACTION_FEE_RATE,
        risk_level: str = 'balanced',
        feature_columns: List[str] = None,
        reward_scaling: float = 1.0,
        normalize_state: bool = True,
        random_start: bool = True
    ):
        """
        Initialize the portfolio environment.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame with price data for multiple assets. Must have a MultiIndex with (symbol, date)
            and columns including at least 'close' and other features.
        lookback_window : int, default 30
            Number of days to look back for state representation
        max_steps : int, optional
            Maximum number of steps in an episode. If None, uses the length of the data.
        initial_cash : float, default 100000
            Initial cash amount
        transaction_fee_rate : float, default 0.001
            Transaction fee rate (e.g., 0.001 = 0.1%)
        risk_level : str, default 'balanced'
            Risk level ('conservative', 'balanced', or 'aggressive')
        feature_columns : list, optional
            List of columns to use as features. If None, uses a default set of features.
        reward_scaling : float, default 1.0
            Scaling factor for rewards
        normalize_state : bool, default True
            Whether to normalize the state
        random_start : bool, default True
            Whether to start at a random point in the data
        """
        super(PortfolioEnv, self).__init__()
        
        # Store parameters
        self.price_data = price_data
        self.lookback_window = lookback_window
        self.initial_cash = initial_cash
        self.transaction_fee_rate = transaction_fee_rate
        self.reward_scaling = reward_scaling
        self.normalize_state = normalize_state
        self.random_start = random_start
        
        # Get risk parameters
        if risk_level not in RISK_LEVELS:
            raise ValueError(f"Invalid risk level: {risk_level}. Must be one of {list(RISK_LEVELS.keys())}")
        self.risk_params = RISK_LEVELS[risk_level]
        
        # Extract unique symbols
        self.symbols = price_data.index.get_level_values('symbol').unique().tolist()
        self.n_assets = len(self.symbols)
        
        # Extract dates and convert to list for easier indexing
        self.dates = price_data.index.get_level_values('date').unique().tolist()
        self.dates.sort()  # Ensure dates are sorted
        
        # Set maximum steps
        if max_steps is None:
            self.max_steps = len(self.dates) - self.lookback_window - 1
        else:
            self.max_steps = min(max_steps, len(self.dates) - self.lookback_window - 1)
        
        # Set default feature columns if not provided
        if feature_columns is None:
            self.feature_columns = [
                'close', 'daily_return', 'macd', 'rsi', 
                'bb_width', 'volatility_20', 'rolling_sharpe_20'
            ]
        else:
            self.feature_columns = feature_columns
        
        # Check if all feature columns exist in the DataFrame
        missing_columns = [col for col in self.feature_columns if col not in price_data.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")
        
        # Define action and observation spaces
        # Action space: allocation percentage for each asset (including cash)
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )
        
        # Observation space: historical price data + current portfolio allocation
        # Each asset has multiple features for each day in the lookback window
        # Plus the current portfolio allocation
        n_features = len(self.feature_columns)
        self.observation_space = spaces.Dict({
            'market_data': spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.lookback_window, self.n_assets, n_features), 
                dtype=np.float32
            ),
            'portfolio': spaces.Box(
                low=0, high=np.inf, shape=(self.n_assets + 1,), dtype=np.float32
            )
        })
        
        # Initialize state variables
        self.reset()
    
    def reset(self, seed=None, options=None):
        """
        Reset the environment to start a new episode.
        
        Parameters:
        -----------
        seed : int, optional
            Random seed
        options : dict, optional
            Additional options
            
        Returns:
        --------
        tuple
            (observation, info)
        """
        super().reset(seed=seed)
        
        # Set the current step
        if self.random_start and self.max_steps < len(self.dates) - self.lookback_window - 1:
            # Start at a random point in the data
            self.current_step = self.np_random.integers(
                0, len(self.dates) - self.lookback_window - self.max_steps - 1
            )
        else:
            # Start at the beginning
            self.current_step = 0
        
        # Initialize portfolio: all cash
        self.portfolio = np.zeros(self.n_assets + 1, dtype=np.float32)
        self.portfolio[-1] = self.initial_cash  # Last element is cash
        
        # Initialize portfolio value history
        self.portfolio_value_history = [self.initial_cash]
        
        # Get the current observation
        observation = self._get_observation()
        
        # Initialize info
        info = {
            'portfolio_value': self.initial_cash,
            'portfolio_return': 0.0,
            'transaction_cost': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'current_date': self.dates[self.current_step + self.lookback_window]
        }
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Parameters:
        -----------
        action : numpy.ndarray
            Action to take (allocation percentages for each asset including cash)
            
        Returns:
        --------
        tuple
            (observation, reward, terminated, truncated, info)
        """
        # Ensure action sums to 1
        action = np.clip(action, 0, 1)
        action = action / np.sum(action)
        
        # Get current portfolio value
        current_portfolio_value = np.sum(self.portfolio)
        
        # Get current prices
        current_date = self.dates[self.current_step + self.lookback_window]
        current_prices = self._get_prices(current_date)
        
        # Calculate current allocation (before rebalancing)
        current_allocation = self.portfolio.copy()
        current_allocation[:-1] = current_allocation[:-1] * current_prices  # Convert shares to value
        current_allocation = current_allocation / np.sum(current_allocation)
        
        # Calculate target allocation
        target_allocation = action * current_portfolio_value
        
        # Calculate transaction costs
        transaction_cost = self._calculate_transaction_cost(
            current_allocation, target_allocation, current_portfolio_value
        )
        
        # Update portfolio after transaction costs
        self.portfolio = self._rebalance_portfolio(
            target_allocation, current_prices, transaction_cost
        )
        
        # Move to the next day
        self.current_step += 1
        
        # Get new prices
        next_date = self.dates[self.current_step + self.lookback_window]
        next_prices = self._get_prices(next_date)
        
        # Update portfolio value based on new prices
        self.portfolio[:-1] = self.portfolio[:-1] * next_prices / current_prices
        
        # Calculate new portfolio value
        new_portfolio_value = np.sum(self.portfolio)
        self.portfolio_value_history.append(new_portfolio_value)
        
        # Calculate return
        portfolio_return = (new_portfolio_value - current_portfolio_value) / current_portfolio_value
        
        # Calculate reward
        reward = self._calculate_reward(portfolio_return, transaction_cost)
        
        # Check if episode is done
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        # Get new observation
        observation = self._get_observation()
        
        # Calculate additional metrics for info
        sharpe_ratio = self._calculate_sharpe_ratio()
        max_drawdown = self._calculate_max_drawdown()
        
        # Prepare info
        info = {
            'portfolio_value': new_portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_cost': transaction_cost,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'current_date': next_date
        }
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self):
        """
        Get the current observation.
        
        Returns:
        --------
        dict
            Observation dictionary with market data and portfolio allocation
        """
        # Get the current date range for the lookback window
        start_idx = self.current_step
        end_idx = self.current_step + self.lookback_window
        date_range = self.dates[start_idx:end_idx]
        
        # Initialize market data array
        market_data = np.zeros(
            (self.lookback_window, self.n_assets, len(self.feature_columns)), 
            dtype=np.float32
        )
        
        # Fill market data array
        for i, date in enumerate(date_range):
            for j, symbol in enumerate(self.symbols):
                try:
                    # Get data for this symbol and date
                    data = self.price_data.loc[(symbol, date), self.feature_columns].values
                    market_data[i, j, :] = data
                except KeyError:
                    # If data is missing, use the previous day's data
                    if i > 0:
                        market_data[i, j, :] = market_data[i-1, j, :]
        
        # Normalize market data if requested
        if self.normalize_state:
            market_data = self._normalize_market_data(market_data)
        
        # Get current portfolio allocation
        portfolio_allocation = self.portfolio.copy()
        
        # Create observation dictionary
        observation = {
            'market_data': market_data.astype(np.float32),
            'portfolio': portfolio_allocation.astype(np.float32)
        }
        
        return observation
    
    def _normalize_market_data(self, market_data):
        """
        Normalize market data using z-score normalization.
        
        Parameters:
        -----------
        market_data : numpy.ndarray
            Market data array of shape (lookback_window, n_assets, n_features)
            
        Returns:
        --------
        numpy.ndarray
            Normalized market data
        """
        # Reshape to (lookback_window * n_assets, n_features)
        shape = market_data.shape
        reshaped = market_data.reshape(-1, shape[2])
        
        # Normalize each feature
        for i in range(shape[2]):
            mean = np.mean(reshaped[:, i])
            std = np.std(reshaped[:, i])
            if std > 0:
                reshaped[:, i] = (reshaped[:, i] - mean) / std
        
        # Reshape back to original shape
        normalized = reshaped.reshape(shape)
        
        return normalized
    
    def _get_prices(self, date):
        """
        Get the closing prices for all assets on the given date.
        
        Parameters:
        -----------
        date : datetime
            Date to get prices for
            
        Returns:
        --------
        numpy.ndarray
            Array of closing prices for each asset
        """
        prices = np.zeros(self.n_assets, dtype=np.float32)
        
        for i, symbol in enumerate(self.symbols):
            try:
                prices[i] = self.price_data.loc[(symbol, date), 'close']
            except KeyError:
                # If price is missing, use the last known price
                prev_dates = [d for d in self.dates if d < date]
                if prev_dates:
                    last_date = max(prev_dates)
                    try:
                        prices[i] = self.price_data.loc[(symbol, last_date), 'close']
                    except KeyError:
                        prices[i] = 1.0  # Default price if no data is available
                else:
                    prices[i] = 1.0  # Default price if no data is available
        
        return prices
    
    def _calculate_transaction_cost(self, current_allocation, target_allocation, portfolio_value):
        """
        Calculate the transaction cost for rebalancing the portfolio.
        
        Parameters:
        -----------
        current_allocation : numpy.ndarray
            Current allocation in value for each asset (including cash)
        target_allocation : numpy.ndarray
            Target allocation in value for each asset (including cash)
        portfolio_value : float
            Current portfolio value
            
        Returns:
        --------
        float
            Transaction cost
        """
        # Calculate the absolute difference in allocation
        allocation_diff = np.abs(target_allocation - current_allocation)
        
        # Calculate transaction cost (excluding cash)
        transaction_cost = np.sum(allocation_diff[:-1]) * self.transaction_fee_rate
        
        return transaction_cost
    
    def _rebalance_portfolio(self, target_allocation, prices, transaction_cost):
        """
        Rebalance the portfolio according to the target allocation.
        
        Parameters:
        -----------
        target_allocation : numpy.ndarray
            Target allocation in value for each asset (including cash)
        prices : numpy.ndarray
            Current prices for each asset
        transaction_cost : float
            Transaction cost
            
        Returns:
        --------
        numpy.ndarray
            New portfolio allocation in shares/value
        """
        # Adjust target allocation for transaction cost
        adjusted_target = target_allocation.copy()
        adjusted_target[-1] -= transaction_cost  # Deduct transaction cost from cash
        
        # Calculate new portfolio
        new_portfolio = np.zeros_like(self.portfolio)
        
        # Convert value to shares for assets
        new_portfolio[:-1] = adjusted_target[:-1] / prices
        
        # Set cash
        new_portfolio[-1] = adjusted_target[-1]
        
        return new_portfolio
    
    def _calculate_reward(self, portfolio_return, transaction_cost):
        """
        Calculate the reward for the current step.
        
        Parameters:
        -----------
        portfolio_return : float
            Portfolio return for the current step
        transaction_cost : float
            Transaction cost for the current step
            
        Returns:
        --------
        float
            Reward
        """
        # Calculate base reward from return
        reward = portfolio_return
        
        # Penalize for transaction costs
        reward -= transaction_cost / self.initial_cash
        
        # Penalize for risk (if portfolio return is negative)
        if portfolio_return < 0:
            risk_penalty = portfolio_return * self.risk_params['risk_penalty_factor']
            reward += risk_penalty  # This will make the reward more negative
        
        # Scale reward
        reward *= self.reward_scaling
        
        return reward
    
    def _calculate_sharpe_ratio(self, window=20):
        """
        Calculate the Sharpe ratio over a window of returns.
        
        Parameters:
        -----------
        window : int, default 20
            Window size for calculating Sharpe ratio
            
        Returns:
        --------
        float
            Sharpe ratio
        """
        if len(self.portfolio_value_history) <= 1:
            return 0.0
        
        # Calculate returns
        values = np.array(self.portfolio_value_history)
        returns = np.diff(values) / values[:-1]
        
        # Use only the last 'window' returns
        returns = returns[-min(window, len(returns)):]
        
        if len(returns) == 0:
            return 0.0
        
        # Calculate Sharpe ratio (annualized, assuming 252 trading days)
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        sharpe_ratio = mean_return / std_return * np.sqrt(252)
        
        return sharpe_ratio
    
    def _calculate_max_drawdown(self):
        """
        Calculate the maximum drawdown.
        
        Returns:
        --------
        float
            Maximum drawdown
        """
        if len(self.portfolio_value_history) <= 1:
            return 0.0
        
        # Calculate drawdown
        values = np.array(self.portfolio_value_history)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        # Calculate maximum drawdown
        max_drawdown = np.min(drawdown)
        
        return max_drawdown
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Parameters:
        -----------
        mode : str, default 'human'
            Rendering mode
        """
        if mode != 'human':
            raise NotImplementedError(f"Rendering mode {mode} is not implemented")
        
        # Get current date
        current_date = self.dates[self.current_step + self.lookback_window]
        
        # Get current portfolio value
        portfolio_value = np.sum(self.portfolio)
        
        # Get current allocation
        current_prices = self._get_prices(current_date)
        allocation = self.portfolio.copy()
        allocation[:-1] = allocation[:-1] * current_prices  # Convert shares to value
        allocation_pct = allocation / portfolio_value * 100
        
        # Print information
        print(f"Date: {current_date}")
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Allocation:")
        for i, symbol in enumerate(self.symbols):
            print(f"  {symbol}: {allocation_pct[i]:.2f}%")
        print(f"  Cash: {allocation_pct[-1]:.2f}%")
        print(f"Sharpe Ratio: {self._calculate_sharpe_ratio():.4f}")
        print(f"Max Drawdown: {self._calculate_max_drawdown():.4f}")
        print("-" * 50)
    
    def close(self):
        """
        Close the environment.
        """
        pass 