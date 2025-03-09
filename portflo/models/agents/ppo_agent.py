"""
PPO (Proximal Policy Optimization) agent for portfolio optimization.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
from typing import Dict, List, Tuple, Optional, Union, Any

from portflo.config.settings import MODELS_DIR


class FeatureExtractor(nn.Module):
    """
    Neural network for extracting features from market data.
    """
    
    def __init__(self, lookback_window: int, n_assets: int, n_features: int, hidden_dim: int = 128):
        """
        Initialize the feature extractor.
        
        Parameters:
        -----------
        lookback_window : int
            Number of days to look back for state representation
        n_assets : int
            Number of assets in the portfolio
        n_features : int
            Number of features per asset
        hidden_dim : int, default 128
            Hidden dimension of the network
        """
        super(FeatureExtractor, self).__init__()
        
        # CNN for extracting temporal features
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.ReLU(),
            nn.Flatten()
        )
        
        # Calculate CNN output dimension
        cnn_output_dim = 32 * lookback_window * n_assets
        
        # MLP for combining features
        self.mlp = nn.Sequential(
            nn.Linear(cnn_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, market_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Parameters:
        -----------
        market_data : torch.Tensor
            Market data tensor of shape (batch_size, lookback_window, n_assets, n_features)
            
        Returns:
        --------
        torch.Tensor
            Extracted features of shape (batch_size, hidden_dim)
        """
        batch_size, lookback, n_assets, n_features = market_data.shape
        
        # Reshape for CNN: (batch_size, 1, lookback_window * n_assets, n_features)
        x = market_data.view(batch_size, 1, lookback * n_assets, n_features)
        
        # Pass through CNN
        x = self.cnn(x)
        
        # Pass through MLP
        x = self.mlp(x)
        
        return x


class ActorNetwork(nn.Module):
    """
    Actor network for the PPO agent.
    """
    
    def __init__(
        self, 
        lookback_window: int, 
        n_assets: int, 
        n_features: int, 
        hidden_dim: int = 128,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0
    ):
        """
        Initialize the actor network.
        
        Parameters:
        -----------
        lookback_window : int
            Number of days to look back for state representation
        n_assets : int
            Number of assets in the portfolio
        n_features : int
            Number of features per asset
        hidden_dim : int, default 128
            Hidden dimension of the network
        log_std_min : float, default -20.0
            Minimum log standard deviation
        log_std_max : float, default 2.0
            Maximum log standard deviation
        """
        super(ActorNetwork, self).__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(lookback_window, n_assets, n_features, hidden_dim)
        
        # Portfolio encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(n_assets + 1, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        
        # Mean and log_std layers
        self.mean_layer = nn.Linear(hidden_dim, n_assets + 1)
        self.log_std_layer = nn.Linear(hidden_dim, n_assets + 1)
        
        # Bounds for log_std
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
    
    def forward(self, market_data: torch.Tensor, portfolio: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the actor network.
        
        Parameters:
        -----------
        market_data : torch.Tensor
            Market data tensor of shape (batch_size, lookback_window, n_assets, n_features)
        portfolio : torch.Tensor
            Portfolio tensor of shape (batch_size, n_assets + 1)
            
        Returns:
        --------
        tuple
            (mean, log_std) of the action distribution
        """
        # Extract features from market data
        market_features = self.feature_extractor(market_data)
        
        # Encode portfolio
        portfolio_features = self.portfolio_encoder(portfolio)
        
        # Combine features
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
        combined_features = self.combined_encoder(combined_features)
        
        # Calculate mean and log_std
        mean = self.mean_layer(combined_features)
        log_std = self.log_std_layer(combined_features)
        
        # Clamp log_std
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample_action(
        self, 
        market_data: torch.Tensor, 
        portfolio: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy.
        
        Parameters:
        -----------
        market_data : torch.Tensor
            Market data tensor of shape (batch_size, lookback_window, n_assets, n_features)
        portfolio : torch.Tensor
            Portfolio tensor of shape (batch_size, n_assets + 1)
        deterministic : bool, default False
            Whether to sample deterministically (use mean)
            
        Returns:
        --------
        tuple
            (action, log_prob) where action is the sampled action and log_prob is its log probability
        """
        mean, log_std = self.forward(market_data, portfolio)
        std = log_std.exp()
        
        if deterministic:
            # Return mean action
            action = mean
            log_prob = None
        else:
            # Sample from Normal distribution
            normal = Normal(mean, std)
            action = normal.rsample()  # Reparameterization trick
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        
        # Apply softmax to ensure action sums to 1 and is non-negative
        action = torch.softmax(action, dim=-1)
        
        return action, log_prob


class CriticNetwork(nn.Module):
    """
    Critic network for the PPO agent.
    """
    
    def __init__(self, lookback_window: int, n_assets: int, n_features: int, hidden_dim: int = 128):
        """
        Initialize the critic network.
        
        Parameters:
        -----------
        lookback_window : int
            Number of days to look back for state representation
        n_assets : int
            Number of assets in the portfolio
        n_features : int
            Number of features per asset
        hidden_dim : int, default 128
            Hidden dimension of the network
        """
        super(CriticNetwork, self).__init__()
        
        # Feature extractor
        self.feature_extractor = FeatureExtractor(lookback_window, n_assets, n_features, hidden_dim)
        
        # Portfolio encoder
        self.portfolio_encoder = nn.Sequential(
            nn.Linear(n_assets + 1, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Combined encoder
        self.combined_encoder = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim),
            nn.ReLU()
        )
        
        # Value layer
        self.value_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, market_data: torch.Tensor, portfolio: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic network.
        
        Parameters:
        -----------
        market_data : torch.Tensor
            Market data tensor of shape (batch_size, lookback_window, n_assets, n_features)
        portfolio : torch.Tensor
            Portfolio tensor of shape (batch_size, n_assets + 1)
            
        Returns:
        --------
        torch.Tensor
            Value estimate of shape (batch_size, 1)
        """
        # Extract features from market data
        market_features = self.feature_extractor(market_data)
        
        # Encode portfolio
        portfolio_features = self.portfolio_encoder(portfolio)
        
        # Combine features
        combined_features = torch.cat([market_features, portfolio_features], dim=1)
        combined_features = self.combined_encoder(combined_features)
        
        # Calculate value
        value = self.value_layer(combined_features)
        
        return value


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) agent for portfolio optimization.
    """
    
    def __init__(
        self,
        lookback_window: int,
        n_assets: int,
        n_features: int,
        hidden_dim: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = None
    ):
        """
        Initialize the PPO agent.
        
        Parameters:
        -----------
        lookback_window : int
            Number of days to look back for state representation
        n_assets : int
            Number of assets in the portfolio
        n_features : int
            Number of features per asset
        hidden_dim : int, default 128
            Hidden dimension of the networks
        lr_actor : float, default 3e-4
            Learning rate for the actor network
        lr_critic : float, default 1e-3
            Learning rate for the critic network
        gamma : float, default 0.99
            Discount factor
        gae_lambda : float, default 0.95
            GAE lambda parameter
        clip_ratio : float, default 0.2
            PPO clip ratio
        target_kl : float, default 0.01
            Target KL divergence
        value_coef : float, default 0.5
            Value loss coefficient
        entropy_coef : float, default 0.01
            Entropy coefficient
        device : str, optional
            Device to use for computation ('cpu' or 'cuda')
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Initialize networks
        self.actor = ActorNetwork(lookback_window, n_assets, n_features, hidden_dim).to(self.device)
        self.critic = CriticNetwork(lookback_window, n_assets, n_features, hidden_dim).to(self.device)
        
        # Initialize optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Store hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
        # Initialize training variables
        self.n_updates = 0
    
    def select_action(
        self, 
        market_data: np.ndarray, 
        portfolio: np.ndarray, 
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select an action from the policy.
        
        Parameters:
        -----------
        market_data : numpy.ndarray
            Market data array of shape (lookback_window, n_assets, n_features)
        portfolio : numpy.ndarray
            Portfolio array of shape (n_assets + 1,)
        deterministic : bool, default False
            Whether to select deterministically (use mean)
            
        Returns:
        --------
        numpy.ndarray
            Selected action
        """
        # Convert to tensors
        market_data = torch.FloatTensor(market_data).unsqueeze(0).to(self.device)
        portfolio = torch.FloatTensor(portfolio).unsqueeze(0).to(self.device)
        
        # Set networks to evaluation mode
        self.actor.eval()
        
        with torch.no_grad():
            # Sample action
            action, _ = self.actor.sample_action(market_data, portfolio, deterministic)
        
        # Convert to numpy
        action = action.cpu().numpy()[0]
        
        return action
    
    def update(
        self,
        market_data: np.ndarray,
        portfolios: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_market_data: np.ndarray,
        next_portfolios: np.ndarray,
        dones: np.ndarray,
        n_epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, float]:
        """
        Update the agent's networks.
        
        Parameters:
        -----------
        market_data : numpy.ndarray
            Market data array of shape (batch_size, lookback_window, n_assets, n_features)
        portfolios : numpy.ndarray
            Portfolio array of shape (batch_size, n_assets + 1)
        actions : numpy.ndarray
            Action array of shape (batch_size, n_assets + 1)
        rewards : numpy.ndarray
            Reward array of shape (batch_size,)
        next_market_data : numpy.ndarray
            Next market data array of shape (batch_size, lookback_window, n_assets, n_features)
        next_portfolios : numpy.ndarray
            Next portfolio array of shape (batch_size, n_assets + 1)
        dones : numpy.ndarray
            Done array of shape (batch_size,)
        n_epochs : int, default 10
            Number of epochs to update
        batch_size : int, default 64
            Batch size for updates
            
        Returns:
        --------
        dict
            Dictionary with training metrics
        """
        # Convert to tensors
        market_data = torch.FloatTensor(market_data).to(self.device)
        portfolios = torch.FloatTensor(portfolios).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_market_data = torch.FloatTensor(next_market_data).to(self.device)
        next_portfolios = torch.FloatTensor(next_portfolios).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        
        # Calculate advantages and returns
        with torch.no_grad():
            # Calculate values
            values = self.critic(market_data, portfolios)
            next_values = self.critic(next_market_data, next_portfolios)
            
            # Calculate advantages using GAE
            advantages = torch.zeros_like(rewards)
            returns = torch.zeros_like(rewards)
            gae = 0
            
            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = next_values[t]
                else:
                    next_non_terminal = 1.0 - dones[t]
                    next_value = values[t + 1]
                
                delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
                gae = delta + self.gamma * self.gae_lambda * next_non_terminal * gae
                advantages[t] = gae
            
            # Calculate returns
            returns = advantages + values
            
            # Normalize advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get old action probabilities
        with torch.no_grad():
            mean, log_std = self.actor(market_data, portfolios)
            std = log_std.exp()
            old_dist = Normal(mean, std)
            old_log_probs = old_dist.log_prob(actions).sum(dim=-1, keepdim=True)
        
        # Update networks for n_epochs
        metrics = {
            'actor_loss': 0,
            'critic_loss': 0,
            'entropy': 0,
            'kl': 0,
            'clip_fraction': 0
        }
        
        for _ in range(n_epochs):
            # Generate random indices
            indices = np.random.permutation(len(rewards))
            
            # Update in batches
            for start_idx in range(0, len(rewards), batch_size):
                # Get batch indices
                batch_indices = indices[start_idx:start_idx + batch_size]
                
                # Get batch data
                batch_market_data = market_data[batch_indices]
                batch_portfolios = portfolios[batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                
                # Calculate new log probabilities
                mean, log_std = self.actor(batch_market_data, batch_portfolios)
                std = log_std.exp()
                new_dist = Normal(mean, std)
                new_log_probs = new_dist.log_prob(batch_actions).sum(dim=-1, keepdim=True)
                
                # Calculate entropy
                entropy = new_dist.entropy().mean()
                
                # Calculate KL divergence
                kl = torch.mean(
                    batch_old_log_probs - new_log_probs
                )
                
                # Calculate ratio
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate surrogate losses
                surrogate1 = ratio * batch_advantages
                surrogate2 = torch.clamp(
                    ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio
                ) * batch_advantages
                
                # Calculate actor loss
                actor_loss = -torch.min(surrogate1, surrogate2).mean()
                
                # Calculate critic loss
                values = self.critic(batch_market_data, batch_portfolios)
                critic_loss = ((values - batch_returns) ** 2).mean()
                
                # Calculate total loss
                loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                # Update actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update critic
                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()
                
                # Update metrics
                metrics['actor_loss'] += actor_loss.item()
                metrics['critic_loss'] += critic_loss.item()
                metrics['entropy'] += entropy.item()
                metrics['kl'] += kl.item()
                metrics['clip_fraction'] += (
                    (ratio < 1.0 - self.clip_ratio).float().mean().item() +
                    (ratio > 1.0 + self.clip_ratio).float().mean().item()
                )
                
                # Early stopping based on KL divergence
                if kl > 1.5 * self.target_kl:
                    break
            
            # Early stopping based on KL divergence
            if kl > 1.5 * self.target_kl:
                break
        
        # Normalize metrics
        n_batches = len(rewards) // batch_size + (1 if len(rewards) % batch_size != 0 else 0)
        for key in metrics:
            metrics[key] /= n_batches
        
        # Increment update counter
        self.n_updates += 1
        
        return metrics
    
    def save(self, path: str = None):
        """
        Save the agent's networks.
        
        Parameters:
        -----------
        path : str, optional
            Path to save the networks. If None, uses the default path.
        """
        if path is None:
            path = os.path.join(MODELS_DIR, f"ppo_agent_{self.n_updates}")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'n_updates': self.n_updates
        }, path)
    
    def load(self, path: str):
        """
        Load the agent's networks.
        
        Parameters:
        -----------
        path : str
            Path to load the networks from
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.n_updates = checkpoint['n_updates'] 