"""
Equal-Weight Portfolio Strategy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union


class EqualWeightStrategy:
    """
    Equal-Weight Portfolio Strategy.
    
    This strategy allocates equal weights to all assets in the portfolio.
    It's a simple baseline strategy that often performs surprisingly well
    compared to more complex strategies.
    """
    
    def __init__(self, include_cash: bool = True, cash_weight: float = 0.0):
        """
        Initialize the Equal-Weight Strategy.
        
        Parameters:
        -----------
        include_cash : bool, default True
            Whether to include a cash position
        cash_weight : float, default 0.0
            Weight allocated to cash (if included)
        """
        self.include_cash = include_cash
        self.cash_weight = cash_weight
    
    def allocate(
        self,
        assets: List[str],
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Allocate equal weights to all assets.
        
        Parameters:
        -----------
        assets : list
            List of asset names
        returns : pandas.DataFrame, optional
            DataFrame with asset returns (not used, included for API consistency)
            
        Returns:
        --------
        dict
            Dictionary with allocation results:
            - 'weights': Asset weights
            - 'expected_return': Expected portfolio return (if returns provided)
            - 'expected_risk': Expected portfolio risk (if returns provided)
            - 'sharpe_ratio': Sharpe ratio of the portfolio (if returns provided)
        """
        n_assets = len(assets)
        
        if self.include_cash:
            # Allocate specified weight to cash
            asset_weight = (1.0 - self.cash_weight) / n_assets
            weights = {asset: asset_weight for asset in assets}
            weights['cash'] = self.cash_weight
        else:
            # Allocate equal weight to all assets
            asset_weight = 1.0 / n_assets
            weights = {asset: asset_weight for asset in assets}
        
        # Create result dictionary
        result = {
            'weights': weights
        }
        
        # Calculate portfolio metrics if returns are provided
        if returns is not None:
            # Extract returns for the assets in the portfolio
            portfolio_returns = returns[assets]
            
            # Calculate expected returns
            expected_returns = portfolio_returns.mean()
            
            # Calculate portfolio expected return
            portfolio_return = 0.0
            for asset in assets:
                portfolio_return += weights.get(asset, 0.0) * expected_returns.get(asset, 0.0)
            
            # Calculate portfolio risk
            cov_matrix = portfolio_returns.cov()
            portfolio_risk = 0.0
            for i, asset_i in enumerate(assets):
                for j, asset_j in enumerate(assets):
                    portfolio_risk += (weights.get(asset_i, 0.0) * weights.get(asset_j, 0.0) * 
                                      cov_matrix.loc[asset_i, asset_j])
            portfolio_risk = np.sqrt(portfolio_risk)
            
            # Calculate Sharpe ratio
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
            
            # Add metrics to result
            result['expected_return'] = portfolio_return
            result['expected_risk'] = portfolio_risk
            result['sharpe_ratio'] = sharpe_ratio
        
        return result
    
    def rebalance(
        self,
        current_weights: Dict[str, float],
        assets: List[str],
        returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Union[Dict[str, float], float]]:
        """
        Rebalance the portfolio to equal weights.
        
        Parameters:
        -----------
        current_weights : dict
            Current asset weights
        assets : list
            List of asset names
        returns : pandas.DataFrame, optional
            DataFrame with asset returns (not used, included for API consistency)
            
        Returns:
        --------
        dict
            Dictionary with allocation results (same as allocate method)
        """
        # Simply reallocate with equal weights
        return self.allocate(assets, returns) 