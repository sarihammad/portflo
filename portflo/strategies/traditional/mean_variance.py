"""
Mean-Variance Portfolio Optimization (Markowitz Model).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union


class MeanVarianceOptimizer:
    """
    Mean-Variance Portfolio Optimizer (Markowitz Model).
    
    This class implements the classic Markowitz portfolio optimization model,
    which aims to maximize the expected return for a given level of risk,
    or minimize risk for a given level of expected return.
    """
    
    def __init__(
        self,
        risk_aversion: float = 1.0,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        target_return: Optional[float] = None,
        target_risk: Optional[float] = None
    ):
        """
        Initialize the Mean-Variance Optimizer.
        
        Parameters:
        -----------
        risk_aversion : float, default 1.0
            Risk aversion parameter (higher values prioritize risk reduction)
        min_weight : float, default 0.0
            Minimum weight for each asset
        max_weight : float, default 1.0
            Maximum weight for each asset
        target_return : float, optional
            Target portfolio return (if specified, will optimize for minimum risk)
        target_risk : float, optional
            Target portfolio risk (if specified, will optimize for maximum return)
        """
        self.risk_aversion = risk_aversion
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.target_return = target_return
        self.target_risk = target_risk
    
    def optimize(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        include_cash: bool = True
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Optimize the portfolio weights.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with asset returns (each column is an asset)
        cov_matrix : pandas.DataFrame, optional
            Covariance matrix of returns. If None, it will be calculated from returns.
        include_cash : bool, default True
            Whether to include a cash position (risk-free asset)
            
        Returns:
        --------
        dict
            Dictionary with optimization results:
            - 'weights': Optimal asset weights
            - 'expected_return': Expected portfolio return
            - 'expected_risk': Expected portfolio risk (volatility)
            - 'sharpe_ratio': Sharpe ratio of the portfolio
        """
        # Get asset names
        assets = returns.columns.tolist()
        n_assets = len(assets)
        
        # Calculate expected returns (mean of historical returns)
        expected_returns = returns.mean().values
        
        # Calculate covariance matrix if not provided
        if cov_matrix is None:
            cov_matrix = returns.cov().values
        else:
            cov_matrix = cov_matrix.values
        
        # Define optimization objective function
        def objective(weights):
            # Portfolio expected return
            portfolio_return = np.sum(weights * expected_returns)
            
            # Portfolio risk (volatility)
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            # Objective: maximize utility (return - risk_aversion * risk)
            # Since we're minimizing, we negate the utility
            if self.target_return is not None:
                # Minimize risk subject to target return
                return portfolio_risk
            elif self.target_risk is not None:
                # Maximize return subject to target risk
                return -portfolio_return
            else:
                # Maximize utility
                return -portfolio_return + self.risk_aversion * portfolio_risk
        
        # Define constraints
        constraints = []
        
        # Constraint: weights sum to 1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # Constraint: target return (if specified)
        if self.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sum(weights * expected_returns) - self.target_return
            })
        
        # Constraint: target risk (if specified)
        if self.target_risk is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) - self.target_risk
            })
        
        # Define bounds for weights
        bounds = tuple((self.min_weight, self.max_weight) for _ in range(n_assets))
        
        # Initial guess: equal weights
        initial_weights = np.ones(n_assets) / n_assets
        
        # Run optimization
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        # Extract optimal weights
        optimal_weights = result['x']
        
        # Calculate portfolio metrics
        portfolio_return = np.sum(optimal_weights * expected_returns)
        portfolio_risk = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
        sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0.0
        
        # Create result dictionary
        optimization_result = {
            'weights': dict(zip(assets, optimal_weights)),
            'expected_return': portfolio_return,
            'expected_risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio
        }
        
        return optimization_result
    
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        n_points: int = 50
    ) -> pd.DataFrame:
        """
        Calculate the efficient frontier.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with asset returns (each column is an asset)
        cov_matrix : pandas.DataFrame, optional
            Covariance matrix of returns. If None, it will be calculated from returns.
        n_points : int, default 50
            Number of points on the efficient frontier
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with efficient frontier points:
            - 'expected_return': Expected portfolio return
            - 'expected_risk': Expected portfolio risk (volatility)
            - 'sharpe_ratio': Sharpe ratio of the portfolio
            - 'weights': Optimal asset weights for each point
        """
        # Calculate expected returns and covariance matrix
        expected_returns = returns.mean()
        if cov_matrix is None:
            cov_matrix = returns.cov()
        
        # Find minimum and maximum returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        
        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_points)
        
        # Calculate efficient frontier
        efficient_frontier = []
        for target_return in target_returns:
            # Set target return
            self.target_return = target_return
            self.target_risk = None
            
            # Optimize portfolio
            result = self.optimize(returns, cov_matrix)
            
            # Store result
            efficient_frontier.append({
                'expected_return': result['expected_return'],
                'expected_risk': result['expected_risk'],
                'sharpe_ratio': result['sharpe_ratio'],
                'weights': result['weights']
            })
        
        # Reset target return
        self.target_return = None
        
        # Convert to DataFrame
        efficient_frontier_df = pd.DataFrame(efficient_frontier)
        
        return efficient_frontier_df
    
    def optimal_portfolio(
        self,
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        criterion: str = 'sharpe'
    ) -> Dict[str, Union[np.ndarray, float]]:
        """
        Find the optimal portfolio based on the specified criterion.
        
        Parameters:
        -----------
        returns : pandas.DataFrame
            DataFrame with asset returns (each column is an asset)
        cov_matrix : pandas.DataFrame, optional
            Covariance matrix of returns. If None, it will be calculated from returns.
        criterion : str, default 'sharpe'
            Criterion for selecting the optimal portfolio:
            - 'sharpe': Maximum Sharpe ratio
            - 'min_risk': Minimum risk
            - 'max_return': Maximum return
            - 'utility': Maximum utility (return - risk_aversion * risk)
            
        Returns:
        --------
        dict
            Dictionary with optimization results:
            - 'weights': Optimal asset weights
            - 'expected_return': Expected portfolio return
            - 'expected_risk': Expected portfolio risk (volatility)
            - 'sharpe_ratio': Sharpe ratio of the portfolio
        """
        # Calculate efficient frontier
        efficient_frontier = self.efficient_frontier(returns, cov_matrix)
        
        # Find optimal portfolio based on criterion
        if criterion == 'sharpe':
            # Maximum Sharpe ratio
            optimal_idx = efficient_frontier['sharpe_ratio'].idxmax()
        elif criterion == 'min_risk':
            # Minimum risk
            optimal_idx = efficient_frontier['expected_risk'].idxmin()
        elif criterion == 'max_return':
            # Maximum return
            optimal_idx = efficient_frontier['expected_return'].idxmax()
        elif criterion == 'utility':
            # Maximum utility
            utility = (efficient_frontier['expected_return'] - 
                      self.risk_aversion * efficient_frontier['expected_risk'])
            optimal_idx = utility.idxmax()
        else:
            raise ValueError(f"Invalid criterion: {criterion}")
        
        # Extract optimal portfolio
        optimal_portfolio = efficient_frontier.loc[optimal_idx].to_dict()
        
        return optimal_portfolio 