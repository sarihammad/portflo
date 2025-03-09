"""
Backtesting framework for portfolio strategies.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import logging

from portflo.config.settings import TRANSACTION_FEE_RATE, INITIAL_PORTFOLIO_VALUE


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BacktestResult:
    """
    Class to store and analyze backtest results.
    """
    
    def __init__(
        self,
        strategy_name: str,
        portfolio_values: pd.Series,
        asset_values: pd.DataFrame,
        weights: pd.DataFrame,
        returns: pd.Series,
        transactions: pd.DataFrame,
        benchmark_values: Optional[pd.Series] = None
    ):
        """
        Initialize the backtest result.
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        portfolio_values : pandas.Series
            Series with portfolio values over time
        asset_values : pandas.DataFrame
            DataFrame with asset values over time
        weights : pandas.DataFrame
            DataFrame with asset weights over time
        returns : pandas.Series
            Series with portfolio returns over time
        transactions : pandas.DataFrame
            DataFrame with transaction details
        benchmark_values : pandas.Series, optional
            Series with benchmark values over time
        """
        self.strategy_name = strategy_name
        self.portfolio_values = portfolio_values
        self.asset_values = asset_values
        self.weights = weights
        self.returns = returns
        self.transactions = transactions
        self.benchmark_values = benchmark_values
        
        # Calculate performance metrics
        self.metrics = self._calculate_metrics()
    
    def _calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
        --------
        dict
            Dictionary with performance metrics
        """
        # Calculate returns if not already calculated
        if self.returns is None or len(self.returns) == 0:
            self.returns = self.portfolio_values.pct_change().dropna()
        
        # Calculate benchmark returns if benchmark is provided
        if self.benchmark_values is not None:
            benchmark_returns = self.benchmark_values.pct_change().dropna()
        else:
            benchmark_returns = None
        
        # Calculate metrics
        metrics = {}
        
        # Total return
        metrics['total_return'] = (
            self.portfolio_values.iloc[-1] / self.portfolio_values.iloc[0] - 1.0
        )
        
        # Annualized return
        n_years = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days / 365.25
        metrics['annualized_return'] = (1.0 + metrics['total_return']) ** (1.0 / n_years) - 1.0
        
        # Volatility (annualized)
        metrics['volatility'] = self.returns.std() * np.sqrt(252)
        
        # Sharpe ratio (annualized, assuming 0 risk-free rate for simplicity)
        metrics['sharpe_ratio'] = metrics['annualized_return'] / metrics['volatility'] if metrics['volatility'] > 0 else 0.0
        
        # Maximum drawdown
        rolling_max = self.portfolio_values.cummax()
        drawdowns = (self.portfolio_values - rolling_max) / rolling_max
        metrics['max_drawdown'] = drawdowns.min()
        
        # Sortino ratio (annualized, assuming 0 risk-free rate)
        downside_returns = self.returns.copy()
        downside_returns[downside_returns > 0] = 0
        downside_deviation = downside_returns.std() * np.sqrt(252)
        metrics['sortino_ratio'] = metrics['annualized_return'] / downside_deviation if downside_deviation > 0 else 0.0
        
        # Calmar ratio (annualized return / max drawdown)
        metrics['calmar_ratio'] = metrics['annualized_return'] / abs(metrics['max_drawdown']) if metrics['max_drawdown'] < 0 else 0.0
        
        # Value at Risk (VaR) - 95% confidence
        metrics['var_95'] = self.returns.quantile(0.05)
        
        # Conditional Value at Risk (CVaR) / Expected Shortfall - 95% confidence
        metrics['cvar_95'] = self.returns[self.returns <= metrics['var_95']].mean()
        
        # Information ratio (if benchmark is provided)
        if benchmark_returns is not None:
            # Align returns
            aligned_returns = pd.concat([self.returns, benchmark_returns], axis=1).dropna()
            if len(aligned_returns) > 0:
                excess_returns = aligned_returns.iloc[:, 0] - aligned_returns.iloc[:, 1]
                tracking_error = excess_returns.std() * np.sqrt(252)
                metrics['information_ratio'] = excess_returns.mean() * 252 / tracking_error if tracking_error > 0 else 0.0
                
                # Beta
                cov = np.cov(aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1])[0, 1]
                var = np.var(aligned_returns.iloc[:, 1])
                metrics['beta'] = cov / var if var > 0 else 0.0
                
                # Alpha (annualized)
                metrics['alpha'] = (
                    metrics['annualized_return'] - 
                    metrics['beta'] * (aligned_returns.iloc[:, 1].mean() * 252)
                )
        
        # Transaction metrics
        if self.transactions is not None and len(self.transactions) > 0:
            metrics['n_transactions'] = len(self.transactions)
            metrics['transaction_costs'] = self.transactions['cost'].sum()
            metrics['turnover'] = self.transactions['value'].sum() / self.portfolio_values.mean()
        
        return metrics
    
    def plot_portfolio_value(
        self,
        benchmark_name: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot portfolio value over time.
        
        Parameters:
        -----------
        benchmark_name : str, optional
            Name of the benchmark
        figsize : tuple, default (12, 6)
            Figure size
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot portfolio value
        plt.plot(self.portfolio_values.index, self.portfolio_values, label=self.strategy_name)
        
        # Plot benchmark value if provided
        if self.benchmark_values is not None:
            plt.plot(self.benchmark_values.index, self.benchmark_values, label=benchmark_name or 'Benchmark')
        
        # Set title and labels
        plt.title(title or f'Portfolio Value - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_asset_weights(
        self,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot asset weights over time.
        
        Parameters:
        -----------
        figsize : tuple, default (12, 6)
            Figure size
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Plot asset weights as stacked area chart
        plt.stackplot(
            self.weights.index, 
            [self.weights[col] for col in self.weights.columns],
            labels=self.weights.columns
        )
        
        # Set title and labels
        plt.title(title or f'Asset Weights - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Weight')
        plt.legend(loc='upper left')
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    
    def plot_drawdown(
        self,
        figsize: Tuple[int, int] = (12, 6),
        title: Optional[str] = None,
        save_path: Optional[str] = None
    ) -> None:
        """
        Plot drawdown over time.
        
        Parameters:
        -----------
        figsize : tuple, default (12, 6)
            Figure size
        title : str, optional
            Plot title
        save_path : str, optional
            Path to save the plot
        """
        plt.figure(figsize=figsize)
        
        # Calculate drawdown
        rolling_max = self.portfolio_values.cummax()
        drawdown = (self.portfolio_values - rolling_max) / rolling_max
        
        # Plot drawdown
        plt.plot(drawdown.index, drawdown)
        
        # Set title and labels
        plt.title(title or f'Drawdown - {self.strategy_name}')
        plt.xlabel('Date')
        plt.ylabel('Drawdown')
        plt.grid(True)
        
        # Save plot if path is provided
        if save_path is not None:
            plt.savefig(save_path)
        
        plt.show()
    
    def print_metrics(self) -> None:
        """
        Print performance metrics.
        """
        print(f"Performance Metrics - {self.strategy_name}")
        print("-" * 50)
        print(f"Total Return: {self.metrics['total_return']:.2%}")
        print(f"Annualized Return: {self.metrics['annualized_return']:.2%}")
        print(f"Volatility: {self.metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {self.metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        print(f"Calmar Ratio: {self.metrics['calmar_ratio']:.2f}")
        print(f"Value at Risk (95%): {self.metrics['var_95']:.2%}")
        print(f"Conditional VaR (95%): {self.metrics['cvar_95']:.2%}")
        
        if 'information_ratio' in self.metrics:
            print(f"Information Ratio: {self.metrics['information_ratio']:.2f}")
            print(f"Beta: {self.metrics['beta']:.2f}")
            print(f"Alpha: {self.metrics['alpha']:.2%}")
        
        if 'n_transactions' in self.metrics:
            print(f"Number of Transactions: {self.metrics['n_transactions']}")
            print(f"Transaction Costs: {self.metrics['transaction_costs']:.2f}")
            print(f"Turnover: {self.metrics['turnover']:.2f}")
        
        print("-" * 50)


class Backtest:
    """
    Backtesting framework for portfolio strategies.
    """
    
    def __init__(
        self,
        price_data: pd.DataFrame,
        strategy: Any,
        strategy_name: str,
        initial_cash: float = INITIAL_PORTFOLIO_VALUE,
        transaction_fee_rate: float = TRANSACTION_FEE_RATE,
        benchmark: Optional[str] = None,
        rebalance_freq: str = 'M',
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Initialize the backtest.
        
        Parameters:
        -----------
        price_data : pandas.DataFrame
            DataFrame with price data for multiple assets
        strategy : object
            Strategy object with allocate and rebalance methods
        strategy_name : str
            Name of the strategy
        initial_cash : float, default 100000
            Initial cash amount
        transaction_fee_rate : float, default 0.001
            Transaction fee rate (e.g., 0.001 = 0.1%)
        benchmark : str, optional
            Benchmark asset (e.g., 'SPY')
        rebalance_freq : str, default 'M'
            Rebalancing frequency ('D' for daily, 'W' for weekly, 'M' for monthly, 'Q' for quarterly)
        start_date : str, optional
            Start date for the backtest (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date for the backtest (format: 'YYYY-MM-DD')
        """
        # Store parameters
        self.price_data = price_data
        self.strategy = strategy
        self.strategy_name = strategy_name
        self.initial_cash = initial_cash
        self.transaction_fee_rate = transaction_fee_rate
        self.benchmark = benchmark
        self.rebalance_freq = rebalance_freq
        
        # Filter data by date range
        if start_date is not None or end_date is not None:
            self.price_data = self._filter_by_date(price_data, start_date, end_date)
        
        # Extract unique assets and dates
        self.assets = price_data.columns.tolist()
        self.dates = price_data.index.tolist()
        
        # Initialize portfolio
        self.portfolio = {
            'cash': initial_cash,
            'assets': {asset: 0.0 for asset in self.assets}
        }
        
        # Initialize transaction history
        self.transactions = []
    
    def _filter_by_date(
        self,
        data: pd.DataFrame,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Filter data by date range.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame to filter
        start_date : str, optional
            Start date (format: 'YYYY-MM-DD')
        end_date : str, optional
            End date (format: 'YYYY-MM-DD')
            
        Returns:
        --------
        pandas.DataFrame
            Filtered DataFrame
        """
        if start_date is not None:
            data = data[data.index >= start_date]
        
        if end_date is not None:
            data = data[data.index <= end_date]
        
        return data
    
    def _calculate_portfolio_value(self, date: pd.Timestamp) -> Dict[str, float]:
        """
        Calculate portfolio value on the given date.
        
        Parameters:
        -----------
        date : pandas.Timestamp
            Date to calculate portfolio value for
            
        Returns:
        --------
        dict
            Dictionary with portfolio value details:
            - 'total': Total portfolio value
            - 'cash': Cash value
            - 'assets': Dictionary with asset values
        """
        # Get prices on the given date
        prices = self.price_data.loc[date]
        
        # Calculate asset values
        asset_values = {}
        for asset in self.assets:
            asset_values[asset] = self.portfolio['assets'][asset] * prices[asset]
        
        # Calculate total value
        total_value = self.portfolio['cash'] + sum(asset_values.values())
        
        return {
            'total': total_value,
            'cash': self.portfolio['cash'],
            'assets': asset_values
        }
    
    def _calculate_weights(self, portfolio_value: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate asset weights.
        
        Parameters:
        -----------
        portfolio_value : dict
            Dictionary with portfolio value details
            
        Returns:
        --------
        dict
            Dictionary with asset weights
        """
        total_value = portfolio_value['total']
        
        if total_value == 0:
            return {'cash': 1.0, **{asset: 0.0 for asset in self.assets}}
        
        weights = {
            'cash': portfolio_value['cash'] / total_value
        }
        
        for asset in self.assets:
            weights[asset] = portfolio_value['assets'][asset] / total_value
        
        return weights
    
    def _rebalance_portfolio(
        self,
        date: pd.Timestamp,
        target_weights: Dict[str, float]
    ) -> None:
        """
        Rebalance the portfolio to the target weights.
        
        Parameters:
        -----------
        date : pandas.Timestamp
            Date to rebalance on
        target_weights : dict
            Target asset weights
        """
        # Calculate current portfolio value
        portfolio_value = self._calculate_portfolio_value(date)
        total_value = portfolio_value['total']
        
        # Calculate target values
        target_values = {
            asset: total_value * weight
            for asset, weight in target_weights.items()
            if asset != 'cash'
        }
        target_values['cash'] = total_value * target_weights.get('cash', 0.0)
        
        # Calculate current values
        current_values = {
            asset: portfolio_value['assets'][asset]
            for asset in self.assets
        }
        current_values['cash'] = portfolio_value['cash']
        
        # Calculate trades
        trades = {}
        for asset in self.assets:
            trades[asset] = target_values[asset] - current_values[asset]
        
        # Calculate transaction costs
        transaction_cost = 0.0
        for asset, trade_value in trades.items():
            if trade_value != 0:
                transaction_cost += abs(trade_value) * self.transaction_fee_rate
        
        # Adjust cash for transaction costs
        target_values['cash'] -= transaction_cost
        
        # Update portfolio
        prices = self.price_data.loc[date]
        for asset in self.assets:
            # Calculate shares to trade
            trade_shares = trades[asset] / prices[asset]
            
            # Update portfolio
            self.portfolio['assets'][asset] += trade_shares
            
            # Record transaction
            if trade_shares != 0:
                self.transactions.append({
                    'date': date,
                    'asset': asset,
                    'shares': trade_shares,
                    'price': prices[asset],
                    'value': trades[asset],
                    'cost': abs(trades[asset]) * self.transaction_fee_rate
                })
        
        # Update cash
        self.portfolio['cash'] = target_values['cash']
    
    def run(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
        --------
        BacktestResult
            Backtest result object
        """
        # Initialize result containers
        portfolio_values = []
        asset_values = []
        weights = []
        returns = []
        dates = []
        
        # Initialize benchmark values if provided
        benchmark_values = [] if self.benchmark is not None else None
        
        # Get rebalancing dates
        if self.rebalance_freq == 'D':
            rebalance_dates = self.dates
        else:
            # Convert to pandas DatetimeIndex for resampling
            date_index = pd.DatetimeIndex(self.dates)
            rebalance_dates = date_index.to_period(self.rebalance_freq).drop_duplicates().to_timestamp()
            # Ensure rebalance_dates are in self.dates
            rebalance_dates = [date for date in rebalance_dates if date in self.dates]
        
        # Initialize portfolio with first allocation
        first_date = self.dates[0]
        
        # Get historical returns for initial allocation
        historical_returns = self.price_data.pct_change().dropna()
        
        # Allocate initial portfolio
        if hasattr(self.strategy, 'allocate'):
            allocation = self.strategy.allocate(self.assets, historical_returns)
            target_weights = allocation['weights']
        else:
            # Default to equal weight if strategy doesn't have allocate method
            target_weights = {asset: 1.0 / len(self.assets) for asset in self.assets}
            target_weights['cash'] = 0.0
        
        # Rebalance to initial allocation
        self._rebalance_portfolio(first_date, target_weights)
        
        # Run backtest
        for date in self.dates:
            # Calculate portfolio value
            portfolio_value = self._calculate_portfolio_value(date)
            
            # Calculate weights
            current_weights = self._calculate_weights(portfolio_value)
            
            # Store results
            portfolio_values.append(portfolio_value['total'])
            asset_values.append(portfolio_value['assets'])
            weights.append(current_weights)
            dates.append(date)
            
            # Store benchmark value if provided
            if self.benchmark is not None:
                benchmark_values.append(self.price_data.loc[date, self.benchmark])
            
            # Calculate return
            if len(portfolio_values) > 1:
                daily_return = portfolio_values[-1] / portfolio_values[-2] - 1.0
                returns.append(daily_return)
            
            # Rebalance if it's a rebalancing date
            if date in rebalance_dates and date != first_date:
                # Get historical returns for rebalancing
                historical_returns = self.price_data.loc[:date].pct_change().dropna()
                
                # Rebalance portfolio
                if hasattr(self.strategy, 'rebalance'):
                    rebalance = self.strategy.rebalance(current_weights, self.assets, historical_returns)
                    target_weights = rebalance['weights']
                elif hasattr(self.strategy, 'allocate'):
                    allocation = self.strategy.allocate(self.assets, historical_returns)
                    target_weights = allocation['weights']
                else:
                    # Default to equal weight if strategy doesn't have rebalance or allocate methods
                    target_weights = {asset: 1.0 / len(self.assets) for asset in self.assets}
                    target_weights['cash'] = 0.0
                
                # Rebalance to target weights
                self._rebalance_portfolio(date, target_weights)
        
        # Convert results to pandas objects
        portfolio_values = pd.Series(portfolio_values, index=dates)
        asset_values = pd.DataFrame(asset_values, index=dates)
        weights = pd.DataFrame(weights, index=dates)
        returns = pd.Series(returns, index=dates[1:])
        transactions = pd.DataFrame(self.transactions)
        
        if benchmark_values is not None:
            benchmark_values = pd.Series(benchmark_values, index=dates)
        
        # Create and return backtest result
        result = BacktestResult(
            strategy_name=self.strategy_name,
            portfolio_values=portfolio_values,
            asset_values=asset_values,
            weights=weights,
            returns=returns,
            transactions=transactions,
            benchmark_values=benchmark_values
        )
        
        return result 