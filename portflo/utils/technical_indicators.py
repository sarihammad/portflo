"""
Technical indicators for feature engineering in portfolio optimization.
"""
import numpy as np
import pandas as pd
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice


def add_technical_indicators(df, price_col='close', volume_col='volume'):
    """
    Add technical indicators to a DataFrame containing price data.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data with columns like 'open', 'high', 'low', 'close', 'volume'
    price_col : str, default 'close'
        Column name for closing prices
    volume_col : str, default 'volume'
        Column name for volume data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original DataFrame
    df_copy = df.copy()
    
    # Check if required columns exist
    required_cols = [price_col]
    if not all(col in df_copy.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df_copy.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Trend indicators
    # MACD
    macd = MACD(close=df_copy[price_col])
    df_copy['macd'] = macd.macd()
    df_copy['macd_signal'] = macd.macd_signal()
    df_copy['macd_diff'] = macd.macd_diff()
    
    # Moving Averages
    df_copy['sma_20'] = SMAIndicator(close=df_copy[price_col], window=20).sma_indicator()
    df_copy['sma_50'] = SMAIndicator(close=df_copy[price_col], window=50).sma_indicator()
    df_copy['sma_200'] = SMAIndicator(close=df_copy[price_col], window=200).sma_indicator()
    
    df_copy['ema_12'] = EMAIndicator(close=df_copy[price_col], window=12).ema_indicator()
    df_copy['ema_26'] = EMAIndicator(close=df_copy[price_col], window=26).ema_indicator()
    
    # Momentum indicators
    # RSI
    df_copy['rsi'] = RSIIndicator(close=df_copy[price_col]).rsi()
    
    # Stochastic Oscillator
    if all(col in df_copy.columns for col in ['high', 'low']):
        stoch = StochasticOscillator(high=df_copy['high'], low=df_copy['low'], close=df_copy[price_col])
        df_copy['stoch_k'] = stoch.stoch()
        df_copy['stoch_d'] = stoch.stoch_signal()
    
    # Volatility indicators
    # Bollinger Bands
    bollinger = BollingerBands(close=df_copy[price_col])
    df_copy['bb_high'] = bollinger.bollinger_hband()
    df_copy['bb_low'] = bollinger.bollinger_lband()
    df_copy['bb_mid'] = bollinger.bollinger_mavg()
    df_copy['bb_width'] = (df_copy['bb_high'] - df_copy['bb_low']) / df_copy['bb_mid']
    
    # ATR
    if all(col in df_copy.columns for col in ['high', 'low']):
        df_copy['atr'] = AverageTrueRange(high=df_copy['high'], low=df_copy['low'], close=df_copy[price_col]).average_true_range()
    
    # Volume indicators
    if volume_col in df_copy.columns and all(col in df_copy.columns for col in ['high', 'low']):
        # VWAP (requires high, low, close, volume)
        vwap = VolumeWeightedAveragePrice(
            high=df_copy['high'],
            low=df_copy['low'],
            close=df_copy[price_col],
            volume=df_copy[volume_col]
        )
        df_copy['vwap'] = vwap.volume_weighted_average_price()
    
    # Calculate rolling volatility (standard deviation)
    df_copy['volatility_20'] = df_copy[price_col].pct_change().rolling(window=20).std()
    
    # Calculate rolling Sharpe ratio (using daily returns)
    df_copy['daily_return'] = df_copy[price_col].pct_change()
    df_copy['rolling_sharpe_20'] = (
        df_copy['daily_return'].rolling(window=20).mean() / 
        df_copy['daily_return'].rolling(window=20).std()
    ) * np.sqrt(252)  # Annualized
    
    # Calculate rolling Sortino ratio (using daily returns)
    # Sortino only penalizes downside volatility
    df_copy['downside_returns'] = df_copy['daily_return'].copy()
    df_copy.loc[df_copy['downside_returns'] > 0, 'downside_returns'] = 0
    df_copy['rolling_sortino_20'] = (
        df_copy['daily_return'].rolling(window=20).mean() / 
        df_copy['downside_returns'].rolling(window=20).std()
    ) * np.sqrt(252)  # Annualized
    
    # Calculate rolling max drawdown
    rolling_max = df_copy[price_col].rolling(window=252, min_periods=1).max()
    df_copy['rolling_drawdown'] = (df_copy[price_col] / rolling_max - 1.0)
    
    # Drop NaN values that may have been introduced
    # df_copy.dropna(inplace=True)
    
    return df_copy


def calculate_risk_metrics(returns, window=20):
    """
    Calculate risk metrics from a series of returns.
    
    Parameters:
    -----------
    returns : pandas.Series
        Series of asset returns
    window : int, default 20
        Rolling window size for calculations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with risk metrics
    """
    metrics = pd.DataFrame(index=returns.index)
    
    # Volatility (annualized)
    metrics['volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
    
    # Sharpe Ratio (annualized, assuming 0 risk-free rate for simplicity)
    metrics['sharpe_ratio'] = (returns.rolling(window=window).mean() / 
                              returns.rolling(window=window).std()) * np.sqrt(252)
    
    # Sortino Ratio (annualized, assuming 0 risk-free rate)
    downside_returns = returns.copy()
    downside_returns[downside_returns > 0] = 0
    metrics['sortino_ratio'] = (returns.rolling(window=window).mean() / 
                               downside_returns.rolling(window=window).std()) * np.sqrt(252)
    
    # Maximum Drawdown
    rolling_max = returns.cumsum().rolling(window=window, min_periods=1).max()
    drawdown = returns.cumsum() - rolling_max
    metrics['max_drawdown'] = drawdown.rolling(window=window).min()
    
    # Value at Risk (VaR) - 95% confidence
    metrics['var_95'] = returns.rolling(window=window).quantile(0.05)
    
    # Conditional Value at Risk (CVaR) / Expected Shortfall - 95% confidence
    def calculate_cvar(x):
        var_95 = np.percentile(x, 5)
        return x[x <= var_95].mean()
    
    metrics['cvar_95'] = returns.rolling(window=window).apply(calculate_cvar, raw=True)
    
    return metrics 