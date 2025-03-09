"""
Data processor for preprocessing market data and adding technical indicators.
"""
import pandas as pd
import numpy as np
from portflo.utils.technical_indicators import add_technical_indicators, calculate_risk_metrics


class DataProcessor:
    """
    Processor for preprocessing market data and adding technical indicators.
    """
    
    def __init__(self, lookback_window=30):
        """
        Initialize the data processor.
        
        Parameters:
        -----------
        lookback_window : int, default 30
            Number of days to look back for feature engineering
        """
        self.lookback_window = lookback_window
    
    def process_data(self, data, add_indicators=True, fill_missing=True, add_returns=True):
        """
        Process market data by adding technical indicators, filling missing values, and calculating returns.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            Market data to process. Can be a DataFrame or a dictionary of DataFrames.
        add_indicators : bool, default True
            Whether to add technical indicators
        fill_missing : bool, default True
            Whether to fill missing values
        add_returns : bool, default True
            Whether to calculate returns
            
        Returns:
        --------
        pandas.DataFrame or dict
            Processed market data
        """
        if isinstance(data, dict):
            # Process each DataFrame in the dictionary
            processed_data = {}
            for symbol, df in data.items():
                processed_data[symbol] = self._process_single_dataframe(
                    df, add_indicators, fill_missing, add_returns
                )
            return processed_data
        else:
            # Process a single DataFrame
            return self._process_single_dataframe(data, add_indicators, fill_missing, add_returns)
    
    def _process_single_dataframe(self, df, add_indicators, fill_missing, add_returns):
        """
        Process a single DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        add_indicators : bool
            Whether to add technical indicators
        fill_missing : bool
            Whether to fill missing values
        add_returns : bool
            Whether to calculate returns
            
        Returns:
        --------
        pandas.DataFrame
            Processed DataFrame
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Sort by date if not already sorted
        if 'date' in df_copy.columns:
            df_copy = df_copy.sort_values('date')
        elif df_copy.index.name == 'date':
            df_copy = df_copy.sort_index()
        
        # Calculate returns if requested
        if add_returns and 'close' in df_copy.columns:
            df_copy['daily_return'] = df_copy['close'].pct_change()
            df_copy['log_return'] = np.log(df_copy['close'] / df_copy['close'].shift(1))
            
            # Calculate cumulative returns
            df_copy['cumulative_return'] = (1 + df_copy['daily_return']).cumprod() - 1
        
        # Add technical indicators if requested
        if add_indicators:
            df_copy = add_technical_indicators(df_copy)
        
        # Fill missing values if requested
        if fill_missing:
            # Forward fill first (for time series data)
            df_copy = df_copy.ffill()
            # Then backfill any remaining NaNs
            df_copy = df_copy.bfill()
            # For any columns that are still NaN, fill with 0
            df_copy = df_copy.fillna(0)
        
        return df_copy
    
    def prepare_for_rl(self, data, feature_columns=None, target_column='daily_return'):
        """
        Prepare data for reinforcement learning by creating state representations.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            Processed market data
        feature_columns : list, optional
            List of columns to use as features. If None, uses a default set of features.
        target_column : str, default 'daily_return'
            Column to use as the target (reward)
            
        Returns:
        --------
        tuple
            (states, rewards) where states is a numpy array of shape (n_samples, lookback_window, n_features)
            and rewards is a numpy array of shape (n_samples,)
        """
        if isinstance(data, dict):
            # Combine all DataFrames in the dictionary
            combined_data = pd.concat([df.assign(symbol=symbol) for symbol, df in data.items()])
            return self._prepare_single_dataframe_for_rl(combined_data, feature_columns, target_column)
        else:
            # Process a single DataFrame
            return self._prepare_single_dataframe_for_rl(data, feature_columns, target_column)
    
    def _prepare_single_dataframe_for_rl(self, df, feature_columns, target_column):
        """
        Prepare a single DataFrame for reinforcement learning.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to process
        feature_columns : list or None
            List of columns to use as features
        target_column : str
            Column to use as the target (reward)
            
        Returns:
        --------
        tuple
            (states, rewards) where states is a numpy array of shape (n_samples, lookback_window, n_features)
            and rewards is a numpy array of shape (n_samples,)
        """
        # Default feature columns if not provided
        if feature_columns is None:
            feature_columns = [
                'close', 'volume', 'daily_return', 'macd', 'rsi', 
                'bb_width', 'volatility_20', 'rolling_sharpe_20'
            ]
        
        # Check if all feature columns exist in the DataFrame
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing feature columns: {missing_columns}")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame")
        
        # Extract features and target
        features = df[feature_columns].values
        target = df[target_column].values
        
        # Create state representations (lookback windows)
        states = []
        rewards = []
        
        for i in range(self.lookback_window, len(df)):
            # Extract the lookback window
            state = features[i - self.lookback_window:i]
            # Extract the reward (next return)
            reward = target[i]
            
            states.append(state)
            rewards.append(reward)
        
        return np.array(states), np.array(rewards)
    
    def normalize_data(self, data, method='zscore'):
        """
        Normalize data using the specified method.
        
        Parameters:
        -----------
        data : pandas.DataFrame or dict
            Data to normalize
        method : str, default 'zscore'
            Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
        --------
        pandas.DataFrame or dict
            Normalized data
        """
        if isinstance(data, dict):
            # Normalize each DataFrame in the dictionary
            normalized_data = {}
            for symbol, df in data.items():
                normalized_data[symbol] = self._normalize_single_dataframe(df, method)
            return normalized_data
        else:
            # Normalize a single DataFrame
            return self._normalize_single_dataframe(data, method)
    
    def _normalize_single_dataframe(self, df, method):
        """
        Normalize a single DataFrame.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to normalize
        method : str
            Normalization method ('zscore', 'minmax', or 'robust')
            
        Returns:
        --------
        pandas.DataFrame
            Normalized DataFrame
        """
        # Make a copy to avoid modifying the original DataFrame
        df_copy = df.copy()
        
        # Columns to exclude from normalization
        exclude_cols = ['date', 'symbol']
        
        # Columns to normalize
        numeric_cols = df_copy.select_dtypes(include=[np.number]).columns
        cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]
        
        # Normalize each column
        for col in cols_to_normalize:
            if method == 'zscore':
                # Z-score normalization (mean=0, std=1)
                mean = df_copy[col].mean()
                std = df_copy[col].std()
                if std != 0:  # Avoid division by zero
                    df_copy[col] = (df_copy[col] - mean) / std
            
            elif method == 'minmax':
                # Min-max normalization (range [0, 1])
                min_val = df_copy[col].min()
                max_val = df_copy[col].max()
                if max_val > min_val:  # Avoid division by zero
                    df_copy[col] = (df_copy[col] - min_val) / (max_val - min_val)
            
            elif method == 'robust':
                # Robust normalization using median and IQR
                median = df_copy[col].median()
                q1 = df_copy[col].quantile(0.25)
                q3 = df_copy[col].quantile(0.75)
                iqr = q3 - q1
                if iqr != 0:  # Avoid division by zero
                    df_copy[col] = (df_copy[col] - median) / iqr
            
            else:
                raise ValueError(f"Unsupported normalization method: {method}")
        
        return df_copy 