"""
Data collector for Yahoo Finance API to fetch stock, ETF, and index data.
"""
import pandas as pd
import yfinance as yf
from datetime import datetime


class YahooDataCollector:
    """
    Collector for fetching stock, ETF, and index data from Yahoo Finance API.
    This is a good fallback option when Alpaca API is not available or for assets not covered by Alpaca.
    """
    
    def __init__(self):
        """
        Initialize the Yahoo Finance data collector.
        """
        pass  # No authentication needed for Yahoo Finance
    
    def _parse_timeframe(self, timeframe):
        """
        Parse the timeframe string to Yahoo Finance interval string.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1d', '1h', '15m')
            
        Returns:
        --------
        str
            Yahoo Finance interval string
        """
        if timeframe == '1d':
            return '1d'
        elif timeframe == '1h':
            return '1h'
        elif timeframe == '15m':
            return '15m'
        elif timeframe == '5m':
            return '5m'
        elif timeframe == '1m':
            return '1m'
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def fetch_data(self, symbols, start_date, end_date=None, timeframe='1d'):
        """
        Fetch historical data for the given symbols.
        
        Parameters:
        -----------
        symbols : list
            List of stock/ETF symbols to fetch
        start_date : str or datetime
            Start date for the data (format: 'YYYY-MM-DD')
        end_date : str or datetime, optional
            End date for the data (format: 'YYYY-MM-DD'). If None, uses current date.
        timeframe : str, default '1d'
            Timeframe for the data ('1d', '1h', '15m', '5m', '1m')
            
        Returns:
        --------
        dict
            Dictionary with symbols as keys and pandas DataFrames as values
        """
        # Parse dates
        if isinstance(start_date, str):
            start_date = start_date
        else:
            start_date = start_date.strftime('%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        elif isinstance(end_date, datetime):
            end_date = end_date.strftime('%Y-%m-%d')
        
        # Parse timeframe
        interval = self._parse_timeframe(timeframe)
        
        data_dict = {}
        for symbol in symbols:
            try:
                # Fetch data from Yahoo Finance
                df = yf.download(
                    tickers=symbol,
                    start=start_date,
                    end=end_date,
                    interval=interval,
                    auto_adjust=True,  # Adjust for splits and dividends
                    progress=False  # Disable progress bar
                )
                
                # Reset index to make date a column
                df = df.reset_index()
                
                # Rename columns to lowercase for consistency
                df.columns = [col.lower() for col in df.columns]
                
                # Store in dictionary
                if not df.empty:
                    data_dict[symbol] = df
                else:
                    print(f"No data found for {symbol}")
            
            except Exception as e:
                print(f"Error fetching data for {symbol}: {e}")
        
        return data_dict
    
    def fetch_and_combine(self, symbols, start_date, end_date=None, timeframe='1d'):
        """
        Fetch data for multiple symbols and combine into a single DataFrame with a MultiIndex.
        
        Parameters:
        -----------
        symbols : list
            List of stock/ETF symbols to fetch
        start_date : str or datetime
            Start date for the data (format: 'YYYY-MM-DD')
        end_date : str or datetime, optional
            End date for the data (format: 'YYYY-MM-DD'). If None, uses current date.
        timeframe : str, default '1d'
            Timeframe for the data ('1d', '1h', '15m', '5m', '1m')
            
        Returns:
        --------
        pandas.DataFrame
            MultiIndex DataFrame with (symbol, date) as index and OHLCV data as columns
        """
        # Fetch data for each symbol
        data_dict = self.fetch_data(symbols, start_date, end_date, timeframe)
        
        if not data_dict:
            return pd.DataFrame()
        
        # Combine into a single DataFrame with MultiIndex
        dfs = []
        for symbol, df in data_dict.items():
            df['symbol'] = symbol
            dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        # Concatenate all DataFrames
        combined_df = pd.concat(dfs, ignore_index=True)
        
        # Set MultiIndex
        combined_df = combined_df.set_index(['symbol', 'date'])
        
        return combined_df 