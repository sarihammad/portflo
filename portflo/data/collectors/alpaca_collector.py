"""
Data collector for Alpaca API to fetch stock and ETF data.
"""
import pandas as pd
from datetime import datetime, timedelta
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from portflo.config.settings import ALPACA_API_KEY, ALPACA_SECRET_KEY


class AlpacaDataCollector:
    """
    Collector for fetching stock and ETF data from Alpaca API.
    """
    
    def __init__(self, api_key=None, secret_key=None):
        """
        Initialize the Alpaca data collector.
        
        Parameters:
        -----------
        api_key : str, optional
            Alpaca API key. If None, uses the key from settings.
        secret_key : str, optional
            Alpaca secret key. If None, uses the key from settings.
        """
        self.api_key = api_key or ALPACA_API_KEY
        self.secret_key = secret_key or ALPACA_SECRET_KEY
        
        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API key and secret key are required. "
                            "Set them in the .env file or pass them as parameters.")
        
        self.client = StockHistoricalDataClient(api_key=self.api_key, secret_key=self.secret_key)
    
    def _parse_timeframe(self, timeframe):
        """
        Parse the timeframe string to Alpaca TimeFrame object.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1d', '1h', '15m')
            
        Returns:
        --------
        alpaca.data.timeframe.TimeFrame
            Alpaca TimeFrame object
        """
        if timeframe == '1d':
            return TimeFrame.Day
        elif timeframe == '1h':
            return TimeFrame.Hour
        elif timeframe == '15m':
            return TimeFrame.Minute(15)
        elif timeframe == '5m':
            return TimeFrame.Minute(5)
        elif timeframe == '1m':
            return TimeFrame.Minute
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
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
        
        if end_date is None:
            end_date = datetime.now()
        elif isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Parse timeframe
        tf = self._parse_timeframe(timeframe)
        
        # Create request
        request_params = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start_date,
            end=end_date
        )
        
        try:
            # Fetch data
            bars = self.client.get_stock_bars(request_params)
            
            # Convert to dictionary of DataFrames
            data_dict = {}
            for symbol in symbols:
                if symbol in bars:
                    # Convert to DataFrame
                    df = bars[symbol].df
                    
                    # Reset index to make timestamp a column
                    df = df.reset_index()
                    
                    # Rename columns to lowercase for consistency
                    df.columns = [col.lower() for col in df.columns]
                    
                    # Rename timestamp column to date for consistency
                    df = df.rename(columns={'timestamp': 'date'})
                    
                    # Store in dictionary
                    data_dict[symbol] = df
                else:
                    print(f"No data found for {symbol}")
            
            return data_dict
        
        except Exception as e:
            print(f"Error fetching data from Alpaca: {e}")
            return {}
    
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