"""
Data collector for Binance API to fetch cryptocurrency data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from binance.client import Client
from binance.exceptions import BinanceAPIException

from portflo.config.settings import BINANCE_API_KEY, BINANCE_SECRET_KEY


class BinanceDataCollector:
    """
    Collector for fetching cryptocurrency data from Binance API.
    """
    
    def __init__(self, api_key=None, secret_key=None):
        """
        Initialize the Binance data collector.
        
        Parameters:
        -----------
        api_key : str, optional
            Binance API key. If None, uses the key from settings.
        secret_key : str, optional
            Binance secret key. If None, uses the key from settings.
        """
        self.api_key = api_key or BINANCE_API_KEY
        self.secret_key = secret_key or BINANCE_SECRET_KEY
        
        # Initialize client (can work with or without API keys for historical data)
        if self.api_key and self.secret_key:
            self.client = Client(api_key=self.api_key, api_secret=self.secret_key)
        else:
            self.client = Client()
    
    def _parse_timeframe(self, timeframe):
        """
        Parse the timeframe string to Binance interval string.
        
        Parameters:
        -----------
        timeframe : str
            Timeframe string (e.g., '1d', '1h', '15m')
            
        Returns:
        --------
        str
            Binance interval string
        """
        if timeframe == '1d':
            return Client.KLINE_INTERVAL_1DAY
        elif timeframe == '1h':
            return Client.KLINE_INTERVAL_1HOUR
        elif timeframe == '15m':
            return Client.KLINE_INTERVAL_15MINUTE
        elif timeframe == '5m':
            return Client.KLINE_INTERVAL_5MINUTE
        elif timeframe == '1m':
            return Client.KLINE_INTERVAL_1MINUTE
        else:
            raise ValueError(f"Unsupported timeframe: {timeframe}")
    
    def _format_symbol(self, symbol):
        """
        Format the symbol for Binance API.
        
        Parameters:
        -----------
        symbol : str
            Symbol in format like 'BTC/USD'
            
        Returns:
        --------
        str
            Formatted symbol for Binance API (e.g., 'BTCUSDT')
        """
        if '/' in symbol:
            base, quote = symbol.split('/')
            # Binance typically uses USDT instead of USD
            if quote == 'USD':
                quote = 'USDT'
            return f"{base}{quote}"
        return symbol
    
    def fetch_data(self, symbols, start_date, end_date=None, timeframe='1d'):
        """
        Fetch historical data for the given cryptocurrency symbols.
        
        Parameters:
        -----------
        symbols : list
            List of cryptocurrency symbols to fetch (e.g., ['BTC/USD', 'ETH/USD'])
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
        
        # Convert to milliseconds timestamp
        start_ts = int(start_date.timestamp() * 1000)
        end_ts = int(end_date.timestamp() * 1000)
        
        # Parse timeframe
        interval = self._parse_timeframe(timeframe)
        
        data_dict = {}
        for symbol in symbols:
            try:
                # Format symbol for Binance API
                formatted_symbol = self._format_symbol(symbol)
                
                # Fetch klines (candlestick data)
                klines = self.client.get_historical_klines(
                    symbol=formatted_symbol,
                    interval=interval,
                    start_str=start_ts,
                    end_str=end_ts
                )
                
                # Convert to DataFrame
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_asset_volume', 'number_of_trades',
                    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
                ])
                
                # Convert timestamp to datetime
                df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
                
                # Convert numeric columns to float
                numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                               'quote_asset_volume', 'taker_buy_base_asset_volume', 
                               'taker_buy_quote_asset_volume']
                df[numeric_cols] = df[numeric_cols].astype(float)
                
                # Drop unnecessary columns
                df = df[['date', 'open', 'high', 'low', 'close', 'volume']]
                
                # Store in dictionary with original symbol as key
                data_dict[symbol] = df
                
            except BinanceAPIException as e:
                print(f"Error fetching data for {symbol}: {e}")
            except Exception as e:
                print(f"Unexpected error for {symbol}: {e}")
        
        return data_dict
    
    def fetch_and_combine(self, symbols, start_date, end_date=None, timeframe='1d'):
        """
        Fetch data for multiple symbols and combine into a single DataFrame with a MultiIndex.
        
        Parameters:
        -----------
        symbols : list
            List of cryptocurrency symbols to fetch (e.g., ['BTC/USD', 'ETH/USD'])
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