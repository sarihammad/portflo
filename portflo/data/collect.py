"""
Script for collecting market data from various sources.
"""
import os
import argparse
import pandas as pd
from datetime import datetime
import logging

from portflo.config.settings import (
    DATA_DIR, DEFAULT_START_DATE, DEFAULT_STOCKS, 
    DEFAULT_ETFS, DEFAULT_CRYPTO, DEFAULT_TIMEFRAME
)
from portflo.data.collectors.alpaca_collector import AlpacaDataCollector
from portflo.data.collectors.binance_collector import BinanceDataCollector
from portflo.data.collectors.yahoo_collector import YahooDataCollector
from portflo.data.processors.data_processor import DataProcessor


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def collect_stock_data(symbols, start_date, end_date=None, timeframe='1d', use_alpaca=True):
    """
    Collect stock and ETF data from Alpaca or Yahoo Finance.
    
    Parameters:
    -----------
    symbols : list
        List of stock/ETF symbols to fetch
    start_date : str
        Start date for the data (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for the data (format: 'YYYY-MM-DD'). If None, uses current date.
    timeframe : str, default '1d'
        Timeframe for the data ('1d', '1h', '15m', '5m', '1m')
    use_alpaca : bool, default True
        Whether to use Alpaca API (True) or Yahoo Finance (False)
        
    Returns:
    --------
    dict
        Dictionary with symbols as keys and pandas DataFrames as values
    """
    try:
        if use_alpaca:
            logger.info(f"Collecting stock data from Alpaca for {len(symbols)} symbols")
            collector = AlpacaDataCollector()
        else:
            logger.info(f"Collecting stock data from Yahoo Finance for {len(symbols)} symbols")
            collector = YahooDataCollector()
        
        data = collector.fetch_data(symbols, start_date, end_date, timeframe)
        logger.info(f"Successfully collected data for {len(data)} symbols")
        return data
    
    except Exception as e:
        logger.error(f"Error collecting stock data: {e}")
        # Fallback to Yahoo Finance if Alpaca fails
        if use_alpaca:
            logger.info("Falling back to Yahoo Finance")
            return collect_stock_data(symbols, start_date, end_date, timeframe, use_alpaca=False)
        else:
            return {}


def collect_crypto_data(symbols, start_date, end_date=None, timeframe='1d'):
    """
    Collect cryptocurrency data from Binance.
    
    Parameters:
    -----------
    symbols : list
        List of cryptocurrency symbols to fetch (e.g., ['BTC/USD', 'ETH/USD'])
    start_date : str
        Start date for the data (format: 'YYYY-MM-DD')
    end_date : str, optional
        End date for the data (format: 'YYYY-MM-DD'). If None, uses current date.
    timeframe : str, default '1d'
        Timeframe for the data ('1d', '1h', '15m', '5m', '1m')
        
    Returns:
    --------
    dict
        Dictionary with symbols as keys and pandas DataFrames as values
    """
    try:
        logger.info(f"Collecting crypto data from Binance for {len(symbols)} symbols")
        collector = BinanceDataCollector()
        data = collector.fetch_data(symbols, start_date, end_date, timeframe)
        logger.info(f"Successfully collected data for {len(data)} symbols")
        return data
    
    except Exception as e:
        logger.error(f"Error collecting crypto data: {e}")
        return {}


def save_data(data, data_dir, asset_type, timeframe):
    """
    Save collected data to CSV files.
    
    Parameters:
    -----------
    data : dict
        Dictionary with symbols as keys and pandas DataFrames as values
    data_dir : str
        Directory to save the data
    asset_type : str
        Type of asset ('stocks', 'etfs', 'crypto')
    timeframe : str
        Timeframe of the data ('1d', '1h', '15m', '5m', '1m')
    """
    # Create directory if it doesn't exist
    save_dir = os.path.join(data_dir, asset_type, timeframe)
    os.makedirs(save_dir, exist_ok=True)
    
    # Save each DataFrame to a CSV file
    for symbol, df in data.items():
        # Clean symbol for filename (replace / with _)
        clean_symbol = symbol.replace('/', '_')
        
        # Save to CSV
        csv_path = os.path.join(save_dir, f"{clean_symbol}.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved data for {symbol} to {csv_path}")


def process_and_save_data(data, data_dir, asset_type, timeframe):
    """
    Process and save collected data.
    
    Parameters:
    -----------
    data : dict
        Dictionary with symbols as keys and pandas DataFrames as values
    data_dir : str
        Directory to save the data
    asset_type : str
        Type of asset ('stocks', 'etfs', 'crypto')
    timeframe : str
        Timeframe of the data ('1d', '1h', '15m', '5m', '1m')
    """
    if not data:
        logger.warning(f"No data to process for {asset_type}")
        return
    
    # Process data
    logger.info(f"Processing {asset_type} data")
    processor = DataProcessor()
    processed_data = processor.process_data(data)
    
    # Save raw data
    logger.info(f"Saving raw {asset_type} data")
    save_data(data, data_dir, f"{asset_type}/raw", timeframe)
    
    # Save processed data
    logger.info(f"Saving processed {asset_type} data")
    save_data(processed_data, data_dir, f"{asset_type}/processed", timeframe)


def main():
    """
    Main function to collect and process market data.
    """
    parser = argparse.ArgumentParser(description='Collect market data')
    parser.add_argument('--source', type=str, choices=['alpaca', 'yahoo', 'binance', 'all'], 
                        default='all', help='Data source')
    parser.add_argument('--start-date', type=str, default=DEFAULT_START_DATE, 
                        help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, 
                        help='End date (YYYY-MM-DD)')
    parser.add_argument('--timeframe', type=str, default=DEFAULT_TIMEFRAME, 
                        choices=['1d', '1h', '15m', '5m', '1m'], 
                        help='Data timeframe')
    parser.add_argument('--stocks', type=str, nargs='+', default=DEFAULT_STOCKS, 
                        help='Stock symbols to collect')
    parser.add_argument('--etfs', type=str, nargs='+', default=DEFAULT_ETFS, 
                        help='ETF symbols to collect')
    parser.add_argument('--crypto', type=str, nargs='+', default=DEFAULT_CRYPTO, 
                        help='Cryptocurrency symbols to collect')
    
    args = parser.parse_args()
    
    # Collect stock data
    if args.source in ['alpaca', 'yahoo', 'all']:
        # Collect stocks
        stock_data = collect_stock_data(
            args.stocks, args.start_date, args.end_date, args.timeframe,
            use_alpaca=(args.source != 'yahoo')
        )
        process_and_save_data(stock_data, DATA_DIR, 'stocks', args.timeframe)
        
        # Collect ETFs
        etf_data = collect_stock_data(
            args.etfs, args.start_date, args.end_date, args.timeframe,
            use_alpaca=(args.source != 'yahoo')
        )
        process_and_save_data(etf_data, DATA_DIR, 'etfs', args.timeframe)
    
    # Collect crypto data
    if args.source in ['binance', 'all']:
        crypto_data = collect_crypto_data(
            args.crypto, args.start_date, args.end_date, args.timeframe
        )
        process_and_save_data(crypto_data, DATA_DIR, 'crypto', args.timeframe)
    
    logger.info("Data collection completed")


if __name__ == '__main__':
    main() 