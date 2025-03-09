"""
Global configuration settings for the portfolio optimization project.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data_storage"
MODELS_DIR = BASE_DIR / "saved_models"

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# API Keys (loaded from environment variables)
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")  # Default to paper trading

BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_SECRET_KEY = os.getenv("BINANCE_SECRET_KEY")

# Data collection settings
DEFAULT_START_DATE = "2018-01-01"
DEFAULT_END_DATE = None  # None means current date
DEFAULT_TIMEFRAME = "1d"  # Daily data

# Asset universe
DEFAULT_STOCKS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "PG", "XOM", "BAC", "DIS"
]

DEFAULT_ETFS = [
    "SPY", "QQQ", "IWM", "VTI", "GLD",
    "TLT", "VGK", "EEM", "VNQ", "XLE"
]

DEFAULT_CRYPTO = [
    "BTC/USD", "ETH/USD", "BNB/USD", "SOL/USD", "ADA/USD"
]

# RL Environment settings
LOOKBACK_WINDOW = 30  # Number of days to look back for state representation
MAX_TRADING_DAYS = 252  # Approximate number of trading days in a year
INITIAL_PORTFOLIO_VALUE = 100000  # Initial portfolio value in USD

# RL Training settings
TRAINING_EPISODES = 1000
VALIDATION_EPISODES = 100
TEST_EPISODES = 100

# Risk management settings
RISK_LEVELS = {
    "conservative": {
        "max_allocation_per_asset": 0.15,
        "min_cash_position": 0.20,
        "risk_penalty_factor": 2.0,
    },
    "balanced": {
        "max_allocation_per_asset": 0.25,
        "min_cash_position": 0.10,
        "risk_penalty_factor": 1.0,
    },
    "aggressive": {
        "max_allocation_per_asset": 0.40,
        "min_cash_position": 0.05,
        "risk_penalty_factor": 0.5,
    }
}

# Backtesting settings
TRANSACTION_FEE_RATE = 0.001  # 0.1% per transaction
SLIPPAGE_RATE = 0.0005  # 0.05% slippage

# Dashboard settings
DASHBOARD_HOST = "127.0.0.1"
DASHBOARD_PORT = 8050
DASHBOARD_DEBUG = True 