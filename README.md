# Portfolio Optimization using Reinforcement Learning

This project implements a reinforcement learning-based portfolio optimization system that dynamically adjusts asset allocation based on market conditions to maximize returns while managing risk.

## Features

- **Reinforcement Learning Models**: Implements PPO, Actor-Critic, and other RL algorithms for asset allocation
- **Risk Management**: Balances risk vs. return through sophisticated reward shaping
- **Multi-Asset Support**: Handles stocks, ETFs, cryptocurrencies, and commodities
- **Live Market Data**: Integrates with Alpaca API and Binance API for real-time data
- **Backtesting**: Compares RL models with traditional portfolio optimization strategies

## Project Structure

```
portflo/
├── data/                  # Data storage and preprocessing
│   ├── collectors/        # API integrations for data collection
│   └── processors/        # Data preprocessing and feature engineering
├── models/                # RL models and training scripts
│   ├── agents/            # RL agent implementations
│   └── environments/      # Custom Gym environments
├── strategies/            # Portfolio optimization strategies
│   ├── rl_strategies/     # RL-based strategies
│   └── traditional/       # Traditional portfolio optimization methods
├── backtesting/           # Backtesting framework
├── dashboard/             # Web dashboard for monitoring
├── utils/                 # Utility functions
└── config/                # Configuration files
```

## Installation

1. Clone the repository:

```bash
git clone https://github.com/sarihammad/portflo.git
cd portflo
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up API keys:
   - Create a `.env` file in the root directory
   - Add your API keys for Alpaca and Binance

## Usage

### Data Collection

```bash
python -m portflo.data.collect --source alpaca --start-date 2018-01-01
```

### Training RL Agent

```bash
python -m portflo.models.train --model ppo --risk-level balanced
```

### Backtesting

```bash
python -m portflo.backtesting.run --strategy rl_ppo --benchmark equal_weight
```

### Running the Dashboard

```bash
python -m portflo.dashboard.app
```
