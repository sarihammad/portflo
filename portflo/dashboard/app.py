"""
Web dashboard for visualizing portfolio performance.
"""
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from datetime import datetime, timedelta
import json
import logging

from portflo.config.settings import (
    DATA_DIR, MODELS_DIR, DASHBOARD_HOST, DASHBOARD_PORT, DASHBOARD_DEBUG
)
from portflo.backtesting.backtest import Backtest
from portflo.strategies.traditional.equal_weight import EqualWeightStrategy
from portflo.strategies.traditional.mean_variance import MeanVarianceOptimizer
from portflo.strategies.rl_strategies.rl_strategy import RLStrategy


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize the Dash app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

# Define the app layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("Portfolio Optimization Dashboard", className="text-center my-4"),
            html.Hr()
        ])
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Data Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Asset Types"),
                            dcc.Checklist(
                                id="asset-types",
                                options=[
                                    {"label": "Stocks", "value": "stocks"},
                                    {"label": "ETFs", "value": "etfs"},
                                    {"label": "Crypto", "value": "crypto"}
                                ],
                                value=["stocks", "etfs"],
                                inline=True
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Timeframe"),
                            dcc.Dropdown(
                                id="timeframe",
                                options=[
                                    {"label": "Daily", "value": "1d"},
                                    {"label": "Hourly", "value": "1h"},
                                    {"label": "15 Minutes", "value": "15m"}
                                ],
                                value="1d"
                            )
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Date Range"),
                            dcc.DatePickerRange(
                                id="date-range",
                                start_date_placeholder_text="Start Date",
                                end_date_placeholder_text="End Date",
                                calendar_orientation="horizontal"
                            )
                        ])
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Benchmark"),
                            dcc.Dropdown(
                                id="benchmark",
                                options=[
                                    {"label": "S&P 500 (SPY)", "value": "SPY"},
                                    {"label": "Nasdaq 100 (QQQ)", "value": "QQQ"},
                                    {"label": "Russell 2000 (IWM)", "value": "IWM"},
                                    {"label": "None", "value": "none"}
                                ],
                                value="SPY"
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Rebalancing Frequency"),
                            dcc.Dropdown(
                                id="rebalance-freq",
                                options=[
                                    {"label": "Daily", "value": "D"},
                                    {"label": "Weekly", "value": "W"},
                                    {"label": "Monthly", "value": "M"},
                                    {"label": "Quarterly", "value": "Q"}
                                ],
                                value="M"
                            )
                        ], width=6)
                    ], className="mt-3"),
                    dbc.Button("Load Data", id="load-data-button", color="primary", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Strategy Selection"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("Strategies"),
                            dcc.Checklist(
                                id="strategies",
                                options=[
                                    {"label": "Equal Weight", "value": "equal_weight"},
                                    {"label": "Mean-Variance", "value": "mean_variance"},
                                    {"label": "RL-PPO", "value": "rl_ppo"}
                                ],
                                value=["equal_weight", "mean_variance"],
                                inline=True
                            )
                        ])
                    ]),
                    dbc.Row([
                        dbc.Col([
                            html.Label("Equal Weight Cash %"),
                            dcc.Slider(
                                id="cash-weight",
                                min=0,
                                max=50,
                                step=5,
                                value=5,
                                marks={i: f"{i}%" for i in range(0, 51, 10)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6),
                        dbc.Col([
                            html.Label("Mean-Variance Risk Aversion"),
                            dcc.Slider(
                                id="risk-aversion",
                                min=0.1,
                                max=5,
                                step=0.1,
                                value=1.0,
                                marks={i: str(i) for i in range(1, 6)},
                                tooltip={"placement": "bottom", "always_visible": True}
                            )
                        ], width=6)
                    ], className="mt-3"),
                    dbc.Row([
                        dbc.Col([
                            html.Label("RL Model Path"),
                            dcc.Input(
                                id="rl-model-path",
                                type="text",
                                placeholder="Path to RL model",
                                className="form-control"
                            )
                        ])
                    ], className="mt-3"),
                    dbc.Button("Run Backtest", id="run-backtest-button", color="success", className="mt-3")
                ])
            ], className="mb-4")
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner(
                dcc.Graph(id="portfolio-value-chart")
            )
        ], width=12)
    ]),
    
    dbc.Row([
        dbc.Col([
            dbc.Spinner(
                dcc.Graph(id="drawdown-chart")
            )
        ], width=6),
        dbc.Col([
            dbc.Spinner(
                dcc.Graph(id="metrics-chart")
            )
        ], width=6)
    ]),
    
    dbc.Row([
        dbc.Col([
            html.H3("Performance Metrics", className="text-center my-4"),
            dbc.Spinner(
                html.Div(id="metrics-table")
            )
        ], width=12)
    ]),
    
    # Store components for intermediate data
    dcc.Store(id="price-data-store"),
    dcc.Store(id="backtest-results-store")
    
], fluid=True)


# Define callbacks
@app.callback(
    Output("price-data-store", "data"),
    Input("load-data-button", "n_clicks"),
    [
        State("asset-types", "value"),
        State("timeframe", "value"),
        State("date-range", "start_date"),
        State("date-range", "end_date")
    ],
    prevent_initial_call=True
)
def load_price_data(n_clicks, asset_types, timeframe, start_date, end_date):
    """
    Load price data based on user selections.
    """
    if n_clicks is None:
        return None
    
    logger.info(f"Loading price data for {asset_types} with timeframe {timeframe}")
    
    all_dfs = []
    
    for asset_type in asset_types:
        # Path to processed data
        data_path = os.path.join(DATA_DIR, f"{asset_type}/processed/{timeframe}")
        
        if not os.path.exists(data_path):
            logger.warning(f"No processed data found for {asset_type} with timeframe {timeframe}")
            continue
        
        # Load all CSV files in the directory
        for file in os.listdir(data_path):
            if file.endswith('.csv'):
                # Load data
                file_path = os.path.join(data_path, file)
                df = pd.read_csv(file_path)
                
                # Add symbol column if not present
                if 'symbol' not in df.columns:
                    symbol = file.replace('.csv', '').replace('_', '/')
                    df['symbol'] = symbol
                
                # Convert date to datetime
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                # Append to list
                all_dfs.append(df)
    
    if not all_dfs:
        return {"error": "No data found. Please run data collection first."}
    
    # Combine all DataFrames
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Pivot to get price data in the format: date x asset
    price_data = combined_df.pivot_table(
        index='date', columns='symbol', values='close'
    )
    
    # Filter by date range
    if start_date is not None:
        price_data = price_data[price_data.index >= start_date]
    
    if end_date is not None:
        price_data = price_data[price_data.index <= end_date]
    
    # Forward fill missing values
    price_data = price_data.ffill()
    
    # Convert to JSON serializable format
    price_data_json = {
        "index": price_data.index.strftime('%Y-%m-%d').tolist(),
        "columns": price_data.columns.tolist(),
        "data": price_data.values.tolist()
    }
    
    return price_data_json


@app.callback(
    Output("backtest-results-store", "data"),
    Input("run-backtest-button", "n_clicks"),
    [
        State("price-data-store", "data"),
        State("strategies", "value"),
        State("benchmark", "value"),
        State("rebalance-freq", "value"),
        State("cash-weight", "value"),
        State("risk-aversion", "value"),
        State("rl-model-path", "value")
    ],
    prevent_initial_call=True
)
def run_backtest(
    n_clicks, price_data_json, strategies, benchmark, 
    rebalance_freq, cash_weight, risk_aversion, rl_model_path
):
    """
    Run backtest for selected strategies.
    """
    if n_clicks is None or price_data_json is None:
        return None
    
    if "error" in price_data_json:
        return {"error": price_data_json["error"]}
    
    # Convert JSON data back to DataFrame
    price_data = pd.DataFrame(
        data=price_data_json["data"],
        index=pd.to_datetime(price_data_json["index"]),
        columns=price_data_json["columns"]
    )
    
    # Set benchmark to None if "none" is selected
    if benchmark == "none":
        benchmark = None
    
    # Initialize strategies
    strategy_objects = []
    
    if "equal_weight" in strategies:
        logger.info("Initializing Equal-Weight strategy")
        equal_weight = EqualWeightStrategy(
            include_cash=True,
            cash_weight=cash_weight / 100.0  # Convert from percentage
        )
        strategy_objects.append((equal_weight, "Equal-Weight"))
    
    if "mean_variance" in strategies:
        logger.info("Initializing Mean-Variance strategy")
        mean_variance = MeanVarianceOptimizer(
            risk_aversion=risk_aversion,
            min_weight=0.0,
            max_weight=0.3  # Limit maximum weight to 30% for diversification
        )
        strategy_objects.append((mean_variance, "Mean-Variance"))
    
    if "rl_ppo" in strategies:
        if rl_model_path is None or rl_model_path == "":
            logger.warning("RL model path not provided, skipping RL strategy")
        else:
            logger.info("Initializing RL strategy")
            try:
                rl_strategy = RLStrategy(
                    model_path=rl_model_path
                )
                strategy_objects.append((rl_strategy, "RL-PPO"))
            except Exception as e:
                logger.error(f"Error initializing RL strategy: {e}")
                return {"error": f"Error initializing RL strategy: {e}"}
    
    # Run backtests
    results = []
    
    for strategy, strategy_name in strategy_objects:
        try:
            # Create backtest
            backtest = Backtest(
                price_data=price_data,
                strategy=strategy,
                strategy_name=strategy_name,
                benchmark=benchmark,
                rebalance_freq=rebalance_freq
            )
            
            # Run backtest
            logger.info(f"Running backtest for {strategy_name}")
            result = backtest.run()
            
            # Convert result to JSON serializable format
            result_json = {
                "strategy_name": result.strategy_name,
                "portfolio_values": {
                    "index": result.portfolio_values.index.strftime('%Y-%m-%d').tolist(),
                    "values": result.portfolio_values.values.tolist()
                },
                "weights": {
                    "index": result.weights.index.strftime('%Y-%m-%d').tolist(),
                    "columns": result.weights.columns.tolist(),
                    "values": result.weights.values.tolist()
                },
                "returns": {
                    "index": result.returns.index.strftime('%Y-%m-%d').tolist(),
                    "values": result.returns.values.tolist()
                },
                "metrics": result.metrics
            }
            
            # Add benchmark values if available
            if result.benchmark_values is not None:
                result_json["benchmark_values"] = {
                    "index": result.benchmark_values.index.strftime('%Y-%m-%d').tolist(),
                    "values": result.benchmark_values.values.tolist()
                }
            
            results.append(result_json)
            
        except Exception as e:
            logger.error(f"Error running backtest for {strategy_name}: {e}")
            return {"error": f"Error running backtest for {strategy_name}: {e}"}
    
    return results


@app.callback(
    [
        Output("portfolio-value-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("metrics-chart", "figure"),
        Output("metrics-table", "children")
    ],
    Input("backtest-results-store", "data"),
    State("benchmark", "value"),
    prevent_initial_call=True
)
def update_charts(results, benchmark):
    """
    Update charts and metrics table based on backtest results.
    """
    if results is None:
        return go.Figure(), go.Figure(), go.Figure(), html.Div()
    
    if "error" in results:
        return (
            go.Figure(layout=dict(title=f"Error: {results['error']}")),
            go.Figure(),
            go.Figure(),
            html.Div(f"Error: {results['error']}")
        )
    
    # Portfolio Value Chart
    portfolio_fig = go.Figure()
    
    for result in results:
        # Normalize portfolio values
        portfolio_values = np.array(result["portfolio_values"]["values"])
        normalized_values = portfolio_values / portfolio_values[0]
        
        portfolio_fig.add_trace(go.Scatter(
            x=pd.to_datetime(result["portfolio_values"]["index"]),
            y=normalized_values,
            mode="lines",
            name=result["strategy_name"]
        ))
    
    # Add benchmark if available
    if "benchmark_values" in results[0]:
        benchmark_values = np.array(results[0]["benchmark_values"]["values"])
        normalized_benchmark = benchmark_values / benchmark_values[0]
        
        portfolio_fig.add_trace(go.Scatter(
            x=pd.to_datetime(results[0]["benchmark_values"]["index"]),
            y=normalized_benchmark,
            mode="lines",
            name=benchmark,
            line=dict(dash="dash")
        ))
    
    portfolio_fig.update_layout(
        title="Portfolio Value (Normalized)",
        xaxis_title="Date",
        yaxis_title="Value (Normalized)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Drawdown Chart
    drawdown_fig = go.Figure()
    
    for result in results:
        # Calculate drawdown
        portfolio_values = np.array(result["portfolio_values"]["values"])
        rolling_max = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        drawdown_fig.add_trace(go.Scatter(
            x=pd.to_datetime(result["portfolio_values"]["index"]),
            y=drawdown,
            mode="lines",
            name=result["strategy_name"]
        ))
    
    drawdown_fig.update_layout(
        title="Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )
    
    # Metrics Chart
    metrics_data = []
    
    for result in results:
        metrics_data.append({
            "Strategy": result["strategy_name"],
            "Annualized Return": result["metrics"]["annualized_return"] * 100,
            "Volatility": result["metrics"]["volatility"] * 100,
            "Sharpe Ratio": result["metrics"]["sharpe_ratio"],
            "Sortino Ratio": result["metrics"]["sortino_ratio"],
            "Max Drawdown": result["metrics"]["max_drawdown"] * 100
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    metrics_fig = px.bar(
        metrics_df,
        x="Strategy",
        y=["Annualized Return", "Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"],
        barmode="group",
        title="Performance Metrics"
    )
    
    metrics_fig.update_layout(
        xaxis_title="Strategy",
        yaxis_title="Value",
        legend_title="Metric",
        template="plotly_white"
    )
    
    # Metrics Table
    metrics_table = dbc.Table.from_dataframe(
        metrics_df,
        striped=True,
        bordered=True,
        hover=True,
        responsive=True
    )
    
    return portfolio_fig, drawdown_fig, metrics_fig, metrics_table


def main():
    """
    Run the dashboard app.
    """
    app.run_server(
        host=DASHBOARD_HOST,
        port=DASHBOARD_PORT,
        debug=DASHBOARD_DEBUG
    )


if __name__ == "__main__":
    main() 