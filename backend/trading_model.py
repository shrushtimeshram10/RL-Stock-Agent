# -*- coding: utf-8 -*-
"""
trading_model.py — RL Trading Agent
Modified from Colab notebook to support Flask web API with progress callbacks.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import pandas_ta as ta
import warnings
warnings.filterwarnings("ignore")

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback


# ── Data ──────────────────────────────────────────────────────────────────────

def download_data(ticker="AAPL", start="2015-01-01", end="2024-01-01"):
    """Download historical OHLCV data from Yahoo Finance."""
    print(f"Downloading {ticker} data from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    df.dropna(inplace=True)
    print(f"Downloaded {len(df)} trading days.")
    return df


def add_technical_indicators(df):
    """Compute and append technical indicators."""
    print("Computing technical indicators...")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["EMA_20"]   = ta.ema(df["Close"], length=20)
    df["EMA_50"]   = ta.ema(df["Close"], length=50)
    df["RSI"]      = ta.rsi(df["Close"], length=14)

    macd           = ta.macd(df["Close"])
    df["MACD"]     = macd["MACD_12_26_9"]
    df["MACD_sig"] = macd["MACDs_12_26_9"]

    bb             = ta.bbands(df["Close"], length=20)
    bb_upper_col   = [c for c in bb.columns if c.startswith("BBU_")][0]
    bb_lower_col   = [c for c in bb.columns if c.startswith("BBL_")][0]
    df["BB_upper"] = bb[bb_upper_col]
    df["BB_lower"] = bb[bb_lower_col]
    df["BB_width"] = (df["BB_upper"] - df["BB_lower"]) / df["Close"]

    df["Volume_norm"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["Return"]      = df["Close"].pct_change()

    df.dropna(inplace=True)
    print(f"Indicators added. Shape: {df.shape}")
    return df


def split_data(df, train_ratio=0.8):
    split    = int(len(df) * train_ratio)
    train_df = df.iloc[:split].copy()
    test_df  = df.iloc[split:].copy()
    print(f"Train: {len(train_df)} days | Test: {len(test_df)} days")
    return train_df, test_df


# ── Environment ───────────────────────────────────────────────────────────────

class StockTradingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    FEATURE_COLS = [
        "Close", "Volume_norm",
        "EMA_20", "EMA_50",
        "RSI", "MACD", "MACD_sig",
        "BB_width", "Return"
    ]

    def __init__(self, df, initial_balance=10_000, render_mode=None):
        super().__init__()
        self.df              = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.render_mode     = render_mode

        n_features = len(self.FEATURE_COLS) + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(n_features,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._max_steps   = len(self.df) - 1
        self.reset()

    def _get_obs(self):
        row      = self.df.iloc[self.current_step]
        close    = row["Close"]
        features = []
        for col in self.FEATURE_COLS:
            val = row[col]
            if col == "Close":
                val = val / self.initial_balance
            elif col in ("EMA_20", "EMA_50"):
                val = (val - close) / (close + 1e-8)
            elif col == "RSI":
                val = val / 100.0
            elif col in ("MACD", "MACD_sig"):
                val = val / (close + 1e-8)
            features.append(float(val))

        cash_ratio       = self.cash / self.initial_balance
        shares_held_norm = (self.shares_held * close) / self.initial_balance
        unrealized_pnl   = (close - self.buy_price) / (self.buy_price + 1e-8) if self.shares_held > 0 else 0.0
        features += [cash_ratio, shares_held_norm, unrealized_pnl]
        return np.array(features, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step    = 0
        self.cash            = self.initial_balance
        self.shares_held     = 0.0
        self.buy_price       = 0.0
        self.portfolio_values = [self.initial_balance]
        self.trade_log       = []
        return self._get_obs(), {}

    def step(self, action):
        row        = self.df.iloc[self.current_step]
        price      = row["Close"]
        prev_value = self._portfolio_value(price)

        if action == 1 and self.cash > 0:
            self.shares_held = self.cash / price
            self.buy_price   = price
            self.cash        = 0.0
            self.trade_log.append((self.current_step, price, "BUY"))
        elif action == 2 and self.shares_held > 0:
            self.cash        = self.shares_held * price
            self.trade_log.append((self.current_step, price, "SELL"))
            self.shares_held = 0.0
            self.buy_price   = 0.0

        self.current_step += 1
        done      = self.current_step >= self._max_steps
        new_price = self.df.iloc[self.current_step]["Close"]
        new_value = self._portfolio_value(new_price)
        self.portfolio_values.append(new_value)

        reward   = (new_value - prev_value) / (prev_value + 1e-8)
        peak     = max(self.portfolio_values)
        drawdown = (peak - new_value) / (peak + 1e-8)
        reward  -= 0.1 * drawdown

        return self._get_obs(), float(reward), done, False, {}

    def _portfolio_value(self, price):
        return self.cash + self.shares_held * price

    def render(self):
        price = self.df.iloc[self.current_step]["Close"]
        pv    = self._portfolio_value(price)
        print(f"Step {self.current_step:4d} | Price: ${price:.2f} | Portfolio: ${pv:,.2f}")


# ── Training Callback ──────────────────────────────────────────────────────────

class TrainingCallback(BaseCallback):
    """Callback that reports progress fraction via optional external callback."""

    def __init__(self, check_freq=5000, total_timesteps=50000,
                 progress_callback=None, verbose=0):
        super().__init__(verbose)
        self.check_freq        = check_freq
        self.total_timesteps   = total_timesteps
        self.progress_callback = progress_callback
        self.rewards           = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            mean_r = np.mean(self.locals.get("rewards", [0]))
            self.rewards.append(mean_r)
            if self.verbose:
                print(f"  Step {self.n_calls:>8,} | Mean reward: {mean_r:.5f}")
            if self.progress_callback:
                frac = min(self.n_calls / max(self.total_timesteps, 1), 1.0)
                self.progress_callback(frac)
        return True


# ── Train ─────────────────────────────────────────────────────────────────────

def train_agent(train_df, total_timesteps=50_000, progress_callback=None):
    print(f"\nTraining PPO agent for {total_timesteps:,} timesteps...")
    env = DummyVecEnv([lambda: StockTradingEnv(train_df)])

    model = PPO(
        policy         = "MlpPolicy",
        env            = env,
        learning_rate  = 3e-4,
        n_steps        = 2048,
        batch_size     = 64,
        n_epochs       = 10,
        gamma          = 0.99,
        gae_lambda     = 0.95,
        clip_range     = 0.2,
        ent_coef       = 0.01,
        verbose        = 0,
    )

    callback = TrainingCallback(
        check_freq      = max(1000, total_timesteps // 50),
        total_timesteps = total_timesteps,
        progress_callback = progress_callback,
    )
    model.learn(total_timesteps=total_timesteps, callback=callback)
    print("Training complete!")
    return model, callback


# ── Backtest ──────────────────────────────────────────────────────────────────

def backtest(model, test_df, initial_balance=10_000):
    print("Running backtest...")
    env  = StockTradingEnv(test_df, initial_balance=initial_balance)
    obs, _ = env.reset()

    portfolio_values = [initial_balance]
    actions_taken    = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done, _, _ = env.step(action)
        price = test_df.iloc[env.current_step]["Close"]
        portfolio_values.append(env._portfolio_value(price))
        actions_taken.append(int(action))
        if done:
            break

    start_price = test_df.iloc[0]["Close"]
    buy_hold    = [initial_balance * (p / start_price) for p in test_df["Close"].values]

    print(f"Backtest complete. Trades: {len(env.trade_log)}")
    return portfolio_values, buy_hold, env.trade_log, test_df


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(portfolio_values, benchmark_values, label="RL Agent"):
    pv = np.array(portfolio_values)
    bv = np.array(benchmark_values)
    n  = min(len(pv), len(bv))
    pv, bv = pv[:n], bv[:n]

    def sharpe(vals):
        rets = np.diff(vals) / vals[:-1]
        return (np.mean(rets) / (np.std(rets) + 1e-8)) * np.sqrt(252)

    def max_drawdown(vals):
        peak = np.maximum.accumulate(vals)
        return ((vals - peak) / peak).min()

    def total_return(vals):
        return (vals[-1] - vals[0]) / vals[0] * 100

    return {
        "rl_total_return":   round(total_return(pv), 2),
        "bh_total_return":   round(total_return(bv), 2),
        "rl_sharpe":         round(sharpe(pv), 3),
        "bh_sharpe":         round(sharpe(bv), 3),
        "rl_max_drawdown":   round(max_drawdown(pv) * 100, 2),
        "bh_max_drawdown":   round(max_drawdown(bv) * 100, 2),
        "rl_final_value":    round(float(pv[-1]), 2),
        "bh_final_value":    round(float(bv[-1]), 2),
    }
