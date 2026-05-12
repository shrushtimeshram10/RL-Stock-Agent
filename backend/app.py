from flask import Flask, jsonify, request
from flask_cors import CORS
import threading
import uuid
import time
import json
import os

from trading_model import (
    download_data, add_technical_indicators, split_data,
    train_agent, backtest, compute_metrics, StockTradingEnv
)
import numpy as np

app = Flask(__name__)
CORS(app)

# In-memory job store
jobs = {}

POPULAR_STOCKS = [
    {"symbol": "AAPL",  "name": "Apple Inc."},
    {"symbol": "MSFT",  "name": "Microsoft Corp."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "AMZN",  "name": "Amazon.com Inc."},
    {"symbol": "TSLA",  "name": "Tesla Inc."},
    {"symbol": "NVDA",  "name": "NVIDIA Corp."},
    {"symbol": "META",  "name": "Meta Platforms"},
    {"symbol": "NFLX",  "name": "Netflix Inc."},
    {"symbol": "BRK-B", "name": "Berkshire Hathaway"},
    {"symbol": "JPM",   "name": "JPMorgan Chase"},
    {"symbol": "V",     "name": "Visa Inc."},
    {"symbol": "JNJ",   "name": "Johnson & Johnson"},
    {"symbol": "WMT",   "name": "Walmart Inc."},
    {"symbol": "PG",    "name": "Procter & Gamble"},
    {"symbol": "DIS",   "name": "Walt Disney Co."},
    {"symbol": "RELIANCE.NS", "name": "Reliance Industries"},
    {"symbol": "TCS.NS",      "name": "Tata Consultancy Services"},
    {"symbol": "INFY.NS",     "name": "Infosys Ltd."},
    {"symbol": "HDFCBANK.NS", "name": "HDFC Bank"},
    {"symbol": "ICICIBANK.NS","name": "ICICI Bank"},
]


def run_pipeline(job_id, ticker, budget, start_date, end_date, timesteps):
    """Full RL training + backtest pipeline run in background thread."""
    try:
        jobs[job_id]["status"] = "downloading"
        jobs[job_id]["progress"] = 5
        raw_df = download_data(ticker, start_date, end_date)

        jobs[job_id]["status"] = "computing_indicators"
        jobs[job_id]["progress"] = 15
        df = add_technical_indicators(raw_df)

        jobs[job_id]["status"] = "splitting_data"
        jobs[job_id]["progress"] = 20
        train_df, test_df = split_data(df, 0.8)

        jobs[job_id]["status"] = "training"
        jobs[job_id]["progress"] = 25

        model, callback = train_agent(
            train_df,
            total_timesteps=timesteps,
            progress_callback=lambda p: _update_training_progress(job_id, p)
        )

        jobs[job_id]["status"] = "backtesting"
        jobs[job_id]["progress"] = 85
        portfolio_values, buy_hold, trade_log, test_df = backtest(model, test_df, budget)

        jobs[job_id]["status"] = "computing_metrics"
        jobs[job_id]["progress"] = 92
        metrics = compute_metrics(portfolio_values, buy_hold)

        # Get current recommendation (last action)
        env_live = StockTradingEnv(test_df, initial_balance=budget)
        obs, _ = env_live.reset()
        last_action = 0
        for _ in range(len(test_df) - 1):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env_live.step(action)
            last_action = int(action)
            if done:
                break

        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        recommendation = action_map[last_action]

        # Serialize chart data
        dates = [str(d)[:10] for d in test_df.index.tolist()]
        close_prices = test_df["Close"].round(2).tolist()
        rsi_values   = test_df["RSI"].round(2).tolist()
        ema20        = test_df["EMA_20"].round(2).tolist()
        ema50        = test_df["EMA_50"].round(2).tolist()
        macd_vals    = test_df["MACD"].round(4).tolist()
        macd_sig     = test_df["MACD_sig"].round(4).tolist()
        bb_upper     = test_df["BB_upper"].round(2).tolist()
        bb_lower     = test_df["BB_lower"].round(2).tolist()

        pv_aligned   = portfolio_values[:len(dates)]
        bh_aligned   = buy_hold[:len(dates)]

        # Drawdown
        pv_arr = np.array(pv_aligned)
        peak   = np.maximum.accumulate(pv_arr)
        dd     = ((pv_arr - peak) / peak * 100).round(2).tolist()

        # Trade signals
        trades = []
        for step, price, ttype in trade_log:
            if step < len(dates):
                trades.append({"date": dates[step], "price": round(float(price), 2), "type": ttype})

        jobs[job_id]["status"] = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["result"] = {
            "ticker":         ticker,
            "recommendation": recommendation,
            "budget":         budget,
            "metrics":        metrics,
            "chart": {
                "dates":      dates,
                "close":      close_prices,
                "rsi":        rsi_values,
                "ema20":      ema20,
                "ema50":      ema50,
                "macd":       macd_vals,
                "macd_signal":macd_sig,
                "bb_upper":   bb_upper,
                "bb_lower":   bb_lower,
                "portfolio":  [round(float(v), 2) for v in pv_aligned],
                "buy_hold":   [round(float(v), 2) for v in bh_aligned],
                "drawdown":   dd,
                "trades":     trades,
            }
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)


def _update_training_progress(job_id, frac):
    """Map training fraction (0..1) to overall progress 25..85."""
    jobs[job_id]["progress"] = int(25 + frac * 60)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/api/stocks", methods=["GET"])
def get_stocks():
    return jsonify(POPULAR_STOCKS)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    data   = request.json
    ticker = data.get("ticker", "AAPL").upper().strip()
    budget = float(data.get("budget", 10000))
    start  = data.get("start_date", "2015-01-01")
    end    = data.get("end_date",   "2024-01-01")
    ts     = int(data.get("timesteps", 50000))

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "queued", "progress": 0, "result": None, "error": None}

    t = threading.Thread(target=run_pipeline, args=(job_id, ticker, budget, start, end, ts), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({
        "status":   job["status"],
        "progress": job["progress"],
        "result":   job["result"],
        "error":    job["error"],
    })


@app.route("/api/quickprice/<ticker>", methods=["GET"])
def quick_price(ticker):
    """Fetch latest price for a ticker via yfinance."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker.upper())
        hist = t.history(period="2d")
        if hist.empty:
            return jsonify({"error": "No data"}), 404
        latest = float(hist["Close"].iloc[-1])
        prev   = float(hist["Close"].iloc[-2]) if len(hist) > 1 else latest
        change_pct = round((latest - prev) / prev * 100, 2)
        return jsonify({"ticker": ticker.upper(), "price": round(latest, 2), "change_pct": change_pct})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
