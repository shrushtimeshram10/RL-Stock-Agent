# NeuralTrade — AI Stock Trading System

A full-stack AI trading dashboard powered by Reinforcement Learning (PPO) and a dark fintech web interface.

## Project Structure

```
rl_trader/
├── backend/
│   ├── app.py               ← Flask REST API (port 5000)
│   ├── trading_model.py     ← RL model (modified from Colab)
│   └── requirements.txt
│
├── frontend/
│   ├── serve.py             ← Flask static server (port 8080)
│   ├── templates/
│   │   └── index.html       ← Main dashboard HTML
│   └── static/
│       ├── css/style.css
│       └── js/app.js
│
└── README.md
```

---

## Setup & Installation

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 2. Install backend dependencies

```bash
cd backend
pip install -r requirements.txt
```

> **Note:** On Mac M1/M2, you may need:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## Running the App

You need **two terminal windows**.

### Terminal 1 — Start the Backend API

```bash
cd rl_trader/backend
python app.py
```

This starts the Flask REST API on `http://localhost:5000`.

### Terminal 2 — Start the Frontend Server

```bash
cd rl_trader/frontend
pip install flask   # if not already installed
python serve.py
```

This serves the dashboard on `http://localhost:8080`.

### Open in Browser

```
http://localhost:8080
```

---

## How to Use

1. **Dashboard** — Overview of the system, popular stocks
2. **Analyze** — Select a stock (search or click), set your budget, training intensity, date range → click **Run AI Analysis**
3. **Results** — View BUY / HOLD / SELL recommendation, performance metrics, 5 interactive charts, and trade log
4. **About Model** — Technical details of the RL algorithm

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/stocks` | List of popular stocks |
| POST | `/api/analyze` | Start an analysis job |
| GET | `/api/status/<job_id>` | Poll job progress & results |
| GET | `/api/quickprice/<ticker>` | Live price for a ticker |

### POST `/api/analyze` body

```json
{
  "ticker": "AAPL",
  "budget": 10000,
  "start_date": "2015-01-01",
  "end_date": "2024-01-01",
  "timesteps": 50000
}
```

---

## Tips

- **Fast mode (20K steps)** is good for a quick test. Use **Deep (150K)** for more reliable results.
- Indian stocks work too — use NSE tickers like `RELIANCE.NS`, `TCS.NS`, `INFY.NS`
- The model trains fresh each time. For production, save and load trained models per ticker.
- The recommendation is based on the agent's final action in the test period, not a guarantee.

---

## Disclaimer

This tool is for **educational purposes only**. Past performance does not guarantee future returns. Do not make real financial decisions based solely on this model.
