/* ── NeuralTrade Frontend ──────────────────────────────────────────────── */

const API = 'http://localhost:5000/api';

let allStocks = [];
let currentJobId = null;
let pollInterval = null;
let currentChart = null;
let lastResult = null;

// ── Init ──────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadStocks();
  bindNav();
  bindSearch();
  bindChartTabs();
  fetchLivePrices();
});

// ── Navigation ────────────────────────────────────────────────────────────
function bindNav() {
  document.querySelectorAll('.nav-item').forEach(el => {
    el.addEventListener('click', e => {
      e.preventDefault();
      switchView(el.dataset.view);
    });
  });
}

function switchView(view) {
  document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById(`view-${view}`)?.classList.add('active');
  document.querySelector(`[data-view="${view}"]`)?.classList.add('active');
  const titles = { dashboard: 'Dashboard', analyze: 'Analyze', results: 'Results', about: 'About Model' };
  document.getElementById('pageTitle').textContent = titles[view] || view;
}

// ── Stock list ────────────────────────────────────────────────────────────
async function loadStocks() {
  try {
    const res = await fetch(`${API}/stocks`);
    allStocks = await res.json();
    renderStocksGrid(allStocks);
  } catch {
    // fallback static list
    allStocks = [
      {symbol:'AAPL',name:'Apple Inc.'},{symbol:'MSFT',name:'Microsoft'},
      {symbol:'GOOGL',name:'Alphabet'},{symbol:'TSLA',name:'Tesla'},
      {symbol:'NVDA',name:'NVIDIA'},{symbol:'AMZN',name:'Amazon'},
      {symbol:'META',name:'Meta'},{symbol:'NFLX',name:'Netflix'},
    ];
    renderStocksGrid(allStocks);
  }
}

function renderStocksGrid(stocks) {
  const grid = document.getElementById('stocksGrid');
  grid.innerHTML = stocks.map(s => `
    <div class="stock-card" onclick="selectStock('${s.symbol}')">
      <div class="stock-sym">${s.symbol}</div>
      <div class="stock-name">${s.name}</div>
      <div class="stock-price" id="price-${s.symbol}">—</div>
    </div>
  `).join('');
}

async function fetchLivePrices() {
  const symbols = ['AAPL', 'TSLA', 'NVDA', 'MSFT', 'GOOGL'];
  for (const sym of symbols) {
    try {
      const res = await fetch(`${API}/quickprice/${sym}`);
      if (!res.ok) continue;
      const d = await res.json();
      updatePriceDisplay(sym, d.price, d.change_pct);
    } catch {}
  }
}

function updatePriceDisplay(sym, price, chg) {
  const el = document.getElementById(`price-${sym}`);
  if (el) {
    const sign = chg >= 0 ? '+' : '';
    const cls  = chg >= 0 ? 'chg-pos' : 'chg-neg';
    el.innerHTML = `$${price} <span class="${cls}">${sign}${chg}%</span>`;
  }
  // topbar
  document.querySelectorAll('.ticker-item').forEach(t => {
    if (t.querySelector('.ticker-sym')?.textContent === sym) {
      const priceEl = t.querySelector('.ticker-price');
      if (priceEl) priceEl.textContent = `$${price}`;
    }
  });
}

function selectStock(symbol) {
  document.getElementById('tickerInput').value = symbol;
  document.getElementById('tickerPreview').textContent = `Selected: ${symbol}`;
  switchView('analyze');
}

// ── Search autocomplete ───────────────────────────────────────────────────
function bindSearch() {
  const input = document.getElementById('tickerInput');
  const dd    = document.getElementById('searchDropdown');

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) { dd.classList.remove('open'); return; }
    const matches = allStocks.filter(s =>
      s.symbol.toLowerCase().includes(q) || s.name.toLowerCase().includes(q)
    ).slice(0, 8);
    if (!matches.length) { dd.classList.remove('open'); return; }
    dd.innerHTML = matches.map(s => `
      <div class="dd-item" onclick="pickStock('${s.symbol}','${s.name}')">
        <span class="dd-sym">${s.symbol}</span>
        <span class="dd-name">${s.name}</span>
      </div>
    `).join('');
    dd.classList.add('open');
  });

  document.addEventListener('click', e => {
    if (!input.contains(e.target) && !dd.contains(e.target)) {
      dd.classList.remove('open');
    }
  });
}

function pickStock(sym, name) {
  document.getElementById('tickerInput').value = sym;
  document.getElementById('tickerPreview').textContent = name;
  document.getElementById('searchDropdown').classList.remove('open');
}

// ── Budget presets ────────────────────────────────────────────────────────
function setBudget(val) {
  document.getElementById('budgetInput').value = val;
  document.querySelectorAll('.preset-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
}

// ── Run Analysis ──────────────────────────────────────────────────────────
async function runAnalysis() {
  const ticker = document.getElementById('tickerInput').value.trim().toUpperCase();
  if (!ticker) { alert('Please enter a stock ticker.'); return; }

  const budget    = parseFloat(document.getElementById('budgetInput').value) || 10000;
  const startDate = document.getElementById('startDate').value;
  const endDate   = document.getElementById('endDate').value;
  const ts        = document.querySelector('input[name="intensity"]:checked')?.value || 50000;

  // Show progress panel
  document.getElementById('progressPanel').style.display = 'block';
  document.getElementById('runBtn').disabled = true;
  document.getElementById('runBtnText').textContent = 'Running...';
  setStageActive('downloading');

  try {
    const res = await fetch(`${API}/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ticker, budget, start_date: startDate, end_date: endDate, timesteps: parseInt(ts) })
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);
    currentJobId = data.job_id;
    startPolling();
  } catch (err) {
    alert('Error starting analysis: ' + err.message);
    resetRunBtn();
  }
}

const STATUS_MAP = {
  downloading:          { title: 'Downloading Data',          sub: 'Fetching historical OHLCV from Yahoo Finance', stage: 'downloading' },
  computing_indicators: { title: 'Computing Indicators',      sub: 'Calculating RSI, MACD, Bollinger Bands, EMA',   stage: 'indicators' },
  splitting_data:       { title: 'Splitting Dataset',         sub: 'Creating train / test split (80/20)',            stage: 'indicators' },
  training:             { title: 'Training PPO Agent',        sub: 'Reinforcement learning in progress…',           stage: 'training' },
  backtesting:          { title: 'Running Backtest',          sub: 'Testing agent on unseen out-of-sample data',    stage: 'backtesting' },
  computing_metrics:    { title: 'Computing Metrics',         sub: 'Calculating Sharpe ratio, drawdown, returns',   stage: 'backtesting' },
  done:                 { title: 'Analysis Complete!',         sub: 'Redirecting to results…',                       stage: 'done' },
  error:                { title: 'Error',                     sub: 'Something went wrong',                          stage: null },
};

function startPolling() {
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`${API}/status/${currentJobId}`);
      const data = await res.json();
      updateProgress(data);
      if (data.status === 'done' || data.status === 'error') {
        clearInterval(pollInterval);
        if (data.status === 'done') {
          lastResult = data.result;
          setTimeout(() => {
            renderResults(data.result);
            switchView('results');
            resetRunBtn();
          }, 800);
        } else {
          alert('Analysis failed: ' + data.error);
          resetRunBtn();
        }
      }
    } catch {}
  }, 1500);
}

function updateProgress(data) {
  const pct  = data.progress || 0;
  const info = STATUS_MAP[data.status] || {};
  document.getElementById('progressBar').style.width = pct + '%';
  document.getElementById('progressPct').textContent  = pct + '%';
  if (info.title) document.getElementById('progressTitle').textContent = info.title;
  if (info.sub)   document.getElementById('progressSub').textContent   = info.sub;
  if (info.stage) setStageActive(info.stage);
}

function setStageActive(active) {
  const order = ['downloading','indicators','training','backtesting','done'];
  const idx   = order.indexOf(active);
  order.forEach((s, i) => {
    const el = document.getElementById(`stage-${s}`);
    if (!el) return;
    el.classList.remove('active','done');
    if (i < idx)  el.classList.add('done');
    if (i === idx) el.classList.add('active');
  });
}

function resetRunBtn() {
  document.getElementById('runBtn').disabled = false;
  document.getElementById('runBtnText').textContent = 'Run AI Analysis';
}

// ── Render Results ────────────────────────────────────────────────────────
function renderResults(result) {
  document.getElementById('noResults').style.display = 'none';
  document.getElementById('resultsContent').style.display = 'block';

  const m = result.metrics;
  document.getElementById('recTicker').textContent = result.ticker;
  const recEl = document.getElementById('recAction');
  recEl.textContent = result.recommendation;
  recEl.className   = 'rec-action ' + result.recommendation;

  const subMap = { BUY: '↑ Model suggests entering a position', HOLD: '→ Model suggests maintaining position', SELL: '↓ Model suggests exiting position' };
  document.getElementById('recSub').textContent = subMap[result.recommendation] || '';

  // Metrics
  const fmtPct = v => (v >= 0 ? '+' : '') + v + '%';
  const fmtDol = v => '$' + v.toLocaleString();

  setMetric('mRlReturn', fmtPct(m.rl_total_return), m.rl_total_return >= 0);
  setMetric('mBhReturn', fmtPct(m.bh_total_return), m.bh_total_return >= 0);
  setMetric('mRlSharpe', m.rl_sharpe, m.rl_sharpe >= 0);
  setMetric('mMaxDD',    m.rl_max_drawdown + '%', false);
  setMetric('mRlFinal',  fmtDol(m.rl_final_value), m.rl_final_value >= result.budget);
  setMetric('mBhFinal',  fmtDol(m.bh_final_value), m.bh_final_value >= result.budget);

  // Trade table
  const tbody = document.getElementById('tradeTableBody');
  tbody.innerHTML = result.chart.trades.map((t, i) => `
    <tr>
      <td class="mono">${i + 1}</td>
      <td>${t.date}</td>
      <td class="mono">$${t.price.toLocaleString()}</td>
      <td><span class="badge-${t.type.toLowerCase()}">${t.type}</span></td>
    </tr>
  `).join('');
  document.getElementById('tradeCount').textContent = result.chart.trades.length + ' trades';

  renderChart('equity', result.chart);
}

function setMetric(id, val, positive) {
  const el = document.getElementById(id);
  el.textContent = val;
  el.className = 'metric-value' + (positive === true ? ' positive' : positive === false ? ' negative' : '');
}

// ── Chart Rendering ───────────────────────────────────────────────────────
function bindChartTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      if (lastResult) renderChart(btn.dataset.chart, lastResult.chart);
    });
  });
}

function renderChart(type, chart) {
  if (currentChart) { currentChart.destroy(); currentChart = null; }
  const ctx = document.getElementById('mainChart').getContext('2d');

  Chart.defaults.color = '#8892a4';
  Chart.defaults.borderColor = 'rgba(255,255,255,0.05)';

  const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    interaction: { mode: 'index', intersect: false },
    plugins: { legend: { position: 'top', labels: { boxWidth: 10, padding: 16, font: { family: 'Space Mono', size: 10 } } }, tooltip: { backgroundColor: '#141720', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1 } },
    scales: { x: { ticks: { maxTicksLimit: 10, font: { family: 'Space Mono', size: 9 } }, grid: { display: false } }, y: { ticks: { font: { family: 'Space Mono', size: 9 } } } }
  };

  const dates = chart.dates;

  if (type === 'equity') {
    currentChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          { label: 'RL Agent',    data: chart.portfolio, borderColor: '#00e5ff', backgroundColor: 'rgba(0,229,255,0.05)', borderWidth: 2, pointRadius: 0, tension: 0.3 },
          { label: 'Buy & Hold', data: chart.buy_hold,  borderColor: '#7c4dff', backgroundColor: 'rgba(124,77,255,0.05)', borderWidth: 2, pointRadius: 0, tension: 0.3, borderDash: [5,3] },
        ]
      },
      options: { ...baseOpts, scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, ticks: { ...baseOpts.scales.y.ticks, callback: v => '$' + Math.round(v).toLocaleString() } } } }
    });

  } else if (type === 'price') {
    const buyPoints  = chart.trades.filter(t => t.type === 'BUY').map(t => ({ x: t.date, y: t.price }));
    const sellPoints = chart.trades.filter(t => t.type === 'SELL').map(t => ({ x: t.date, y: t.price }));

    currentChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          { label: 'Close Price', data: chart.close, borderColor: '#8892a4', borderWidth: 1.5, pointRadius: 0, tension: 0.1, fill: false },
          { label: 'EMA-20',      data: chart.ema20,  borderColor: '#00e5ff', borderWidth: 1, pointRadius: 0, tension: 0.1, borderDash: [3,2], fill: false },
          { label: 'EMA-50',      data: chart.ema50,  borderColor: '#7c4dff', borderWidth: 1, pointRadius: 0, tension: 0.1, borderDash: [3,2], fill: false },
          { label: 'BUY',  data: buyPoints,  type: 'scatter', borderColor: '#00e676', backgroundColor: '#00e676', pointRadius: 6, pointStyle: 'triangle' },
          { label: 'SELL', data: sellPoints, type: 'scatter', borderColor: '#ff1744', backgroundColor: '#ff1744', pointRadius: 6, pointStyle: 'triangle', rotation: 180 },
        ]
      },
      options: { ...baseOpts, scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, ticks: { ...baseOpts.scales.y.ticks, callback: v => '$' + v } } } }
    });

  } else if (type === 'rsi') {
    currentChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          { label: 'RSI-14', data: chart.rsi, borderColor: '#7c4dff', backgroundColor: 'rgba(124,77,255,0.05)', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true },
        ]
      },
      options: {
        ...baseOpts,
        scales: {
          x: baseOpts.scales.x,
          y: { ...baseOpts.scales.y, min: 0, max: 100,
            ticks: { ...baseOpts.scales.y.ticks, stepSize: 20 }
          }
        },
        plugins: {
          ...baseOpts.plugins,
          annotation: {
            annotations: {
              ob: { type: 'line', yMin: 70, yMax: 70, borderColor: 'rgba(255,23,68,0.5)',   borderWidth: 1, borderDash: [4,3] },
              os: { type: 'line', yMin: 30, yMax: 30, borderColor: 'rgba(0,230,118,0.5)',   borderWidth: 1, borderDash: [4,3] },
            }
          }
        }
      }
    });

  } else if (type === 'macd') {
    const hist = chart.macd.map((v, i) => v - chart.macd_signal[i]);
    currentChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: dates,
        datasets: [
          { label: 'MACD',        data: chart.macd,        type: 'line',  borderColor: '#00e5ff', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'Signal',      data: chart.macd_signal, type: 'line',  borderColor: '#ff9100', borderWidth: 1.5, pointRadius: 0, tension: 0.3 },
          { label: 'Histogram',   data: hist,               backgroundColor: hist.map(v => v >= 0 ? 'rgba(0,230,118,0.4)' : 'rgba(255,23,68,0.4)') },
        ]
      },
      options: baseOpts
    });

  } else if (type === 'drawdown') {
    currentChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: dates,
        datasets: [
          { label: 'Drawdown (%)', data: chart.drawdown, borderColor: '#ff1744', backgroundColor: 'rgba(255,23,68,0.1)', borderWidth: 1.5, pointRadius: 0, tension: 0.3, fill: true },
        ]
      },
      options: { ...baseOpts, scales: { ...baseOpts.scales, y: { ...baseOpts.scales.y, ticks: { ...baseOpts.scales.y.ticks, callback: v => v + '%' } } } }
    });
  }
}
