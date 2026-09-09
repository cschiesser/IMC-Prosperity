# IMC Prosperity Algorithmic Trading Bot

Python trading algorithms built for IMC Prosperity,
a global algorithmic trading competition run on a simulated exchange. Each round
introduces new tradeable products with different price dynamics. The goal is to
design strategies that maximise PnL under per-product position limits.

## Approach

The bot applies a different strategy per product, chosen from its price behaviour:

- **Market making around a fair value** for stable, mean-reverting products
  (e.g. `EMERALDS`, fair value ≈ 10,000): quoting symmetric bids/asks and skewing
  quotes as inventory builds toward the position limit.
- **Mean reversion with an EMA** for noisy products (e.g. `ASH_COATED_OSMIUM`):
  tracking an exponential moving average and fading deviations from it.
- **Trend following** for products with persistent drift
  (e.g. `INTARIAN_PEPPER_ROOT`, ≈ +0.001 / timestamp): leaning into the trend
  while respecting limits.
- **Inventory management** every strategy enforces IMC's per-product position
  limits (±80) and manages inventory risk explicitly.

Strategy parameters are tuned offline with a **grid search** (`grid_search.py`)
run against a **custom backtester** (`backtest.py`) on historical round data,
rather than hand-picked.

## Repository structure

```
prosperity-round1/     Round 1: market making (EMERALDS, TOMATOES)
prosperity-round2/     Round 2: + grid-search parameter tuning
prosperity-round3/     Round 3: EMA mean-reversion + trend following, multi-day state
```

Each round folder contains:

```
traderN.py       submitted strategy for that round
backtest.py      local backtester
grid_search.py   parameter optimisation
research.py      exploratory analysis
datamodel.py     IMC-provided exchange data model
data/            round market data
```

## Running

```bash
cd prosperity-round3
pip install -r requirements.txt
python backtest.py
```
