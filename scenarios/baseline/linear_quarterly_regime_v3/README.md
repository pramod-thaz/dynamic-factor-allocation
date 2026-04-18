# Regime Detection v3 - Score-Based with Hysteresis

## Overview

Enhanced regime detection using score-based approach with hysteresis to reduce whipsawing:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly (every 3 months)
- **Regime Detection**: Score-based with 5 features + leverage ETFs
- **Hysteresis**: Stay in regime for 2 consecutive periods before switching

## Results

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 13.8% |
| **Annual Volatility** | 8.4% |
| **Sharpe Ratio** | 1.65 |
| **Max Drawdown** | -4.3% |
| **Final Return** | 219.0% |
| **SPY (aligned)** | 218.2% |

## Approach

### Score-Based Regime Detection
Combines 5 features into a composite score:
1. **Trend**: SPY above 200-day SMA
2. **Breadth**: % ETFs above 50-day MA
3. **Vol Regime**: USMV - SPY volatility spread
4. **Momentum**: (EMA20 - EMA50) / StdDev
5. **Credit Spreads**: HYG/LQD ratio (z-score normalized)

### Leverage Selection
- Score >= 1.0 → SSO (2x leveraged)
- Score >= 0.0 → QLD (1.5x leveraged)
- Score < 0 → SHV (safe)

### Hysteresis
Require 2 consecutive periods in new regime before switching to reduce whipsawing.

## Leverage Distribution
- SSO (2x): 7 quarters
- QLD (1.5x): 28 quarters
- SHV (safe): 0 quarters

## Comparison with Baseline

| Scenario | Annual Return | Sharpe | Max DD | Final Return |
|----------|---------------|--------|--------|--------------|
| RA=0.5 (baseline) | 14.4% | 1.86 | -3.4% | 237.0% |
| RA=1.0 | 10.8% | 1.52 | -4.3% | 149.7% |
| **Regime v3** | **13.8%** | **1.65** | **-4.3%** | **219.0%** |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts
  - `factor_rotation_backtest_results.csv`
  - `weights_history.csv`
  - `regime_history.csv`
  - `factor_rotation_charts.png`