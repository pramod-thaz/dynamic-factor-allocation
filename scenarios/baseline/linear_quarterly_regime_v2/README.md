# Regime Detection v2 - Score-Based with Leverage

## Overview

Enhanced regime detection using a simplified score-based approach:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly (every 3 months)
- **Regime Detection**: Score-based with 5 features + leverage ETFs
- **Fast Exit**: 3% SPY drop triggers exit to safety

## Results

| Metric | Value |
|--------|-------|
| **Period** | Apr 2018 - Apr 2026 |
| **Total Quarters** | 33 |
| **Annual Return** | 17.8% |
| **Annual Volatility** | 8.4% |
| **Sharpe Ratio** | 2.12 |
| **Max Drawdown** | -4.3% |
| **Final Return** | 309.7% |

## Approach

### Score-Based Regime Detection
Combines 5 features into a composite score:
1. **Trend**: SPY above 200-day SMA
2. **Breadth**: % ETFs above 50-day MA
3. **Vol Regime**: USMV - SPY volatility spread
4. **Momentum**: (EMA20 - EMA50) / StdDev
5. **Credit Spreads**: HYG/LQD ratio (z-score normalized)

### Leverage Selection
- Score >= 3 → SSO (2x leveraged)
- Score >= 1 → QLD (1.5x leveraged)
- Score < 1 → SHV (safe)

### Fast Exit
Between quarterly rebalances, monitors for 3%+ SPY drops and exits to safety.

## Comparison

| Scenario | Annual Return | Sharpe | Max DD | Final Return |
|----------|---------------|--------|--------|--------------|
| RA=0.5 (baseline) | 14.4% | 1.86 | -3.4% | 237.0% |
| RA=1.0 | 10.8% | 1.52 | -4.3% | 149.7% |
| **Regime v2** | **17.8%** | **2.12** | **-4.3%** | **309.7%** |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts