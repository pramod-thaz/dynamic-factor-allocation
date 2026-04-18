# Baseline: Linear Quarterly Model (Regime-Aware Risk Aversion)

## Overview

This scenario implements a dynamic factor rotation model with adaptive risk aversion based on market regime:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly (every 3 months)
- **Risk Aversion**: Dynamic (0.5 in bull/transition, 1.0 in bear)

## Results (Corrected)

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 13.7% |
| **Annual Volatility** | 7.8% |
| **Sharpe Ratio** | 1.76 |
| **Max Drawdown** | -3.6% |
| **Final Return** | 217.3% |
| **SPY (aligned)** | 218.2% |

### RA Usage

- **Bull/Transition periods** (state 0, 1): 26 quarters (RA = 0.5)
- **Bear periods** (state 2): 9 quarters (RA = 1.0)

## Comparison

| Metric | RA=0.5 | RA=1.0 | Regime-Aware |
|--------|--------|--------|--------------|
| Annual Return | 14.4% | 10.8% | 13.7% |
| Sharpe | 1.86 | 1.52 | 1.76 |
| Max Drawdown | -3.4% | -4.3% | -3.6% |
| Final Return | 237.0% | 149.7% | 217.3% |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts
- `output/ra_history.csv`: RA usage history