# Baseline: Linear Quarterly Model (RA=1.0)

## Overview

This scenario implements a dynamic factor rotation model using ETFs with the following configuration:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly (every 3 months)
- **Risk Aversion**: 1.0 (balanced/risk averse)

## Results (Corrected)

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 10.8% |
| **Annual Volatility** | 7.1% |
| **Sharpe Ratio** | 1.52 |
| **Max Drawdown** | -4.3% |
| **Final Return** | 149.7% |
| **SPY (aligned)** | 218.2% |

## Comparison with RA=0.5

| Metric | RA=0.5 | RA=1.0 |
|--------|--------|--------|
| Annual Return | 14.4% | 10.8% |
| Sharpe | 1.86 | 1.52 |
| Max Drawdown | -3.4% | -4.3% |
| Final Return | 237.0% | 149.7% |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts

## Notes

- Risk aversion of 1.0 is the "standard" mean-variance optimal
- Higher risk aversion leads to more conservative allocations
- Lower Sharpe than RA=0.5 but still positive