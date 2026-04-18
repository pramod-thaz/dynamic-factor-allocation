# Baseline: Linear Quarterly SSO Proxy (Leverage)

## Overview

This scenario uses SPY for prediction/optimization but swaps SPY → SSO in the portfolio for bull and transition regimes, effectively adding 2x leverage:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly
- **Risk Aversion**: 0.5 for bull/transition, 1.0 for bear
- **Leverage**: SSO (2x S&P 500) replaces SPY in bull/transition states

## How It Works

| State | Regime | Optimization | Portfolio After Swap | Return |
|-------|--------|--------------|---------------------|--------|
| 0 | Bull | 9 tickers | SSO replaces SPY | SSO return |
| 1 | Transition | 9 tickers | SSO replaces SPY | SSO return |
| 2 | Bear | 9 tickers | No swap | SPY return |

## Results (Corrected)

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 14.4% |
| **Annual Volatility** | 8.4% |
| **Sharpe Ratio** | 1.72 |
| **Max Drawdown** | -3.6% |
| **Final Return** | 234.8% |
| **SPY (aligned)** | 218.2% |

### Regime Usage
- Bull periods: 9
- Transition periods: 17
- Bear periods: 9
- **SSO Swapped**: 26 periods (bull + transition)

## Comparison

| Scenario | Ann Ret | Sharpe | Max DD | Final |
|----------|---------|--------|--------|-------|
| RA=0.5 | 14.4% | 1.86 | -3.4% | 237.0% |
| RA=1.0 | 10.8% | 1.52 | -4.3% | 149.7% |
| Regime-Aware | 13.7% | 1.76 | -3.6% | 217.3% |
| SSO in Opt | 13.7% | 1.73 | -3.6% | 217.9% |
| **SSO Proxy** | 14.4% | 1.72 | -3.6% | 234.8% |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts