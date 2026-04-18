# Baseline: Linear Quarterly Model with SSO in Optimization

## Overview

This scenario extends the regime-aware model by adding **SSO (2x S&P 500)** leveraged ETF in bull markets:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly
- **Risk Aversion**: 0.5 for bull/transition, 1.0 for bear
- **SSO**: Added to ticker pool in bull regime only (state 0)

## Results (Corrected)

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 13.7% |
| **Annual Volatility** | 7.9% |
| **Sharpe Ratio** | 1.73 |
| **Max Drawdown** | -3.6% |
| **Final Return** | 217.9% |
| **SPY (aligned)** | 218.2% |

### Regime Usage
- Bull periods (SSO active): 9
- Transition periods: 17
- Bear periods: 9

## Comparison

| Metric | RA=0.5 | Regime-Aware | SSO in Opt |
|--------|--------|--------------|------------|
| Annual Return | 14.4% | 13.7% | 13.7% |
| Sharpe | 1.86 | 1.76 | 1.73 |
| Max Drawdown | -3.4% | -3.6% | -3.6% |
| Final Return | 237.0% | 217.3% | 217.9% |

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/`: Results and charts