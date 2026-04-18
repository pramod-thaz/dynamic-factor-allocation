# Baseline: Linear Quarterly Model (RA=0.5)

## Overview

This scenario implements a dynamic factor rotation model using ETFs with the following configuration:
- **Model**: Linear surrogate with PyMC MAP inference
- **Rebalancing**: Quarterly (every 3 months)
- **Risk Aversion**: 0.5 (aggressive/risk-neutral)

## Feature Engineering

### Factors Comparison

| Factor | Old Baseline | This Scenario |
|--------|--------------|----------------|
| **Features** | Only 2 (SPY_vol6, SPY_mom12) | 7 features + HMM regime |
| **Spread features** | None | 5 spreads (value, lowvol, growth, bond, small) |
| **HMM** | Fit once, predict forward (data leakage) | Refit at each rebalance |
| **HMM smoothing** | None | 3-month rolling mode |
| **Inference** | MAP | MAP |
| **Rebalancing** | Quarterly | Quarterly |
| **Risk Aversion** | 1.0 | 0.5 |

### Feature Details

1. **Base Features** (per ETF):
   - `{t}_ret`: Monthly returns
   - `{t}_vol6`: 6-month rolling volatility
   - `{t}_mom12`: 12-month rolling momentum

2. **Spread Features** (regime-aware):
   - `value_mom_spread` = VTV - MTUM
   - `lowvol_market_spread` = USMV - SPY
   - `growth_value_spread` = VUG - VTV
   - `bond_gold_spread` = TLT - GLD
   - `small_market_spread` = IJR - SPY

3. **Dynamic HMM Regime**:
   - 3-component GaussianHMM refitted at each rebalance
   - 3-month rolling mode smoothing
   - Appended as additional feature for prediction

## Model Architecture

- **Surrogate Model**: Linear regression with PyMC
- **Inference**: MAP (Maximum A Posteriori) estimation
- **Likelihood**: Student-t (nu=4) for fat tails
- **Portfolio Optimization**: Mean-variance with risk aversion 0.5

## Results (Corrected)

| Metric | Value |
|--------|-------|
| **Period** | Aug 2017 - Feb 2026 |
| **Total Quarters** | 35 |
| **Annual Return** | 14.4% |
| **Annual Volatility** | 7.7% |
| **Sharpe Ratio** | 1.86 |
| **Max Drawdown** | -3.4% |
| **Final Return** | 237.0% |
| **SPY (aligned)** | 218.2% |

## Key Improvements

1. **More Features**: 7 features + HMM regime vs 2 features
2. **Spread Features**: Captures factor relationships
3. **No Data Leakage**: HMM refitted at each rebalance
4. **Smoothing**: 3-month rolling mode reduces regime noise
5. **Lower RA**: 0.5 allows more aggressive portfolio allocation

## Files

- `run_model.py`: Main model code
- `charting.py`: Separated charting module
- `output/factor_rotation_backtest_results.csv`: Detailed results
- `output/factor_rotation_charts.png`: Consolidated 4-panel chart
- `output/weights_history.csv`: Portfolio weights over time