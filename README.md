# Dynamic Factor Allocation - Session Context

## Problem Statement
Build a dynamic factor rotation model using ETFs (SSO, SPY, SHV) for live trading using:
- HSGP for GP prediction (later replaced with simpler EMA)
- HMM for regime detection  
- Configurable hysteresis to reduce whipsawing
- Target: Sharpe > 1, Excess vs SPY > 40%, Max DD < 25%

## Data Source
- `/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv`
- Auto-fetch via yfinance (last 60 days appended on each run)
- All ETFs in cache: 'GLD', 'IJR', 'MTUM', 'QUAL', 'SPY', 'TLT', 'USMV', 'VTV', 'VUG', 'SSO', 'HYG', 'LQD', 'SHV', 'QLD'

## Historical Evolution

### Early Iterations (9+ ETFs)
- Used 9+ ETFs originally (GLD, IJR, MTUM, QUAL, SPY, TLT, USMV, VTV, VUG, etc.)
- Later simplified to 3 ETFs: SSO (2x leveraged SPY), SPY, SHV (cash/safe)
- Reasoning: SSO already 2x leveraged, don't multiply returns again

### Key Discoveries & Fixes
1. **SSO is 2x leveraged** - Don't multiply returns by leverage again
2. **HMM full-data fit caused look-ahead bias** - Probabilities were extreme because model fit on all data then predicted on same data
3. **ADVI with GP times out** - Falls back to rolling mean/std
4. **HMM returns binary features** - Caused extreme 0/1 probabilities - need continuous features
5. **Drawdown calculation bug** - Was dividing by tiny initial cum_ret values, causing fake -988% drawdown. Fixed by using dollar portfolio value method
6. **Vol spike at 45% too aggressive** - Triggered on normal volatility, worsened results
7. **10-day crash at -10% too sensitive** - -15% is better
8. **proc_bull always 0 or 1** - HMM converges to extreme probabilities, fixed with:
   - Z-score normalized features (continuous instead of binary)
   - Added noise injection (0.05 std)
   - Minimum regime duration filter (18 days)

## Folder Structure
```
/home/realdomarp/PYMC/FACTOR ROTATION/
├── data/
│   └── etf_data.csv
├── scenarios/
│   └── baseline/
│       ├── linear_quarterly_hsgp_hysteresis/
│       ├── linear_quarterly_hsgp_optimized_hysteresis/
│       ├── linear_quarterly_hsgp_optimized_v2/
│       ├── linear_quarterly_ra05/
│       ├── linear_quarterly_ra10/
│       ├── linear_quarterly_regime_aware/
│       ├── linear_quarterly_regime_aware_sso/
│       ├── linear_quarterly_regime_aware_sso_proxy/
│       ├── linear_quarterly_regime_v2/
│       ├── linear_quarterly_regime_v3/
│       ├── linear_quarterly_regime_v4/
│       ├── linear_quarterly_sklearn_gp/
│       ├── quarterly_ema_smoother/          (Sharpe 1.12, Excess +117%)
│       └── quarterly_ema_smoothing_v2/        (CURRENT BEST - Sharpe 1.18, Excess +146%)
```

## Scenario Results Summary

| Scenario | Sharpe | Excess vs SPY | Max DD | Notes |
|----------|--------|--------------|--------|-------|
| linear_quarterly_hsgp_optimized_hysteresis | 1.12 | +117% | -21% | Previous best before v2 |
| quarterly_ema_smoother | 1.12 | +117% | -21% | Added smoother features |
| **quarterly_ema_smoothing_v2** | **1.18** | **+146%** | **-21%** | Current best |

## Current Best: quarterly_ema_smoothing_v2

### HMM Features (7 continuous features)

| # | Feature | Formula |
|---|---------|---------|
| 1 | trend_fast | (SPY - MA50) / σ(SPY, 50d) |
| 2 | trend_slow | (SPY - MA150) / σ(SPY, 150d) |
| 3 | momentum | Σ(SPY, 63d) / σ(SPY, 63d) |
| 4 | breadth | MA50 / MA50(20d ago) - 1 |
| 5 | vol_regime | SHV - SPY |
| 6 | adx | ADX(14) EWM(span=10) |
| 7 | market_vol | σ(SPY, 21d) × √252 |

### Exit Triggers

| Trigger | Condition | Action |
|---------|-----------|--------|
| Vol Spike Exit | 5-day vol > 40% | → 100% SHV |
| Crash Exit (30d) | Drop from 30d high < -10.5% | → 100% SHV |
| Crash Exit (10d) | 10-day return < -8.5% | → 100% SHV |
| HMM Fast Exit | prob_bear > 0.64 for 4 days | → 100% SHV |
| HMM Slow Exit | prob_bear > 0.47 for 12 days | → 25% SSO / 25% SPY / 50% SHV |

### Re-Entry Trigger

| Trigger | Condition | Action |
|---------|-----------|--------|
| Bull Re-entry | prob_bull > 0.62 for 8 consecutive days | → Quarterly rebalance |

### Minimum Regime Duration Filter
- **18 days** minimum before HMM can switch regimes
- Prevents whipsawing during uncertain periods

### Quarterly MVO

| Parameter | Value |
|-----------|-------|
| Objective | Minimize: risk_aversion × σ² - μ + turnover_penalty |
| Risk Aversion | 0.5 |
| Turnover Penalty | 0.015 |
| Frequency | Every 63 days |
| Leverage Rules | |
| - 2x leverage | If prob_bull > 0.70 AND σ_SSO < 0.018 |
| - Exit to SHV | If σ_SSO > 0.025 |

### Risk Metrics (Raw Ratios)

| Metric | Strategy | SPY |
|--------|----------|-----|
| Ann Return | 19.6% | 15.7% |
| Volatility | 16.6% | 19.0% |
| Beta | 0.44 | 1.00 |
| Sharpe | 1.18 | 0.83 |
| Sortino | 1.16 | 1.01 |
| Jensen Alpha | +12.7% | - |

### Issues Identified
- **2018-2019 spent in SHV** - Crash exit in Oct 2018, slow HMM re-entry post-crash
- Re-entry required prob_bull > 0.62 for 8 consecutive days - too conservative
- Min regime 18d filter delays re-entry after crashes

## GitHub
- Repo: https://github.com/pramod-thaz/dynamic-factor-allocation.git

## Live Trading Instructions (per scenario)
1. Run: `python run_model.py` (fetches latest data, runs backtest)
2. Check: `cat output/next_trade.csv`
3. Generate charts: `python charting.py`

## Notes for Next Session

### Potential Improvements
1. Lower REENTRY_PROB from 0.62 to 0.55 for faster re-entry
2. Reduce REENTRY_DAYS from 8 to 5 for faster re-entry
3. Raise vol spike threshold from 40% to 45% to reduce false exits
4. Consider adding more ETFs back to portfolio for diversification

### Pending Fixes
- Post-crash re-entry is too slow (stays in SHV too long)
- Could benefit from more diversification beyond 3 ETFs