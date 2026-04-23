# Dynamic Factor Allocation - Session Context

## Folder Structure

## Scenario Results Summary


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

### Re-Entry Trigger

| Trigger | Condition | Action |
|---------|-----------|--------|

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

### Risk Metrics (Raw Ratios)

| Metric | Strategy | SPY |
|--------|----------|-----|

### Issues Identified
- **2018-2019 spent in SHV** - Crash exit in Oct 2018, slow HMM re-entry post-crash
- Re-entry required prob_bull > 0.62 for 8 consecutive days - too conservative
- Min regime 18d filter delays re-entry after crashes

## GitHub
- Repo: https://github.com/pramod-thaz/dynamic-factor-allocation.git

## Live Trading Instructions (per scenario)

## Notes for Next Session

### Potential Improvements
