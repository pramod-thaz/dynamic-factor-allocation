# Quarterly EMA + Smoothing V2

## Strategy Overview

- EMA predictor (span=40) replacing ADVI/HSGP
- Smoothed HMM with z-score normalized features
- 18-day minimum regime duration filter
- Turnover penalty to reduce unnecessary switches

## Live Trading Instructions

### How to Run

```bash
cd /home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/quarterly_ema_smoothing_v2
python run_model.py
```

### Check the Trade Signal

```bash
cat output/next_trade.csv
```

### Actionable Tasks

| Action Type | What to Do |
|------------|------------|
| **NO CHANGE** | No action needed |
| **CRASH EXIT → SHV** | Sell all, move to 100% SHV |
| **VOL SPIKE EXIT → SHV** | Vol spike detected, move to SHV |
| **REBAL** | Rebalance to shown allocation |

---

## Results (2017-08 to 2026-04)

| Metric | Value |
|--------|-------|
| Annual Return | 19.6% |
| Annual Vol | 16.6% |
| Sharpe | 1.18 |
| Max Drawdown | -21.3% |
| Final Return | 377.1% |
| SPY Buy-Hold | 231.3% |
| Excess vs SPY | +145.9% |

## Key Parameters

- MIN_REGIME_DAYS = 18
- TURNOVER_PENALTY = 0.015
- CRASH 30d = -10.5%
- CRASH 10d = -8.5%
- Vol spike = 40%
- FAST_EXIT_PROB = 0.64
- SLOW_EXIT_PROB = 0.47
- REENTRY_PROB = 0.62 (for 8 days)

## Triggers

- Quarterly rebalances: 85
- Fast exits: 6
- Re-entries: 47

## Notes

- 2018-2019 spent in SHV due to crash exit and slow HMM re-entry post-crash
- Re-entry required prob_bull > 0.62 for 8 consecutive days
- Min regime duration filter prevents over-trading but can delay re-entry after crashes