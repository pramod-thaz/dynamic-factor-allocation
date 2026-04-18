# Linear Quarterly HSGP Optimized (With Vol Spike)

## Strategy Version: v2 (Current with Vol Spike Adjustments)

This version has:
- Rolling 63-day mean/std predictions
- Quarterly rebalancing
- HMM-based regime detection
- Enhanced crash detector (30-day + 10-day triggers)
- Volatility spike detector (60% threshold)

## Results (2017-08 to 2026-02)

| Metric | Value |
|--------|-------|
| Annual Return | 16.4% |
| Annual Vol | 18.4% |
| Sharpe | 0.89 |
| Max Drawdown | -22.6% |
| Final Return | 247.6% |
| SPY Buy-Hold | 217.2% |
| Excess vs SPY | +30.3% |

## Key Parameters

- FAST_EXIT_PROB = 0.65
- SLOW_EXIT_PROB = 0.48
- REENTRY_PROB = 0.60
- CRASH_DROP = 0.12 (30-day)
- 10-day trigger = -15%
- Vol spike = 60%

## Triggers

- Quarterly rebalances: 52
- Fast exits: 8
- Re-entries: 6
- Vol spike exit: 0 (threshold not reached)