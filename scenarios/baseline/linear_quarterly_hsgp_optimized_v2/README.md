# Linear Quarterly HSGP Optimized (Original)

## Strategy Version: Original

This is the **original baseline** version with:
- Rolling 63-day mean/std predictions
- Quarterly rebalancing
- HMM-based regime detection
- Simple crash detector (30-day, -12% threshold)

## Results (2017-08 to 2026-02)

| Metric | Value |
|--------|-------|
| Annual Return | 16.4% |
| Annual Vol | 18.4% |
| Sharpe | 0.89 |
| Max Drawdown | -22.6% |
| Final Return | 247.5% |
| SPY Buy-Hold | 219.1% |
| Excess vs SPY | +28.4% |

## Key Parameters

- FAST_EXIT_PROB = 0.65
- SLOW_EXIT_PROB = 0.48
- REENTRY_PROB = 0.60
- CRASH_DROP = 0.12 (30-day threshold only)

## Triggers

- Quarterly rebalances: 52
- Fast exits: 8
- Re-entries: 6