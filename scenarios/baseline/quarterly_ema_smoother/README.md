# Quarterly EMA + Smoothed HMM Strategy

## Strategy Overview

- Simple EMA predictor (span=40) replacing ADVI/HSGP
- Smoothed HMM with z-score normalized features + market_vol
- Turnover penalty + tighter exit triggers

## Live Trading Instructions

### How to Run

```bash
cd /home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/quarterly_ema_smoother
python run_model.py
```

### Check the Trade Signal

After running, check `output/next_trade.csv`:

```bash
cat output/next_trade.csv
```

### Actionable Tasks

| Action Type | What to Do |
|------------|------------|
| **REBAL to SSO:XX% SPY:XX% SHV:XX%** | Sell current positions and buy new allocation as specified |
| **BUY SSO X% / SELL SSO X%** | Buy or sell SSO to reach target allocation |
| **BUY SPY X% / SELL SPY X%** | Buy or sell SPY to reach target allocation |
| **BUY SHV X% / SELL SHV X%** | Buy or sell SHV to reach target allocation |
| **NO CHANGE** | No action needed - keep current positions |
| **CRASH EXIT → SHV** | Sell all and move to 100% SHV (cash) |
| **VOL SPIKE EXIT → SHV** | Sell all and move to 100% SHV (cash) - volatility spike detected |

---

## Results (2017-08 to 2026-04)

| Metric | Value |
|--------|-------|
| Annual Return | 18.8% |
| Annual Vol | 16.8% |
| Sharpe | 1.12 |
| Max Drawdown | -21.3% |
| Final Return | 346.5% |
| SPY Buy-Hold | 229.3% |
| Excess vs SPY | +117.2% |

## Key Changes from Previous

| Change | Previous | This Version |
|-------|----------|-------------|
| **Predictor** | ADVI GP | EMA (span=40) |
| **HMM Features** | Binary | Z-score normalized + market_vol |
| **Turnover penalty** | None | 0.01 |
| **Vol spike exit** | 35% | 45% |

## Key Parameters

- FAST_EXIT_PROB = 0.64
- SLOW_EXIT_PROB = 0.47
- REENTRY_PROB = 0.62
- CRASH 30d = -10.5%
- CRASH 10d = -8.5%
- Vol spike = 45%

## Triggers

- Quarterly rebalances: 82
- Fast exits: 6
- Re-entries: 44