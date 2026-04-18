# Linear Quarterly HSGP Optimized (With Vol Spike)

## Strategy Version: v2 (Current with Vol Spike Adjustments)

This version has:
- Rolling 63-day mean/std predictions
- Quarterly rebalancing
- HMM-based regime detection
- Enhanced crash detector (30-day + 10-day triggers)
- Volatility spike detector (60% threshold)

## Live Trading Instructions

### How to Run

```bash
cd /home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_hsgp_optimized_hysteresis
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

### Allocation Actions Format

The model outputs trades in `next_trade.csv`:
- `date`: Trade date
- `action`: Plain English action (e.g., "REBAL to SSO:0% SPY:0% SHV:100%", "NO CHANGE")
- `buy_sell`: Specific buy/sell instructions (e.g., "SELL SSO 100%, BUY SHV 100%")
- `sso`, `spy`, `shv`: Target allocation percentages
- `regime`: Current HMM regime (0=bear, 1=normal, 2=bull)
- `prob_bull`: Probability of bull regime (0-1)

---

## Results (2017-08 to 2026-04)

| Metric | Value |
|--------|-------|
| Annual Return | 17.4% |
| Annual Vol | 19.2% |
| Sharpe | 0.91 |
| Max Drawdown | -20.1% |
| Final Return | 279.9% |
| SPY Buy-Hold | 231.3% |
| Excess vs SPY | +48.6% |

> Note: HMM disabled - was causing 0.9 proc_bull issues; using fixed prob with vol/crash exit triggers

## Key Parameters

- CRASH_DROP = 0.12 (30-day)
- 10-day trigger = -15%
- Vol spike = 60%

## Triggers

- Quarterly rebalances: 49
- Fast exits: 5
- Vol spike exit: 0 (threshold not reached)