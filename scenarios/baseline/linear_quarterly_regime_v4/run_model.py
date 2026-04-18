# -*- coding: utf-8 -*-
"""
Regime v4 - Simplified 3-ETF (SSO, QLD, SHV) with Daily Regime Monitoring

Features:
- Simplified portfolio: Only SSO, QLD, SHV
- Daily regime score (5 features combined + normalized)
- Fast exit: Score ≤ 0.3 for 3 consecutive days → SHV
- Slow exit: Score ≤ 0.5 for 15 consecutive days → SHV
- Re-entry: All 3 signals for 10 consecutive days
"""

import pandas as pd
import numpy as np
import pymc as pm
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

# === CONFIG ===
DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_regime_v4/output'

# Portfolio ETFs
PORTFOLIO_ETFS = ['SSO', 'QLD', 'SHV']
PROXY_ETFS = ['SPY', 'USMV', 'HYG', 'LQD', 'VTV', 'MTUM', 'QUAL', 'VUG', 'IJR', 'TLT', 'GLD']

# Risk aversion for optimization
RA = 0.5

# HMM settings
HMM_STATES = 3
HMM_WINDOW = 252  # 1 year rolling window

# Exit triggers
FAST_EXIT_PROB = 0.3
FAST_EXIT_DAYS = 3
SLOW_EXIT_PROB = 0.5
SLOW_EXIT_DAYS = 15

# Re-entry triggers
REENTRY_PROB = 0.6
REENTRY_DAYS = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("=== Regime v4: 3-ETF with Daily Regime Monitoring ===")
print("Loading data...")

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
daily_returns = data.pct_change().dropna()

print(f"Daily data: {daily_returns.index[0]} to {daily_returns.index[-1]}")
print(f"Total days: {len(daily_returns)}")

# Verify tickers
for t in PORTFOLIO_ETFS + PROXY_ETFS:
    if t not in daily_returns.columns:
        raise ValueError(f"Ticker {t} not found!")

# === BUILD DAILY REGIME FEATURES ===
print("Building daily regime features...")

features = pd.DataFrame(index=daily_returns.index)

# 1. Trend: SPY > 200-day SMA
features['trend'] = (daily_returns['SPY'] > daily_returns['SPY'].rolling(200).mean()).astype(float)

# 2. Breadth: % of equity ETFs above 50-day MA
equity_proxies = [t for t in PROXY_ETFS if t not in ['TLT', 'GLD']]
features['breadth'] = (daily_returns[equity_proxies].rolling(50).mean() > 0).mean(axis=1)

# 3. Vol Regime: USMV - SPY volatility spread (6-day rolling)
features['vol_regime'] = daily_returns['USMV'].rolling(6).std() - daily_returns['SPY'].rolling(6).std()

# 4. Momentum: (EMA20 - EMA50) / StdDev
ema_20 = daily_returns['SPY'].ewm(span=20).mean()
ema_50 = daily_returns['SPY'].ewm(span=50).mean()
features['momentum'] = (ema_20 - ema_50) / daily_returns['SPY'].rolling(20).std()

# 5. Credit Spreads: HYG/LQD ratio z-score
features['credit_spread'] = (daily_returns['HYG'] + 1).cumprod() / (daily_returns['LQD'] + 1).cumprod()
features['credit_z'] = (features['credit_spread'] - features['credit_spread'].rolling(60).mean()) / features['credit_spread'].rolling(60).std()

# 6. Additional re-entry signals
features['hyg_ma20'] = (daily_returns['HYG'] + 1).cumprod() / (daily_returns['HYG'] + 1).cumprod().rolling(20).mean()
features['vol_20'] = daily_returns['SPY'].rolling(20).std()
features['vol_60'] = daily_returns['SPY'].rolling(60).std()

features = features.dropna()
print(f"Features period: {features.index[0]} to {features.index[-1]}")

# === BUILD DAILY REGIME SCORE ===
print("Computing daily regime score...")

# Use a simple score-based approach instead of HMM (more robust)
# Score components (same as before)
regime_score = (
    features['trend'] * 1.0 +
    features['breadth'] * 1.0 +
    features['momentum'].clip(-2, 2) / 2 +
    features['vol_regime'].clip(-0.02, 0.02) / 0.02 * 0.5 +
    (-features['credit_z'].clip(-2, 2) / 2) * 0.5
)

# Normalize to 0-1 range using percentile
regime_score_norm = regime_score.rank(pct=True)
features['regime_score'] = regime_score

# Map to probability-like (higher score = more bullish)
# Score > 0.7 = Bull (prob ~1)
# Score 0.3-0.7 = Normal (prob ~0.5)
# Score < 0.3 = Bear (prob ~0)
features['bull_prob'] = regime_score_norm.clip(0, 1)

# === BACKTEST ===
print("Running backtest with daily HMM monitoring...")

# Start after we have enough data for features
start_date = features.index[max(252, 50)]
backtest_start = start_date

# Get quarterly rebalance dates
monthly = data.resample('ME').last()
monthly_returns = monthly.pct_change().dropna()
quarterly_dates = monthly_returns.index[monthly_returns.index.month.isin([2, 5, 8, 11])]

# Filter to backtest period
quarterly_dates = quarterly_dates[quarterly_dates >= backtest_start]

# Track state
current_position = 'EQUAL'  # Start with equal weights
portfolio_value = 1.0
portfolio_history = []
daily_regime_history = []
exit_events = []
reentry_events = []

# Daily tracking variables
bull_prob_consecutive = 0
reentry_consecutive = 0
in_reentry_mode = False

# Monthly/quarterly state tracking
last_quarterly_rebalance = None

# Main loop: iterate through each day
backtest_dates = features.loc[backtest_start:].index

for date in backtest_dates:
    # Skip weekends/holidays
    if date not in daily_returns.index:
        continue
    
    # === QUARTERLY REBALANCE CHECK ===
    is_quarterly_rebalance = date in quarterly_dates
    
    # === GET DAILY REGIME INFO ===
    if pd.notna(features.loc[date, 'bull_prob']):
        bull_prob = features.loc[date, 'bull_prob']
    else:
        bull_prob = 0.5  # Neutral
    
    # === CHECK EXIT CONDITIONS ===
    if bull_prob < FAST_EXIT_PROB:
        bull_prob_consecutive += 1
        if bull_prob_consecutive >= FAST_EXIT_DAYS and current_position != 'SHV':
            current_position = 'SHV'
            exit_events.append({'date': date, 'trigger': 'fast_exit', 'bull_prob': bull_prob})
    elif bull_prob < SLOW_EXIT_PROB:
        bull_prob_consecutive += 1
        if bull_prob_consecutive >= SLOW_EXIT_DAYS and current_position != 'SHV':
            current_position = 'SHV'
            exit_events.append({'date': date, 'trigger': 'slow_exit', 'bull_prob': bull_prob})
    else:
        bull_prob_consecutive = 0
    
    # === CHECK RE-ENTRY CONDITIONS ===
    # Need all 3 signals for REENTRY_DAYS days
    credit_ok = (daily_returns.loc[date, 'HYG'] > 0) or (features.loc[date, 'credit_z'] > -1)
    vol_ok = features.loc[date, 'vol_20'] < features.loc[date, 'vol_60'] if pd.notna(features.loc[date, 'vol_60']) else False
    trend_ok = bull_prob > REENTRY_PROB
    
    if current_position == 'SHV' and credit_ok and vol_ok and trend_ok:
        if not in_reentry_mode:
            in_reentry_mode = True
            reentry_consecutive = 1
        else:
            reentry_consecutive += 1
            
        if reentry_consecutive >= REENTRY_DAYS:
            # Re-entry: go to QLD first
            current_position = 'QLD'
            in_reentry_mode = False
            reentry_consecutive = 0
            reentry_events.append({'date': date, 'bull_prob': bull_prob, 'new_position': 'QLD'})
    elif current_position == 'QLD' and bull_prob > 0.75 and trend_ok:
        # After being in QLD for a while, can upgrade to SSO if strong bull
        current_position = 'SSO'
        reentry_events.append({'date': date, 'bull_prob': bull_prob, 'new_position': 'SSO'})
    else:
        in_reentry_mode = False
        reentry_consecutive = 0
    
    # === QUARTERLY REBALANCE ===
    # Check quarterly if we should switch between SSO/QLD based on regime score
    if is_quarterly_rebalance:
        # Use regime score for allocation decision
        if features.loc[date, 'bull_prob'] > 0.7:
            current_position = 'SSO'
        elif features.loc[date, 'bull_prob'] < 0.3:
            current_position = 'SHV'
        else:
            current_position = 'QLD'
    
    # === CALCULATE DAILY RETURN ===
    if current_position == 'SHV':
        daily_ret = daily_returns.loc[date, 'SHV'] if 'SHV' in daily_returns.columns else 0
    elif current_position == 'QLD':
        daily_ret = daily_returns.loc[date, 'QLD'] if 'QLD' in daily_returns.columns else 0
    elif current_position == 'SSO':
        daily_ret = daily_returns.loc[date, 'SSO'] if 'SSO' in daily_returns.columns else 0
    elif current_position == 'EQUAL':
        daily_ret = daily_returns.loc[date, PORTFOLIO_ETFS].mean()
    else:
        daily_ret = 0
    
    portfolio_value *= (1 + daily_ret)
    
    # === RECORD HISTORY ===
    portfolio_history.append({
        'date': date,
        'position': current_position,
        'return': daily_ret,
        'portfolio_value': portfolio_value,
        'bull_prob': bull_prob,
        'regime_score': bull_prob,
        'is_rebalance': is_quarterly_rebalance
    })

print(f"Completed backtest: {len(portfolio_history)} days")
print(f"Exit events: {len(exit_events)}")
print(f"Re-entry events: {len(reentry_events)}")

# === RESULTS ===
results_df = pd.DataFrame(portfolio_history).set_index('date')

# Calculate metrics
daily_returns_series = results_df['return']
ann_ret = daily_returns_series.mean() * 252 * 100
ann_vol = daily_returns_series.std() * np.sqrt(252) * 100
sharpe = daily_returns_series.mean() / daily_returns_series.std() * np.sqrt(252)
max_dd = ((results_df['portfolio_value'] / results_df['portfolio_value'].cummax() - 1) * 100).min()

# Monthly aggregation for comparison
monthly_results = results_df.resample('ME').agg({'return': lambda x: (1 + x).prod() - 1})
monthly_results['cum_ret'] = (1 + monthly_results['return']).cumprod() - 1

# SPY comparison
spy_returns = daily_returns['SPY'].loc[backtest_start:]
spy_cum = (1 + spy_returns).cumprod() - 1

print("\n=== RESULTS (Regime v4 - 3-ETF Daily Regime) ===")
print(f"Period: {results_df.index[0].strftime('%Y-%m-%d')} to {results_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Total days: {len(results_df)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final Return: {(portfolio_value - 1) * 100:.1f}%")
print(f"SPY (same period): {spy_cum.iloc[-1] * 100:.1f}%")

# Position breakdown
position_counts = results_df['position'].value_counts()
print(f"\nPosition distribution:")
for pos, count in position_counts.items():
    print(f"  {pos}: {count} days ({count/len(results_df)*100:.1f}%)")

# === SAVE OUTPUTS ===
results_df.to_csv(f'{OUTPUT_DIR}/daily_results.csv')

# Save events
if exit_events:
    pd.DataFrame(exit_events).to_csv(f'{OUTPUT_DIR}/exit_events.csv', index=False)
if reentry_events:
    pd.DataFrame(reentry_events).to_csv(f'{OUTPUT_DIR}/reentry_events.csv', index=False)

# Monthly summary
monthly_results.to_csv(f'{OUTPUT_DIR}/monthly_results.csv')

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")