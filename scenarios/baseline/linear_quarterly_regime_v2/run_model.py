# -*- coding: utf-8 -*-
"""
Enhanced Regime Detection v2 - Simplified Score-Based
- 5 features for regime scoring (trend, breadth, vol, momentum, credit spreads)
- Conditional leverage based on score thresholds
- Fast exit monitoring between quarterly rebalances
"""

import pandas as pd
import numpy as np
import pymc as pm
from scipy.optimize import minimize
import warnings
import os

warnings.filterwarnings('ignore')

# === CONFIG ===
BASE_TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
LEVERAGE_TICKERS = ['SSO', 'QLD', 'SHV']
SAFE_TICKER = 'SHV'

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_regime_v2/output'

# Risk aversion
RA_BULL = 0.5
RA_BEAR = 1.0

# Leverage thresholds
SCORE_STRONG_BULL = 3   # Score >= 3 = SSO (2x)
SCORE_MODERATE_BULL = 1  # Score >= 1 = QLD (1.5x)
# Score < 1 = SHV (0x)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("=== Enhanced Regime Detection v2 (Score-Based) ===")
print("Loading data...")

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
monthly = data.resample('ME').last()
returns = monthly.pct_change().dropna()

print(f"Data period: {returns.index[0]} to {returns.index[-1]}")

# Verify tickers
all_tickers = BASE_TICKERS + LEVERAGE_TICKERS + ['HYG', 'LQD']
for t in all_tickers:
    if t not in returns.columns:
        raise ValueError(f"Ticker {t} not found!")

# === BUILD REGIME FEATURES ===
print("Building 5 regime features...")

regime_features = pd.DataFrame(index=returns.index)

# 1. Trend: SPY > 200-day SMA
regime_features['trend'] = (returns['SPY'] > returns['SPY'].rolling(200).mean()).astype(float)

# 2. Breadth: % ETFs above 50-day MA
equity_etfs = [t for t in BASE_TICKERS if t not in ['TLT', 'GLD']]
regime_features['breadth'] = (returns[equity_etfs].rolling(50).mean() > 0).mean(axis=1)

# 3. Vol Regime: USMV - SPY volatility spread
regime_features['vol_regime'] = returns['USMV'].rolling(6).std() - returns['SPY'].rolling(6).std()

# 4. Momentum: (EMA20 - EMA50) / StdDev
ema_20 = returns['SPY'].ewm(span=20).mean()
ema_50 = returns['SPY'].ewm(span=50).mean()
regime_features['momentum'] = (ema_20 - ema_50) / returns['SPY'].rolling(20).std()

# 5. Credit Spreads: HYG/LQD
regime_features['credit_spread'] = (returns['HYG'] + 1).cumprod() / (returns['LQD'] + 1).cumprod()
# Normalize credit spread (z-score)
regime_features['credit_z'] = (regime_features['credit_spread'] - regime_features['credit_spread'].mean()) / regime_features['credit_spread'].std()

# Build score from features
regime_features['score'] = (
    regime_features['trend'] * 1 +
    regime_features['breadth'] * 1 +
    regime_features['momentum'].clip(-2, 2) / 2 +  # Normalize to -1 to 1
    regime_features['vol_regime'].clip(-0.02, 0.02) / 0.02 * 0.5 +  # Vol regime positive = good
    -regime_features['credit_z'].clip(-2, 2) / 2 * 0.5  # Credit spread up = bad
)

regime_features = regime_features.dropna()

# Determine regime based on score
def get_regime(score):
    if score >= SCORE_STRONG_BULL:
        return 2, 'SSO'  # Bull, use SSO
    elif score >= SCORE_MODERATE_BULL:
        return 1, 'QLD'  # Moderate, use QLD
    else:
        return 0, 'SHV'  # Bear, use SHV

# === BUILD ROTATION FEATURES ===
print("Building rotation features...")

features = pd.DataFrame(index=returns.index)
for t in BASE_TICKERS:
    features[f'{t}_ret'] = returns[t]
    features[f'{t}_vol6'] = returns[t].rolling(6).std()
    features[f'{t}_mom12'] = returns[t].rolling(12).mean()

# Add leveraged ETF features
for t in LEVERAGE_TICKERS:
    features[f'{t}_ret'] = returns[t]
    features[f'{t}_vol6'] = returns[t].rolling(6).std()
    features[f'{t}_mom12'] = returns[t].rolling(12).mean()

# Spread features
features['value_mom_spread'] = features['VTV_ret'] - features['MTUM_ret']
features['lowvol_market_spread'] = features['USMV_ret'] - features['SPY_ret']
features['growth_value_spread'] = features['VUG_ret'] - features['VTV_ret']
features['bond_gold_spread'] = features['TLT_ret'] - features['GLD_ret']
features['small_market_spread'] = features['IJR_ret'] - features['SPY_ret']

features = features.dropna()

ROTATION_FEATURES = [
    'SPY_vol6', 'SPY_mom12',
    'value_mom_spread', 'lowvol_market_spread', 'growth_value_spread',
    'bond_gold_spread', 'small_market_spread'
]

print(f"Feature period: {features.index[0]} to {features.index[-1]}")

# === BACKTEST ===
print("Running backtest with score-based regime detection...")

rebal_dates = features.index
start_idx = 36
quarterly_indices = list(range(start_idx, len(rebal_dates) - 1, 3))

n = len(quarterly_indices)
print(f"Running {n} quarterly iterations...")

portfolio_returns = []
weights_history = []
dates = []
regime_states = []
leverage_used = []
fast_exit_events = []

current_position = SAFE_TICKER

for q_idx, i in enumerate(quarterly_indices):
    if q_idx % 10 == 0:
        print(f"  Iteration {q_idx+1}/{n}")
    
    train_end = rebal_dates[i]
    test_start = rebal_dates[i + 1]
    
    train = features.loc[:train_end].copy()
    test_row = features.loc[test_start:test_start].copy()
    
    # Get regime score at test_start
    if test_start in regime_features.index:
        score = regime_features.loc[test_start, 'score']
    else:
        # Find closest date
        diffs = abs(regime_features.index - test_start)
        score = regime_features.iloc[diffs.argmin()]['score']
    
    regime, leverage_etf = get_regime(score)
    regime_states.append(regime)
    leverage_used.append(leverage_etf)
    
    # Determine current RA based on regime
    current_ra = RA_BULL if regime >= 1 else RA_BEAR
    
    # === DAILY FAST EXIT CHECK ===
    # Check if we need to exit based on daily data
    daily_data = data.loc[train_end:test_start]
    if len(daily_data) > 5:
        daily_returns = daily_data.pct_change().dropna()
        
        # Simple fast exit: check if SPY drops significantly
        for d in daily_returns.index:
            if daily_returns.loc[d, 'SPY'] < -0.03:  # 3% drop
                fast_exit_events.append(d)
                leverage_etf = SAFE_TICKER
                current_position = SAFE_TICKER
                break
    
    # Build ticker list for this period
    if leverage_etf != SAFE_TICKER:
        rotation_tickers = BASE_TICKERS + [leverage_etf]
    else:
        rotation_tickers = BASE_TICKERS
    
    # Build features
    X_train_base = train[ROTATION_FEATURES].copy()
    X_train_base['dynamic_regime'] = regime
    X_train = X_train_base.values
    
    X_test_base = test_row[ROTATION_FEATURES].copy()
    X_test_base['dynamic_regime'] = regime
    X_test = X_test_base.values
    
    y_train = {t: train[f'{t}_ret'].values for t in rotation_tickers}
    
    # Linear model
    preds_mean = {}
    preds_std = {}
    
    for t in rotation_tickers:
        with pm.Model() as linear_model:
            alpha = pm.Normal("alpha", mu=0, sigma=0.1)
            beta = pm.Normal("beta", mu=0, sigma=0.1, shape=X_train.shape[1])
            mu = alpha + pm.math.dot(X_train, beta)
            
            sigma = pm.HalfNormal("sigma", sigma=0.02)
            obs = pm.StudentT("obs", mu=mu, nu=4, observed=y_train[t], sigma=sigma)
            
            X_new = pm.Data("X_new", X_test)
            f_pred = pm.Deterministic("f_pred", alpha + pm.math.dot(X_new, beta))
            
            result = pm.find_MAP(maxeval=30)
            preds_mean[t] = result['f_pred'].item()
            preds_std[t] = result['sigma'] * 2
    
    # Portfolio optimization
    mu_arr = np.array([preds_mean.get(t, 0) for t in rotation_tickers])
    sigma_arr = np.array([preds_std.get(t, 0.02) for t in rotation_tickers])
    cov = np.diag(sigma_arr ** 2)
    
    def objective(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w.T @ cov @ w)
        return -(port_ret - current_ra * port_vol)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in rotation_tickers]
    
    res = minimize(objective, np.ones(len(rotation_tickers)) / len(rotation_tickers), 
                   bounds=bounds, constraints=constraints)
    weights = res.x
    
    # Calculate realized return
    if leverage_etf == SAFE_TICKER:
        # In safe mode - use SHV or minimal return
        if 'SHV' in returns.columns:
            port_ret = returns.loc[test_start, 'SHV']
        else:
            port_ret = 0.001
    else:
        actual_ret = returns.loc[test_start, rotation_tickers].values
        port_ret = np.dot(weights[:len(actual_ret)], actual_ret)
    
    # Reset position for next quarter
    current_position = leverage_etf
    
    portfolio_returns.append(port_ret)
    weights_history.append(weights)
    dates.append(test_start)

print(f"Completed {n} iterations")
print(f"Fast exit events: {len(fast_exit_events)}")

# === RESULTS ===
results = pd.DataFrame({
    'date': dates,
    'return': portfolio_returns,
    'cum_ret': np.cumprod(1 + np.array(portfolio_returns)) - 1
}).set_index('date')

# Corrected metrics for quarterly
ann_ret = results['return'].mean() * 4 * 100
ann_vol = results['return'].std() * np.sqrt(4) * 100
sharpe = results['return'].mean() / results['return'].std() * np.sqrt(4)
portfolio_value = 1 + results['cum_ret']
max_dd = ((portfolio_value / portfolio_value.cummax() - 1) * 100).min()

print(f"\n=== RESULTS (Regime v2 - Score-Based) ===")
print(f"Period: {results.index[0]} to {results.index[-1]}")
print(f"Total quarters: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final Return: {results['cum_ret'].iloc[-1]:.1%}")
print(f"Fast Exit Events: {len(fast_exit_events)}")

# Summary
bull_count = sum(1 for r in regime_states if r == 2)
mod_count = sum(1 for r in regime_states if r == 1)
bear_count = sum(1 for r in regime_states if r == 0)
sso_count = sum(1 for l in leverage_used if l == 'SSO')
qld_count = sum(1 for l in leverage_used if l == 'QLD')
shv_count = sum(1 for l in leverage_used if l == 'SHV')

print(f"\\nRegime: {bull_count} bull, {mod_count} moderate, {bear_count} bear")
print(f"Leverage: SSO={sso_count}, QLD={qld_count}, SHV={shv_count}")

# SPY benchmark
spy_returns = returns['SPY'].dropna()
strategy_start = results.index[0]
spy_from_start = spy_returns.loc[strategy_start:]
spy_cum = (1 + spy_from_start).cumprod() - 1

spy_aligned = []
for d in results.index:
    diffs = abs(spy_cum.index - d)
    spy_aligned.append(spy_cum.iloc[diffs.argmin()])
spy_aligned = pd.Series(spy_aligned, index=results.index)
print(f"SPY: {spy_aligned.iloc[-1]:.1%}")

# Save
results.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv')

# Save weights
all_cols = BASE_TICKERS + LEVERAGE_TICKERS
weights_df = pd.DataFrame(0.0, index=dates, columns=all_cols)
for i, w in enumerate(weights_history):
    for j, t in enumerate(all_cols):
        if j < len(w):
            weights_df.iloc[i][t] = w[j]
weights_df.to_csv(f'{OUTPUT_DIR}/weights_history.csv')

# Save regime history
reg_df = pd.DataFrame({'date': dates, 'regime': regime_states, 'leverage': leverage_used})
reg_df.to_csv(f'{OUTPUT_DIR}/regime_history.csv', index=False)

# Charts
print("\nGenerating charts...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charting

charting.plot_all_charts(
    results_df=results,
    spy_aligned=spy_aligned,
    weights_history=weights_df[BASE_TICKERS].values.tolist(),
    tickers=BASE_TICKERS,
    dates=dates,
    hmm_states=regime_states,
    output_path=f'{OUTPUT_DIR}/factor_rotation_charts.png',
    title_prefix='Regime v2'
)

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")