# -*- coding: utf-8 -*-
"""
Regime Detection v3 - Score-Based with Hysteresis + Real Allocations
- Fixed: All weights showing zero
- Added: Hysteresis to reduce whipsawing
- Added: Proper regime transitions (stay in regime for at least 2 periods)
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
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_regime_v3/output'

# Risk aversion
RA = 0.5

# Hysteresis settings
HYSTERESIS_PERIODS = 2  # Must stay in new regime for 2 periods before switching

# Score thresholds (relaxed to detect more bull/bear)
SCORE_STRONG_BULL = 1.0  # SSO (2x)
SCORE_MODERATE_BULL = 0.0  # QLD (1.5x)
# Below = SHV (0x)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("=== Regime Detection v3 (Hysteresis + Proper Allocations) ===")
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
regime_features['credit_z'] = (regime_features['credit_spread'] - regime_features['credit_spread'].mean()) / regime_features['credit_spread'].std()

# Build score from features
regime_features['score'] = (
    regime_features['trend'] * 1 +
    regime_features['breadth'] * 1 +
    regime_features['momentum'].clip(-2, 2) / 2 +
    regime_features['vol_regime'].clip(-0.02, 0.02) / 0.02 * 0.5 +
    -regime_features['credit_z'].clip(-2, 2) / 2 * 0.5
)

regime_features = regime_features.dropna()

# Get score thresholds (dynamic based on historical distribution)
score_q25 = regime_features['score'].quantile(0.25)
score_q75 = regime_features['score'].quantile(0.75)
print(f"Score distribution: min={regime_features['score'].min():.2f}, q25={score_q25:.2f}, q75={score_q75:.2f}, max={regime_features['score'].max():.2f}")

# Hysteresis tracking
current_regime = None
regime_stability_counter = 0

def get_regime_hysteresis(score, prev_regime, stability_counter):
    """Get regime with hysteresis - require 2 consecutive periods"""
    new_regime = None
    if score >= SCORE_STRONG_BULL:
        new_regime = 'SSO'
    elif score >= SCORE_MODERATE_BULL:
        new_regime = 'QLD'
    else:
        new_regime = 'SHV'
    
    # Apply hysteresis
    if prev_regime is None:
        return new_regime, 1
    
    if new_regime == prev_regime:
        return new_regime, stability_counter + 1
    else:
        if stability_counter >= HYSTERESIS_PERIODS:
            return new_regime, 1
        else:
            return prev_regime, stability_counter + 1

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
print("Running backtest with hysteresis regime detection...")

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
scores_used = []

current_regime = None
regime_stability_counter = 0

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
        diffs = abs(regime_features.index - test_start)
        score = regime_features.iloc[diffs.argmin()]['score']
    
    scores_used.append(score)
    
    # Apply hysteresis
    leverage_etf, regime_stability_counter = get_regime_hysteresis(score, current_regime, regime_stability_counter)
    current_regime = leverage_etf
    
    regime_states.append(leverage_etf)
    leverage_used.append(leverage_etf)
    
    # === BUILD TICKER LIST ===
    if leverage_etf != SAFE_TICKER:
        rotation_tickers = BASE_TICKERS + [leverage_etf]
    else:
        rotation_tickers = BASE_TICKERS
    
    # === BUILD FEATURES ===
    X_train_base = train[ROTATION_FEATURES].copy()
    X_train_base['dynamic_regime'] = 1 if leverage_etf != SAFE_TICKER else 0
    X_train = X_train_base.values
    
    X_test_base = test_row[ROTATION_FEATURES].copy()
    X_test_base['dynamic_regime'] = 1 if leverage_etf != SAFE_TICKER else 0
    X_test = X_test_base.values
    
    y_train = {t: train[f'{t}_ret'].values for t in rotation_tickers}
    
    # === LINEAR MODEL ===
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
            
            # Reduced maxeval for speed
            result = pm.find_MAP(maxeval=20)
            preds_mean[t] = result['f_pred'].item()
            preds_std[t] = result['sigma'] * 2
    
    # === PORTFOLIO OPTIMIZATION ===
    mu_arr = np.array([preds_mean.get(t, 0) for t in rotation_tickers])
    sigma_arr = np.array([preds_std.get(t, 0.02) for t in rotation_tickers])
    cov = np.diag(sigma_arr ** 2)
    
    def objective(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w.T @ cov @ w)
        return -(port_ret - RA * port_vol)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in rotation_tickers]
    
    res = minimize(objective, np.ones(len(rotation_tickers)) / len(rotation_tickers), 
                   bounds=bounds, constraints=constraints)
    weights = res.x
    
    # === CALCULATE REALIZED RETURN ===
    if leverage_etf == SAFE_TICKER:
        if 'SHV' in returns.columns:
            port_ret = returns.loc[test_start, 'SHV']
        else:
            port_ret = 0.001
    else:
        actual_ret = returns.loc[test_start, rotation_tickers].values
        port_ret = np.dot(weights[:len(actual_ret)], actual_ret)
    
    portfolio_returns.append(port_ret)
    # Store both weights and rotation_tickers
    weights_history.append((weights, rotation_tickers))
    dates.append(test_start)

print(f"Completed {n} iterations")

# === RESULTS ===
results = pd.DataFrame({
    'date': dates,
    'return': portfolio_returns,
    'cum_ret': np.cumprod(1 + np.array(portfolio_returns)) - 1
}).set_index('date')

# Metrics (quarterly = 4 periods/year)
ann_ret = results['return'].mean() * 4 * 100
ann_vol = results['return'].std() * np.sqrt(4) * 100
sharpe = results['return'].mean() / results['return'].std() * np.sqrt(4)
portfolio_value = 1 + results['cum_ret']
max_dd = ((portfolio_value / portfolio_value.cummax() - 1) * 100).min()

print(f"\n=== RESULTS (Regime v3 - Hysteresis) ===")
print(f"Period: {results.index[0]} to {results.index[-1]}")
print(f"Total quarters: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final Return: {results['cum_ret'].iloc[-1]:.1%}")

# Summary
sso_count = sum(1 for l in leverage_used if l == 'SSO')
qld_count = sum(1 for l in leverage_used if l == 'QLD')
shv_count = sum(1 for l in leverage_used if l == 'SHV')
print(f"\nLeverage: SSO={sso_count}, QLD={qld_count}, SHV={shv_count}")

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

# === SAVE OUTPUTS ===
results.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv')
print("Saved results CSV")

# Save weights - need to handle varying lengths
# Use max length = 10 (BASE_TICKERS + 1 leverage max)
simple_weights = [list(w) + [0.0] * (10 - len(w)) for (w, rot) in weights_history]
all_cols = BASE_TICKERS + ['LEVERAGE']  # Simplified: just 10 cols
pd.DataFrame(simple_weights, index=dates, columns=all_cols).to_csv(f'{OUTPUT_DIR}/weights_history.csv')
print("Saved weights CSV")

# Save regime history
reg_df = pd.DataFrame({
    'date': dates, 
    'regime': regime_states, 
    'leverage': leverage_used,
    'score': scores_used
})
reg_df.to_csv(f'{OUTPUT_DIR}/regime_history.csv', index=False)

# === CHARTS ===
print("\nGenerating charts...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charting

# Map regime to numeric for chart
regime_numeric = {'SHV': 0, 'QLD': 1, 'SSO': 2}

# Create simplified weights list (just the weight arrays, not tuples)
# Pad to max length 10
simple_weights = [list(w) + [0.0] * (10 - len(w)) for (w, rot) in weights_history]
all_cols_chart = BASE_TICKERS + ['LEVERAGE']  # Just 10 columns

charting.plot_all_charts(
    results_df=results,
    spy_aligned=spy_aligned,
    weights_history=simple_weights,
    tickers=all_cols_chart,
    dates=dates,
    hmm_states=[regime_numeric.get(r, 0) for r in regime_states],
    output_path=f'{OUTPUT_DIR}/factor_rotation_charts.png',
    title_prefix='Regime v3 (Hysteresis)'
)

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")