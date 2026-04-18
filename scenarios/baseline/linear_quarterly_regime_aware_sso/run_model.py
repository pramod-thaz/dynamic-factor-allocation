# -*- coding: utf-8 -*-
"""
Baseline Linear Model (Quarterly): Regime-Aware with SSO in Bull Markets
Risk Aversion: 0.5 for bull/transition, 1.0 for bear
SSO (2x S&P 500) added in bull regime only
"""

import pandas as pd
import numpy as np
import pymc as pm
from scipy.optimize import minimize
from hmmlearn import hmm
import warnings
import os

warnings.filterwarnings('ignore')

# === CONFIG ===
BASE_TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMC', 'VUG', 'IJR', 'TLT', 'GLD']
# Fix: correct ticker is 'USMV' not 'USMC'
BASE_TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
SSO_TICKER = 'SSO'

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_regime_aware_sso/output'

RA_BULL = 0.5
RA_TRANSITION = 0.5
RA_BEAR = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("=== Baseline Linear Model (Quarterly, Regime-Aware with SSO) ===")
print("Loading data...")

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
monthly = data.resample('ME').last()
returns = monthly.pct_change().dropna()

print(f"Data period: {returns.index[0]} to {returns.index[-1]}")

# Verify all tickers exist
all_tickers = BASE_TICKERS + [SSO_TICKER]
for t in all_tickers:
    if t not in returns.columns:
        raise ValueError(f"Ticker {t} not found in data!")

# === FEATURE ENGINEERING ===
print("Building features...")

features = pd.DataFrame(index=returns.index)

# Build features for ALL tickers including SSO
for t in all_tickers:
    features[f'{t}_ret'] = returns[t]
    features[f'{t}_vol6'] = returns[t].rolling(6).std()
    features[f'{t}_mom12'] = returns[t].rolling(12).mean()

# Spread features (only using base tickers)
features['value_mom_spread'] = features['VTV_ret'] - features['MTUM_ret']
features['lowvol_market_spread'] = features['USMV_ret'] - features['SPY_ret']
features['growth_value_spread'] = features['VUG_ret'] - features['VTV_ret']
features['bond_gold_spread'] = features['TLT_ret'] - features['GLD_ret']
features['small_market_spread'] = features['IJR_ret'] - features['SPY_ret']

features = features.dropna()

# Base features (without SSO for regular regime)
BASE_FEATURES = [
    'SPY_vol6', 'SPY_mom12',
    'value_mom_spread', 'lowvol_market_spread', 'growth_value_spread',
    'bond_gold_spread', 'small_market_spread'
]

# Features when SSO is included
FEATURES_WITH_SSO = BASE_FEATURES + ['SSO_vol6', 'SSO_mom12']

print(f"Feature period: {features.index[0]} to {features.index[-1]}")

# === BACKTEST ===
rebal_dates = features.index
start_idx = 36
indices = list(range(start_idx, len(rebal_dates) - 1, 3))

n = len(indices)
print(f"\nRunning backtest with {n} quarterly iterations...")

portfolio_returns = []
weights_history = []
dates = []
hmm_states = []
ra_used = []
sso_used = []

for idx, i in enumerate(indices):
    if idx % 10 == 0:
        print(f"  Iteration {idx+1}/{n}")
    
    train_end = rebal_dates[i]
    test_start = rebal_dates[i + 1]
    
    train = features.loc[:train_end].copy()
    test_row = features.loc[test_start:test_start].copy()
    
    # HMM
    hmm_train_spy_data = returns['SPY'].loc[:train_end].values.reshape(-1, 1)
    
    try:
        hmm_model = hmm.GaussianHMM(
            n_components=3, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42, 
            min_covar=1e-6
        )
        hmm_model.fit(hmm_train_spy_data)
        
        full_hmm_pred = hmm_model.predict(returns['SPY'].loc[:test_start].values.reshape(-1, 1))
        full_hmm_pred_series = pd.Series(full_hmm_pred, index=returns.loc[:test_start].index)
        
        smoothed_hmm = full_hmm_pred_series.rolling(window=3, min_periods=1).apply(
            lambda x: x.mode()[0], raw=False
        )
        smoothed_hmm = smoothed_hmm.fillna(full_hmm_pred_series).astype(int)
        
        hmm_states_for_X_train = smoothed_hmm.loc[train.index].values
        hmm_state_for_X_test = smoothed_hmm.loc[test_start]
        
    except Exception as e:
        hmm_state_for_X_test = 0
        hmm_states_for_X_train = np.zeros(len(train))
    
    hmm_states.append(hmm_state_for_X_test)
    
    # Determine tickers and RA based on regime
    if hmm_state_for_X_test == 0:
        # Bull regime: add SSO
        current_tickers = BASE_TICKERS + [SSO_TICKER]
        current_features = FEATURES_WITH_SSO
        current_ra = RA_BULL
        sso_included = True
    elif hmm_state_for_X_test == 1:
        # Transition: no SSO
        current_tickers = BASE_TICKERS
        current_features = BASE_FEATURES
        current_ra = RA_TRANSITION
        sso_included = False
    else:  # bear
        current_tickers = BASE_TICKERS
        current_features = BASE_FEATURES
        current_ra = RA_BEAR
        sso_included = False
    
    ra_used.append(current_ra)
    sso_used.append(sso_included)
    
    # Build X with correct features
    X_train_base = train[current_features].copy()
    X_train_base['dynamic_hmm_regime'] = hmm_states_for_X_train
    X_train = X_train_base.values
    
    X_test_base = test_row[current_features].copy()
    X_test_base['dynamic_hmm_regime'] = hmm_state_for_X_test
    X_test = X_test_base.values
    
    # Prepare y
    y_train = {t: train[f'{t}_ret'].values for t in current_tickers}
    
    # Linear model
    preds_mean = {}
    preds_std = {}
    
    for t in current_tickers:
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
    
    # Portfolio construction
    mu_arr = np.array([preds_mean[t] for t in current_tickers])
    sigma_arr = np.array([preds_std[t] for t in current_tickers])
    cov = np.diag(sigma_arr ** 2)
    
    def objective(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w.T @ cov @ w)
        return -(port_ret - current_ra * port_vol)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in current_tickers]
    
    res = minimize(objective, np.ones(len(current_tickers)) / len(current_tickers), 
                   bounds=bounds, constraints=constraints)
    weights = res.x
    
    # Realized return
    next_ret = returns.loc[test_start, current_tickers].values
    port_ret = weights @ next_ret
    
    portfolio_returns.append(port_ret)
    weights_history.append(weights)
    dates.append(test_start)

print(f"Completed {n} iterations")

# === RESULTS ===
results = pd.DataFrame({
    'date': dates,
    'return': portfolio_returns,
    'cum_ret': np.cumprod(1 + np.array(portfolio_returns)) - 1
}).set_index('date')

ann_ret = results['return'].mean() * 4 * 100  # quarterly data: 4 quarters per year
ann_vol = results['return'].std() * np.sqrt(4) * 100
sharpe = results['return'].mean() / results['return'].std() * np.sqrt(4)
# Correct drawdown: based on portfolio value, not percentage
portfolio_value = 1 + results['cum_ret']
max_dd = ((portfolio_value / portfolio_value.cummax() - 1) * 100).min()

print(f"\n=== RESULTS (Regime-Aware with SSO) ===")
print(f"Period: {results.index[0]} to {results.index[-1]}")
print(f"Total quarters: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final Return: {results['cum_ret'].iloc[-1]:.1%}")

# Summary
bull_count = sum(1 for s in hmm_states if s == 0)
trans_count = sum(1 for s in hmm_states if s == 1)
bear_count = sum(1 for s in hmm_states if s == 2)
sso_count = sum(1 for s in sso_used if s)

print(f"\nRegime Usage: {bull_count} bull, {trans_count} transition, {bear_count} bear")
print(f"SSO Used: {sso_count} periods")

# SPY benchmark
spy_returns = returns['SPY'].dropna()
strategy_start = results.index[0]
spy_from_start = spy_returns.loc[strategy_start:]
spy_cum_from_strategy = (1 + spy_from_start).cumprod() - 1

spy_aligned = []
for d in results.index:
    diffs = abs(spy_cum_from_strategy.index - d)
    closest_idx = diffs.argmin()
    spy_aligned.append(spy_cum_from_strategy.iloc[closest_idx])

spy_aligned = pd.Series(spy_aligned, index=results.index)
print(f"SPY (same period): {spy_aligned.iloc[-1]:.1%}")

# Save results
results.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv')

# Save weights with consistent columns
all_cols = BASE_TICKERS + [SSO_TICKER]
weights_df = pd.DataFrame(0.0, index=dates, columns=all_cols)
for i, w in enumerate(weights_history):
    period_tickers = (BASE_TICKERS + [SSO_TICKER]) if sso_used[i] else BASE_TICKERS
    for j, t in enumerate(period_tickers):
        weights_df.loc[dates[i], t] = w[j]
weights_df.to_csv(f'{OUTPUT_DIR}/weights_history.csv')

# RA history
ra_df = pd.DataFrame({
    'date': dates, 
    'hmm_state': hmm_states, 
    'risk_aversion': ra_used,
    'sso_included': sso_used
})
ra_df.to_csv(f'{OUTPUT_DIR}/ra_history.csv', index=False)

# Charts
print("\nGenerating charts...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charting

# Use BASE_TICKERS for charting
charting.plot_all_charts(
    results_df=results,
    spy_aligned=spy_aligned,
    weights_history=weights_df[BASE_TICKERS].values.tolist(),
    tickers=BASE_TICKERS,
    dates=dates,
    hmm_states=hmm_states,
    output_path=f'{OUTPUT_DIR}/factor_rotation_charts.png',
    title_prefix='Linear Quarterly Regime-Aware SSO'
)

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")