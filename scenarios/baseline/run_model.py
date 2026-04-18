# -*- coding: utf-8 -*-
"""
Baseline Linear Model (Monthly): Fixed SPY comparison
"""

import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/baseline'

print("=== Baseline Linear Model (Monthly) ===")

# Load data
data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
monthly = data.resample('ME').last()
returns = monthly.pct_change().dropna()

# Features
features = pd.DataFrame(index=returns.index)
for t in TICKERS:
    features[f'{t}_ret'] = returns[t]
    features[f'{t}_vol6'] = returns[t].rolling(6).std()
    features[f'{t}_mom12'] = returns[t].rolling(12).mean()
features = features.dropna()

# Backtest
rebal_dates = features.index[36:]
indices = list(range(0, len(rebal_dates) - 1, 3))

n = len(indices)
print(f"Running {n} iterations...")

portfolio_returns, weights_history, dates = [], [], []
hmm_states = []

for idx, i in enumerate(indices):
    if idx % 10 == 0:
        print(f"  {idx+1}/{n}")
    
    train_end = rebal_dates[i]
    test_start = rebal_dates[i + 1]
    train = features.loc[:train_end]
    test_row = features.loc[test_start:test_start]
    
    # HMM
    spy_train = returns['SPY'].loc[:train_end].values.reshape(-1, 1)
    try:
        hm = hmm.GaussianHMM(n_components=3, covariance_type='full', n_iter=50, random_state=42, min_covar=1e-4)
        hm.fit(spy_train)
        hmm_state = hm.predict(returns['SPY'].loc[:test_start].values.reshape(-1, 1))[-1]
    except:
        hmm_state = 0
    hmm_states.append(hmm_state)
    
    X_train = np.column_stack([train[['SPY_vol6', 'SPY_mom12']].values, np.full(len(train), hmm_state)])
    X_test = np.column_stack([test_row[['SPY_vol6', 'SPY_mom12']].values, [[hmm_state]]])
    
    preds_mean, preds_std = {}, {}
    for t in TICKERS:
        y = train[f'{t}_ret'].values
        with pm.Model() as m:
            alpha = pm.Normal('alpha', mu=0, sigma=0.01)
            beta = pm.Normal('beta', mu=0, sigma=0.1, shape=3)
            sigma = pm.HalfNormal('sigma', sigma=0.02)
            mu = alpha + pm.math.dot(X_train, beta)
            obs = pm.StudentT('obs', mu=mu, nu=4, observed=y, sigma=sigma)
            pred = pm.Deterministic('pred', alpha + pm.math.dot(X_test, beta))
            result = pm.find_MAP(maxeval=30)
            preds_mean[t] = result['pred'].item()
            preds_std[t] = result['sigma'] * 2
    
    mu_arr = np.array([preds_mean[t] for t in TICKERS])
    sigma_arr = np.array([preds_std[t] for t in TICKERS])
    cov = np.diag(sigma_arr ** 2)
    
    res = minimize(lambda w: -(w @ mu_arr - 1.0 * np.sqrt(w @ cov @ w)),
                 np.ones(9)/9, method='SLSQP', bounds=[(0,1)]*9,
                 constraints={'type':'eq', 'fun': lambda w: sum(w)-1})
    
    weights = res.x
    port_ret = weights @ returns.loc[test_start, TICKERS].values
    
    portfolio_returns.append(port_ret)
    weights_history.append(weights)
    dates.append(test_start)

results = pd.DataFrame({'date': dates, 'return': portfolio_returns,
                        'cum_ret': np.cumprod(1 + np.array(portfolio_returns)) - 1}).set_index('date')

# Verify results
print(f"\n=== VERIFICATION ===")
print(f"Period: {results.index[0]} to {results.index[-1]}")
print(f"Total periods: {len(results)}")

# Calculate metrics
ann_ret = results['return'].mean() * 12 * 100
ann_vol = results['return'].std() * np.sqrt(12) * 100
sharpe = results['return'].mean() / results['return'].std() * np.sqrt(12)
max_dd = ((results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100).min()

print(f"\n=== RESULTS ===")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%") 
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final: {results['cum_ret'].iloc[-1]:.1%}")

# SPY aligned to same dates - get SPY cumulative at each strategy date
spy_returns = returns['SPY'].dropna()
spy_cum = (1 + spy_returns).cumprod() - 1

# For each strategy date, get the closest SPY cumulative return
spy_aligned = []
for d in results.index:
    # Find closest date in spy_cum
    diffs = abs(spy_cum.index - d)
    closest_idx = diffs.argmin()
    spy_aligned.append(spy_cum.iloc[closest_idx])
spy_aligned = pd.Series(spy_aligned, index=results.index)

print(f"\nSPY (same period): {spy_aligned.iloc[-1]:.1%}")

# Save
results.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv')
pd.DataFrame(weights_history, index=dates, columns=TICKERS).to_csv(f'{OUTPUT_DIR}/weights_history.csv')

# Charts
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
regime_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
hmm_series = pd.Series(hmm_states, index=results.index)

# Equity with CORRECTED SPY
axes[0].plot(results.index, results['cum_ret'].values * 100, 
            label=f'Strategy ({results["cum_ret"].iloc[-1]:.1%})', linewidth=2.5, color='blue')
axes[0].plot(spy_aligned.index, spy_aligned.values * 100, 
            label=f'SPY ({spy_aligned.iloc[-1]:.1%})', linewidth=2, color='gray', alpha=0.7)
axes[0].set_ylabel('Cumulative Return (%)')
axes[0].set_title('Equity Curve - Baseline Linear (Monthly)')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)

# Drawdown
drawdown = (results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100
axes[1].fill_between(results.index, drawdown.values, 0, color='red', alpha=0.5)
axes[1].set_ylabel('Drawdown (%)')
axes[1].grid(True, alpha=0.3)

# Regime
axes[2].fill_between(results.index, 0, hmm_series.values * 30, 
                     color=[regime_colors.get(r, 'gray') for r in hmm_series], alpha=0.6)
axes[2].set_ylabel('HMM Regime')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/factor_rotation_equity_curve.png', dpi=150)
plt.close()

# Weights
wdf = pd.DataFrame(weights_history, index=dates, columns=TICKERS)
wdf.plot(kind='area', stacked=True, alpha=0.7, figsize=(14, 8), colormap='tab10')
plt.ylabel('Portfolio Weight')
plt.title('Portfolio Allocation')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/factor_rotation_weights.png', dpi=150)
plt.close()

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")