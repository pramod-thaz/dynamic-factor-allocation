# -*- coding: utf-8 -*-
"""
Baseline Linear Model (Quarterly): SSO as Leverage Proxy
- Use SPY for prediction/optimization (9 tickers)
- In bull/transition states, replace SPY with SSO in portfolio
- SSO acts as 2x leverage proxy
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
BASE_TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
SSO_TICKER = 'SSO'

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_regime_aware_sso_proxy/output'

# Risk aversion by regime
RA_BULL = 0.5
RA_TRANSITION = 0.5
RA_BEAR = 1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
print("=== Linear Quarterly SSO Proxy (Leverage) ===")
print("Loading data...")

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
monthly = data.resample('ME').last()
returns = monthly.pct_change().dropna()

print(f"Data period: {returns.index[0]} to {returns.index[-1]}")

# Verify all tickers
for t in BASE_TICKERS + [SSO_TICKER]:
    if t not in returns.columns:
        raise ValueError(f"Ticker {t} not found!")

# === FEATURE ENGINEERING ===
print("Building features...")

features = pd.DataFrame(index=returns.index)
for t in BASE_TICKERS:
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

BASE_FEATURES = [
    'SPY_vol6', 'SPY_mom12',
    'value_mom_spread', 'lowvol_market_spread', 'growth_value_spread',
    'bond_gold_spread', 'small_market_spread'
]

print(f"Feature period: {features.index[0]} to {features.index[-1]}")

# === BACKTEST ===
rebal_dates = features.index
start_idx = 36
indices = list(range(start_idx, len(rebal_dates) - 1, 3))

n = len(indices)
print(f"\nRunning backtest with {n} quarterly iterations...")

portfolio_returns = []
weights_history = []  # Store with SSO column for chart compatibility
weights_for_opt = []  # Store just 9-ticker weights for optimization
dates = []
hmm_states = []
ra_used = []
sso_swapped = []  # Track which periods used SSO swap

for idx, i in enumerate(indices):
    if idx % 10 == 0:
        print(f"  Iteration {idx+1}/{n}")
    
    train_end = rebal_dates[i]
    test_start = rebal_dates[i + 1]
    
    train = features.loc[:train_end].copy()
    test_row = features.loc[test_start:test_start].copy()
    
    # HMM refit
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
    
    # Determine RA based on regime
    if hmm_state_for_X_test == 0:
        current_ra = RA_BULL
        swap_to_sso = True
    elif hmm_state_for_X_test == 1:
        current_ra = RA_TRANSITION
        swap_to_sso = True
    else:  # bear
        current_ra = RA_BEAR
        swap_to_sso = False
    
    ra_used.append(current_ra)
    sso_swapped.append(swap_to_sso)
    
    # Build X (always 9 features for optimization)
    X_train_base = train[BASE_FEATURES].copy()
    X_train_base['dynamic_hmm_regime'] = hmm_states_for_X_train
    X_train = X_train_base.values
    
    X_test_base = test_row[BASE_FEATURES].copy()
    X_test_base['dynamic_hmm_regime'] = hmm_state_for_X_test
    X_test = X_test_base.values
    
    # y training (always 9 tickers for optimization)
    y_train = {t: train[f'{t}_ret'].values for t in BASE_TICKERS}
    
    # Linear model for prediction
    preds_mean = {}
    preds_std = {}
    
    for t in BASE_TICKERS:
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
    
    # Portfolio optimization (always 9 tickers)
    mu_arr = np.array([preds_mean[t] for t in BASE_TICKERS])
    sigma_arr = np.array([preds_std[t] for t in BASE_TICKERS])
    cov = np.diag(sigma_arr ** 2)
    
    def objective(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w.T @ cov @ w)
        return -(port_ret - current_ra * port_vol)
    
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in BASE_TICKERS]
    
    res = minimize(objective, np.ones(len(BASE_TICKERS)) / len(BASE_TICKERS), 
                   bounds=bounds, constraints=constraints)
    opt_weights = res.x  # These are for 9 tickers
    
    # === SSO SWAP LOGIC ===
    # If bull or transition, replace SPY weight with SSO
    if swap_to_sso:
        # Find SPY index
        spy_idx = BASE_TICKERS.index('SPY')
        sso_weight = opt_weights[spy_idx]
        
        # Create final weights with SSO column
        # Start with all zeros
        final_weights = np.zeros(len(BASE_TICKERS) + 1)  # 9 + SSO
        # Copy all non-SPY weights
        for j, t in enumerate(BASE_TICKERS):
            if t != 'SPY':
                final_weights[j] = opt_weights[j]
        # Put SPY weight into SSO position (last column)
        final_weights[-1] = sso_weight
        # Set SPY to 0 (swapped to SSO)
        final_weights[spy_idx] = 0
        
        # Use SSO return instead of SPY for actual return calculation
        actual_returns = returns.loc[test_start, BASE_TICKERS].copy()
        actual_returns['SPY'] = returns.loc[test_start, SSO_TICKER]
    else:
        # No swap - keep SPY
        final_weights = np.zeros(len(BASE_TICKERS) + 1)
        for j, t in enumerate(BASE_TICKERS):
            final_weights[j] = opt_weights[j]
        
        actual_returns = returns.loc[test_start, BASE_TICKERS]
    
    # Calculate realized return
    if swap_to_sso:
        # Create returns array with SSO instead of SPY
        all_returns = []
        for t in BASE_TICKERS:
            if t == 'SPY':
                all_returns.append(returns.loc[test_start, SSO_TICKER])
            else:
                all_returns.append(returns.loc[test_start, t])
        # Add SSO return (same as SPY when swapped - just for weight calculation)
        all_returns.append(returns.loc[test_start, SSO_TICKER])
        port_ret = final_weights @ np.array(all_returns)
    else:
        # No swap - use original SPY, SSO weight is 0
        all_returns = [returns.loc[test_start, t] for t in BASE_TICKERS]
        all_returns.append(0)  # SSO weight is 0
        port_ret = final_weights @ np.array(all_returns)
    
    portfolio_returns.append(port_ret)
    # Store weights with SSO column for consistent chart
    weights_history.append(final_weights)
    weights_for_opt.append(opt_weights)
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

print(f"\n=== RESULTS (SSO Proxy - Leverage) ===")
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
sso_count = sum(1 for s in sso_swapped if s)

print(f"\nRegime Usage: {bull_count} bull, {trans_count} transition, {bear_count} bear")
print(f"SSO Swapped: {sso_count} periods")

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

# Save weights (10 columns: 9 tickers + SSO)
all_cols = BASE_TICKERS + [SSO_TICKER]
weights_df = pd.DataFrame(weights_history, index=dates, columns=all_cols)
weights_df.to_csv(f'{OUTPUT_DIR}/weights_history.csv')

# RA history
ra_df = pd.DataFrame({
    'date': dates, 
    'hmm_state': hmm_states, 
    'risk_aversion': ra_used,
    'sso_swapped': sso_swapped
})
ra_df.to_csv(f'{OUTPUT_DIR}/ra_history.csv', index=False)

# Charts
print("\nGenerating charts...")
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import charting

# Use BASE_TICKERS for charting (show actual SPY position, not SSO)
# But for SSO swapped periods, this will show the swapped weight in SPY slot
charting.plot_all_charts(
    results_df=results,
    spy_aligned=spy_aligned,
    weights_history=weights_df[BASE_TICKERS].values.tolist(),
    tickers=BASE_TICKERS,
    dates=dates,
    hmm_states=hmm_states,
    output_path=f'{OUTPUT_DIR}/factor_rotation_charts.png',
    title_prefix='Linear Quarterly SSO Proxy'
)

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")