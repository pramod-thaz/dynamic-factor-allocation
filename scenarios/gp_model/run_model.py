# -*- coding: utf-8 -*-
"""
GP Model: Replace Linear with GP in baseline
Same settings as baseline, just GP instead of Linear
"""

import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from hmmlearn import hmm
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os

TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = './'

print("=== GP Model Test ===")

if __name__ == "__main__":
    # Load data
    if os.path.exists(DATA_FILE):
        data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        data = yf.download(TICKERS, start='2013-01-01', end=datetime.today().strftime('%Y-%m-%d'))['Close']
        data.to_csv(DATA_FILE)
    
    monthly = data.resample('ME').last()
    returns = monthly.pct_change().dropna()
    returns.index = returns.index.map(lambda x: x.replace(day=x.days_in_month))
    
    # Features
    features = pd.DataFrame(index=returns.index)
    for t in TICKERS:
        features[f'{t}_ret'] = returns[t]
        features[f'{t}_vol6'] = returns[t].rolling(6).std()
        features[f'{t}_mom12'] = returns[t].rolling(12).mean()
    features = features.dropna()
    
    # Backtest - FULL iterations like baseline
    rebal_dates = features.index[36:]
    indices = list(range(0, len(rebal_dates) - 1, 3))  # Quarterly, full
    
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
        
        # Gaussian HMM
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
            
            # GP MODEL
            with pm.Model() as m:
                mean_func = pm.gp.mean.Zero()
                cov = pm.gp.cov.Matern52(input_dim=X_train.shape[1], ls=1.0)
                gp = pm.gp.Latent(mean_func=mean_func, cov_func=cov)
                sigma = pm.HalfNormal('sigma', sigma=0.05)
                yGP = gp.prior('yGP', X=X_train)
                obs = pm.StudentT('obs', mu=yGP, nu=4, observed=y, sigma=sigma)
                f_pred = gp.conditional('f_pred', X_test)
                result = pm.find_MAP(maxeval=20)
                preds_mean[t] = result['f_pred'].item()
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
    
    ann_ret = results['return'].mean() * 12 * 100
    ann_vol = results['return'].std() * np.sqrt(12) * 100
    sharpe = results['return'].mean() / results['return'].std() * np.sqrt(12)
    max_dd = ((results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100).min()
    
    print(f"\n=== GP RESULTS ===")
    print(f"Annual Return: {ann_ret:.1f}%")
    print(f"Annual Vol: {ann_vol:.1f}%") 
    print(f"Sharpe: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.1f}%")
    print(f"Final: {results['cum_ret'].iloc[-1]:.1%}")
    
    # Save results
    results.to_csv('factor_rotation_backtest_results.csv')
    pd.DataFrame(weights_history, index=dates, columns=TICKERS).to_csv('weights_history.csv')
    
    # SPY benchmark
    spy_cum = (1 + returns['SPY']).cumprod() - 1
    spy_aligned = spy_cum.reindex(results.index)
    
    # Chart
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    regime_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
    hmm_series = pd.Series(hmm_states, index=results.index)
    
    # Equity curve
    axes[0].plot(results.index, results['cum_ret'].values * 100, label=f'GP ({results["cum_ret"].iloc[-1]:.1%})', linewidth=2.5, color='blue')
    
    # Overlay SPY Buy & Hold
    axes[0].plot(spy_aligned.index, spy_aligned.values * 100, label=f'SPY ({spy_aligned.iloc[-1]:.1%})', linewidth=2, color='gray', alpha=0.7)
    axes[0].set_ylabel('Cumulative Return (%)')
    axes[0].set_title('GP Model Equity Curve')
    axes[0].legend(loc='upper left')
    axes[0].grid(True, alpha=0.3)
    
    # Drawdown
    drawdown = (results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100
    axes[1].fill_between(results.index, drawdown.values, 0, color='red', alpha=0.5)
    axes[1].set_ylabel('Drawdown (%)')
    axes[1].set_title('Drawdown')
    axes[1].grid(True, alpha=0.3)
    
    # Regime timeline  
    axes[2].fill_between(results.index, 0, hmm_series.values * 30, color=[regime_colors.get(r, 'gray') for r in hmm_series], alpha=0.6)
    axes[2].set_ylabel('HMM Regime')
    axes[2].set_title('Regime Timeline')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('factor_rotation_equity_curve.png', dpi=150)
    plt.close()
    
    # Weights chart
    wdf = pd.DataFrame(weights_history, index=dates, columns=TICKERS)
    wdf.plot(kind='area', stacked=True, alpha=0.7, figsize=(14, 8), colormap='tab10')
    plt.ylabel('Portfolio Weight')
    plt.title('ETF Allocation Weights Over Time')
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()
    plt.savefig('factor_rotation_weights.png', dpi=150)
    plt.close()
    
    print("\nOutput saved:")
    print("  factor_rotation_backtest_results.csv")
    print("  factor_rotation_equity_curve.png")
    print("  factor_rotation_weights.png")
    print("=== DONE ===")