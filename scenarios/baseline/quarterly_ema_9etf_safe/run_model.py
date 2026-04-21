import pandas as pd
import numpy as np
from scipy.optimize import minimize
import yfinance as yf
import warnings
from datetime import timedelta, datetime
warnings.filterwarnings('ignore')

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/quarterly_ema_9etf_safe/output'

TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV', 'VUG', 'IJR', 'TLT', 'GLD']
SAFE_TICKERS = ['TLT', 'GLD', 'SHV']

TURNOVER_PENALTY = 0.015

def fetch_and_merge_data():
    print("\nFetching latest data...")
    try:
        cached = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        last_cached_date = cached.index[-1]
        
        if last_cached_date.date() >= (datetime.now() - timedelta(days=2)).date():
            return cached
        
        start_date = last_cached_date + timedelta(days=1)
        new_data = yf.download(list(cached.columns), start=start_date.strftime('%Y-%m-%d'), 
                           end=datetime.now().strftime('%Y-%m-%d'), progress=False)
        
        if new_data.empty:
            return cached
        
        if 'Adj Close' in new_data.columns.get_level_values(0):
            new_data = new_data['Adj Close']
        else:
            new_data = new_data['Close']
        
        combined = pd.concat([cached, new_data]).dropna(how='all')
        combined = combined[~combined.index.duplicated(keep='last')].sort_index()
        combined.to_csv(DATA_FILE)
        return combined
    except:
        return pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

print("=== Quarterly EMA 9-ETF with Safe Exit ===")

data = fetch_and_merge_data()
all_tickers = list(set(TICKERS + SAFE_TICKERS))
daily_prices = data[all_tickers]
daily_returns = daily_prices.pct_change().dropna()

print(f"Data: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

def predict_next(returns_df, ticker, span=40):
    try:
        arr = returns_df[ticker].dropna()
        if len(arr) < 40:
            return arr.mean(), arr.std() if len(arr) > 0 else 0.01
        return arr.iloc[-span:].mean(), arr.iloc[-40:].std()
    except:
        return 0.0, 0.01

def mean_variance_optimize(mu, sigma, previous=None, risk_aversion=0.5, turnover_penalty=0.015):
    mu_arr = np.array([mu.get(t, 0) for t in TICKERS])
    sigma_arr = np.diag([sigma.get(t, 0.01) ** 2 for t in TICKERS])
    
    def obj(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w @ sigma_arr @ w)
        turnover_cost = turnover_penalty * np.sum(np.abs(w - previous)) if previous is not None else 0
        return risk_aversion * port_vol**2 - port_ret + turnover_cost
    
    result = minimize(obj, np.ones(len(TICKERS)) / len(TICKERS), method='SLSQP',
                    bounds=[(0, 1) for _ in TICKERS],
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    return result.x if result.success else np.ones(len(TICKERS)) / len(TICKERS)

def safe_allocation(returns_df):
    mu = {}
    sigma = {}
    for t in SAFE_TICKERS:
        m, s = predict_next(returns_df, t)
        mu[t] = m
        sigma[t] = s
    
    mu_arr = np.array([mu.get(t, 0) for t in SAFE_TICKERS])
    sigma_arr = np.diag([(sigma.get(t, 0.01)) ** 2 for t in SAFE_TICKERS])
    
    def obj(w):
        return np.sqrt(w @ sigma_arr @ w)**2 - w @ mu_arr
    
    result = minimize(obj, np.ones(len(SAFE_TICKERS)) / len(SAFE_TICKERS), method='SLSQP',
                    bounds=[(0, 1) for _ in SAFE_TICKERS],
                    constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    return result.x if result.success else np.ones(len(SAFE_TICKERS)) / len(SAFE_TICKERS)

print("\n=== Running Backtest ===")

start_date = pd.Timestamp('2018-01-01')
d_idx = 0
while d_idx < len(daily_returns) and daily_returns.index[d_idx] < start_date:
    d_idx += 1

portfolio_returns = []
dates = []
allocation_history = []

current_allocation = np.zeros(len(TICKERS))
current_allocation[0] = 1.0

quarterly_rebals = 0
fast_exits_count = 0
reentries = 0

last_rebal = daily_returns.index[d_idx]
in_safe_mode = False

print(f"Period: {daily_returns.index[d_idx].date()} to {daily_returns.index[-1].date()}")

while d_idx < len(daily_returns):
    current_date = daily_returns.index[d_idx]
    
    quarterly_due = (current_date - last_rebal).days >= 63
    
    if quarterly_due:
        preds_mean = {t: predict_next(daily_returns, t)[0] for t in TICKERS}
        preds_std = {t: predict_next(daily_returns, t)[1] for t in TICKERS}
        
        current_allocation = mean_variance_optimize(preds_mean, preds_std, previous=current_allocation)
        
        last_rebal = current_date
        quarterly_rebals += 1
        in_safe_mode = False
        
        top3 = sorted(zip(TICKERS, current_allocation), key=lambda x: -x[1])[:3]
        print(f"REBAL → {current_date.date()}: {[(t, f'{w:.0%}') for t, w in top3]}")
    
    vol_5d = daily_returns['SPY'].iloc[max(0, d_idx-5):d_idx].std() * np.sqrt(252) if d_idx > 5 else 0
    
    if not in_safe_mode and vol_5d > 0.55:
        safe_weights = safe_allocation(daily_returns)
        current_allocation = np.zeros(len(TICKERS))
        for i, t in enumerate(SAFE_TICKERS):
            if t in TICKERS:
                current_allocation[TICKERS.index(t)] = safe_weights[i]
        in_safe_mode = True
        fast_exits_count += 1
        print(f"SAFE EXIT → {current_date.date()} → vol={vol_5d:.1%}")
    
    lookback = current_date - pd.Timedelta(days=30)
    if lookback in daily_prices.index and not in_safe_mode:
        recent_high = daily_prices['SPY'].loc[lookback:current_date].max()
        drop_30d = (daily_prices['SPY'].loc[current_date] / recent_high) - 1
        if drop_30d < -0.15:
            safe_weights = safe_allocation(daily_returns)
            current_allocation = np.zeros(len(TICKERS))
            for i, t in enumerate(SAFE_TICKERS):
                if t in TICKERS:
                    current_allocation[TICKERS.index(t)] = safe_weights[i]
            in_safe_mode = True
            fast_exits_count += 1
            print(f"CRASH EXIT → {current_date.date()} → drop={drop_30d:.1%}")
    
    if in_safe_mode and vol_5d < 0.20:
        preds_mean = {t: predict_next(daily_returns, t)[0] for t in TICKERS}
        preds_std = {t: predict_next(daily_returns, t)[1] for t in TICKERS}
        
        current_allocation = mean_variance_optimize(preds_mean, preds_std, previous=None)
        
        in_safe_mode = False
        reentries += 1
        last_rebal = current_date
        quarterly_rebals += 1
        print(f"RE-ENTRY → {current_date.date()} → vol={vol_5d:.1%}")
    
    daily_ret = sum(current_allocation[i] * daily_returns[TICKERS[i]].iloc[d_idx] for i in range(len(TICKERS)))
    portfolio_returns.append(daily_ret)
    dates.append(current_date)
    allocation_history.append(list(current_allocation))
    
    d_idx += 1

results = pd.DataFrame({'return': portfolio_returns, 'allocation': allocation_history}, index=dates)
results['cum_ret'] = (1 + results['return']).cumprod() - 1

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol

portfolio_value = pd.Series(1.0, index=results.index)
for i in range(1, len(results)):
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + results['return'].iloc[i])
max_dd = ((portfolio_value / portfolio_value.expanding().max() - 1) * 100).min()

spy_ret = (1 + daily_returns['SPY'].loc[results.index[0]:results.index[-1]]).cumprod().iloc[-1] - 1

print("\n" + "=" * 50)
print("RESULTS")
print("=" * 50)
print(f"Quarterly Rebalances: {quarterly_rebals}")
print(f"Fast Exits: {fast_exits_count}")
print(f"Re-entries: {reentries}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Strategy: {results['cum_ret'].iloc[-1]*100:.1f}%")
print(f"SPY: {spy_ret*100:.1f}%")
print(f"Excess: {(results['cum_ret'].iloc[-1] - spy_ret)*100:.1f}%")

results.to_csv(f'{OUTPUT_DIR}/daily_results.csv')
print("\nSaved!")
print("=== DONE ===")