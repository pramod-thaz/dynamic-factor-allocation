import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_sklearn_gp/output'

TICKERS = ['SSO', 'SPY', 'SHV']

FAST_EXIT_PROB = 0.65
FAST_EXIT_DAYS = 4

SLOW_EXIT_PROB = 0.48
SLOW_EXIT_DAYS = 12

REENTRY_PROB = 0.60
REENTRY_DAYS = 5

CRASH_DROP = 0.12

print("=== sklearn GP + Probability Exits + HMM ===", flush=True)

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
daily_prices = data[TICKERS]
daily_returns = daily_prices.pct_change().dropna()

print(f"Data: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

def calculate_hmm_features(returns_df, prices_df):
    features = pd.DataFrame(index=returns_df.index)
    
    ma50 = returns_df['SPY'].rolling(50).mean()
    ma150 = returns_df['SPY'].rolling(150).mean()
    
    features['trend_fast'] = (returns_df['SPY'] > ma50).astype(float)
    features['trend_slow'] = (returns_df['SPY'] > ma150).astype(float)
    features['momentum'] = returns_df['SPY'].rolling(63).sum()
    features['breadth'] = (returns_df['SPY'].rolling(50).mean() > 0).astype(float)
    features['vol_regime'] = returns_df['SHV'] - returns_df['SPY']
    
    def adx(price, period=14):
        high = price.rolling(period).max()
        low = price.rolling(period).min()
        tr = pd.concat([high - low, abs(high - price.shift()), abs(low - price.shift())], axis=1).max(axis=1)
        dm_plus = high - high.shift()
        dm_minus = low.shift() - low
        di_plus = 100 * dm_plus.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean()
        di_minus = 100 * dm_minus.ewm(alpha=1/period).mean() / tr.ewm(alpha=1/period).mean()
        dx = abs(di_plus - di_minus) / (di_plus + di_minus + 1e-8) * 100
        return dx.rolling(period).mean()
    
    features['adx'] = adx(prices_df['SPY'].cumsum())
    return features.dropna()

def sklearn_gp_predict(returns_df, ticker, current_date, lookback=252):
    train_mask = returns_df.index <= current_date
    y = returns_df[ticker][train_mask].values
    
    if len(y) < 100:
        return y[-63:].mean(), y[-63:].std()
    
    recent_vol = y[-63:].std()
    recent_mean = y[-63:].mean()
    
    y_train = y[-lookback:] if len(y) > lookback else y
    X_train = np.arange(len(y_train)).reshape(-1, 1)
    
    try:
        kernel = Matern(length_scale=0.8, nu=2.5) + WhiteKernel(noise_level=0.01)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=0.01, n_restarts_optimizer=2, random_state=42)
        gp.fit(X_train, y_train)
        
        X_new = np.array([[len(y_train)]])
        pred_mean, pred_std = gp.predict(X_new, return_std=True)
        pred_std = float(pred_std[0]) if isinstance(pred_std, np.ndarray) else float(pred_std)
        
        return float(pred_mean), max(pred_std, recent_vol * 0.3)
    except:
        return recent_mean, recent_vol

def mean_variance_optimize(preds_mean, preds_std, risk_aversion=0.5):
    mu = np.array(preds_mean)
    sigma = np.diag(preds_std ** 2)
    
    def obj(w):
        port_return = w @ mu
        port_vol = np.sqrt(w @ sigma @ w)
        return risk_aversion * port_vol ** 2 - port_return
    
    n = len(mu)
    w0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}
    
    result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else w0

print("\nCalculating HMM features...")
features = calculate_hmm_features(daily_returns, daily_prices)

print("\n=== Running Backtest ===")

d_idx = 63
while d_idx < len(daily_returns) and daily_returns.index[d_idx].year < 2017:
    d_idx += 1
while d_idx < len(daily_returns) and daily_returns.index[d_idx] < pd.Timestamp('2017-08-31'):
    d_idx += 1

first_strat_date = daily_returns.index[d_idx]
last_quarterly_rebal = first_strat_date
last_hmm_refit = first_strat_date

portfolio_returns = []
dates = []
allocation_history = []
regime_history = []
prob_bull_history = []

current_allocation = np.array([0.0, 0.0, 1.0])
consecutive_fast = 0
consecutive_slow = 0
consecutive_reentry = 0

quarterly_rebals = 0
fast_exits_count = 0
slow_exits_count = 0
reentries = 0

current_hmm = None

while d_idx < len(daily_returns) and daily_returns.index[d_idx] <= pd.Timestamp('2026-02-28'):
    current_date = daily_returns.index[d_idx]
    
    days_since_last_rebal = (current_date - last_quarterly_rebal).days
    quarterly_due = days_since_last_rebal >= 63
    
    if quarterly_due:
        features_expanding = features.loc[:current_date]
        try:
            hmm_quarterly = GaussianHMM(n_components=3, covariance_type='diag', n_iter=500, random_state=42)
            hmm_quarterly.fit(features_expanding.values)
            current_hmm = hmm_quarterly
            last_hmm_refit = current_date
        except:
            pass
    
    if current_hmm is not None and current_date in features.index:
        try:
            today_features = features.loc[[current_date]].values
            probs = current_hmm.predict_proba(today_features)[0]
            today_regime = pd.Series({
                'regime': probs.argmax(),
                'prob_bear': probs[0],
                'prob_normal': probs[1],
                'prob_bull': probs[2]
            })
        except:
            today_regime = pd.Series({'regime': 2, 'prob_bear': 0, 'prob_normal': 0, 'prob_bull': 1})
    else:
        today_regime = pd.Series({'regime': 2, 'prob_bear': 0, 'prob_normal': 0, 'prob_bull': 1})
    
    prob_bull = today_regime['prob_bull']
    prob_bear = today_regime['prob_bear']
    
    force_rebalance = False
    override_allocation = None
    
    # Layer 1: Price crash detector
    if current_allocation[2] < 0.85:
        lookback_start = current_date - pd.Timedelta(days=30)
        if lookback_start in daily_prices.index:
            recent_high = daily_prices['SPY'].loc[lookback_start:current_date].max()
            price_drop_30d = (daily_prices['SPY'].loc[current_date] / recent_high) - 1
            if price_drop_30d < -CRASH_DROP:
                override_allocation = np.array([0.0, 0.0, 1.0])
                force_rebalance = True
                fast_exits_count += 1
                print(f"CRASH EXIT → {current_date.date()} → SHV (price drop = {price_drop_30d:.1%})")
                consecutive_fast = 0
                consecutive_slow = 0
                consecutive_reentry = 0
    
    # Layer 2: HMM Fast Exit
    if not force_rebalance and prob_bear > FAST_EXIT_PROB:
        consecutive_fast += 1
        if consecutive_fast >= FAST_EXIT_DAYS and current_allocation[2] < 0.85:
            override_allocation = np.array([0.0, 0.0, 1.0])
            force_rebalance = True
            fast_exits_count += 1
            print(f"HMM FAST EXIT → {current_date.date()} → SHV (bear prob = {prob_bear:.2f})")
            consecutive_fast = 0
            consecutive_slow = 0
            consecutive_reentry = 0
    else:
        consecutive_fast = 0
    
    # Layer 3: HMM Slow Exit
    if not force_rebalance and prob_bear > SLOW_EXIT_PROB:
        consecutive_slow += 1
        if consecutive_slow >= SLOW_EXIT_DAYS and current_allocation[2] < 0.85:
            override_allocation = np.array([0.25, 0.25, 0.50])
            force_rebalance = True
            slow_exits_count += 1
            print(f"HMM SLOW EXIT → {current_date.date()} → Safer mix")
            consecutive_slow = 0
            consecutive_reentry = 0
    else:
        consecutive_slow = 0
    
    # Re-entry
    if current_allocation[2] > 0.7:
        if prob_bull > REENTRY_PROB:
            consecutive_reentry += 1
            if consecutive_reentry >= REENTRY_DAYS:
                force_rebalance = True
                reentries += 1
                print(f"RE-ENTRY → {current_date.date()} → Bull mode (bull prob={prob_bull:.2f})")
                consecutive_reentry = 0
        else:
            consecutive_reentry = 0
    
    if quarterly_due or force_rebalance:
        if override_allocation is not None:
            current_allocation = override_allocation
        else:
            preds_mean = {}
            preds_std = {}
            for ticker in TICKERS:
                pred_mean, pred_std = sklearn_gp_predict(daily_returns, ticker, current_date)
                preds_mean[ticker] = pred_mean
                preds_std[ticker] = pred_std
            
            mu_arr = np.array([preds_mean[t] for t in TICKERS])
            sigma_arr = np.array([preds_std[t] for t in TICKERS])
            
            leverage = 1.0
            if today_regime['prob_bull'] > 0.70 and preds_std['SSO'] < 0.018:
                leverage = 2.0
                print(f"LEVERAGE → {current_date.date()} 2.0x")
            elif preds_std['SSO'] > 0.025:
                leverage = 1.0
            
            current_allocation = mean_variance_optimize(mu_arr, sigma_arr, risk_aversion=0.5)
            current_allocation[0] *= leverage
            current_allocation[1] *= leverage
            current_allocation = current_allocation / current_allocation.sum()
            
            last_quarterly_rebal = current_date
            quarterly_rebals += 1
            if not force_rebalance:
                print(f"REBAL → {current_date.date()} SSO:{current_allocation[0]:.0%} SPY:{current_allocation[1]:.0%} SHV:{current_allocation[2]:.0%}")
    
    daily_ret = sum(current_allocation[i] * daily_returns[TICKERS[i]].iloc[d_idx] for i in range(3))
    portfolio_returns.append(daily_ret)
    dates.append(current_date)
    allocation_history.append(list(current_allocation))
    regime_history.append(today_regime['regime'])
    prob_bull_history.append(today_regime['prob_bull'])
    
    d_idx += 1

results = pd.DataFrame({
    'date': dates,
    'return': portfolio_returns,
    'allocation': allocation_history,
    'regime': regime_history,
    'prob_bull': prob_bull_history
})
results.set_index('date', inplace=True)

results['cum_ret'] = (1 + results['return']).cumprod() - 1

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

portfolio_value = pd.Series(1.0, index=results.index)
for i in range(1, len(results)):
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + results['return'].iloc[i])
peak = portfolio_value.expanding().max()
max_dd = ((portfolio_value / peak - 1) * 100).min()

spy_ret = (1 + daily_returns['SPY'].loc[results.index[0]:results.index[-1]]).cumprod()
spy_final = spy_ret.iloc[-1] - 1

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
print(f"Days: {len(results)}")
print(f"Quarterly Rebalances: {quarterly_rebals}")
print(f"Fast Exits: {fast_exits_count}")
print(f"Slow Exits: {slow_exits_count}")
print(f"Re-entries: {reentries}")
print(f"\nAnnual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Strategy: {results['cum_ret'].iloc[-1]*100:.1f}%")
print(f"SPY: {spy_final*100:.1f}%")
print(f"Excess: {(results['cum_ret'].iloc[-1] - spy_final)*100:.1f}%")

results.to_csv(f'{OUTPUT_DIR}/daily_results.csv')

results_df = pd.DataFrame({
    'Metric': ['Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown', 'Final Return', 'SPY Buy-Hold', 'Excess Return'],
    'Value': [f'{ann_ret:.1f}%', f'{ann_vol:.1f}%', f'{sharpe:.2f}', f'{max_dd:.1f}%', 
             f'{results["cum_ret"].iloc[-1]*100:.1f}%', f'{spy_final*100:.1f}%',
             f'{(results["cum_ret"].iloc[-1] - spy_final)*100:.1f}%']
})
results_df.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv', index=False)

print("\nSaved!")
print("=== DONE ===")