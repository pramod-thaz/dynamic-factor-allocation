import pandas as pd
import numpy as np
import pymc as pm
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_hsgp_optimized_hysteresis/output'

TICKERS = ['SSO', 'SPY', 'SHV']

HSGP_M = 20
HSGP_C = 5.0

FAST_EXIT_PROB = 0.65
FAST_EXIT_DAYS = 4

SLOW_EXIT_PROB = 0.48
SLOW_EXIT_DAYS = 12

REENTRY_PROB = 0.60
REENTRY_DAYS = 5

CRASH_DROP = 0.12  # -12% from 30-day high

print("=== ADVI GP + Probability Exits + HMM ===", flush=True)

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

def advi_predict_next(returns_df, ticker, current_date, n_iter=2000):
    train_mask = returns_df.index <= current_date
    y = returns_df[ticker][train_mask].values
    
    if len(y) < 100:
        return y[-63:].mean(), y[-63:].std()
    
    recent_vol = y[-63:].std()
    recent_mean = y[-63:].mean()
    
    # Use last 252 points (1 year)
    y_advi = y[-252:] if len(y) > 252 else y
    X = np.arange(len(y_advi)).reshape(-1, 1)
    
    try:
        with pm.Model() as m:
            a = pm.Normal('a', mu=0, sigma=0.01)
            b = pm.Normal('b', mu=0, sigma=0.001)
            sigma = pm.HalfNormal('sigma', sigma=0.02)
            mu = a + b * X.flatten()
            y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y_advi)
            approx = pm.fit(n=n_iter, method='advi', progressbar=False, random_seed=42)
        
        a_mean = approx.eval(approx.mean)['a']
        b_mean = approx.eval(approx.mean)['b']
        sigma_mean = approx.eval(approx.mean)['sigma']
        
        X_pred = np.array([[len(y_advi)]])
        pred_mean = float(a_mean + b_mean * X_pred[0, 0])
        pred_std = float(sigma_mean)
        
        return pred_mean, max(pred_std, recent_vol * 0.3)
    except:
        return recent_mean, recent_vol

def hsgp_predict(returns_df, ticker, current_date, m=20, c=5.0):
    train_mask = returns_df.index <= current_date
    n = sum(train_mask)
    if n < 100:
        return returns_df[ticker].iloc[:63].mean()
    
    X_train = np.linspace(0, 1, n).reshape(-1, 1)
    y_train = returns_df[ticker][train_mask].values
    
    try:
        with pm.Model() as model:
            gp = pm.gp.HSGP(m=[m], L=[1.0], c=c, cov_func=pm.gp.cov.Matern52(1, ls=0.5))
            sigma = pm.HalfCauchy('sigma', beta=0.05)
            f = gp.marginal_likelihood('f', X=X_train, y=y_train, noise=sigma)
            trace = pm.sample(200, tune=100, chains=2, target_accept=0.95,
                          progressbar=False, random_seed=42)
        
        X_new = np.array([[1.0]])
        with model:
            f_pred = gp.conditional('f_pred', Xnew=X_new)
            post_pred = pm.sample_posterior_predictive(trace, var_names=['f_pred'], progressbar=False)
        
        samples = post_pred.posterior_predictive['f_pred'].values.flatten()
        return float(samples.mean())
    except:
        return y_train[-63:].mean()

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
    days_since_refit = (current_date - last_hmm_refit).days
    
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
    
    # === LAYER 1: PRICE CRASH DETECTOR (immediate) ===
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
    
    # === LAYER 2: HMM FAST EXIT ===
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
    
    # === LAYER 3: HMM SLOW EXIT ===
    if not force_rebalance and prob_bear > SLOW_EXIT_PROB:
        consecutive_slow += 1
        if consecutive_slow >= SLOW_EXIT_DAYS and current_allocation[2] < 0.85:
            override_allocation = np.array([0.25, 0.25, 0.50])
            force_rebalance = True
            slow_exits_count += 1
            print(f"HMM SLOW EXIT → {current_date.date()} → Safer mix (bear prob = {prob_bear:.2f})")
            consecutive_slow = 0
            consecutive_reentry = 0
    else:
        consecutive_slow = 0
    
    # === RE-ENTRY ===
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
                pred_mean, pred_std = advi_predict_next(daily_returns, ticker, current_date)
                preds_mean[ticker] = pred_mean
                preds_std[ticker] = pred_std
            
            mu_arr = np.array([preds_mean[t] for t in TICKERS])
            sigma_arr = np.array([preds_std[t] for t in TICKERS])
            
            leverage = 1.0
            if today_regime['prob_bull'] > 0.70 and preds_std['SSO'] < 0.018:
                leverage = 2.0
                print(f"LEVERAGE → {current_date.date()} 2.0x (low uncertainty)")
            elif preds_std['SSO'] > 0.025:
                leverage = 1.0
                print(f"LEVERAGE → {current_date.date()} 1.0x (high volatility)")
            
            current_allocation = mean_variance_optimize(mu_arr, sigma_arr, risk_aversion=0.5)
            current_allocation[0] *= leverage
            current_allocation[1] *= leverage
            current_allocation = current_allocation / current_allocation.sum()
            
            last_quarterly_rebal = current_date
            quarterly_rebals += 1
            if not force_rebalance:
                print(f"REBAL → {current_date.date()} SSO:{current_allocation[0]:.0%} SPY:{current_allocation[1]:.0%} SHV:{current_allocation[2]:.0%} (μ_SSO={preds_mean['SSO']:.4f}, σ_SSO={preds_std['SSO']:.4f})")
    
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