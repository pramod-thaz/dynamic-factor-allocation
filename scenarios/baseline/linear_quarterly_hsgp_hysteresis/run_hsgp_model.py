import pandas as pd
import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import warnings
warnings.filterwarnings('ignore')

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_hsgp_hysteresis/output'

TICKERS = ['SSO', 'SPY', 'SHV']

HSGP_M = 20
HSGP_C = 5.0

FAST_EXIT_SCORE = -1.5
FAST_EXIT_DAYS = 3

SLOW_EXIT_SCORE = -0.5
SLOW_EXIT_DAYS = 15

REENTRY_SCORE = 0.5
REENTRY_DAYS = 10

print("=== HSGP + HMM + Hysteresis Model (3 ETFs) ===")

data = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
daily_prices = data[TICKERS]
daily_returns = daily_prices.pct_change().dropna()

print(f"Data: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")
print(f"Trading days: {len(daily_returns)}")

def calculate_hmm_features(returns_df, prices_df):
    features = pd.DataFrame(index=returns_df.index)
    
    features['trend'] = (returns_df['SPY'] > returns_df['SPY'].rolling(200).mean()).astype(float)
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

def detect_regimes_hmm(features_df):
    X = features_df.values
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=1000, random_state=42)
    hmm.fit(X)
    
    states = hmm.predict(X)
    probs = hmm.predict_proba(X)
    
    regime_df = pd.DataFrame({
        'regime': states,
        'prob_bear': probs[:, 0],
        'prob_normal': probs[:, 1],
        'prob_bull': probs[:, 2]
    }, index=features_df.index)
    
    return regime_df, hmm

def detect_regimes_hmm_rolling(features_df, lookback=252):
    X = features_df.values[-lookback:]
    hmm = GaussianHMM(n_components=3, covariance_type='diag', n_iter=500, random_state=42)
    hmm.fit(X)
    
    states = hmm.predict(X)
    probs = hmm.predict_proba(X)
    
    return probs[-1], hmm

def hsgp_predict_next(returns_df, ticker, current_date, m=20, c=5.0):
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
    except Exception as e:
        return y_train[-63:].mean()

print("\nCalculating HMM features...")
features = calculate_hmm_features(daily_returns, daily_prices)
print(f"Features calculated: {len(features)} days")

print("\nHMM will be fitted daily using rolling window to avoid look-ahead bias")

print("\n=== Running Backtest ===")

d_idx = 63
# Skip to match ra10 start date (2017-08-31)
while d_idx < len(daily_returns) and daily_returns.index[d_idx].year < 2017:
    d_idx += 1
while d_idx < len(daily_returns) and daily_returns.index[d_idx] < pd.Timestamp('2017-08-31'):
    d_idx += 1

portfolio_returns = []
dates = []
etf_history = []
regime_history = []
score_history = []

current_etf = 'SHV'
first_strat_date = daily_returns.index[d_idx]
last_rebal = first_strat_date
in_shadow = False

consecutive_fast = 0
consecutive_slow = 0
consecutive_reentry = 0

fast_exits = 0
slow_exits = 0
quarterly_rebals = 0
reentries = 0

HMM_LOOKBACK = 252  # 1 year rolling window

while d_idx < len(daily_returns) and daily_returns.index[d_idx] <= pd.Timestamp('2026-02-28'):
    current_date = daily_returns.index[d_idx]
    
    # Fit HMM daily with rolling lookback to avoid look-ahead bias
    features_for_hmm = features.loc[:current_date]
    if len(features_for_hmm) >= HMM_LOOKBACK:
        features_window = features_for_hmm[-HMM_LOOKBACK:]
    else:
        features_window = features_for_hmm
    
    try:
        hmm_daily = GaussianHMM(n_components=3, covariance_type='diag', n_iter=200, random_state=42)
        hmm_daily.fit(features_window.values)
        probs = hmm_daily.predict_proba(features_window.values[-1:])
        today_regime = pd.Series({
            'regime': probs[0].argmax(),
            'prob_bear': probs[0, 0],
            'prob_normal': probs[0, 1],
            'prob_bull': probs[0, 2]
        })
    except:
        today_regime = pd.Series({'regime': 2, 'prob_bear': 0, 'prob_normal': 0, 'prob_bull': 1})
    
    score = today_regime['prob_bull'] * 2 + today_regime['prob_normal'] - today_regime['prob_bear'] * 2
    
    if in_shadow:
        if score > REENTRY_SCORE:
            consecutive_reentry += 1
            if consecutive_reentry >= REENTRY_DAYS:
                in_shadow = False
                current_etf = 'SPY'
                reentries += 1
                print(f"RE-ENTRY  → {current_date.date()} → SPY (score={score:.2f})")
                consecutive_reentry = 0
        else:
            consecutive_reentry = 0
    else:
        if score <= FAST_EXIT_SCORE:
            consecutive_fast += 1
            if consecutive_fast >= FAST_EXIT_DAYS:
                current_etf = 'SHV'
                in_shadow = True
                fast_exits += 1
                print(f"FAST EXIT  → {current_date.date()} → SHV (score={score:.2f})")
                consecutive_fast = 0
                consecutive_slow = 0
        else:
            consecutive_fast = 0
        
        if score <= SLOW_EXIT_SCORE:
            consecutive_slow += 1
            if consecutive_slow >= SLOW_EXIT_DAYS:
                current_etf = 'SHV'
                in_shadow = True
                slow_exits += 1
                print(f"SLOW EXIT → {current_date.date()} → SHV (score={score:.2f})")
                consecutive_slow = 0
        else:
            consecutive_slow = 0
    
    quarterly_due = (current_date - last_rebal).days >= 63
    
    if quarterly_due:
        preds = {}
        for ticker in TICKERS:
            preds[ticker] = hsgp_predict_next(daily_returns, ticker, current_date, HSGP_M, HSGP_C)
        
        if today_regime['prob_bull'] > 0.70:
            target_etf = 'SSO'
        elif today_regime['prob_bull'] > 0.45:
            target_etf = 'SSO'
        elif today_regime['regime'] == 0:
            target_etf = 'SHV'
        else:
            target_etf = 'SPY'
        
        current_etf = target_etf
        in_shadow = (target_etf == 'SHV')
        last_rebal = current_date
        quarterly_rebals += 1
        
        preds_str = ', '.join([f'{t}:{v:.4f}' for t, v in preds.items()])
        print(f"REBAL    → {current_date.date()} → {target_etf} | preds: {preds_str}")
    
    daily_ret = daily_returns[current_etf].iloc[d_idx]
    portfolio_returns.append(daily_ret)
    dates.append(current_date)
    etf_history.append(current_etf)
    regime_history.append(today_regime['regime'])
    score_history.append(score)
    
    d_idx += 1

results = pd.DataFrame({
    'date': dates,
    'return': portfolio_returns,
    'etf': etf_history,
    'regime': regime_history,
    'score': score_history
})
results.set_index('date', inplace=True)

results['cum_ret'] = (1 + results['return']).cumprod() - 1

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

# Proper max drawdown using dollar portfolio value
portfolio_value = pd.Series(1.0, index=results.index)
for i in range(1, len(results)):
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + results['return'].iloc[i])

peak = portfolio_value.expanding().max()
max_dd = ((portfolio_value / peak - 1) * 100).min()

# Calculate SPY buy-hold aligned to strategy dates
spy_ret = (1 + daily_returns['SPY'].loc[results.index[0]:results.index[-1]]).cumprod()
spy_final = spy_ret.iloc[-1] - 1

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
print(f"Days: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%")
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Final Return: {results['cum_ret'].iloc[-1]:.1%}")
print(f"\nSPY Buy-Hold: {spy_final:.1%}")
print(f"\nFast Exits: {fast_exits}")
print(f"Slow Exits: {slow_exits}")
print(f"Re-entries: {reentries}")
print(f"Quarterly Rebalances: {quarterly_rebals}")

results.to_csv(f'{OUTPUT_DIR}/daily_results.csv')
results.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv')

print(f"\nSaved to {OUTPUT_DIR}/")
print("=== DONE ===")