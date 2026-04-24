import pandas as pd
import numpy as np
from scipy.optimize import minimize
from hmmlearn.hmm import GaussianHMM
import yfinance as yf
import warnings
from datetime import timedelta, datetime
import argparse
warnings.filterwarnings('ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Quarterly EMA + Smoothing V2')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date for backtest (YYYY-MM-DD). Default: latest available data')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run mode - validate inputs without running backtest')
    parser.add_argument('--as-of', type=str, default=None,
                       help='Alias for --end-date')
    return parser.parse_args()

args = parse_args()
END_DATE_ARG = args.end_date or args.as_of
DRY_RUN = args.dry_run

DATA_FILE = '/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv'
OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/quarterly_ema_smoothing_v2/output'

TICKERS = ['SSO', 'SPY', 'SHV']

FAST_EXIT_PROB = 0.64
FAST_EXIT_DAYS = 4

SLOW_EXIT_PROB = 0.47
SLOW_EXIT_DAYS = 12

REENTRY_PROB = 0.62
REENTRY_DAYS = 8

TURNOVER_PENALTY = 0.015
MIN_REGIME_DAYS = 18

def fetch_and_merge_data():
    print("\nFetching latest data from yfinance...")
    
    try:
        cached = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
        all_tickers = list(cached.columns)
        print(f"Cache tickers: {all_tickers}")
        print(f"Cache date range: {cached.index[0].date()} to {cached.index[-1].date()}")
        
        last_cached_date = cached.index[-1]
        
        # Always try to fetch new data if cache is older than 1 day
        if last_cached_date.date() >= (datetime.now() - timedelta(days=1)).date():
            print("Cache is recent, but fetching latest data anyway...")
        
        start_date = last_cached_date + timedelta(days=1)
        end_date = min(datetime.now(), start_date + timedelta(days=10))
        
        print(f"Fetching {start_date.date()} to {end_date.date()}...")
        
        new_data = yf.download(all_tickers, start=start_date.strftime('%Y-%m-%d'), 
                           end=end_date.strftime('%Y-%m-%d'), progress=False)
        
        if new_data.empty:
            print("No new data fetched.")
            return cached
        
        if 'Adj Close' in new_data.columns.get_level_values(0):
            new_data = new_data['Adj Close']
        else:
            new_data = new_data['Close']
        
        new_data = new_data.dropna(how='all')
        
        if len(new_data) > 0:
            combined = pd.concat([cached, new_data])
            combined = combined[all_tickers]
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            combined.to_csv(DATA_FILE)
            print(f"Updated cache: {combined.index[0].date()} to {combined.index[-1].date()}")
            return combined
        else:
            print("No new data available.")
            return cached
            
    except FileNotFoundError:
        print("Cache not found, fetching full history from yfinance...")
        all_tickers = ['GLD', 'IJR', 'MTUM', 'QUAL', 'SPY', 'TLT', 'USMV', 'VTV', 'VUG', 'SSO', 'HYG', 'LQD', 'SHV', 'QLD']
        start_date = '2017-08-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False)
        if 'Adj Close' in data.columns.get_level_values(0):
            data = data['Adj Close']
        else:
            data = data['Close']
        
        data = data.dropna(how='all')
        data.to_csv(DATA_FILE)
        print(f"Created cache: {data.index[0].date()} to {data.index[-1].date()}")
        return data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("Falling back to cached data.")
        if cached is not None:
            return cached
        raise

print("=== Quarterly EMA + Smoothing V2 ===", flush=True)

if END_DATE_ARG:
    try:
        END_DATE_PARSED = pd.Timestamp(END_DATE_ARG)
        print(f"Running with end_date: {END_DATE_PARSED.date()}")
    except:
        print(f"ERROR: Invalid date format '{END_DATE_ARG}'. Use YYYY-MM-DD")
        exit(1)

if DRY_RUN:
    print("Dry run mode - exiting after validation")
    exit(0)

data = fetch_and_merge_data()
daily_prices = data[TICKERS]
daily_returns = daily_prices.pct_change().dropna()

print(f"Data: {daily_returns.index[0].date()} to {daily_returns.index[-1].date()}")

def calculate_hmm_features_improved(returns_df, prices_df):
    features = pd.DataFrame(index=returns_df.index)
    
    ma50 = returns_df['SPY'].rolling(50).mean()
    ma150 = returns_df['SPY'].rolling(150).mean()
    
    features['trend_fast'] = (returns_df['SPY'] - ma50) / returns_df['SPY'].rolling(50).std()
    features['trend_slow'] = (returns_df['SPY'] - ma150) / returns_df['SPY'].rolling(150).std()
    features['momentum'] = returns_df['SPY'].rolling(63).sum() / returns_df['SPY'].rolling(63).std()
    features['breadth'] = returns_df['SPY'].rolling(50).mean() / returns_df['SPY'].rolling(50).mean().shift(20) - 1
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
    
    features['adx'] = adx(prices_df['SPY'].cumsum()).ewm(span=10).mean()
    features['market_vol'] = returns_df['SPY'].rolling(21).std() * np.sqrt(252)
    
    return features.dropna()

def predict_next(returns_df, ticker, current_date, span=40):
    train_mask = returns_df.index <= current_date
    y = returns_df[ticker][train_mask]
    
    if len(y) < 100:
        return y[-40:].mean(), y[-40:].std()
    
    ema_mean = y.ewm(span=span, adjust=False).mean().iloc[-1]
    recent_std = y[-40:].std()
    
    return float(ema_mean), float(recent_std)

def mean_variance_optimize(mu, sigma, previous_allocation=None, risk_aversion=0.5, turnover_penalty=0.015):
    mu_arr = np.array([mu[t] for t in TICKERS])
    sigma_arr = np.diag([sigma[t] ** 2 for t in TICKERS])
    
    def obj(w):
        port_ret = w @ mu_arr
        port_vol = np.sqrt(w @ sigma_arr @ w)
        turnover_cost = 0
        if previous_allocation is not None and len(previous_allocation) == len(w):
            turnover_cost = turnover_penalty * np.sum(np.abs(w - previous_allocation))
        return risk_aversion * port_vol**2 - port_ret + turnover_cost
    
    n = len(mu_arr)
    w0 = np.ones(n) / n
    bounds = [(0, 1) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    
    result = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x if result.success else w0

print("\nCalculating HMM features...")
features = calculate_hmm_features_improved(daily_returns, daily_prices)

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

forced_regime = 2
days_in_forced_regime = 0

quarterly_rebals = 0
fast_exits_count = 0
slow_exits_count = 0
reentries = 0

current_hmm = None

if END_DATE_ARG:
    end_date = END_DATE_PARSED
    if end_date not in daily_returns.index:
        available_dates = daily_returns.index
        end_date = available_dates[available_dates <= end_date][-1]
        print(f"Adjusted end_date to nearest trading day: {end_date.date()}")
else:
    end_date = daily_returns.index[-1]

print(f"Backtest period: {daily_returns.index[63].date()} to {end_date.date()}")

while d_idx < len(daily_returns) and daily_returns.index[d_idx] <= end_date:
    current_date = daily_returns.index[d_idx]
    
    days_since_last_rebal = (current_date - last_quarterly_rebal).days
    
    quarterly_due = days_since_last_rebal >= 63
    
    if quarterly_due:
        features_expanding = features.loc[:current_date]
        try:
            np.random.seed(42)
            features_noisy = features_expanding.values + np.random.normal(0, 0.05, features_expanding.shape)
            hmm_quarterly = GaussianHMM(n_components=3, covariance_type='diag', 
                                    n_iter=150, random_state=42, covars_prior=0.01)
            hmm_quarterly.fit(features_noisy)
            current_hmm = hmm_quarterly
            last_hmm_refit = current_date
        except Exception as e:
            pass
    
    raw_regime = 2
    raw_prob_bear = 0.33
    raw_prob_normal = 0.34
    raw_prob_bull = 0.33
    
    if current_hmm is not None and current_date in features.index:
        try:
            idx_pos = features.index.get_indexer([current_date])[0]
            if idx_pos >= 0:
                today_features = features.iloc[[idx_pos]].values
                probs = current_hmm.predict_proba(today_features)[0]
                raw_regime = probs.argmax()
                raw_prob_bear = probs[0]
                raw_prob_normal = probs[1]
                raw_prob_bull = probs[2]
        except Exception as e:
            pass
    
    today_regime = pd.Series({
        'regime': raw_regime,
        'prob_bear': raw_prob_bear,
        'prob_normal': raw_prob_normal,
        'prob_bull': raw_prob_bull
    })
    
    if days_in_forced_regime == 0:
        forced_regime = raw_regime
    
    if today_regime['regime'] != forced_regime:
        if days_in_forced_regime >= MIN_REGIME_DAYS:
            forced_regime = today_regime['regime']
            days_in_forced_regime = 0
        else:
            today_regime['regime'] = forced_regime
    
    days_in_forced_regime += 1
    
    prob_bull = today_regime['prob_bull']
    prob_bear = today_regime['prob_bear']
    
    force_rebalance = False
    override_allocation = None
    
    if current_allocation[2] < 0.85:
        vol_5d = daily_returns['SPY'].loc[current_date - pd.Timedelta(days=5):current_date].std() * np.sqrt(252)
        if vol_5d > 0.40:
            override_allocation = np.array([0.0, 0.0, 1.0])
            force_rebalance = True
            fast_exits_count += 1
            print(f"VOL SPIKE EXIT → {current_date.date()} → SHV (5d vol = {vol_5d:.1%})")
    
    if current_allocation[2] < 0.85 and not force_rebalance:
        lookback_start = current_date - pd.Timedelta(days=30)
        if lookback_start in daily_prices.index:
            recent_high = daily_prices['SPY'].loc[lookback_start:current_date].max()
            price_drop_30d = (daily_prices['SPY'].loc[current_date] / recent_high) - 1
            price_drop_10d = daily_returns['SPY'].loc[current_date - pd.Timedelta(days=10):current_date].sum()
            
            if price_drop_30d < -0.105 or price_drop_10d < -0.085:
                override_allocation = np.array([0.0, 0.0, 1.0])
                force_rebalance = True
                fast_exits_count += 1
                print(f"CRASH EXIT → {current_date.date()} → SHV (30d drop = {price_drop_30d:.1%})")
                consecutive_fast = 0
                consecutive_slow = 0
                consecutive_reentry = 0
    
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
    
    if current_allocation[2] > 0.70:
        if prob_bull > REENTRY_PROB:
            consecutive_reentry += 1
            if consecutive_reentry >= REENTRY_DAYS:
                force_rebalance = True
                reentries += 1
                print(f"RE-ENTRY → {current_date.date()} → Bull mode")
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
                pred_mean, pred_std = predict_next(daily_returns, ticker, current_date)
                preds_mean[ticker] = pred_mean
                preds_std[ticker] = pred_std
            
            current_allocation = mean_variance_optimize(
                preds_mean, preds_std, 
                previous_allocation=current_allocation if quarterly_due else None,
                risk_aversion=0.5,
                turnover_penalty=TURNOVER_PENALTY
            )
            
            if today_regime['prob_bull'] > 0.70 and preds_std['SSO'] < 0.018:
                leverage = 2.0
                current_allocation[0] *= leverage
                current_allocation[1] *= leverage
                current_allocation = current_allocation / current_allocation.sum()
                print(f"LEVERAGE → {current_date.date()} 2.0x (low uncertainty)")
            elif preds_std['SSO'] > 0.025:
                current_allocation = np.array([0.0, 0.0, 1.0])
                print(f"LEVERAGE → {current_date.date()} 1.0x (high volatility)")
            
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

final_date = results.index[-1]
final_alloc = results['allocation'].iloc[-1]
final_action = "NO CHANGE"
final_buy_sell = "NONE"

prev_alloc = results['allocation'].iloc[-2] if len(results) > 1 else np.array([0.0, 0.0, 1.0])
sso_diff = final_alloc[0] - prev_alloc[0]
spy_diff = final_alloc[1] - prev_alloc[1]
shv_diff = final_alloc[2] - prev_alloc[2]

if abs(sso_diff) > 0.01 or abs(spy_diff) > 0.01 or abs(shv_diff) > 0.01:
    action_signals = []
    if sso_diff > 0.01:
        action_signals.append(f"BUY SSO {sso_diff:.0%}")
    elif sso_diff < -0.01:
        action_signals.append(f"SELL SSO {abs(sso_diff):.0%}")
    if spy_diff > 0.01:
        action_signals.append(f"BUY SPY {spy_diff:.0%}")
    elif spy_diff < -0.01:
        action_signals.append(f"SELL SPY {abs(spy_diff):.0%}")
    if shv_diff > 0.01:
        action_signals.append(f"BUY SHV {shv_diff:.0%}")
    elif shv_diff < -0.01:
        action_signals.append(f"SELL SHV {abs(shv_diff):.0%}")
    final_action = f"REBAL to SSO:{final_alloc[0]:.0%} SPY:{final_alloc[1]:.0%} SHV:{final_alloc[2]:.0%}"
    final_buy_sell = ", ".join(action_signals)

trade_signal = pd.DataFrame({
    'date': [final_date],
    'action': [final_action],
    'buy_sell': [final_buy_sell],
    'sso': [final_alloc[0]],
    'spy': [final_alloc[1]],
    'shv': [final_alloc[2]],
    'regime': [results['regime'].iloc[-1]],
    'prob_bull': [results['prob_bull'].iloc[-1]]
})
trade_signal.to_csv(f'{OUTPUT_DIR}/next_trade.csv')

print(f"\nNEXT TRADE (as of {final_date.date()}):")
print(f"  Action: {final_action}")
print(f"  Trade: {final_buy_sell}")
print(f"  Allocation: SSO {final_alloc[0]:.0%} | SPY {final_alloc[1]:.0%} | SHV {final_alloc[2]:.0%}")

print("\nSaved!")
print("=== DONE ===")