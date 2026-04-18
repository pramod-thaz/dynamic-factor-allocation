import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ast

OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_hsgp_optimized_hysteresis/output'

results = pd.read_csv(f'{OUTPUT_DIR}/daily_results.csv', index_col=0, parse_dates=True)

def parse_allocation(x):
    s = str(x)
    import re
    nums = re.findall(r'np\.float64\(([^)]+)\)', s)
    result = [0.0, 0.0, 0.0]
    for i, v in enumerate(nums[:3]):
        try:
            result[i] = float(v)
        except:
            pass
    return result

results['allocation_parsed'] = results['allocation'].apply(parse_allocation)

allocation_df = pd.DataFrame(results['allocation_parsed'].tolist(), index=results.index, columns=['SSO', 'SPY', 'SHV'])

data = pd.read_csv('/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv', index_col=0, parse_dates=True)
spy = data['SPY'].loc[results.index[0]:results.index[-1]]
spy_returns = spy.pct_change().dropna()
spy_cum = (1 + spy_returns).cumprod() - 1

portfolio_value = pd.Series(1.0, index=results.index)
for i in range(1, len(results)):
    portfolio_value.iloc[i] = portfolio_value.iloc[i-1] * (1 + results['return'].iloc[i])
cum_ret = portfolio_value - 1

peak = portfolio_value.expanding().max()
drawdown = (portfolio_value / peak - 1) * 100

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
max_dd = drawdown.min()

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

regime_colors = {0: 'red', 1: 'gray', 2: 'green'}
current_regime = int(results['regime'].iloc[0])
start_idx = 0

for i in range(1, len(results)):
    new_regime = int(results['regime'].iloc[i])
    if new_regime != current_regime or i == len(results) - 1:
        end_idx = i
        axes[0].axvspan(results.index[start_idx], results.index[end_idx],
                        alpha=0.2, color=regime_colors[current_regime], zorder=0)
        start_idx = i
        current_regime = new_regime

axes[0].plot(results.index, cum_ret.values * 100, 
            label=f'Strategy ({cum_ret.iloc[-1]*100:.1f}%)', linewidth=2, color='blue', zorder=1)
axes[0].plot(spy_returns.index, spy_cum.values * 100, 
            label=f'SPY Buy-Hold ({spy_cum.iloc[-1]*100:.1f}%)', linewidth=2, color='gray', alpha=0.7, linestyle='--', zorder=1)
axes[0].set_ylabel('Cumulative Return (%)')
axes[0].set_title('Equity Curve - ADVI GP + Probability Exits + HMM Strategy')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].fill_between(results.index, 0, cum_ret.values * 100, alpha=0.1, color='blue', zorder=1)

axes[1].fill_between(results.index, drawdown.values, 0, color='red', alpha=0.5)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title(f'Drawdown (Max: {max_dd:.1f}%)')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

axes[2].fill_between(results.index, 0, allocation_df['SSO'].values * 100, alpha=0.7, color='green', label='SSO')
axes[2].fill_between(results.index, allocation_df['SSO'].values * 100, 
                    (allocation_df['SSO'] + allocation_df['SPY']).values * 100, alpha=0.7, color='blue', label='SPY')
axes[2].fill_between(results.index, (allocation_df['SSO'] + allocation_df['SPY']).values * 100, 100, 
                    alpha=0.7, color='red', label='SHV')
axes[2].set_ylabel('Allocation (%)')
axes[2].set_title('Portfolio Allocation')
axes[2].legend(loc='upper right')
axes[2].set_ylim(0, 100)
axes[2].grid(True, alpha=0.3)

axes[3].plot(results.index, results['prob_bull'].values, color='purple', linewidth=1, alpha=0.7, label='prob_bull')
axes[3].plot(results.index, results['regime'].values, color='blue', linewidth=1, alpha=0.5, linestyle='--', label='regime')
axes[3].axhline(y=0.65, color='red', linestyle='--', linewidth=1, label='Fast exit (0.65)')
axes[3].axhline(y=0.45, color='orange', linestyle='--', linewidth=1, label='Slow exit (0.45)')
axes[3].set_ylabel('Probability')
axes[3].set_title('HMM Regime Probabilities')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(-0.1, 1.1)

axes[3].xaxis.set_major_locator(mdates.YearLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/factor_rotation_charts.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
print(f"Days: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%") 
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Strategy Final: {cum_ret.iloc[-1]*100:.1f}%")
print(f"SPY Final: {spy_cum.iloc[-1]*100:.1f}%")
print(f"Excess Return: {(cum_ret.iloc[-1] - spy_cum.iloc[-1])*100:.1f}%")
print(f"Charts saved to {OUTPUT_DIR}/factor_rotation_charts.png")