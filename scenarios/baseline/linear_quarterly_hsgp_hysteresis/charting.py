import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

OUTPUT_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION/scenarios/baseline/linear_quarterly_hsgp_hysteresis/output'

results = pd.read_csv(f'{OUTPUT_DIR}/daily_results.csv', index_col=0, parse_dates=True)

data = pd.read_csv('/home/realdomarp/PYMC/FACTOR ROTATION/data/etf_data.csv', index_col=0, parse_dates=True)
spy = data['SPY'].loc[results.index[0]:results.index[-1]]
spy_returns = spy.pct_change().dropna()
spy_cum = (1 + spy_returns).cumprod() - 1
spy_dates = spy_returns.index

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
max_dd = ((results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100).min()

results_df = pd.DataFrame({
    'Metric': ['Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown', 'Final Return', 'SPY Buy-Hold'],
    'Value': [f'{ann_ret:.1f}%', f'{ann_vol:.1f}%', f'{sharpe:.2f}', f'{max_dd:.1f}%', 
             f'{results["cum_ret"].iloc[-1]*100:.1f}%', f'{spy_cum.iloc[-1]*100:.1f}%']
})
results_df.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv', index=False)

etf_allocation = results['etf'].value_counts(normalize=True) * 100
etf_df = pd.DataFrame({'ETF': etf_allocation.index, 'Days': etf_allocation.values})
etf_df.to_csv(f'{OUTPUT_DIR}/etf_allocation.csv', index=False)

print(f"Period: {results.index[0].date()} to {results.index[-1].date()}")
print(f"Days: {len(results)}")
print(f"Annual Return: {ann_ret:.1f}%")
print(f"Annual Vol: {ann_vol:.1f}%") 
print(f"Sharpe: {sharpe:.2f}")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Strategy Final: {results['cum_ret'].iloc[-1]*100:.1f}%")
print(f"SPY Final: {spy_cum.iloc[-1]*100:.1f}%")
print(f"Excess Return: {(results['cum_ret'].iloc[-1] - spy_cum.iloc[-1])*100:.1f}%")

fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)

axes[0].plot(results.index, results['cum_ret'].values * 100, 
            label=f'Strategy ({results["cum_ret"].iloc[-1]*100:.1f}%)', linewidth=2, color='blue')
axes[0].plot(spy_dates, spy_cum.values * 100, 
            label=f'SPY ({spy_cum.iloc[-1]*100:.1f}%)', linewidth=2, color='gray', alpha=0.7)
axes[0].set_ylabel('Cumulative Return (%)')
axes[0].set_title('Equity Curve - HSGP + HMM + Hysteresis Strategy')
axes[0].legend(loc='upper left')
axes[0].grid(True, alpha=0.3)
axes[0].fill_between(results.index, 0, results['cum_ret'].values * 100, alpha=0.1, color='blue')

drawdown = (results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100
axes[1].fill_between(results.index, drawdown.values, 0, color='red', alpha=0.5)
axes[1].set_ylabel('Drawdown (%)')
axes[1].set_title('Drawdown')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

position_numeric = results['etf'].map({'SSO': 2, 'SPY': 1, 'SHV': 0})
colors = results['etf'].map({'SSO': 'green', 'SPY': 'blue', 'SHV': 'red'})
axes[2].bar(results.index, position_numeric.values, color=colors.values, alpha=0.6, width=1)
axes[2].set_ylabel('Position')
axes[2].set_title('ETF Position (2=SSO, 1=SPY, 0=SHV)')
axes[2].set_yticks([0, 1, 2])
axes[2].set_yticklabels(['SHV', 'SPY', 'SSO'])
axes[2].grid(True, alpha=0.3)

axes[3].plot(results.index, results['score'].values, color='purple', linewidth=1, alpha=0.7)
axes[3].axhline(y=0.5, color='green', linestyle='--', linewidth=1, label='Re-entry (0.5)')
axes[3].axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, label='Slow exit (-0.5)')
axes[3].axhline(y=-1.5, color='red', linestyle='--', linewidth=1, label='Fast exit (-1.5)')
axes[3].set_ylabel('Score')
axes[3].set_title('HMM Score')
axes[3].legend(loc='upper right')
axes[3].grid(True, alpha=0.3)
axes[3].set_ylim(-2.5, 2.5)

axes[3].xaxis.set_major_locator(mdates.YearLocator())
axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/factor_rotation_charts.png', dpi=150, bbox_inches='tight')
plt.close()

ann_ret = results['return'].mean() * 252 * 100
ann_vol = results['return'].std() * np.sqrt(252) * 100
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
max_dd = ((results['cum_ret'] / results['cum_ret'].cummax() - 1) * 100).min()

results_df = pd.DataFrame({
    'Metric': ['Annual Return', 'Annual Vol', 'Sharpe', 'Max Drawdown', 'Final Return', 'SPY Buy-Hold'],
    'Value': [f'{ann_ret:.1f}%', f'{ann_vol:.1f}%', f'{sharpe:.2f}', f'{max_dd:.1f}%', 
             f'{results["cum_ret"].iloc[-1]*100:.1f}%', f'{spy_cum.iloc[-1]*100:.1f}%']
})
results_df.to_csv(f'{OUTPUT_DIR}/factor_rotation_backtest_results.csv', index=False)

etf_allocation = results['etf'].value_counts(normalize=True) * 100
etf_df = pd.DataFrame({'ETF': etf_allocation.index, 'Days': etf_allocation.values})
etf_df.to_csv(f'{OUTPUT_DIR}/etf_allocation.csv', index=False)

print(f"Charts saved to {OUTPUT_DIR}/factor_rotation_charts.png")