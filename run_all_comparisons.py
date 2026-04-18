# -*- coding: utf-8 -*-
"""
RUN ALL COMPARISONS
Executes baseline + all 8 tests + SPY, outputs comparison chart
"""

import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

TICKERS = ['SPY', 'VTV', 'MTUM', 'QUAL', 'USMV']

# Files to run
FILES = [
    'baseline_simple_linear_monthly.py',
    'test_01_simple_linear_monthly.py',
    'test_02_simple_gp_monthly.py',
    'test_03_hmm_linear_monthly.py',
    'test_04_hmm_gp_monthly.py',
    'test_05_simple_linear_daily.py',
    'test_06_simple_gp_daily.py',
    'test_07_hmm_linear_daily.py',
    'test_08_hmm_gp_daily.py',
]

print("="*60)
print("RUNNING ALL COMPARISONS")
print("="*60)

# Run each file
results = {}
for f in FILES:
    name = f.replace('.py', '')
    print(f"\n>>> Running {name}...")
    
    try:
        result = subprocess.run(['python', f], capture_output=True, timeout=300)
        
        # Read results
        if os.path.exists(f'{name}_results.csv'):
            df = pd.read_csv(f'{name}_results.csv', index_col=0)
            results[name] = {
                'return': df['return'],
                'cum_ret': df['cum_ret'],
                'final': df['cum_ret'].iloc[-1]
            }
            ann_ret = df['return'].mean() * 12 * 100 if 'monthly' in name else df['return'].mean() * 252 * 100
            ann_vol = df['return'].std() * np.sqrt(12) * 100 if 'monthly' in name else df['return'].std() * np.sqrt(252) * 100
            sharpe = df['return'].mean() / df['return'].std() * np.sqrt(12) if 'monthly' in name else df['return'].mean() / df['return'].std() * np.sqrt(252)
            
            results[name]['ann_return'] = ann_ret
            results[name]['ann_vol'] = ann_vol
            results[name]['sharpe'] = sharpe
            
            print(f"  Final: {results[name]['final']:.1%}, Sharpe: {sharpe:.2f}")
        else:
            print(f"  WARNING: No results file found")
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT - skipping")
    except Exception as e:
        print(f"  ERROR: {e}")

# Compute SPY benchmark
if os.path.exists('etf_data.csv'):
    data = pd.read_csv('etf_data.csv', index_col=0, parse_dates=True)
    returns = data.pct_change().dropna()
    
    # Monthly benchmark
    monthly = returns.resample('ME').last()
    monthly_ret = monthly.pct_change()['SPY'].dropna()
    spy_monthly = (1 + monthly_ret).cumprod() - 1
    
    results['SPY_benchmark'] = {
        'cum_ret': spy_monthly,
        'final': spy_monthly.iloc[-1],
        'ann_return': monthly_ret.mean() * 12 * 100,
        'ann_vol': monthly_ret.std() * np.sqrt(12) * 100,
        'sharpe': monthly_ret.mean() / monthly_ret.std() * np.sqrt(12)
    }

print("\n" + "="*60)
print("ALL RESULTS")
print("="*60)

# Print comparison table
print(f"\n{'Model':<35} {'Ann Ret':>10} {'Ann Vol':>10} {'Sharpe':>8} {'Final':>10}")
print("-" * 75)

for name, r in results.items():
    print(f"{name:<35} {r['ann_return']:>9.1f}% {r['ann_vol']:>9.1f}% {r['sharpe']:>8.2f} {r['final']:>9.1%}")

# Save metrics
metrics = []
for name, r in results.items():
    metrics.append({
        'Model': name,
        'Ann_Return': r['ann_return'],
        'Ann_Vol': r['ann_vol'],
        'Sharpe': r['sharpe'],
        'Final_Return': r['final']
    })

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv('all_models_metrics.csv', index=False)
print(f"\nMetrics saved to all_models_metrics.csv")

# ========================== COMPARISON CHART ==========================
print("\nGenerating comparison chart...")

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

# Equity curves
ax = axes[0]
for name, r in results.items():
    label = f"{name} ({r['final']:.1%})"
    ax.plot(r['cum_ret'].index, r['cum_ret'].values * 100, label=label, linewidth=1.5)

ax.set_ylabel('Cumulative Return (%)')
ax.set_title('Factor Rotation Strategies vs SPY Buy & Hold')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

# Metrics bar chart
ax2 = axes[1]
model_names = list(results.keys())
sharpes = [results[m]['sharpe'] for m in model_names]
colors = ['blue' if 'baseline' in m else 'green' if 'simple' in m and 'hmm' not in m else 'orange' if 'hmm' in m else 'red' for m in model_names]

ax2.barh(model_names, sharpes, color=colors)
ax2.set_xlabel('Sharpe Ratio')
ax2.set_title('Sharpe Ratio Comparison')
ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)

for i, v in enumerate(sharpes):
    ax2.text(v + 0.02, i, f'{v:.2f}', va='center')

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=150)
plt.close()

print("Chart saved to all_models_comparison.png")

print("\n=== DONE ===")
print("All files generated and comparison chart created!")