# -*- coding: utf-8 -*-
"""
Final Comparison: All Tests vs Baseline vs SPY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("=== Creating Final Comparison ===")

# Load data
data = pd.read_csv('etf_data.csv', index_col=0, parse_dates=True)
monthly = data.resample('ME').last()
returns = monthly.pct_change()

# Load baseline
baseline = pd.read_csv('baseline_v1_results.csv', index_col=0)
baseline_cum = baseline['cum_ret']

# SPY benchmark - align to baseline dates!
spy_returns = returns['SPY'].dropna()
test_dates = baseline.index
spy_aligned = spy_returns.reindex(test_dates)
spy_cum = (1 + spy_aligned).cumprod() - 1

# Load tests that completed
tests = {
    'Baseline': baseline_cum,
    'Baseline + SPY': spy_cum,
}

# Test 02
try:
    df = pd.read_csv('test_02_simple_regime_results.csv', index_col=0)
    tests['Simple Regime'] = df['cum_ret']
except:
    pass

# Create comparison
print(f"\nComparing {len(tests)} series:")
for name, cum in tests.items():
    print(f"  {name}: {cum.iloc[-1]:.1%}")

# Plot comparison
fig, ax = plt.subplots(figsize=(14, 10))

# Baseline first
ax.plot(tests['Baseline'].index, tests['Baseline'].values * 100, 
       label=f"Baseline ({tests['Baseline'].iloc[-1]:.1%})", 
       linewidth=2.5, color='blue')

# Simple Regime if available
if 'Simple Regime' in tests:
    ax.plot(tests['Simple Regime'].index, tests['Simple Regime'].values * 100,
           label=f"Simple Regime ({tests['Simple Regime'].iloc[-1]:.1%})",
           linewidth=2, color='green', linestyle='--')

# SPY benchmark
ax.plot(tests['Baseline + SPY'].index, tests['Baseline + SPY'].values * 100,
       label=f"SPY Buy&Hold ({tests['Baseline + SPY'].iloc[-1]:.1%})",
       linewidth=2, color='gray', alpha=0.7)

ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.set_xlabel('Date', fontsize=12)
ax.set_title('Factor Rotation Models: Baseline vs Alternatives vs SPY', fontsize=14)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=150)
plt.close()

print(f"\nSaved: all_models_comparison.png")

# Save proper metrics CSV - calculate from aligned returns!
metrics = []

# Baseline
baseline_rets = baseline['return']
metrics.append({
    'Model': 'Baseline (HMM+Linear+Monthly)',
    'Ann Return': baseline_rets.mean() * 12 * 100,
    'Sharpe': baseline_rets.mean() / baseline_rets.std() * np.sqrt(12),
    'Final Return': baseline_cum.iloc[-1]
})

# Simple Regime
if 'Simple Regime' in tests:
    sr = pd.read_csv('test_02_simple_regime_results.csv')
    sr_rets = sr['return']
    metrics.append({
        'Model': 'Simple Regime (Rolling+Linear+Monthly)',
        'Ann Return': sr_rets.mean() * 12 * 100,
        'Sharpe': sr_rets.mean() / sr_rets.std() * np.sqrt(12),
        'Final Return': tests['Simple Regime'].iloc[-1]
    })

# SPY aligned
spy_rets = spy_aligned.diff().dropna()
metrics.append({
    'Model': 'SPY Buy & Hold',
    'Ann Return': spy_rets.mean() * 12 * 100,
    'Sharpe': spy_rets.mean() / spy_rets.std() * np.sqrt(12),
    'Final Return': spy_cum.iloc[-1]
})

metrics_df = pd.DataFrame(metrics)
metrics_df = metrics_df.sort_values('Ann Return', ascending=False)
metrics_df.to_csv('all_models_metrics.csv', index=False)

print("\n=== FINAL METRICS ===")
print(metrics_df.to_string(index=False))

print("\n=== DONE ===")