# -*- coding: utf-8 -*-
"""
Separated Charting Module for Factor Rotation
Produces 4-subplot consolidated chart (equity, drawdown, regime, weights)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_all_charts(results_df, spy_aligned, weights_history, tickers, dates, hmm_states,
                    output_path, title_prefix='Factor Rotation'):
    """
    Generate consolidated 4-subplot chart.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    dates_idx = list(range(len(dates)))
    x_labels = [str(d)[:7] for d in dates]  # YYYY-MM format
    
    # 1. Equity curve with SPY comparison
    ax1 = axes[0]
    ax1.plot(dates_idx, results_df['cum_ret'].values * 100, 
             label=f'Strategy ({results_df["cum_ret"].iloc[-1]:.1%})', 
             linewidth=2.5, color='blue')
    ax1.plot(dates_idx, spy_aligned.values * 100, 
             label=f'SPY ({spy_aligned.iloc[-1]:.1%})', 
             linewidth=2, color='gray', alpha=0.7)
    ax1.set_xticks(dates_idx[::4])
    ax1.set_xticklabels(x_labels[::4], rotation=45, ha='right')
    ax1.set_ylabel('Cumulative Return (%)')
    ax1.set_title(f'{title_prefix} - Equity Curve')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown
    ax2 = axes[1]
    # Use actual portfolio value (1 + cum_ret), not percentage
    portfolio_value = 1 + results_df['cum_ret']
    drawdown = (portfolio_value / portfolio_value.cummax() - 1) * 100
    ax2.fill_between(dates_idx, drawdown.values, 0, color='red', alpha=0.5)
    ax2.set_xticks(dates_idx[::4])
    ax2.set_xticklabels(x_labels[::4], rotation=45, ha='right')
    ax2.set_ylabel('Drawdown (%)')
    ax2.set_title(f'{title_prefix} - Drawdown')
    ax2.grid(True, alpha=0.3)
    
    # 3. Regime timeline
    ax3 = axes[2]
    regime_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
    regime_values = [r * 30 for r in hmm_states]
    colors = [regime_colors.get(r, 'gray') for r in hmm_states]
    ax3.scatter(dates_idx, regime_values, c=colors, s=150, alpha=0.8, marker='s')
    ax3.set_xticks(dates_idx[::4])
    ax3.set_xticklabels(x_labels[::4], rotation=45, ha='right')
    ax3.set_ylabel('HMM Regime')
    ax3.set_title(f'{title_prefix} - Regime Timeline')
    ax3.set_yticks([0, 30, 60])
    ax3.set_yticklabels(['0 (Bull)', '1 (Transition)', '2 (Bear)'])
    ax3.grid(True, alpha=0.3)
    
    # 4. ETF weights (stacked area)
    ax4 = axes[3]
    wdf = pd.DataFrame(weights_history, index=dates_idx, columns=tickers)
    wdf.plot(kind='area', stacked=True, alpha=0.7, ax=ax4, colormap='tab10')
    ax4.set_xticks(dates_idx[::4])
    ax4.set_xticklabels(x_labels[::4], rotation=45, ha='right')
    ax4.set_ylabel('Portfolio Weight')
    ax4.set_title(f'{title_prefix} - ETF Allocation')
    ax4.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlabel('Date')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_metrics(results_df, frequency=12):
    """
    Calculate performance metrics.
    """
    ann_ret = results_df['return'].mean() * frequency * 100
    ann_vol = results_df['return'].std() * np.sqrt(frequency) * 100
    sharpe = results_df['return'].mean() / results_df['return'].std() * np.sqrt(frequency)
    # Use correct drawdown: based on portfolio value, not percentage
    portfolio_value = 1 + results_df['cum_ret']
    max_dd = ((portfolio_value / portfolio_value.cummax() - 1) * 100).min()
    
    return {
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'final_return': results_df['cum_ret'].iloc[-1]
    }