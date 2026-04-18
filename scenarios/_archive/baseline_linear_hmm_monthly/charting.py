# -*- coding: utf-8 -*-
"""
Separated Charting Module for Factor Rotation
Import this module and call plotting functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_equity_curve(results_df, spy_aligned, output_path, title='Equity Curve'):
    """
    Plot equity curve with SPY comparison.
    
    Args:
        results_df: DataFrame with 'cum_ret' column (decimal, not percentage)
        spy_aligned: Series of SPY cumulative returns aligned to strategy dates
        output_path: Full path to save figure
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(results_df.index, results_df['cum_ret'].values * 100, 
           label=f'Strategy ({results_df["cum_ret"].iloc[-1]:.1%})', 
           linewidth=2.5, color='blue')
    ax.plot(spy_aligned.index, spy_aligned.values * 100, 
           label=f'SPY ({spy_aligned.iloc[-1]:.1%})', 
           linewidth=2, color='gray', alpha=0.7)
    
    ax.set_ylabel('Cumulative Return (%)')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown(results_df, output_path, title='Drawdown'):
    """
    Plot drawdown chart.
    
    Args:
        results_df: DataFrame with 'cum_ret' column
        output_path: Full path to save figure
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    drawdown = (results_df['cum_ret'] / results_df['cum_ret'].cummax() - 1) * 100
    ax.fill_between(results_df.index, drawdown.values, 0, color='red', alpha=0.5)
    ax.set_ylabel('Drawdown (%)')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_regime(hmm_states, dates, output_path, title='Regime Timeline'):
    """
    Plot regime timeline.
    
    Args:
        hmm_states: List or array of HMM states (0, 1, 2)
        dates: List of dates corresponding to states
        output_path: Full path to save figure
        title: Chart title
    """
    fig, ax = plt.subplots(figsize=(14, 5))
    
    regime_colors = {0: 'lightblue', 1: 'lightgreen', 2: 'salmon'}
    hmm_series = pd.Series(hmm_states, index=pd.to_datetime(dates))
    
    ax.fill_between(hmm_series.index, 0, hmm_series.values * 30, 
                   color=[regime_colors.get(r, 'gray') for r in hmm_series], alpha=0.6)
    ax.set_ylabel('HMM Regime')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_weights(weights_history, tickers, dates, output_path, title='Portfolio Allocation'):
    """
    Plot portfolio weights over time.
    
    Args:
        weights_history: List of weight arrays (one per period)
        tickers: List of ticker symbols
        dates: List of dates corresponding to weights
        output_path: Full path to save figure
        title: Chart title
    """
    wdf = pd.DataFrame(weights_history, index=pd.to_datetime(dates), columns=tickers)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    wdf.plot(kind='area', stacked=True, alpha=0.7, ax=ax, colormap='tab10')
    ax.set_ylabel('Portfolio Weight')
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_all(results_df, spy_aligned, weights_history, tickers, dates, hmm_states, 
            output_dir, title_prefix='Factor Rotation'):
    """
    Generate all charts at once.
    
    Args:
        results_df: DataFrame with 'cum_ret' column
        spy_aligned: Series of SPY cumulative returns
        weights_history: List of weight arrays
        tickers: List of ticker symbols
        dates: List of dates
        hmm_states: List of HMM states
        output_dir: Directory to save charts
        title_prefix: Prefix for chart titles
    """
    import os
    
    plot_equity_curve(
        results_df, spy_aligned, 
        os.path.join(output_dir, 'factor_rotation_equity_curve.png'),
        f'{title_prefix} - Equity Curve'
    )
    
    plot_drawdown(
        results_df,
        os.path.join(output_dir, 'factor_rotation_drawdown.png'),
        f'{title_prefix} - Drawdown'
    )
    
    plot_regime(
        hmm_states, dates,
        os.path.join(output_dir, 'factor_rotation_regime.png'),
        f'{title_prefix} - Regime Timeline'
    )
    
    plot_weights(
        weights_history, tickers, dates,
        os.path.join(output_dir, 'factor_rotation_weights.png'),
        f'{title_prefix} - Portfolio Allocation'
    )


def calculate_metrics(results_df, frequency=12):
    """
    Calculate performance metrics.
    
    Args:
        results_df: DataFrame with 'return' and 'cum_ret' columns
        frequency: 12 for monthly, 252 for daily
    
    Returns:
        dict: Dictionary of metrics
    """
    ann_ret = results_df['return'].mean() * frequency * 100
    ann_vol = results_df['return'].std() * np.sqrt(frequency) * 100
    sharpe = results_df['return'].mean() / results_df['return'].std() * np.sqrt(frequency)
    max_dd = ((results_df['cum_ret'] / results_df['cum_ret'].cummax() - 1) * 100).min()
    
    return {
        'annual_return': ann_ret,
        'annual_vol': ann_vol,
        'sharpe': sharpe,
        'max_drawdown': max_dd,
        'final_return': results_df['cum_ret'].iloc[-1]
    }