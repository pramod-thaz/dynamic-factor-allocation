# Keep the existing baseline as the original Linear model
# Copy baseline_results to the right places, verify and run
import os

BASE = '/home/realdomarp/PYMC/FACTOR ROTATION'

# Current baseline in baseline/ folder - copy to scenarios/linear for easy access
# Already have baseline_v1.py which does exactly this

# Copy the original runs, then run full iterations on GP vs Linear to get comparable outputs
# Already did baseline_v1.py (Linear)

# Need to run a version with the exact same settings in test scenario 

print("The baseline (Linear) is in baseline/, GP scenario in scenarios/gp_model/")

# The issues are:
# 1. Need the same time period for valid comparison
# 2. Need to check if the date ranges match 
# 3. Chart format needs to match exactly what original output looked like

# To correctly debug - compare actual dates:
import pandas as pd

baseline = pd.read_csv(f'{BASE}/baseline_v1_results.csv')
print(f"Baseline dates: {baseline['date'].iloc[0]} to {baseline['date'].iloc[-1]}")

# Check GP output
if os.path.exists(f'{BASE}/output_gp_model/factor_rotation_backtest_results.csv'):
    gp = pd.read_csv(f'{BASE}/output_gp_model/factor_rotation_backtest_results.csv')
    print(f"GP dates: {gp['date'].iloc[0]} to {gp['date'].iloc[-1]}")
else:
    print("GP results: NOT FOUND")