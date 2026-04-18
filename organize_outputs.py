# Create properly organized file structure
import os
import pandas as pd
import numpy as np

BASE_DIR = '/home/realdomarp/PYMC/FACTOR ROTATION'

configs = [
    ('baseline_v1', 'Linear + GaussianHMM + Monthly'), 
    ('output_gp_model', 'GP + GaussianHMM + Monthly'),
    ('test_02_simple_regime', 'Linear + SimpleRegime + Monthly'),
]

print("=== ORGANIZING OUTPUT FILES ===\n")

for folder, desc in configs:
    src_path = os.path.join(BASE_DIR, folder)
    target_path = os.path.join(BASE_DIR, f'output_{desc.replace(" ", "_").replace("+", "")}')
    
    # Create target folder
    os.makedirs(target_path, exist_ok=True)
    
    # Copy files if exist
    for fname in ['factor_rotation_backtest_results.csv', 'factor_rotation_equity_curve.png', 
                'factor_rotation_weights.png', 'weights_history.csv']:
        src = os.path.join(src_path, fname) if folder == 'output_gp_model' else os.path.join(BASE_DIR, fname.replace('.csv', f'_{folder}.csv'))
        src_alt = os.path.join(src_path, fname)
        
        # Try different source paths
        if os.path.exists(src_alt):
            dest = os.path.join(target_path, fname)
            try:
                import shutil
                shutil.copy2(src_alt, dest)
                print(f"Copied: {fname} -> {target_path}")
            except:
                pass
        elif os.path.exists(src):
            try:
                import shutil
                shutil.copy2(src, dest := os.path.join(target_path, fname))
                print(f"Copied: {fname}")
            except:
                pass
    
    print(f"✓ {desc}: {target_path}\n")

print("=== FILE ORGANIZATION COMPLETE ===")
print("\nFolder Structure:")
print("  output_Linear_GaussianHMM_Monthly/")
print("  output_GP_GaussianHMM_Monthly/")
print("  output_Linear_SimpleRegime_Monthly/")