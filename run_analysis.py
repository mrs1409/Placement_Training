"""Run student performance analysis and modeling.

This script expects `student-por.csv` in the same folder.
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from joblib import dump

from analysis_utils import load_data, clean_and_encode, train_test_split_df

ROOT = os.path.dirname(__file__)
DATA_PATH = os.path.join(ROOT, 'student-por.csv')
OUTDIR = os.path.join(ROOT, 'outputs')
os.makedirs(OUTDIR, exist_ok=True)

print('Loading data from', DATA_PATH)
df = load_data(DATA_PATH)
print('Data shape:', df.shape)

# Quick EDA plots
plt.figure(figsize=(12, 6))
if 'studytime' in df.columns and 'G3' in df.columns:
    sns.boxplot(x='studytime', y='G3', data=df)
    plt.title('G3 by studytime (1: <2 hours, 2: 2-5h, 3:5-10h, 4:>10h)')
else:
    plt.text(0.5, 0.5, 'studytime or G3 not in dataset', ha='center')

plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'studytime_vs_G3.png'))
plt.close()

# Correlations for numeric columns
numeric = df.select_dtypes(include=['number'])
cor = numeric.corr()
if 'G3' in cor.columns:
    studytime_corr = cor.loc['G3', 'studytime'] if 'studytime' in cor.columns else np.nan
    absences_corr = cor.loc['G3', 'absences'] if 'absences' in cor.columns else np.nan
else:
    studytime_corr = np.nan
    absences_corr = np.nan

# Prepare features: baseline features G1, G2, studytime, failures, absences
keep_vars = ['G1', 'G2', 'studytime', 'failures', 'absences', 'Medu', 'Fedu', 'famsup', 'schoolsup']
existing = [c for c in keep_vars if c in df.columns]
print('Key variables found in dataset:', existing)

# Build dataset for two models: base (without Medu/Fedu/famsup/schoolsup) and extended (with family support variables)
base_vars = [v for v in ['G1', 'G2', 'studytime', 'failures', 'absences'] if v in df.columns]
extended_vars = base_vars + [v for v in ['Medu', 'Fedu', 'famsup', 'schoolsup'] if v in df.columns]

results = {}

for name, vars_list in [('base', base_vars), ('extended', extended_vars)]:
    print(f'Preparing {name} model with vars:', vars_list)
    subset = df.copy()
    # Keep only selected vars + target
    use_cols = [c for c in vars_list if c in subset.columns] + ['G3']
    subset = subset[use_cols].copy()

    X, y = clean_and_encode(subset, target='G3')
    # Scale numeric features for coefficient comparison
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split_df(pd.DataFrame(X_scaled, columns=X.columns), y)

    # Linear regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    r2_lr = r2_score(y_test, y_pred_lr)

    # Random forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    r2_rf = r2_score(y_test, y_pred_rf)

    results[name] = {
        'linear_r2': r2_lr,
        'linear_mse': mean_squared_error(y_test, y_pred_lr),
        'rf_r2': r2_rf,
        'rf_mse': mean_squared_error(y_test, y_pred_rf),
        'lr_coefficients': dict(zip(X.columns, lr.coef_)),
        'rf_importances': dict(zip(X.columns, rf.feature_importances_))
    }

    # Save models
    dump(lr, os.path.join(OUTDIR, f'linear_{name}.joblib'))
    dump(rf, os.path.join(OUTDIR, f'rf_{name}.joblib'))

# Compare studytime correlation, parental education diff, absences
# 1) studytime correlation
studytime_corr_val = studytime_corr

# 2) parental education comparison: define high vs low as Medu+Fedu >= 6 (i.e., both >=3) if both exist
parent_diff = None
if 'Medu' in df.columns and 'Fedu' in df.columns:
    df['parent_high'] = ((df['Medu'].fillna(0) + df['Fedu'].fillna(0)) >= 6)
    means = df.groupby('parent_high')['G3'].mean()
    parent_diff = means.get(True, np.nan) - means.get(False, np.nan)

# 3) absences effect: compute correlation and average drop per 10 absences via linear fit
abs_effect = None
if 'absences' in df.columns and 'G3' in df.columns:
    abs_corr = df['absences'].corr(df['G3'])
    # simple linear slope
    slope = np.polyfit(df['absences'].fillna(0), df['G3'].fillna(0), 1)[0]
    abs_effect = {'corr': abs_corr, 'slope_per_absence': slope, 'drop_per_10': slope * 10}

# 4) failures vs studytime impact: compare standardized LR coefficients from extended model (if available)
impact = {}
if 'extended' in results:
    coeffs = results['extended']['lr_coefficients']
    # look for keys that contain studytime or failures
    study_coef = None
    fail_coef = None
    for k, v in coeffs.items():
        if 'studytime' in k:
            study_coef = v
        if 'failures' in k:
            fail_coef = v
    impact = {'studytime_coef': study_coef, 'failures_coef': fail_coef}

# Save summary
summary_path = os.path.join(OUTDIR, 'model_comparison.txt')
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('Model comparison results\n')
    for k, v in results.items():
        f.write(f'\n-- {k} model --\n')
        f.write(f"Linear R2: {v['linear_r2']:.4f}, RF R2: {v['rf_r2']:.4f}\n")

    f.write('\nSpecific analyses:\n')
    f.write(f'studytime-G3 Pearson corr: {studytime_corr_val:.4f}\n')
    if parent_diff is not None:
        f.write(f'Avg G3 difference (high vs low parental education): {parent_diff:.3f}\n')
    if abs_effect is not None:
        f.write(f"Absences corr: {abs_effect['corr']:.4f}, slope per absence: {abs_effect['slope_per_absence']:.4f}, drop per 10 absences: {abs_effect['drop_per_10']:.4f}\n")
    if impact:
        f.write(f"Std linear coefs -> studytime: {impact.get('studytime_coef')}, failures: {impact.get('failures_coef')}\n")

print('\n--- SUMMARY ---')
print('Results written to', summary_path)
print('Studytime-G3 corr:', studytime_corr_val)
if parent_diff is not None:
    print('Avg G3 diff (high vs low parental education):', parent_diff)
if abs_effect is not None:
    print('Absences corr:', abs_effect['corr'], 'drop per 10 absences:', abs_effect['drop_per_10'])
if impact:
    print('Studytime coef (std):', impact.get('studytime_coef'), 'Failures coef (std):', impact.get('failures_coef'))

print('\nCompleted.')
