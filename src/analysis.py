"""
Follow-up analysis: Feature-selected classifiers and deeper statistical analysis.

The initial experiment showed significant correlations but classifiers overfit
due to high-dimensional features with only 20 samples.

This script:
1. Uses top-k feature selection
2. Tests simple threshold classifiers
3. Performs bootstrap analysis of correlations
4. Runs permutation tests for classifier significance
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from pathlib import Path

SEED = 42
np.random.seed(SEED)
FIGURES_DIR = Path('results/figures')

# Load data from experiment
df = pd.read_csv('results/feature_data.csv')
with open('results/final_results.json') as f:
    prev_results = json.load(f)

print(f"Loaded {len(df)} prompt formats")
print(f"Accuracy range: {df['accuracy'].min():.3f} - {df['accuracy'].max():.3f}")
print(f"Accuracy spread: {df['accuracy'].max() - df['accuracy'].min():.3f}")

# ═══════════════════════════════════════════════════════════════
# 1. Feature Selection: Use only top correlated features
# ═══════════════════════════════════════════════════════════════

# Get core geometric features (not std_across_examples variants)
core_features = [c for c in df.columns if c not in [
    'format', 'accuracy', 'n_examples', 'mean_logit_diff', 'std_logit_diff',
    'mean_confidence', 'std_confidence', 'is_failure'
] and '_std_across_examples' not in c]

print(f"\nCore geometric features: {len(core_features)}")

# Compute correlations for core features
correlations = {}
for feat in core_features:
    if df[feat].std() < 1e-10:
        continue
    rho, p = stats.spearmanr(df[feat], df['accuracy'])
    correlations[feat] = {'rho': rho, 'p_value': p}

# Sort by absolute rho
sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]['rho']), reverse=True)

print("\nCore feature correlations with accuracy:")
for feat, info in sorted_corr:
    sig = "***" if info['p_value'] < 0.001 else "**" if info['p_value'] < 0.01 else "*" if info['p_value'] < 0.05 else ""
    print(f"  {feat:35s}: ρ={info['rho']:+.3f}  p={info['p_value']:.4f} {sig}")

# ═══════════════════════════════════════════════════════════════
# 2. Bootstrap Confidence Intervals for Top Correlations
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Bootstrap confidence intervals for top correlations")
print("=" * 60)

n_bootstrap = 10000
top_features = [f for f, _ in sorted_corr[:5]]

bootstrap_results = {}
for feat in top_features:
    boot_rhos = []
    x = df[feat].values
    y = df['accuracy'].values
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x), len(x), replace=True)
        rho, _ = stats.spearmanr(x[idx], y[idx])
        if not np.isnan(rho):
            boot_rhos.append(rho)

    ci_low = np.percentile(boot_rhos, 2.5)
    ci_high = np.percentile(boot_rhos, 97.5)
    bootstrap_results[feat] = {
        'rho': correlations[feat]['rho'],
        'ci_low': ci_low,
        'ci_high': ci_high,
        'p_value': correlations[feat]['p_value'],
    }
    print(f"  {feat:35s}: ρ={correlations[feat]['rho']:+.3f}  95% CI [{ci_low:+.3f}, {ci_high:+.3f}]")

# ═══════════════════════════════════════════════════════════════
# 3. Simple Threshold Classifiers (top-1 feature)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Simple threshold classifiers (single feature)")
print("=" * 60)

median_acc = df['accuracy'].median()
y = (df['accuracy'] < median_acc).astype(int).values

threshold_results = {}
for feat in top_features:
    x = df[feat].values
    rho = correlations[feat]['rho']

    # Use LOO with optimal threshold on training set
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))

    for train_idx, test_idx in loo.split(x):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test = x[test_idx]

        # Find best threshold on training set
        best_acc = 0
        best_thresh = 0
        best_dir = 1

        for thresh in x_train:
            for direction in [1, -1]:
                pred = (direction * x_train > direction * thresh).astype(int)
                acc = np.mean(pred == y_train)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
                    best_dir = direction

        y_pred[test_idx] = (best_dir * x_test > best_dir * best_thresh).astype(int)

    acc = np.mean(y_pred == y)
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)

    threshold_results[feat] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(f"  {feat:35s}: acc={acc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")

# ═══════════════════════════════════════════════════════════════
# 4. Feature-selected classifier (top-3 features)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Feature-selected classifiers (top-k features)")
print("=" * 60)

for k in [2, 3, 5]:
    selected_feats = [f for f, _ in sorted_corr[:k]]
    X = df[selected_feats].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # LOO with Logistic Regression
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y))
    y_proba = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_scaled):
        clf = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
        clf.fit(X_scaled[train_idx], y[train_idx])
        y_pred[test_idx] = clf.predict(X_scaled[test_idx])
        y_proba[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]

    acc = np.mean(y_pred == y)
    try:
        auc = roc_auc_score(y, y_proba)
    except:
        auc = 0.5
    prec, rec, f1, _ = precision_recall_fscore_support(y, y_pred, average='binary', zero_division=0)

    print(f"\n  Top-{k} LR: acc={acc:.3f}  AUC={auc:.3f}  prec={prec:.3f}  rec={rec:.3f}  f1={f1:.3f}")
    print(f"    Features: {selected_feats}")

    # LOO with Random Forest (depth=2 to prevent overfitting)
    y_pred_rf = np.zeros(len(y))
    y_proba_rf = np.zeros(len(y))

    for train_idx, test_idx in loo.split(X_scaled):
        clf = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=SEED)
        clf.fit(X_scaled[train_idx], y[train_idx])
        y_pred_rf[test_idx] = clf.predict(X_scaled[test_idx])
        y_proba_rf[test_idx] = clf.predict_proba(X_scaled[test_idx])[:, 1]

    acc_rf = np.mean(y_pred_rf == y)
    try:
        auc_rf = roc_auc_score(y, y_proba_rf)
    except:
        auc_rf = 0.5
    prec_rf, rec_rf, f1_rf, _ = precision_recall_fscore_support(y, y_pred_rf, average='binary', zero_division=0)
    print(f"  Top-{k} RF: acc={acc_rf:.3f}  AUC={auc_rf:.3f}  prec={prec_rf:.3f}  rec={rec_rf:.3f}  f1={f1_rf:.3f}")


# ═══════════════════════════════════════════════════════════════
# 5. Permutation Test for Best Classifier
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Permutation test for correlation significance")
print("=" * 60)

n_perms = 5000
top_feat = sorted_corr[0][0]
observed_rho = abs(correlations[top_feat]['rho'])
x_perm = df[top_feat].values
y_perm = df['accuracy'].values

perm_rhos = []
for _ in range(n_perms):
    perm_y = np.random.permutation(y_perm)
    rho, _ = stats.spearmanr(x_perm, perm_y)
    perm_rhos.append(abs(rho))

perm_p = np.mean(np.array(perm_rhos) >= observed_rho)
print(f"Top feature: {top_feat}")
print(f"Observed |ρ| = {observed_rho:.3f}")
print(f"Permutation p-value: {perm_p:.4f} (n_perms={n_perms})")

# ═══════════════════════════════════════════════════════════════
# 6. Continuous prediction (regression)
# ═══════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("Continuous accuracy prediction (regression)")
print("=" * 60)

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

for k in [2, 3, 5]:
    selected_feats = [f for f, _ in sorted_corr[:k]]
    X = df[selected_feats].values
    y_reg = df['accuracy'].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    loo = LeaveOneOut()
    y_pred_reg = np.zeros(len(y_reg))

    for train_idx, test_idx in loo.split(X_scaled):
        reg = Ridge(alpha=1.0)
        reg.fit(X_scaled[train_idx], y_reg[train_idx])
        y_pred_reg[test_idx] = reg.predict(X_scaled[test_idx])

    mse = mean_squared_error(y_reg, y_pred_reg)
    r2 = r2_score(y_reg, y_pred_reg)
    mae = np.mean(np.abs(y_reg - y_pred_reg))

    print(f"  Top-{k} Ridge: R²={r2:.3f}  MSE={mse:.4f}  MAE={mae:.3f}")

# ═══════════════════════════════════════════════════════════════
# 7. Visualizations
# ═══════════════════════════════════════════════════════════════

# A. Correlation heatmap of top features
fig, ax = plt.subplots(figsize=(10, 8))
top_feats_for_plot = [f for f, _ in sorted_corr[:8]]
plot_df = df[top_feats_for_plot + ['accuracy']].copy()
# Shorten column names for readability
rename_map = {c: c.replace('mean_', 'm_').replace('std_', 's_').replace('_layer_', '_l_')
              for c in plot_df.columns}
plot_df = plot_df.rename(columns=rename_map)
corr_matrix = plot_df.corr(method='spearman')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, ax=ax, vmin=-1, vmax=1)
ax.set_title('Spearman Correlation Matrix: Top Geometric Features + Accuracy')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("\nSaved: results/figures/correlation_heatmap.png")

# B. Bootstrap CI plot
fig, ax = plt.subplots(figsize=(10, 5))
feats_for_ci = list(bootstrap_results.keys())
rhos = [bootstrap_results[f]['rho'] for f in feats_for_ci]
ci_lows = [bootstrap_results[f]['ci_low'] for f in feats_for_ci]
ci_highs = [bootstrap_results[f]['ci_high'] for f in feats_for_ci]
errors = [[r - l for r, l in zip(rhos, ci_lows)],
          [h - r for r, h in zip(rhos, ci_highs)]]

short_names = [f.replace('mean_', 'm_').replace('variance_', 'var_').replace('entropy_', 'ent_')
               for f in feats_for_ci]
ax.barh(short_names, rhos, xerr=errors, color='steelblue', ecolor='black', capsize=5)
ax.axvline(0, color='black', linewidth=0.5)
ax.set_xlabel('Spearman ρ')
ax.set_title('Top Feature Correlations with Accuracy (95% Bootstrap CI)')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'bootstrap_ci.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/bootstrap_ci.png")

# C. Detailed scatter of top 2 features with format labels
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for i, (feat, info) in enumerate(sorted_corr[:2]):
    ax = axes[i]
    ax.scatter(df[feat], df['accuracy'], s=80, alpha=0.7, edgecolors='black', linewidth=0.5, c='steelblue')
    for j, row in df.iterrows():
        ax.annotate(row['format'], (row[feat], row['accuracy']),
                   fontsize=6, alpha=0.7, ha='center', va='bottom')
    ax.set_xlabel(feat)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'ρ = {info["rho"]:.3f} (p = {info["p_value"]:.4f})')
    ax.grid(True, alpha=0.3)
    # Add trend line
    z = np.polyfit(df[feat], df['accuracy'], 1)
    p_line = np.poly1d(z)
    x_range = np.linspace(df[feat].min(), df[feat].max(), 100)
    ax.plot(x_range, p_line(x_range), 'r--', alpha=0.5)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'top2_detailed_scatter.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/top2_detailed_scatter.png")

# D. Summary comparison bar chart
fig, ax = plt.subplots(figsize=(12, 6))
methods = ['Random', 'Confidence Baseline', 'Top-1 Threshold', 'Top-3 LR', 'Top-3 RF']
# Collect the AUC or accuracy values
method_accs = [0.5, 0.0,
               threshold_results[sorted_corr[0][0]]['accuracy'],
               0, 0]  # Will fill from LOO results

# Recompute for the summary
# Top-3 LR
k = 3
selected_feats = [f for f, _ in sorted_corr[:k]]
X = df[selected_feats].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
loo = LeaveOneOut()
y_pred_lr3 = np.zeros(len(y))
for train_idx, test_idx in loo.split(X_scaled):
    clf = LogisticRegression(C=0.1, max_iter=1000, random_state=SEED)
    clf.fit(X_scaled[train_idx], y[train_idx])
    y_pred_lr3[test_idx] = clf.predict(X_scaled[test_idx])
method_accs[3] = np.mean(y_pred_lr3 == y)

# Top-3 RF
y_pred_rf3 = np.zeros(len(y))
for train_idx, test_idx in loo.split(X_scaled):
    clf = RandomForestClassifier(n_estimators=50, max_depth=2, random_state=SEED)
    clf.fit(X_scaled[train_idx], y[train_idx])
    y_pred_rf3[test_idx] = clf.predict(X_scaled[test_idx])
method_accs[4] = np.mean(y_pred_rf3 == y)

colors = ['gray', 'orange', 'green', 'steelblue', 'darkblue']
bars = ax.bar(methods, method_accs, color=colors, edgecolor='black', linewidth=0.5)
ax.axhline(0.5, color='red', linestyle='--', linewidth=1, label='Random baseline')
ax.set_ylabel('LOO Classification Accuracy')
ax.set_title('Prompt Failure Prediction: Method Comparison')
ax.set_ylim(0, 1)
ax.legend()
for bar, val in zip(bars, method_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{val:.3f}', ha='center', fontsize=10)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'method_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: results/figures/method_comparison.png")

# ═══════════════════════════════════════════════════════════════
# 8. Save complete results
# ═══════════════════════════════════════════════════════════════

analysis_results = {
    'core_correlations': {k: v for k, v in sorted_corr},
    'bootstrap_ci': bootstrap_results,
    'threshold_classifiers': threshold_results,
    'permutation_test': {
        'feature': top_feat,
        'observed_rho': float(observed_rho),
        'permutation_p': float(perm_p),
        'n_permutations': n_perms,
    },
}

with open('results/analysis_results.json', 'w') as f:
    json.dump(analysis_results, f, indent=2, default=str)

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
