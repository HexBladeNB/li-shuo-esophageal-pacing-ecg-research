# -*- coding: utf-8 -*-
"""
 XGBoost Baseline — Addressing Reviewer 1 & 2
 Grid Search + 5-fold CV + Feature Importance Analysis
 Usage: python train_xgboost.py          (full grid search, ~12 min)
        python train_xgboost.py --fast   (use known best params, ~10 sec)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from terminal_ui import *
from train_rf import (
    load_features, compute_metrics_per_class,
    plot_roc_curves, plot_confusion_matrix
)
from paths import CACHE_DIR, DAY4_DIR

OUTPUT_DIR = DAY4_DIR

FAST_MODE = '--fast' in sys.argv

# Pre-computed best params (from initial run, seed=42)
KNOWN_BEST_PARAMS = {
    '3class': {'learning_rate': 0.1, 'max_depth': 3, 'n_estimators': 300, 'subsample': 0.8},
    '4class': {'learning_rate': 0.3, 'max_depth': 3, 'n_estimators': 100, 'subsample': 0.8},
}
KNOWN_CV_SCORES = {'3class': 0.9995, '4class': 0.9989}


def plot_feature_importance(model, feat_cols, task, top_n=20):
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1][:top_n]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    top_features = [feat_cols[i] for i in indices]
    top_importances = importance[indices]
    
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
    bars = ax.barh(range(top_n), top_importances[::-1], color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features[::-1], fontsize=9)
    ax.set_xlabel('Feature Importance (Gain)', fontsize=13)
    ax.set_title(f'XGBoost Top {top_n} Feature Importance ({task})', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.2, axis='x')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    save_path = OUTPUT_DIR / f"xgb_feature_importance_{task}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    importance_df = pd.DataFrame({
        'Feature': [feat_cols[i] for i in np.argsort(importance)[::-1]],
        'Importance': importance[np.argsort(importance)[::-1]]
    })
    importance_df.to_csv(OUTPUT_DIR / f"xgb_feature_importance_{task}.csv", index=False)
    return save_path, top_features[:5]


def train_and_evaluate_xgb(task='3class'):
    section(f"XGBoost — {task.upper()}", "◆")
    t0 = time.time()
    
    # Load
    step(1, 5, "Loading feature matrix")
    X_train, y_train, X_test, y_test, feat_cols, class_names = load_features(task)
    n_classes = len(class_names)
    kv("Train samples", X_train.shape[0])
    kv("Test samples", X_test.shape[0])
    kv("Features", X_train.shape[1])
    kv("Classes", ', '.join(class_names))
    
    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Grid search
    step(2, 5, "Hyperparameter optimization (Grid Search + 5-fold CV)")
    
    if FAST_MODE and task in KNOWN_BEST_PARAMS:
        info("FAST MODE: Using pre-computed best parameters (same seed=42)")
        best_params = KNOWN_BEST_PARAMS[task]
        cv_score = KNOWN_CV_SCORES[task]
        best = XGBClassifier(
            **best_params, objective='multi:softprob', eval_metric='mlogloss',
            use_label_encoder=False, random_state=42, n_jobs=-1, tree_method='hist'
        )
        best.fit(X_train_s, y_train)
    else:
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
        }
        total_combos = 3 * 3 * 3 * 2
        info(f"Search space: {total_combos} combinations x 5 folds = {total_combos * 5} fits")
        xgb = XGBClassifier(
            objective='multi:softprob', eval_metric='mlogloss',
            use_label_encoder=False, random_state=42, n_jobs=-1, tree_method='hist'
        )
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(xgb, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0, refit=True)
        grid.fit(X_train_s, y_train)
        best = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = grid.best_score_
    
    success(f"Best CV accuracy: {cv_score*100:.2f}%")
    for k, v in best_params.items():
        kv(k, v, indent=6)
    
    # Evaluate
    step(3, 5, "Test set evaluation")
    y_pred = best.predict(X_test_s)
    y_prob = best.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    
    metrics = compute_metrics_per_class(y_test, y_pred, y_prob, class_names, n_classes)
    
    table(
        ["Class", "SEN (%)", "SPE (%)", "PPV (%)", "NPV (%)", "F1 (%)", "AUC"],
        [[m['Class'], m['SEN (%)'], m['SPE (%)'], m['PPV (%)'], m['NPV (%)'], m['F1 (%)'], m['AUC']] for m in metrics]
    )
    
    # Plots
    step(4, 5, "Generating ROC curves & confusion matrix")
    macro_auc, roc_path = plot_roc_curves(y_test, y_prob, class_names, n_classes, task, 'XGBoost', OUTPUT_DIR)
    cm_path = plot_confusion_matrix(y_test, y_pred, class_names, task, 'XGBoost', OUTPUT_DIR)
    success(f"ROC curve:        {roc_path.name}")
    success(f"Confusion matrix: {cm_path.name}")
    
    # Feature importance
    step(5, 5, "Feature importance analysis")
    fi_path, top_5 = plot_feature_importance(best, feat_cols, task, top_n=20)
    success(f"Feature importance: {fi_path.name}")
    info(f"Top 5 features: {', '.join(top_5)}")
    
    # Save
    results = {
        'model': 'XGBoost', 'task': task,
        'accuracy': f"{acc*100:.2f}", 'macro_auc': f"{macro_auc:.4f}",
        'best_params': best_params, 'cv_accuracy': f"{cv_score:.4f}",
        'per_class_metrics': metrics
    }
    with open(CACHE_DIR / f"xgb_results_{task}.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / f"xgb_metrics_{task}.csv", index=False)
    
    result_box(f"XGBOOST — {task.upper()}", {
        "Test Accuracy": f"{acc*100:.2f}%",
        "Macro AUC (OVR)": f"{macro_auc:.4f}",
        "Best learning_rate": str(best_params.get('learning_rate', '')),
        "Best max_depth": str(best_params.get('max_depth', '')),
        "Best n_estimators": str(best_params.get('n_estimators', '')),
        "Elapsed": elapsed_time(t0),
    })
    
    return results, best


def main():
    t0 = time.time()
    mode_str = "FAST MODE (pre-computed params)" if FAST_MODE else "FULL (Grid Search)"
    banner(
        "XGBOOST BASELINE EXPERIMENT",
        f"Addressing Reviewer 1 & 2 | {mode_str}"
    )
    
    for task in ['3class', '4class']:
        try:
            train_and_evaluate_xgb(task)
        except FileNotFoundError:
            error(f"Feature file not found for {task}. Run feature_engineering.py first.")
    
    success(f"XGBoost experiment complete ({elapsed_time(t0)})")


if __name__ == "__main__":
    main()
