# -*- coding: utf-8 -*-
"""
 Random Forest Baseline — Addressing Reviewer 1 & 2
 Grid Search + 5-fold Stratified CV + Full Metrics Suite
 Usage: python train_rf.py          (full grid search, ~6 min)
        python train_rf.py --fast   (use known best params, ~10 sec)
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve, auc,
    confusion_matrix
)
from sklearn.preprocessing import label_binarize, StandardScaler
import json
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from terminal_ui import *
from paths import CACHE_DIR, DAY3_DIR

OUTPUT_DIR = DAY3_DIR

FAST_MODE = '--fast' in sys.argv

# Pre-computed best params (from initial run, seed=42)
KNOWN_BEST_PARAMS = {
    '3class': {'max_depth': 20, 'min_samples_split': 5, 'n_estimators': 300},
    '4class': {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 300},
}
KNOWN_CV_SCORES = {'3class': 0.9977, '4class': 0.9980}


def load_features(task='3class'):
    train_path = CACHE_DIR / f"features_train_{task}.csv"
    test_path = CACHE_DIR / f"features_test_{task}.csv"
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    feat_cols = [c for c in df_train.columns if c not in ['label', 'label_name', 'patient_id']]
    X_train = df_train[feat_cols].values
    y_train = df_train['label'].values
    X_test = df_test[feat_cols].values
    y_test = df_test['label'].values
    class_names = sorted(df_train['label_name'].unique())
    return X_train, y_train, X_test, y_test, feat_cols, class_names


def compute_metrics_per_class(y_true, y_pred, y_prob, class_names, n_classes):
    results = []
    for i, cls_name in enumerate(class_names):
        y_true_i = (y_true == i).astype(int)
        y_pred_i = (y_pred == i).astype(int)
        tp = np.sum((y_pred_i == 1) & (y_true_i == 1))
        tn = np.sum((y_pred_i == 0) & (y_true_i == 0))
        fp = np.sum((y_pred_i == 1) & (y_true_i == 0))
        fn = np.sum((y_pred_i == 0) & (y_true_i == 1))
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * ppv * sen / (ppv + sen) if (ppv + sen) > 0 else 0
        try:
            cls_auc = roc_auc_score(y_true_i, y_prob[:, i])
        except:
            cls_auc = 0
        results.append({
            'Class': cls_name,
            'SEN (%)': f"{sen*100:.2f}",
            'SPE (%)': f"{spe*100:.2f}",
            'PPV (%)': f"{ppv*100:.2f}",
            'NPV (%)': f"{npv*100:.2f}",
            'F1 (%)': f"{f1*100:.2f}",
            'AUC': f"{cls_auc:.4f}"
        })
    return results


def plot_roc_curves(y_test, y_prob, class_names, n_classes, task, model_name, save_dir):
    y_test_bin = label_binarize(y_test, classes=range(n_classes))
    fig, ax = plt.subplots(1, 1, figsize=(8, 7))
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
    for i, cls_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2.5,
                label=f'{cls_name} (AUC = {roc_auc:.3f})')
    macro_auc = roc_auc_score(y_test_bin, y_prob, multi_class='ovr', average='macro')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13)
    ax.set_ylabel('True Positive Rate', fontsize=13)
    ax.set_title(f'{model_name} ROC Curve ({task})\nMacro AUC = {macro_auc:.3f}', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    save_path = save_dir / f"roc_{model_name.lower().replace(' ', '_')}_{task}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    return macro_auc, save_path


def plot_confusion_matrix(y_test, y_pred, class_names, task, model_name, save_dir):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax,
                annot_kws={"size": 14}, linewidths=0.5)
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('True', fontsize=13)
    ax.set_title(f'{model_name} Confusion Matrix ({task})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = save_dir / f"cm_{model_name.lower().replace(' ', '_')}_{task}.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    return save_path


def train_and_evaluate_rf(task='3class'):
    section(f"Random Forest — {task.upper()}", "◆")
    t0 = time.time()
    
    # Load
    step(1, 4, "Loading feature matrix")
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
    
    # Grid search or fast mode
    step(2, 4, "Hyperparameter optimization (Grid Search + 5-fold CV)")
    
    if FAST_MODE and task in KNOWN_BEST_PARAMS:
        info("FAST MODE: Using pre-computed best parameters (same seed=42)")
        best_params = KNOWN_BEST_PARAMS[task]
        cv_score = KNOWN_CV_SCORES[task]
        best = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1, class_weight='balanced')
        best.fit(X_train_s, y_train)
    else:
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
        }
        total_combos = 3 * 4 * 3
        info(f"Search space: {total_combos} combinations x 5 folds = {total_combos * 5} fits")
        rf = RandomForestClassifier(random_state=42, n_jobs=-1, class_weight='balanced')
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        grid = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=0, refit=True)
        grid.fit(X_train_s, y_train)
        best = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = grid.best_score_
    
    success(f"Best CV accuracy: {cv_score*100:.2f}%")
    for k, v in best_params.items():
        kv(k, v, indent=6)
    
    # Evaluate
    step(3, 4, "Test set evaluation")
    y_pred = best.predict(X_test_s)
    y_prob = best.predict_proba(X_test_s)
    acc = accuracy_score(y_test, y_pred)
    
    metrics = compute_metrics_per_class(y_test, y_pred, y_prob, class_names, n_classes)
    
    table(
        ["Class", "SEN (%)", "SPE (%)", "PPV (%)", "NPV (%)", "F1 (%)", "AUC"],
        [[m['Class'], m['SEN (%)'], m['SPE (%)'], m['PPV (%)'], m['NPV (%)'], m['F1 (%)'], m['AUC']] for m in metrics]
    )
    
    # Plots
    step(4, 4, "Generating visualizations")
    macro_auc, roc_path = plot_roc_curves(y_test, y_prob, class_names, n_classes, task, 'Random Forest', OUTPUT_DIR)
    cm_path = plot_confusion_matrix(y_test, y_pred, class_names, task, 'Random Forest', OUTPUT_DIR)
    success(f"ROC curve:        {roc_path.name}")
    success(f"Confusion matrix: {cm_path.name}")
    
    # Save
    results = {
        'model': 'Random Forest', 'task': task,
        'accuracy': f"{acc*100:.2f}", 'macro_auc': f"{macro_auc:.4f}",
        'best_params': best_params, 'cv_accuracy': f"{cv_score:.4f}",
        'per_class_metrics': metrics
    }
    with open(CACHE_DIR / f"rf_results_{task}.json", 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame(metrics).to_csv(OUTPUT_DIR / f"rf_metrics_{task}.csv", index=False)
    
    result_box(f"RANDOM FOREST — {task.upper()}", {
        "Test Accuracy": f"{acc*100:.2f}%",
        "Macro AUC (OVR)": f"{macro_auc:.4f}",
        "Best n_estimators": str(best_params.get('n_estimators', '')),
        "Best max_depth": str(best_params.get('max_depth', '')),
        "Elapsed": elapsed_time(t0),
    })
    
    return results, best


def main():
    t0 = time.time()
    mode_str = "FAST MODE (pre-computed params)" if FAST_MODE else "FULL (Grid Search)"
    banner(
        "RANDOM FOREST BASELINE EXPERIMENT",
        f"Addressing Reviewer 1 & 2 | {mode_str}"
    )
    
    for task in ['3class', '4class']:
        try:
            train_and_evaluate_rf(task)
        except FileNotFoundError:
            error(f"Feature file not found for {task}. Run feature_engineering.py first.")
    
    success(f"Random Forest experiment complete ({elapsed_time(t0)})")


if __name__ == "__main__":
    main()

