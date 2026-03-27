# -*- coding: utf-8 -*-
"""
Generate ALL tables and figures from the retrained model predictions.
Reads y_true.npy + y_pred_proba.npy from each config, outputs:
  1. Figure 5: Confusion matrices (TIFF 600DPI + preview PNG)
  2. Table 2: ACC + AUC + 95% CI (Markdown + CSV)
  3. Table 3: SEN/SPE/PPV/NPV/F1 + 95% CI per class per config (Markdown + CSV)
  4. Statistical comparisons between configs (P values, matching original Table 2 stars)

Usage:
    cd C:\\Users\\Administrator\\Desktop\\李烁数据
    python 深度学习模型图片修复\\resnet_experiment\\generate_all_tables.py
"""

import os
import sys
import numpy as np
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)

# ==============================================================================
# Config
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'tables_and_figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

N_BOOT = 2000
SEED = 42
CLASSES = ['AVNRT', 'AVRT-L', 'AVRT-R', 'N']
NUM_CLASSES = 4

CONFIGS = [
    {'name': 'II+V1',         'panel': 'A', 'dir': 'II_V1'},
    {'name': 'II+V1+EB',      'panel': 'B', 'dir': 'II_V1_EB'},
    {'name': 'aVF',           'panel': 'C', 'dir': 'aVF'},
    {'name': 'aVF+EB',        'panel': 'D', 'dir': 'aVF_EB'},
    {'name': 'II+V1+aVF',     'panel': 'E', 'dir': 'II_V1_aVF'},
    {'name': 'II+V1+aVF+EB',  'panel': 'F', 'dir': 'II_V1_aVF_EB'},
    {'name': '12-lead',       'panel': 'G', 'dir': '12-Surface'},
    {'name': '12-lead+EB',    'panel': 'H', 'dir': '12-Surface_EB'},
    {'name': 'EB',            'panel': 'I', 'dir': 'EB'},
]


def load_predictions(cfg):
    d = os.path.join(OUTPUT_DIR, cfg['dir'])
    y_true = np.load(os.path.join(d, 'y_true.npy'))
    y_proba = np.load(os.path.join(d, 'y_pred_proba.npy'))
    y_pred = y_proba.argmax(axis=1)
    return y_true, y_pred, y_proba


def wilson_ci(p, n, z=1.96):
    """Wilson score interval for proportions."""
    if n == 0:
        return 0, 0
    denom = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denom
    spread = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return max(0, center - spread), min(1, center + spread)


def bootstrap_auc(y_true, y_proba, n_boot=N_BOOT, seed=SEED):
    """Bootstrap AUC with 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_proba[idx]
        y_oh = np.eye(NUM_CLASSES)[yt]
        fpr, tpr, _ = roc_curve(y_oh.ravel(), yp.ravel())
        aucs.append(auc(fpr, tpr))
    aucs = np.array(aucs)
    return aucs.mean(), np.percentile(aucs, 2.5), np.percentile(aucs, 97.5), aucs


def bootstrap_acc(y_true, y_pred, n_boot=N_BOOT, seed=SEED):
    """Bootstrap ACC with 95% CI."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    accs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        accs.append(accuracy_score(y_true[idx], y_pred[idx]))
    accs = np.array(accs)
    return accs.mean(), np.percentile(accs, 2.5), np.percentile(accs, 97.5)


def per_class_metrics(y_true, y_pred, cls_idx, n_samples):
    """Compute SEN, SPE, PPV, NPV, F1 with Wilson CI for one class."""
    tp = np.sum((y_pred == cls_idx) & (y_true == cls_idx))
    fp = np.sum((y_pred == cls_idx) & (y_true != cls_idx))
    fn = np.sum((y_pred != cls_idx) & (y_true == cls_idx))
    tn = np.sum((y_pred != cls_idx) & (y_true != cls_idx))

    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1 = 2*tp / (2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0

    sen_lo, sen_hi = wilson_ci(sen, tp + fn)
    spe_lo, spe_hi = wilson_ci(spe, tn + fp)
    ppv_lo, ppv_hi = wilson_ci(ppv, tp + fp)
    npv_lo, npv_hi = wilson_ci(npv, tn + fn)
    f1_lo, f1_hi = wilson_ci(f1, n_samples)  # approximate

    return {
        'SEN': sen, 'SEN_lo': sen_lo, 'SEN_hi': sen_hi,
        'SPE': spe, 'SPE_lo': spe_lo, 'SPE_hi': spe_hi,
        'PPV': ppv, 'PPV_lo': ppv_lo, 'PPV_hi': ppv_hi,
        'NPV': npv, 'NPV_lo': npv_lo, 'NPV_hi': npv_hi,
        'F1': f1, 'F1_lo': f1_lo, 'F1_hi': f1_hi,
    }


def per_class_auc(y_true, y_proba, cls_idx, n_boot=N_BOOT, seed=SEED):
    """One-vs-rest AUC with bootstrap CI for one class."""
    y_bin = (y_true == cls_idx).astype(int)
    scores = y_proba[:, cls_idx]
    fpr, tpr, _ = roc_curve(y_bin, scores)
    auc_val = auc(fpr, tpr)

    rng = np.random.RandomState(seed)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb, sb = y_bin[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        f, t, _ = roc_curve(yb, sb)
        aucs.append(auc(f, t))
    aucs = np.array(aucs)
    return auc_val, np.percentile(aucs, 2.5), np.percentile(aucs, 97.5)


def fmt_pct(val, lo, hi):
    return f"{val*100:.2f} ({lo*100:.2f}-{hi*100:.2f})"


def fmt_auc(val, lo, hi):
    return f"{val:.3f} ({lo:.3f}-{hi:.3f})"


# ==============================================================================
# 1. Figure 5: Confusion Matrices
# ==============================================================================
def generate_figure5():
    print("Generating Figure 5 (Confusion Matrices)...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    for idx, cfg in enumerate(CONFIGS):
        y_true, y_pred, _ = load_predictions(cfg)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES, ax=ax,
            cbar=True, cbar_kws={'shrink': 0.8},
            annot_kws={'size': 11, 'fontweight': 'bold'},
            linewidths=0.5, linecolor='white',
        )
        ax.set_xlabel('Predicted label', fontsize=11)
        ax.set_ylabel('True label', fontsize=11)
        ax.set_title('Confusion matrix', fontsize=12, fontweight='bold')
        ax.set_xticklabels(CLASSES, rotation=45, ha='right', fontsize=9)
        ax.set_yticklabels(CLASSES, rotation=0, fontsize=9)
        ax.text(-0.15, 1.05, cfg['panel'], transform=ax.transAxes,
                fontsize=16, fontweight='bold', va='bottom', ha='left')

    plt.tight_layout(pad=2.5)

    tiff_path = os.path.join(RESULTS_DIR, 'Figure5_confusion_matrices.tiff')
    plt.savefig(tiff_path, dpi=600, bbox_inches='tight', format='tiff')
    png_path = os.path.join(RESULTS_DIR, 'Figure5_confusion_matrices_preview.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', format='png')
    plt.close()
    print(f"  Saved: {tiff_path}")
    print(f"  Preview: {png_path}")


# ==============================================================================
# 2. Table 2: ACC + AUC + 95% CI
# ==============================================================================
def generate_table2():
    print("\nGenerating Table 2 (ACC + AUC + CI)...")
    rows = []
    all_auc_bootstraps = {}

    for cfg in CONFIGS:
        y_true, y_pred, y_proba = load_predictions(cfg)

        # ACC + CI
        acc = accuracy_score(y_true, y_pred)
        _, acc_lo, acc_hi = bootstrap_acc(y_true, y_pred)

        # AUC + CI
        auc_val, auc_lo, auc_hi, auc_boots = bootstrap_auc(y_true, y_proba)
        all_auc_bootstraps[cfg['panel']] = auc_boots

        rows.append({
            'name': cfg['name'], 'panel': cfg['panel'],
            'acc': acc, 'acc_lo': acc_lo, 'acc_hi': acc_hi,
            'auc': auc_val, 'auc_lo': auc_lo, 'auc_hi': auc_hi,
        })

    # Statistical comparisons vs EB (Panel I)
    eb_boots = all_auc_bootstraps['I']
    eb_acc_row = [r for r in rows if r['panel'] == 'I'][0]

    for r in rows:
        if r['panel'] == 'I':
            r['p_vs_eb'] = 1.0
            r['sig'] = ''
        else:
            boots = all_auc_bootstraps[r['panel']]
            diff = eb_boots - boots
            p_val = (diff <= 0).mean()
            r['p_vs_eb'] = p_val
            r['sig'] = '*' if p_val < 0.05 else ''

    # Write Markdown
    md_path = os.path.join(RESULTS_DIR, 'Table2_ACC_AUC.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Table 2: Classification Accuracy and AUC Values\n\n")
        f.write("| Input Leads | ACC (%) | 95% CI for ACC | AUC | 95% CI for AUC |\n")
        f.write("| --- | --- | --- | --- | --- |\n")
        for r in rows:
            sig = r['sig']
            f.write(f"| {r['name']}{' ' + sig if sig else ''} "
                    f"| {r['acc']*100:.2f} "
                    f"| {r['acc_lo']*100:.2f}-{r['acc_hi']*100:.2f} "
                    f"| {r['auc']:.3f}{sig} "
                    f"| {r['auc_lo']:.3f}-{r['auc_hi']:.3f} |\n")
        f.write(f"\n*P<0.05 when compared to EB lead configurations.\n")
        f.write(f"\nCI = Bootstrap {N_BOOT} resamples.\n")

    # Also save JSON for downstream use
    json_path = os.path.join(RESULTS_DIR, 'Table2_data.json')
    with open(json_path, 'w') as f:
        json.dump([{k: v for k, v in r.items()} for r in rows], f, indent=2)

    print(f"  Saved: {md_path}")

    # Print to console
    print(f"\n  {'Leads':<18} {'ACC%':>8} {'95%CI':>16} {'AUC':>7} {'95%CI':>16} {'vs EB':>6}")
    print(f"  {'-'*75}")
    for r in rows:
        print(f"  {r['name']:<18} {r['acc']*100:7.2f}  ({r['acc_lo']*100:.2f}-{r['acc_hi']*100:.2f})"
              f"  {r['auc']:.3f}  ({r['auc_lo']:.3f}-{r['auc_hi']:.3f})"
              f"  {r['sig'] if r['sig'] else 'ns':>4}")

    return rows, all_auc_bootstraps


# ==============================================================================
# 3. Table 3: Per-class metrics + 95% CI
# ==============================================================================
def generate_table3():
    print("\nGenerating Table 3 (Per-class metrics + CI)...")
    md_path = os.path.join(RESULTS_DIR, 'Table3_per_class_metrics.md')

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Table 3: Diagnostic Metrics for ECG Across Various Lead Configurations\n\n")
        f.write("| Input Leads | Diagnosis | SEN (95%CI, %) | SPE (95%CI, %) | "
                "PPV (95%CI, %) | NPV (95%CI, %) | F1-Score (95%CI, %) |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")

        for cfg in CONFIGS:
            y_true, y_pred, y_proba = load_predictions(cfg)
            n = len(y_true)

            for cls_idx, cls_name in enumerate(CLASSES):
                m = per_class_metrics(y_true, y_pred, cls_idx, n)
                lead_col = cfg['name'] if cls_idx == 0 else ''
                f.write(f"| {lead_col} | {cls_name} "
                        f"| {fmt_pct(m['SEN'], m['SEN_lo'], m['SEN_hi'])} "
                        f"| {fmt_pct(m['SPE'], m['SPE_lo'], m['SPE_hi'])} "
                        f"| {fmt_pct(m['PPV'], m['PPV_lo'], m['PPV_hi'])} "
                        f"| {fmt_pct(m['NPV'], m['NPV_lo'], m['NPV_hi'])} "
                        f"| {fmt_pct(m['F1'], m['F1_lo'], m['F1_hi'])} |\n")
            f.write("| | | | | | | |\n")

    print(f"  Saved: {md_path}")

    # Print summary to console
    for cfg in CONFIGS:
        y_true, y_pred, y_proba = load_predictions(cfg)
        print(f"\n  Panel {cfg['panel']}: {cfg['name']}")
        for cls_idx, cls_name in enumerate(CLASSES):
            m = per_class_metrics(y_true, y_pred, cls_idx, len(y_true))
            print(f"    {cls_name:<7} SEN={m['SEN']*100:5.1f}%  SPE={m['SPE']*100:5.1f}%  "
                  f"PPV={m['PPV']*100:5.1f}%  NPV={m['NPV']*100:5.1f}%  F1={m['F1']*100:5.1f}%")


# ==============================================================================
# 4. Config comparison P-values (pairwise, matching Table 2 footnote)
# ==============================================================================
def generate_comparisons(all_auc_bootstraps):
    print("\nGenerating statistical comparisons...")
    md_path = os.path.join(RESULTS_DIR, 'statistical_comparisons.md')

    # Key comparisons
    comparisons = [
        ('I', 'G', 'EB vs 12-lead'),
        ('I', 'H', 'EB vs 12-lead+EB'),
        ('H', 'G', '12-lead+EB vs 12-lead'),
        ('I', 'A', 'EB vs II+V1'),
        ('I', 'C', 'EB vs aVF'),
        ('D', 'C', 'aVF+EB vs aVF'),
        ('B', 'A', 'II+V1+EB vs II+V1'),
        ('D', 'I', 'aVF+EB vs EB'),
    ]

    with open(md_path, 'w', encoding='utf-8') as f:
        f.write("# Statistical Comparisons (Bootstrap AUC)\n\n")
        f.write("| Comparison | AUC_A | AUC_B | Δ AUC | 95% CI for Δ | P-value |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")

        for pa, pb, label in comparisons:
            a_boots = all_auc_bootstraps[pa]
            b_boots = all_auc_bootstraps[pb]
            diff = a_boots - b_boots
            d_mean = diff.mean()
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            p_val = (diff <= 0).mean()
            p_str = f"p<0.001" if p_val < 0.001 else f"p={p_val:.3f}"
            sig = "*" if p_val < 0.05 else "ns"

            f.write(f"| {label} | {a_boots.mean():.3f} | {b_boots.mean():.3f} "
                    f"| {d_mean:+.4f} | [{d_lo:+.4f}, {d_hi:+.4f}] "
                    f"| {p_str} {sig} |\n")

            print(f"  {label:<25} Δ={d_mean:+.4f} [{d_lo:+.4f},{d_hi:+.4f}] {p_str} {sig}")

    print(f"\n  Saved: {md_path}")


# ==============================================================================
# Main
# ==============================================================================
def main():
    print("=" * 60)
    print("  Generating ALL tables and figures from retrained model")
    print("=" * 60)

    generate_figure5()
    rows, all_auc_bootstraps = generate_table2()
    generate_table3()
    generate_comparisons(all_auc_bootstraps)

    print(f"\n{'='*60}")
    print(f"  All outputs saved to: {RESULTS_DIR}")
    print(f"{'='*60}")
    print("\nFiles generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"  {f:<50} {size/1024:.1f} KB")


if __name__ == '__main__':
    main()
