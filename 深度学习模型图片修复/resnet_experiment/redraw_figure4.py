# -*- coding: utf-8 -*-
"""Redraw Figure 4 ROC from saved predictions (no retraining needed)."""
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')

CONFIGS = [
    {'panel': 'A', 'dir': 'II_V1'},
    {'panel': 'B', 'dir': 'II_V1_EB'},
    {'panel': 'C', 'dir': 'aVF'},
    {'panel': 'D', 'dir': 'aVF_EB'},
    {'panel': 'E', 'dir': 'II_V1_aVF'},
    {'panel': 'F', 'dir': 'II_V1_aVF_EB'},
    {'panel': 'G', 'dir': '12-Surface'},
    {'panel': 'H', 'dir': '12-Surface_EB'},
    {'panel': 'I', 'dir': 'EB'},
]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
for idx, cfg in enumerate(CONFIGS):
    d = os.path.join(OUTPUT_DIR, cfg['dir'])
    yt = np.load(os.path.join(d, 'y_true.npy'))
    yp = np.load(os.path.join(d, 'y_pred_proba.npy'))
    yoh = np.eye(4)[yt]
    fpr, tpr, _ = roc_curve(yoh.ravel(), yp.ravel())
    auc_val = auc(fpr, tpr)

    ax = axes[idx // 3, idx % 3]
    ax.plot(fpr, tpr, color='red', lw=1.5)
    ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)
    ax.text(0.02, 0.98, cfg['panel'], transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='top', ha='left')
    ax.text(0.95, 0.05, f"AUC = {auc_val:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight='bold',
            va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))
    ax.set_title('ROC', fontsize=12, fontweight='bold')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=9)
    ax.set_ylabel('True Positive Rate', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, color='gray')

plt.tight_layout(pad=2.0)
tiff_path = os.path.join(OUTPUT_DIR, 'Figure4_ROC_all_configs.tiff')
plt.savefig(tiff_path, dpi=600, bbox_inches='tight', format='tiff')
png_path = os.path.join(OUTPUT_DIR, 'Figure4_ROC_all_configs_preview.png')
plt.savefig(png_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {tiff_path}")
print(f"Preview: {png_path}")
