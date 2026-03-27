# -*- coding: utf-8 -*-
"""
Train ALL 9 lead configs using the ORIGINAL architecture & training parameters.
Matches Keras specs from 总体数据预测结果.md.

Usage:
    cd C:\\Users\\Administrator\\Desktop\\李烁数据
    python 深度学习模型图片修复\\resnet_experiment\\train_all_original.py
"""

import sys
import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from model_original import OriginalResNet
from dataset import load_all_data, get_dataloaders_from_cache

PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'output_original')

# ===== ORIGINAL PARAMETERS =====
RANDOM_SEED = 0
EPOCHS = 200
BATCH_SIZE = 60
LR = 0.002
NUM_CLASSES = 4
EARLY_STOP_PATIENCE = 11
LR_REDUCE_PATIENCE = 10
LR_REDUCE_FACTOR = 0.1

CONFIGS = [
    {'name': 'II+V1',           'panel': 'A', 'indices': [2, 4],                                     'paper_auc': 0.917},
    {'name': 'II+V1+EB',        'panel': 'B', 'indices': [0, 2, 4],                                  'paper_auc': 0.979},
    {'name': 'aVF',             'panel': 'C', 'indices': [10],                                       'paper_auc': 0.960},
    {'name': 'aVF+EB',          'panel': 'D', 'indices': [0, 10],                                    'paper_auc': 0.975},
    {'name': 'II+V1+aVF',       'panel': 'E', 'indices': [2, 4, 10],                                 'paper_auc': 0.884},
    {'name': 'II+V1+aVF+EB',    'panel': 'F', 'indices': [0, 2, 4, 10],                              'paper_auc': 0.972},
    {'name': '12-Surface',      'panel': 'G', 'indices': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],   'paper_auc': 0.921},
    {'name': '12-Surface+EB',   'panel': 'H', 'indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],'paper_auc': 0.958},
    {'name': 'EB',              'panel': 'I', 'indices': [0],                                        'paper_auc': 0.980},
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def train_one_epoch(model, device, optimizer, criterion, loader):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for X, Y, _ in loader:
        X, Y = X.to(device), Y.to(device)
        Y_cls = Y.argmax(dim=1)
        optimizer.zero_grad()
        Z = model(X)
        loss = criterion(Z, Y_cls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
        preds = Z.argmax(dim=1)
        correct += (preds == Y_cls).sum().item()
        total += X.size(0)
    return total_loss / total, correct / total


def evaluate(model, device, criterion, loader):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for X, Y, _ in loader:
            X, Y = X.to(device), Y.to(device)
            Y_cls = Y.argmax(dim=1)
            Z = model(X)
            loss = criterion(Z, Y_cls)
            total_loss += loss.item() * X.size(0)
            probs = torch.softmax(Z, dim=1)
            all_probs.append(probs.cpu().numpy())
            preds = Z.argmax(dim=1)
            all_labels.append(Y_cls.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            correct += (preds == Y_cls).sum().item()
            total += X.size(0)
    return (total_loss / total, correct / total,
            np.concatenate(all_labels),
            np.concatenate(all_preds),
            np.concatenate(all_probs))


def compute_micro_roc(y_true, y_proba):
    n_classes = y_proba.shape[1]
    y_true_onehot = np.eye(n_classes)[y_true]
    fpr, tpr, _ = roc_curve(y_true_onehot.ravel(), y_proba.ravel())
    roc_auc = auc(fpr, tpr)
    return fpr, tpr, roc_auc


def train_config(config, device, cached_data):
    name = config['name']
    indices = config['indices']
    panel = config['panel']
    n_leads = len(indices)

    print(f"\n{'='*60}")
    print(f"  Panel {panel}: {name} ({n_leads} leads)")
    print(f"{'='*60}")

    set_seed(RANDOM_SEED)

    train_loader, test_loader = get_dataloaders_from_cache(
        *cached_data,
        lead_indices=indices,
        batch_size=BATCH_SIZE,
        num_workers=0,
    )

    model = OriginalResNet(in_channels=n_leads, num_classes=NUM_CLASSES, dropout=0.2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=LR_REDUCE_FACTOR,
        patience=LR_REDUCE_PATIENCE, min_lr=LR / 100
    )

    best_val_loss = float('inf')
    best_acc = 0.0
    best_labels = best_preds = best_probs = None
    patience_counter = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, device, optimizer, criterion, train_loader)
        val_loss, test_acc, labels, preds, probs = evaluate(model, device, criterion, test_loader)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_acc = test_acc
            best_labels = copy.deepcopy(labels)
            best_preds = copy.deepcopy(preds)
            best_probs = copy.deepcopy(probs)
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 20 == 0 or patience_counter >= EARLY_STOP_PATIENCE:
            lr = optimizer.param_groups[0]['lr']
            print(f'  Ep {epoch:3d} | TrACC:{train_acc:.3f} | TsACC:{test_acc:.3f} | '
                  f'VLoss:{val_loss:.4f} | LR:{lr:.1e} | Pat:{patience_counter}/{EARLY_STOP_PATIENCE}')

        if patience_counter >= EARLY_STOP_PATIENCE:
            print(f"  >> Early stop at epoch {epoch}")
            break

    elapsed = time.time() - start_time
    fpr, tpr, roc_auc = compute_micro_roc(best_labels, best_probs)
    print(f"  >> ACC={best_acc:.4f}, AUC={roc_auc:.4f} (paper={config['paper_auc']}), "
          f"Δ={roc_auc - config['paper_auc']:+.4f}, Time={elapsed:.0f}s")

    cfg_dir = os.path.join(OUTPUT_DIR, name.replace('+', '_'))
    os.makedirs(cfg_dir, exist_ok=True)
    np.save(os.path.join(cfg_dir, 'y_true.npy'), best_labels)
    np.save(os.path.join(cfg_dir, 'y_pred_proba.npy'), best_probs)

    return {
        'panel': panel, 'name': name,
        'fpr': fpr, 'tpr': tpr,
        'auc': roc_auc, 'acc': best_acc,
        'paper_auc': config['paper_auc'],
    }


def plot_combined_roc(results, output_path):
    """3x3 ROC figure: red curves + grid, NO AUC labels."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for idx, res in enumerate(results):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]

        ax.plot(res['fpr'], res['tpr'], color='red', lw=1.5)
        ax.plot([0, 1], [0, 1], 'k--', lw=0.8, alpha=0.5)

        ax.text(0.02, 0.98, res['panel'], transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='top', ha='left')
        ax.set_title('ROC', fontsize=12, fontweight='bold')

        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=9)
        ax.set_ylabel('True Positive Rate', fontsize=9)
        ax.tick_params(labelsize=8)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.grid(True, linestyle='-', linewidth=0.3, alpha=0.5, color='gray')
        ax.text(0.95, 0.05, f"AUC = {res['auc']:.3f}",
                transform=ax.transAxes, fontsize=10, fontweight='bold',
                va='bottom', ha='right',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.8))

    plt.tight_layout(pad=2.0)
    # TIFF 600 DPI for journal submission
    tiff_path = output_path.replace('.png', '.tiff')
    plt.savefig(tiff_path, dpi=600, bbox_inches='tight', format='tiff')
    # Preview PNG 150 DPI for quick viewing
    plt.savefig(output_path.replace('.png', '_preview.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nROC figure saved: {tiff_path} (TIFF 600DPI)")
    print(f"Preview saved: {output_path.replace('.png', '_preview.png')} (PNG 150DPI)")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load ALL data once (cached in memory, ~500MB for 64GB RAM = trivial)
    cached_data = load_all_data(PROJECT_ROOT)

    total_start = time.time()
    results = []

    for config in CONFIGS:
        res = train_config(config, device, cached_data)
        results.append(res)

    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All 9 configs complete! Total time: {total_time/60:.1f} min")
    print(f"{'='*60}")

    print(f"\n{'Panel':<6} {'Config':<20} {'AUC':>8} {'Paper':>8} {'Δ':>8} {'ACC':>8}")
    print('-' * 60)
    for r in results:
        delta = r['auc'] - r['paper_auc']
        print(f"{r['panel']:<6} {r['name']:<20} {r['auc']:8.4f} {r['paper_auc']:8.3f} {delta:+8.4f} {r['acc']:8.4f}")

    plot_combined_roc(results, os.path.join(OUTPUT_DIR, 'Figure4_ROC_all_configs.png'))

    summary = [{'panel': r['panel'], 'name': r['name'], 'auc': float(r['auc']),
                'acc': float(r['acc']), 'paper_auc': r['paper_auc']} for r in results]
    with open(os.path.join(OUTPUT_DIR, 'all_configs_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nDone!")


if __name__ == '__main__':
    main()
