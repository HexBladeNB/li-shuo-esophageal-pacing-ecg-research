# -*- coding: utf-8 -*-
"""
Pairwise Bootstrap comparisons (AUC + ACC) with Benjamini-Hochberg correction.
Generates 36 pairs × 2 metrics = 72 comparisons.

Usage:
    cd C:\\Users\\Administrator\\Desktop\\李烁数据
    python 深度学习模型图片修复\\resnet_experiment\\pairwise_comparisons.py
"""
import numpy as np
import os
from sklearn.metrics import roc_curve, auc, accuracy_score
from itertools import combinations

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'tables_and_figures')
N_BOOT = 2000

CONFIGS = [
    ('A', 'II_V1',       'II+V1'),
    ('B', 'II_V1_EB',    'II+V1+EB'),
    ('C', 'aVF',         'aVF'),
    ('D', 'aVF_EB',      'aVF+EB'),
    ('E', 'II_V1_aVF',   'II+V1+aVF'),
    ('F', 'II_V1_aVF_EB','II+V1+aVF+EB'),
    ('G', '12-Surface',  '12-lead'),
    ('H', '12-Surface_EB','12-lead+EB'),
    ('I', 'EB',          'EB'),
]


def bh_correct(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.zeros(n)
    for i in range(n - 1, -1, -1):
        rank = i + 1
        raw = pvals[order[i]] * n / rank
        if i == n - 1:
            adjusted[order[i]] = min(raw, 1.0)
        else:
            adjusted[order[i]] = min(raw, adjusted[order[i + 1]], 1.0)
    return adjusted


def fmt_p(p):
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def main():
    rng = np.random.RandomState(42)
    data = {}

    print("Loading data and bootstrapping...")
    for panel, dirname, name in CONFIGS:
        yt = np.load(os.path.join(OUTPUT_DIR, dirname, 'y_true.npy'))
        yp = np.load(os.path.join(OUTPUT_DIR, dirname, 'y_pred_proba.npy'))
        ypred = yp.argmax(axis=1)
        n = len(yt)

        auc_boots, acc_boots = [], []
        for _ in range(N_BOOT):
            idx = rng.choice(n, n, replace=True)
            yoh = np.eye(4)[yt[idx]]
            fpr, tpr, _ = roc_curve(yoh.ravel(), yp[idx].ravel())
            auc_boots.append(auc(fpr, tpr))
            acc_boots.append(accuracy_score(yt[idx], ypred[idx]))

        data[panel] = {
            'name': name,
            'auc_boots': np.array(auc_boots),
            'acc_boots': np.array(acc_boots),
            'auc_mean': np.mean(auc_boots),
            'acc_mean': np.mean(acc_boots),
        }
        print(f"  {panel} {name:<16} AUC={data[panel]['auc_mean']:.4f} ACC={data[panel]['acc_mean']*100:.2f}%")

    # All pairwise
    panels = [c[0] for c in CONFIGS]
    pairs = list(combinations(panels, 2))
    auc_pvals_raw, acc_pvals_raw = [], []
    pair_info = []

    for pa, pb in pairs:
        diff_auc = data[pa]['auc_boots'] - data[pb]['auc_boots']
        p_auc = 2 * min((diff_auc <= 0).mean(), (diff_auc >= 0).mean())
        auc_pvals_raw.append(min(p_auc, 1.0))

        diff_acc = data[pa]['acc_boots'] - data[pb]['acc_boots']
        p_acc = 2 * min((diff_acc <= 0).mean(), (diff_acc >= 0).mean())
        acc_pvals_raw.append(min(p_acc, 1.0))

        pair_info.append((pa, pb))

    auc_adj = bh_correct(np.array(auc_pvals_raw))
    acc_adj = bh_correct(np.array(acc_pvals_raw))

    # Write Markdown
    out_path = os.path.join(RESULTS_DIR, 'pairwise_comparisons_BH.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('# Pairwise Comparisons (Bootstrap + BH Correction)\n\n')

        # AUC table
        f.write('## AUC Comparisons\n\n')
        f.write('| Config A | Config B | AUC_A | AUC_B | Δ AUC | P_raw | P_adj | Sig |\n')
        f.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for i, (pa, pb) in enumerate(pair_info):
            da, db = data[pa], data[pb]
            delta = da['auc_mean'] - db['auc_mean']
            sig = '*' if auc_adj[i] < 0.05 else 'ns'
            f.write(f"| {da['name']} | {db['name']} "
                    f"| {da['auc_mean']:.3f} | {db['auc_mean']:.3f} "
                    f"| {delta:+.4f} | {fmt_p(auc_pvals_raw[i])} "
                    f"| {fmt_p(auc_adj[i])} | {sig} |\n")

        # ACC table
        f.write('\n## ACC Comparisons\n\n')
        f.write('| Config A | Config B | ACC_A | ACC_B | Δ ACC | P_raw | P_adj | Sig |\n')
        f.write('| --- | --- | --- | --- | --- | --- | --- | --- |\n')
        for i, (pa, pb) in enumerate(pair_info):
            da, db = data[pa], data[pb]
            delta = da['acc_mean'] - db['acc_mean']
            sig = '*' if acc_adj[i] < 0.05 else 'ns'
            f.write(f"| {da['name']} | {db['name']} "
                    f"| {da['acc_mean']*100:.2f}% | {db['acc_mean']*100:.2f}% "
                    f"| {delta*100:+.2f}% | {fmt_p(acc_pvals_raw[i])} "
                    f"| {fmt_p(acc_adj[i])} | {sig} |\n")

        # Verify original manuscript claims
        f.write('\n---\n\n## 原文 Results 关键表述验证\n\n')

        def find_p(pa, pb, metric='auc'):
            for j, (a, b) in enumerate(pair_info):
                if (a == pa and b == pb) or (a == pb and b == pa):
                    return auc_adj[j] if metric == 'auc' else acc_adj[j]
            return None

        claims = [
            ('Results §1: 12-lead ACC ≈ aVF ACC (P>0.05)', 'G', 'C', 'acc', '>'),
            ('Results §1: aVF AUC > other surface (vs II+V1, P<0.05)', 'C', 'A', 'auc', '<'),
            ('Results §1: 12-lead AUC ≈ II+V1+aVF (P>0.05)', 'G', 'E', 'auc', '>'),
            ('Results §2: 12-lead ACC ≈ 12-lead+EB ACC (P>0.05)', 'G', 'H', 'acc', '>'),
            ('Results §2: aVF AUC ≈ aVF+EB AUC (P>0.05)', 'C', 'D', 'auc', '>'),
            ('Results §3: EB ACC ≈ aVF+EB ACC (P>0.05)', 'I', 'D', 'acc', '>'),
            ('Results §3: EB AUC > all surface (vs 12-lead, P<0.05)', 'I', 'G', 'auc', '<'),
        ]

        f.write('| 原文表述 | P_adj(BH) | 原文要求 | 仍然成立？ |\n')
        f.write('| --- | --- | --- | --- |\n')
        for desc, pa, pb, metric, direction in claims:
            p = find_p(pa, pb, metric)
            if direction == '>':
                holds = '✅' if p >= 0.05 else '❌ 需修改'
            else:
                holds = '✅' if p < 0.05 else '❌ 需修改'
            f.write(f"| {desc} | {fmt_p(p)} | P{direction}0.05 | {holds} |\n")

    print(f"\nSaved: {out_path}")
    print(f"Total: {len(pairs)} pairs × 2 metrics = {len(pairs)*2} comparisons")


if __name__ == '__main__':
    main()
