# -*- coding: utf-8 -*-
"""
ECG Dataset loading for the ResNet experiment.
Handles two data sources:
  1. my_folder/   — per-patient raw CSV files (13 leads, continuous, needs segmentation)
  2. 湖南大学/    — per-lead pre-segmented CSV files
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, Sampler


# ==============================================================================
# Lead configuration
# ==============================================================================
# Column order in my_folder CSVs (after encoding fix):
# Ⅰ, Ⅱ, Ⅲ, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6, EB
# We map to a canonical order matching 湖南大学 format:
# EB, I, II, III, V1, V2, V3, V4, V5, V6, aVF, aVL, aVR
ALL_LEADS = ['EB', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
LEAD_NAME_MAP = {
    'EB': 'EB', 'I': 'Ⅰ', 'II': 'Ⅱ', 'III': 'Ⅲ',
    'V1': 'V1', 'V2': 'V2', 'V3': 'V3', 'V4': 'V4', 'V5': 'V5', 'V6': 'V6',
    'aVF': 'aVF', 'aVL': 'aVL', 'aVR': 'aVR',
}

# Lead name mapping for 湖南大学 CSV filenames
HNU_LEAD_NAMES = {
    'EB': 'Lead EB', 'Ⅰ': 'Lead I', 'Ⅱ': 'Lead II', 'Ⅲ': 'Lead III',
    'V1': 'Lead V1', 'V2': 'Lead V2', 'V3': 'Lead V3', 'V4': 'Lead V4',
    'V5': 'Lead V5', 'V6': 'Lead V6',
    'aVF': 'Lead aVF', 'aVL': 'Lead aVL', 'aVR': 'Lead aVR',
}

CLASSES = ['AVNRT', 'AVRT-L', 'AVRT-R', 'N']
# One-hot labels (original code convention):
# AVNRT=[0,0,0,1], AVRT-L=[0,0,1,0], AVRT-R=[0,1,0,0], N=[1,0,0,0]
CLASS_LABELS = {
    'AVNRT': [0, 0, 0, 1],
    'AVRT-L': [0, 0, 1, 0],
    'AVRT-R': [0, 1, 0, 0],
    'N': [1, 0, 0, 0],
}

SEGMENT_LEN = 5000  # 10 seconds at 500 Hz


# ==============================================================================
# Data loading: my_folder format (per-patient raw CSV)
# ==============================================================================
def load_my_folder_data(base_path, split, lead_indices=None):
    """
    Load data from my_folder/{split}/4class/{class}/*.CSV

    Each file contains continuous ECG data at 500Hz with 13 columns.
    We segment into non-overlapping 5000-point segments.

    Args:
        base_path: path to my_folder/
        split: 'train' or 'test'
        lead_indices: list of column indices to select (None = all 13)

    Returns:
        segments: np.array [N, n_leads, 5000]
        labels: np.array [N, 4] (one-hot)
    """
    all_segments = []
    all_labels = []

    data_dir = os.path.join(base_path, split, '4class')
    if not os.path.isdir(data_dir):
        print(f"  [WARN] Directory not found: {data_dir}")
        return np.empty((0,)), np.empty((0,))

    for cls_name in CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = CLASS_LABELS[cls_name]
        files = sorted([f for f in os.listdir(cls_dir) if f.upper().endswith('.CSV')])

        for fname in files:
            fpath = os.path.join(cls_dir, fname)
            try:
                # Read with all 13 leads in canonical order
                df = pd.read_csv(fpath, encoding='gbk',
                                 usecols=['Ⅰ', 'Ⅱ', 'Ⅲ', 'aVR', 'aVL', 'aVF',
                                          'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'EB'])
                df = df[ALL_LEADS]  # reorder to canonical
                data = df.values  # [time_steps, 13]

                # Segment into 5000-point chunks
                n_segments = len(data) // SEGMENT_LEN
                for i in range(n_segments):
                    seg = data[i * SEGMENT_LEN : (i + 1) * SEGMENT_LEN]  # [5000, 13]
                    all_segments.append(seg)
                    all_labels.append(label)
            except Exception as e:
                print(f"  [WARN] Failed to read {fpath}: {e}")

    if not all_segments:
        return np.empty((0,)), np.empty((0,))

    segments = np.array(all_segments, dtype=np.float32)  # [N, 5000, 13]
    segments = segments.transpose(0, 2, 1)                # -> [N, 13, 5000]
    labels = np.array(all_labels, dtype=np.float32)       # [N, 4]
    return segments, labels


# ==============================================================================
# Data loading: 湖南大学 format (per-lead pre-segmented CSV)
# ==============================================================================
def load_hnu_data(base_path, split, lead_indices=None):
    """
    Load data from 湖南大学/{split}/4分类/{class}/

    Each file is named like `AVNRT_train_Lead EB.csv` and has rows = segments,
    cols = [patient_name, class_name, 5000 data points].

    Args:
        base_path: path to 湖南大学/
        split: '训练集' or '测试集'
        lead_indices: list of lead indices to select (None = all 13)

    Returns:
        segments: np.array [N, n_leads, 5000]
        labels: np.array [N, 4] (one-hot)
    """
    all_segments = []
    all_labels = []

    data_dir = os.path.join(base_path, split, '4分类')
    if not os.path.isdir(data_dir):
        print(f"  [WARN] Directory not found: {data_dir}")
        return np.empty((0,)), np.empty((0,))

    for cls_name in CLASSES:
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue

        label = CLASS_LABELS[cls_name]
        split_label = 'train' if split == '训练集' else 'test'

        # Load all 13 leads for this class
        lead_data = {}
        n_samples = None

        for lead_idx, lead_name in enumerate(ALL_LEADS):
            hnu_name = HNU_LEAD_NAMES[lead_name]
            fname = f"{cls_name}_{split_label}_{hnu_name}.csv"
            fpath = os.path.join(cls_dir, fname)

            if not os.path.isfile(fpath):
                print(f"  [WARN] File not found: {fpath}")
                continue

            try:
                df = pd.read_csv(fpath)
                # Auto-detect first numeric column (skip all string metadata columns)
                data_start = 0
                for ci in range(min(10, df.shape[1])):
                    if pd.api.types.is_numeric_dtype(df.iloc[:, ci]):
                        data_start = ci
                        break
                data = df.iloc[:, data_start:data_start + SEGMENT_LEN].values.astype(np.float32)
                # Truncate or pad to exactly 5000 points
                if data.shape[1] > SEGMENT_LEN:
                    data = data[:, :SEGMENT_LEN]
                elif data.shape[1] < SEGMENT_LEN:
                    pad_width = SEGMENT_LEN - data.shape[1]
                    data = np.pad(data, ((0, 0), (0, pad_width)), mode='constant')

                lead_data[lead_idx] = data
                if n_samples is None:
                    n_samples = data.shape[0]
            except Exception as e:
                print(f"  [WARN] Failed to read {fpath}: {e}")

        if n_samples is None or len(lead_data) == 0:
            continue

        # Stack all leads: [n_samples, 13, 5000]
        combined = np.zeros((n_samples, len(ALL_LEADS), SEGMENT_LEN), dtype=np.float32)
        for lead_idx, data in lead_data.items():
            combined[:, lead_idx, :] = data[:n_samples]

        for i in range(n_samples):
            all_segments.append(combined[i])  # [13, 5000]
            all_labels.append(label)

    if not all_segments:
        return np.empty((0,)), np.empty((0,))

    segments = np.array(all_segments, dtype=np.float32)  # [N, 13, 5000]
    labels = np.array(all_labels, dtype=np.float32)
    return segments, labels


# ==============================================================================
# Combined dataset
# ==============================================================================
class ECGDataset(Dataset):
    """PyTorch Dataset for ECG segments."""

    def __init__(self, signals, labels, lead_indices=None):
        """
        Args:
            signals: np.array [N, 13, 5000] or [N, 5000, 13]
            labels: np.array [N, 4] (one-hot)
            lead_indices: list of ints, which lead channels to keep (None = all)
        """
        # Ensure shape is [N, n_leads, 5000]
        if signals.ndim == 3 and signals.shape[2] == len(ALL_LEADS):
            signals = signals.transpose(0, 2, 1)  # [N, 5000, 13] -> [N, 13, 5000]

        if lead_indices is not None:
            signals = signals[:, lead_indices, :]

        self.signals = torch.tensor(signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        return self.signals[idx], self.labels[idx], idx


# ==============================================================================
# Class-balanced sampler (simplified reimplementation)
# ==============================================================================
class ClassBalancedSampler(Sampler):
    """Oversamples minority classes to balance mini-batches."""

    def __init__(self, labels, shuffle=True):
        """
        Args:
            labels: np.array [N, num_classes] (one-hot)
        """
        self.shuffle = shuffle
        # Convert one-hot to class indices
        class_indices = np.argmax(labels, axis=1)
        self.num_classes = labels.shape[1]

        # Group sample indices by class
        self.class_groups = {}
        for cls in range(self.num_classes):
            self.class_groups[cls] = np.where(class_indices == cls)[0]

        # Target: max class size
        self.max_size = max(len(v) for v in self.class_groups.values())
        self.total = self.max_size * self.num_classes

    def __iter__(self):
        indices = []
        for cls in range(self.num_classes):
            cls_indices = self.class_groups[cls].copy()
            if self.shuffle:
                np.random.shuffle(cls_indices)
            # Oversample to max_size
            if len(cls_indices) < self.max_size:
                repeats = self.max_size // len(cls_indices) + 1
                cls_indices = np.tile(cls_indices, repeats)[:self.max_size]
            else:
                cls_indices = cls_indices[:self.max_size]
            indices.append(cls_indices)

        indices = np.concatenate(indices)
        if self.shuffle:
            np.random.shuffle(indices)
        return iter(indices.tolist())

    def __len__(self):
        return self.total


# ==============================================================================
# Main data loading function
# ==============================================================================
def get_dataloaders(project_root, lead_indices=None, batch_size=64, num_workers=0):
    """
    Load and combine data from both sources.

    Args:
        project_root: path to 李烁数据/
        lead_indices: list of lead indices to use (None = all 13)
        batch_size: batch size for DataLoader
        num_workers: number of data loading workers

    Returns:
        train_loader, test_loader
    """
    my_folder_path = os.path.join(project_root, 'my_folder')
    hnu_path = os.path.join(project_root, '湖南大学')

    print("Loading data...")

    # --- Train ---
    print("  Loading my_folder/train ...")
    seg1_train, lab1_train = load_my_folder_data(my_folder_path, 'train')
    print(f"    my_folder train: {seg1_train.shape if seg1_train.ndim > 1 else '(empty)'}")

    print("  Loading 湖南大学/训练集 ...")
    seg2_train, lab2_train = load_hnu_data(hnu_path, '训练集')
    print(f"    湖南大学 train: {seg2_train.shape if seg2_train.ndim > 1 else '(empty)'}")

    # --- Test ---
    print("  Loading my_folder/test ...")
    seg1_test, lab1_test = load_my_folder_data(my_folder_path, 'test')
    print(f"    my_folder test: {seg1_test.shape if seg1_test.ndim > 1 else '(empty)'}")

    print("  Loading 湖南大学/测试集 ...")
    seg2_test, lab2_test = load_hnu_data(hnu_path, '测试集')
    print(f"    湖南大学 test: {seg2_test.shape if seg2_test.ndim > 1 else '(empty)'}")

    # Combine
    train_parts_seg = [s for s in [seg1_train, seg2_train] if s.ndim > 1]
    train_parts_lab = [l for l in [lab1_train, lab2_train] if l.ndim > 1]
    test_parts_seg = [s for s in [seg1_test, seg2_test] if s.ndim > 1]
    test_parts_lab = [l for l in [lab1_test, lab2_test] if l.ndim > 1]

    train_signals = np.concatenate(train_parts_seg, axis=0) if train_parts_seg else np.empty((0,))
    train_labels = np.concatenate(train_parts_lab, axis=0) if train_parts_lab else np.empty((0,))
    test_signals = np.concatenate(test_parts_seg, axis=0) if test_parts_seg else np.empty((0,))
    test_labels = np.concatenate(test_parts_lab, axis=0) if test_parts_lab else np.empty((0,))

    print(f"  Combined train: {train_signals.shape}, test: {test_signals.shape}")

    # Create datasets
    train_dataset = ECGDataset(train_signals, train_labels, lead_indices=lead_indices)
    test_dataset = ECGDataset(test_signals, test_labels, lead_indices=lead_indices)

    # Create dataloaders
    sampler = ClassBalancedSampler(train_labels, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"  Train loader: {len(train_loader)} batches, Test loader: {len(test_loader)} batches")
    print("Data loading complete.")
    return train_loader, test_loader


# ==============================================================================
# Optimized: Load ALL data once, reuse for multiple configs
# ==============================================================================
def load_all_data(project_root):
    """
    Load all 13-lead data once into memory.
    Returns: (train_signals, train_labels, test_signals, test_labels)
    All shapes: signals=[N, 13, 5000], labels=[N, 4]
    """
    my_folder_path = os.path.join(project_root, 'my_folder')
    hnu_path = os.path.join(project_root, '湖南大学')

    print("Loading ALL data into memory (one-time)...")

    seg1_train, lab1_train = load_my_folder_data(my_folder_path, 'train')
    print(f"  my_folder train: {seg1_train.shape if seg1_train.ndim > 1 else '(empty)'}")

    seg2_train, lab2_train = load_hnu_data(hnu_path, '训练集')
    print(f"  湖南大学 train: {seg2_train.shape if seg2_train.ndim > 1 else '(empty)'}")

    seg1_test, lab1_test = load_my_folder_data(my_folder_path, 'test')
    print(f"  my_folder test: {seg1_test.shape if seg1_test.ndim > 1 else '(empty)'}")

    seg2_test, lab2_test = load_hnu_data(hnu_path, '测试集')
    print(f"  湖南大学 test: {seg2_test.shape if seg2_test.ndim > 1 else '(empty)'}")

    train_parts_seg = [s for s in [seg1_train, seg2_train] if s.ndim > 1]
    train_parts_lab = [l for l in [lab1_train, lab2_train] if l.ndim > 1]
    test_parts_seg = [s for s in [seg1_test, seg2_test] if s.ndim > 1]
    test_parts_lab = [l for l in [lab1_test, lab2_test] if l.ndim > 1]

    train_signals = np.concatenate(train_parts_seg, axis=0)
    train_labels = np.concatenate(train_parts_lab, axis=0)
    test_signals = np.concatenate(test_parts_seg, axis=0)
    test_labels = np.concatenate(test_parts_lab, axis=0)

    print(f"  Cached: train={train_signals.shape}, test={test_signals.shape}")
    print("Data loading complete. Cached in memory.\n")
    return train_signals, train_labels, test_signals, test_labels


def get_dataloaders_from_cache(train_signals, train_labels, test_signals, test_labels,
                                lead_indices=None, batch_size=64, num_workers=0):
    """
    Create DataLoaders from pre-loaded cached data (no disk I/O).
    """
    train_dataset = ECGDataset(train_signals, train_labels, lead_indices=lead_indices)
    test_dataset = ECGDataset(test_signals, test_labels, lead_indices=lead_indices)

    sampler = ClassBalancedSampler(train_labels, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
