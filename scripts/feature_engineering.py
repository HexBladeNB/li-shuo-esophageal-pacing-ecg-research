# -*- coding: utf-8 -*-
"""
 ECG Feature Engineering — Time-domain + Frequency-domain
 Extracts 169-dimensional feature vectors for traditional ML baselines.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import fft, fftfreq
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))
from data_pipeline import load_dataset, get_class_names, LEAD_ORDER_13
from terminal_ui import *
from paths import CACHE_DIR

SAMPLING_RATE = 500


def extract_time_domain_features(signal):
    features = {}
    features['mean'] = np.mean(signal)
    features['std'] = np.std(signal)
    features['max'] = np.max(signal)
    features['min'] = np.min(signal)
    features['peak_to_peak'] = np.ptp(signal)
    features['rms'] = np.sqrt(np.mean(signal**2))
    features['skewness'] = stats.skew(signal)
    features['kurtosis'] = stats.kurtosis(signal)
    zero_crossings = np.sum(np.diff(np.sign(signal - np.mean(signal))) != 0)
    features['zero_crossing_rate'] = zero_crossings / len(signal)
    features['energy'] = np.sum(signal**2) / len(signal)
    return features


def extract_freq_domain_features(signal):
    features = {}
    n = len(signal)
    yf = np.abs(fft(signal))[:n//2]
    xf = fftfreq(n, 1.0/SAMPLING_RATE)[:n//2]
    features['dominant_freq'] = xf[np.argmax(yf[1:]) + 1] if len(yf) > 1 else 0.0
    power = yf**2
    power_norm = power / (np.sum(power) + 1e-12)
    power_norm = power_norm[power_norm > 0]
    features['spectral_entropy'] = -np.sum(power_norm * np.log2(power_norm + 1e-12))
    features['total_power'] = np.sum(power) / n
    return features


def extract_features_single_sample(ecg_sample, lead_names=None):
    n_leads = ecg_sample.shape[1]
    if lead_names is None:
        lead_names = [f"lead_{i}" for i in range(n_leads)]
    
    all_features = {}
    for lead_idx in range(n_leads):
        signal = ecg_sample[:, lead_idx]
        clean_name = lead_names[lead_idx].replace(' ', '_').replace('Ⅰ', 'I').replace('Ⅱ', 'II').replace('Ⅲ', 'III')
        
        for feat_name, value in extract_time_domain_features(signal).items():
            all_features[f"{clean_name}_{feat_name}"] = value
        for feat_name, value in extract_freq_domain_features(signal).items():
            all_features[f"{clean_name}_{feat_name}"] = value
    
    return all_features


def extract_features_batch(X, lead_names=None):
    n_samples = X.shape[0]
    all_rows = []
    
    for i in range(n_samples):
        features = extract_features_single_sample(X[i], lead_names)
        all_rows.append(features)
        progress_bar(i + 1, n_samples, prefix="Extracting")
    
    df = pd.DataFrame(all_rows)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def main():
    t0 = time.time()
    
    banner(
        "ECG FEATURE ENGINEERING",
        "Time-domain + Frequency-domain | 13 features x 13 leads = 169 dim"
    )
    
    # Check cache
    all_exist = all(
        (CACHE_DIR / f"features_{s}_{t}.csv").exists()
        for s in ['train', 'test'] for t in ['3class', '4class']
    )
    if all_exist:
        success("All feature files already cached in _cache/ — skipping extraction")
        for t in ['3class', '4class']:
            for s in ['train', 'test']:
                p = CACHE_DIR / f"features_{s}_{t}.csv"
                info(f"{p.name} ({p.stat().st_size/1024:.0f} KB)")
        return
    
    for task in ['3class', '4class']:
        section(f"Task: {task.upper()}", "◆")
        class_names = get_class_names(task)
        
        for split in ['train', 'test']:
            step_label = f"{split.upper()} set"
            info(f"Loading {step_label} data...")
            
            X, y, patients = load_dataset(
                split=split, task=task, 
                lead_config='all_13', source='myfolder'
            )
            
            if len(X) == 0:
                warning(f"No data found for {split}/{task}")
                continue
            
            kv("Samples", X.shape[0])
            kv("Shape", str(X.shape))
            
            info(f"Extracting features from {X.shape[0]} segments...")
            features_df = extract_features_batch(X, lead_names=LEAD_ORDER_13)
            
            features_df['label'] = y
            features_df['label_name'] = [class_names[l] for l in y]
            features_df['patient_id'] = patients
            
            output_path = CACHE_DIR / f"features_{split}_{task}.csv"
            features_df.to_csv(output_path, index=False)
            
            feat_count = features_df.shape[1] - 3
            success(f"Saved {output_path.name} ({features_df.shape[0]} samples x {feat_count} features)")
    
    result_box("FEATURE ENGINEERING COMPLETE", {
        "Feature dimensions": "169 per segment",
        "Time-domain":        "10 features x 13 leads",
        "Frequency-domain":   "3 features x 13 leads",
        "Elapsed":            elapsed_time(t0),
    })


if __name__ == "__main__":
    main()
