# -*- coding: utf-8 -*-
"""
===============================================================================
 统一数据读取管道 (Unified Data Pipeline)
 ---------------------------------------------------------------
 功能：
   1. 读取两套数据源（湖南大学 切片格式 + my_folder 原始患者文件）
   2. 输出标准格式 (n_samples, 5000, n_leads) + 标签
   3. 支持 3分类/4分类 切换
   4. 支持选择特定导联组合
 ---------------------------------------------------------------
 数据源说明：
   - 湖南大学/: 每个导联一个CSV，每行一个样本（5000个采样点）
   - my_folder/: 每个患者一个CSV（13列导联），每5000点切一个样本
===============================================================================
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path

# ==============================================================================
# 配置
# ==============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent

# 导联顺序（标准化）
LEAD_ORDER_13 = ['EB', 'Ⅰ', 'Ⅱ', 'Ⅲ', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'aVF', 'aVL', 'aVR']
LEAD_CSV_NAMES_HNU = [
    'Lead EB', 'Lead I', 'Lead II', 'Lead III',
    'Lead V1', 'Lead V2', 'Lead V3', 'Lead V4', 'Lead V5', 'Lead V6',
    'Lead aVF', 'Lead aVL', 'Lead aVR'
]

# 3分类标签映射
LABEL_3CLASS = {'AVNRT': 0, 'AVRT': 1, 'N': 2}
# 4分类标签映射
LABEL_4CLASS = {'AVNRT': 0, 'AVRT-L': 1, 'AVRT-R': 2, 'N': 3}

# 导联组合预设
LEAD_CONFIGS = {
    'all_13':       list(range(13)),  # 全部13导联
    '12_surface':   list(range(1, 13)),  # 12导联体表（去掉EB）
    'EB':           [0],
    'aVF':          [10],
    'aVF_EB':       [0, 10],
    'II_V1':        [2, 4],
    'II_V1_EB':     [0, 2, 4],
    'II_V1_aVF':    [2, 4, 10],
    'II_V1_aVF_EB': [0, 2, 4, 10],
    '12_EB':        list(range(13)),  # 同 all_13
}


def read_hnu_data(split, task, class_name, lead_indices=None):
    """
    读取湖南大学切片格式数据。
    
    Parameters:
        split: 'train' or 'test' -> '训练集' or '测试集'
        task: '3class' or '4class' -> '3分类' or '4分类'
        class_name: 'AVNRT', 'AVRT', 'N', 'AVRT-L', 'AVRT-R'
        lead_indices: 要选择的导联索引列表，None=全部
    
    Returns:
        X: ndarray (n_samples, 5000, n_leads)
        n_samples: int
    """
    split_dir = '训练集' if split == 'train' else '测试集'
    task_dir = '3分类' if task == '3class' else '4分类'
    data_dir = BASE_DIR / '湖南大学' / split_dir / task_dir / class_name
    
    if not data_dir.exists():
        return np.array([]), 0
    
    # 获取排序后的文件列表
    files = sorted([f for f in data_dir.iterdir() if f.suffix.upper() == '.CSV'])
    
    if len(files) == 0:
        return np.array([]), 0
    
    # 读取第一个文件确定样本数
    first_df = pd.read_csv(files[0])
    n_samples = first_df.values.shape[0]
    
    # 读取所有导联数据
    all_lead_data = []
    for f in files:
        try:
            df = pd.read_csv(f)
            data = df.values[:, 3:]  # 跳过前3列（序号等元数据）
            all_lead_data.append(data)
        except Exception:
            pass
    
    if len(all_lead_data) == 0:
        return np.array([]), 0
    
    # 组装: (n_samples, 5000, n_leads)
    ecg_data = np.zeros((n_samples, 5000, len(all_lead_data)))
    for lead_idx, lead_data in enumerate(all_lead_data):
        for i in range(n_samples):
            if i < lead_data.shape[0]:
                length = min(5000, lead_data.shape[1] if len(lead_data.shape) > 1 else len(lead_data[i]))
                ecg_data[i, :length, lead_idx] = lead_data[i, :length] if len(lead_data.shape) > 1 else lead_data[i][:length]
    
    # 选择导联
    if lead_indices is not None:
        ecg_data = ecg_data[:, :, lead_indices]
    
    return ecg_data, n_samples


def read_myfolder_data(split, task, class_name, lead_indices=None):
    """
    读取 my_folder 原始患者文件格式数据。
    
    每个文件是一个患者的完整记录（13列导联,  N行数据），
    按5000点（10秒@500Hz）切分为多个样本。
    
    Parameters:
        split: 'train' or 'test'
        task: '3class' or '4class'
        class_name: 'AVNRT', 'AVRT', 'N', 'AVRT-L', 'AVRT-R'
        lead_indices: 要选择的导联索引列表
    
    Returns:
        X: ndarray (n_samples, 5000, n_leads)
        n_samples: int
        patient_names: list[str]
    """
    data_dir = BASE_DIR / 'my_folder' / split / task / class_name
    
    if not data_dir.exists():
        return np.array([]), 0, []
    
    all_segments = []
    patient_names = []
    
    for f in sorted(data_dir.iterdir()):
        if f.suffix.upper() != '.CSV':
            continue
        
        try:
            # 提取患者姓名
            match = re.match(r'^\d{4}-\d{2}-\d{2}-([a-zA-Z]+)', f.stem)
            p_name = match.group(1).lower() if match else f.stem
            
            # 读取数据
            df = pd.read_csv(f, encoding='gbk',
                           usecols=['Ⅰ', 'Ⅱ', 'Ⅲ', 'aVR', 'aVL', 'aVF', 
                                   'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'EB'])
            
            # 重排列顺序为标准顺序
            df = df[LEAD_ORDER_13]
            data = df.values
            
            # 按5000点切分
            n_points = data.shape[0]
            for start in range(0, n_points, 5000):
                segment = data[start:start+5000]
                if segment.shape[0] == 5000:  # 只保留完整的10秒片段
                    all_segments.append(segment)
                    patient_names.append(p_name)
        except Exception as e:
            print(f"  Warning: Failed to read {f.name}: {e}")
    
    if len(all_segments) == 0:
        return np.array([]), 0, []
    
    X = np.array(all_segments)  # (n_samples, 5000, 13)
    
    # 选择导联
    if lead_indices is not None:
        X = X[:, :, lead_indices]
    
    return X, len(all_segments), patient_names


def load_dataset(split='train', task='3class', lead_config='all_13', 
                 source='both', verbose=True):
    """
    统一数据加载接口。
    
    Parameters:
        split: 'train' or 'test'
        task: '3class' or '4class'
        lead_config: 导联组合名称（见 LEAD_CONFIGS）
        source: 'hnu', 'myfolder', 'both'
        verbose: 是否打印信息
    
    Returns:
        X: ndarray (total_samples, 5000, n_leads)
        y: ndarray (total_samples,) — 整数标签
        patient_ids: list[str] — 每个样本对应的患者标识
    """
    lead_indices = LEAD_CONFIGS.get(lead_config, None)
    labels = LABEL_4CLASS if task == '4class' else LABEL_3CLASS
    
    all_X = []
    all_y = []
    all_patients = []
    
    for class_name, label_id in labels.items():
        class_X_parts = []
        class_count = 0
        
        # 读取湖南大学数据
        if source in ('hnu', 'both'):
            X_hnu, n_hnu = read_hnu_data(split, task, class_name, lead_indices)
            if n_hnu > 0:
                class_X_parts.append(X_hnu)
                class_count += n_hnu
                all_patients.extend([f"hnu_{class_name}_{i}" for i in range(n_hnu)])
        
        # 读取 my_folder 数据
        if source in ('myfolder', 'both'):
            X_mf, n_mf, p_names = read_myfolder_data(split, task, class_name, lead_indices)
            if n_mf > 0:
                class_X_parts.append(X_mf)
                class_count += n_mf
                all_patients.extend(p_names)
        
        if class_count > 0:
            class_X = np.vstack(class_X_parts)
            all_X.append(class_X)
            all_y.extend([label_id] * class_count)
        
        if verbose:
            print(f"  [{split}] {class_name}: {class_count} segments")
    
    if len(all_X) == 0:
        return np.array([]), np.array([]), []
    
    X = np.vstack(all_X)
    y = np.array(all_y)
    
    if verbose:
        n_leads = X.shape[2] if len(X.shape) == 3 else 0
        print(f"  Total: {X.shape[0]} segments, shape={X.shape}, "
              f"leads={n_leads}, classes={len(labels)}")
    
    return X, y, all_patients


def get_class_names(task='3class'):
    """返回类别名称列表（按标签ID排序）。"""
    labels = LABEL_4CLASS if task == '4class' else LABEL_3CLASS
    return [name for name, _ in sorted(labels.items(), key=lambda x: x[1])]


if __name__ == "__main__":
    print("=" * 60)
    print("  Data Pipeline Test")
    print("=" * 60)
    
    # 测试 my_folder 源（这是有患者姓名的数据源）
    for task in ['3class']:
        for split in ['train', 'test']:
            print(f"\n--- {task} / {split} (my_folder only) ---")
            X, y, patients = load_dataset(split=split, task=task, 
                                          lead_config='all_13',
                                          source='myfolder')
            if len(X) > 0:
                print(f"  Shape: {X.shape}, Labels: {np.unique(y, return_counts=True)}")
                unique_patients = set(patients)
                print(f"  Unique patients: {len(unique_patients)}")
