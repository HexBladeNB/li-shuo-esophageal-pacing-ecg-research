# -*- coding: utf-8 -*-
"""
 Shared path configuration for all scripts.
 Centralizes output directories so results go to the right Day folders.
"""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"
CACHE_DIR = RESULTS_DIR / "_cache"

# Day folders for organized delivery
DAY1_DIR = RESULTS_DIR / "Day1_患者验证_0315"
DAY2_DIR = RESULTS_DIR / "Day2_特征工程_0316"
DAY3_DIR = RESULTS_DIR / "Day3_RF基线_0317"
DAY4_DIR = RESULTS_DIR / "Day4_XGBoost基线_0318"
DAY5_DIR = RESULTS_DIR / "Day5_对比报告_0319"

# Create all directories
for d in [CACHE_DIR, DAY1_DIR, DAY2_DIR, DAY3_DIR, DAY4_DIR, DAY5_DIR]:
    d.mkdir(parents=True, exist_ok=True)
