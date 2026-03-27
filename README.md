# Li Shuo Research Project: Esophageal Pacing ECG Analysis for SVT Classification

[English Version](README_EN.md)

[![Code-Only Public Release](https://img.shields.io/badge/Release-Code--Only-success)](#2-scope-of-this-public-repository)
[![Clinical Research](https://img.shields.io/badge/Domain-Clinical%20ECG-blue)](#1-scientific-context)
[![Pipeline](https://img.shields.io/badge/Pipeline-Reproducible-informational)](#6-recommended-execution-order)
[![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)](#5-installation)
[![Privacy](https://img.shields.io/badge/Data-Privacy%20Protected-critical)](#9-data-governance-and-privacy)

A reproducible machine-learning and deep-learning codebase for multi-lead ECG analysis in the context of **esophageal pacing** and supraventricular tachycardia (SVT) phenotype classification.

## 1. Scientific Context
This repository supports a clinical research workflow focused on arrhythmia discrimination from esophageal pacing-related ECG recordings.

### Clinical/Methodological Targets
- Patient-level split verification to prevent data leakage
- Feature-engineered ML baselines (Random Forest, XGBoost)
- Deep-learning experiment scripts (ResNet-based and legacy pipeline)
- Manuscript-facing figure/table regeneration utilities

### Research Tasks
- **3-class classification**: AVNRT / AVRT / N
- **4-class classification**: AVNRT / AVRT-L / AVRT-R / N

## 2. Scope of This Public Repository
This public repository is intentionally **code-first** and **privacy-preserving**.

Included:
- Source code and reproducible pipeline scripts
- Lightweight technical documentation

Excluded:
- Raw ECG datasets
- Patient-identifiable artifacts
- Generated local results and derivative files

This design follows data minimization and publication-readiness principles.

## 3. Repository Layout
```text
.
|-- scripts/
|   |-- data_pipeline.py
|   |-- feature_engineering.py
|   |-- verify_patient_split.py
|   |-- train_rf.py
|   |-- train_xgboost.py
|   |-- paths.py
|   `-- requirements.txt
|-- 源代码/
|   `-- *.py (legacy deep-learning source)
|-- 深度学习模型图片修复/
|   `-- resnet_experiment/
|       `-- *.py (retraining + figure/table regeneration)
|-- doc/
|   `-- REPO_STRUCTURE.md
|-- .gitignore
`-- README.md
```

## 4. End-to-End Pipeline
```mermaid
flowchart LR
    A[Patient-Level Data Sources] --> B[Patient Split Verification]
    B --> C[Unified Data Pipeline]
    C --> D[Feature Engineering 169-D]
    D --> E1[Random Forest Baseline]
    D --> E2[XGBoost Baseline]
    C --> F[ResNet Experiment Scripts]
    E1 --> G[Metrics + Curves + Tables]
    E2 --> G
    F --> G
```

## 5. Methodological Highlights
### A. Patient-level isolation audit
`verify_patient_split.py` audits train/test boundaries at patient level and reports overlap diagnostics.

### B. Feature engineering (per ECG segment)
`feature_engineering.py` extracts **169 features**:
- Time-domain: 10 features x 13 leads
- Frequency-domain: 3 features x 13 leads

### C. Baseline modeling
- `train_rf.py`: Random Forest + grid search + stratified CV
- `train_xgboost.py`: XGBoost + grid search + stratified CV + feature importance

### D. Deep-learning support
- Legacy scripts under `源代码/`
- ResNet retraining and figure/table scripts under `深度学习模型图片修复/resnet_experiment/`

## 6. Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r scripts/requirements.txt
```

## 7. Recommended Execution Order
Run from project root:
```bash
python scripts/verify_patient_split.py
python scripts/feature_engineering.py
python scripts/train_rf.py --fast
python scripts/train_xgboost.py --fast
```

For full hyperparameter search, remove `--fast`.

## 8. Reproducibility Checklist
- Fixed random seeds in baseline training (`random_state=42`)
- Stratified cross-validation in model selection
- Centralized path management in `scripts/paths.py`
- Deterministic pipeline order for reporting

## 9. Data Governance and Privacy
This repository does **not** publish:
- Raw CSV ECG datasets
- Intermediate files containing patient identifiers
- Generated local result artifacts

Any clinical data use must follow local IRB/ethics and institutional data-governance policies.

## 10. Publication-Facing Positioning
This codebase is organized to support:
- Transparent method reporting
- Reviewer-facing baseline comparisons
- Reproducible manuscript figure/table regeneration

## 11. Citation
If you use this repository in academic work, please cite the associated manuscript (to be updated upon publication).

## 12. Acknowledgment
Project lead: **Dr. Li Shuo**

## 13. Contact
For technical issues, open a GitHub issue in this repository.
