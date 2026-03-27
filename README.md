# Li Shuo Research Project: Esophageal Pacing ECG Analysis for SVT Classification

A reproducible machine-learning and deep-learning codebase for multi-lead ECG analysis in the context of **esophageal pacing** and supraventricular tachycardia (SVT) phenotype classification.

## 1. Scientific Context
This repository supports a clinical research workflow focused on arrhythmia discrimination from esophageal pacing-related ECG recordings. The project includes:
- Patient-level split verification to prevent leakage
- Feature-engineered machine-learning baselines (Random Forest, XGBoost)
- Deep-learning experimentation scripts (ResNet-based and legacy notebook pipelines)
- Figure/table regeneration utilities for manuscript and rebuttal workflows

## 2. Scope of This Public Repository
This public repository is intentionally **code-first**:
- Included: source code, reproducible pipeline scripts, project documentation
- Excluded: raw ECG data, patient-identifiable artifacts, generated result files

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

## 4. Methodological Pipeline
### Step A: Patient-level isolation check
`verify_patient_split.py` audits train/test boundaries at patient level and reports overlap diagnostics.

### Step B: Feature engineering
`feature_engineering.py` extracts **169 features per segment**:
- Time-domain: 10 features x 13 leads
- Frequency-domain: 3 features x 13 leads

### Step C: ML baselines
- `train_rf.py`: Random Forest with grid search and stratified CV
- `train_xgboost.py`: XGBoost with grid search, stratified CV, and feature-importance outputs

### Step D: Deep-learning support
- Legacy code under `源代码/`
- ResNet experiment and figure/table utilities under `深度学习模型图片修复/resnet_experiment/`

## 5. Installation
```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r scripts/requirements.txt
```

## 6. Typical Execution Order
Run from project root:
```bash
python scripts/verify_patient_split.py
python scripts/feature_engineering.py
python scripts/train_rf.py --fast
python scripts/train_xgboost.py --fast
```

Use full grid search by removing `--fast`.

## 7. Reproducibility Notes
- Most training scripts use fixed random seed (`random_state=42`)
- Stratified cross-validation is applied in baseline model selection
- Paths are centralized in `scripts/paths.py`

## 8. Data Governance and Privacy
This repository does **not** publish:
- Raw CSV ECG datasets
- Intermediate files containing patient identifiers
- Generated local results and figures

Any clinical data use must follow local IRB/ethics and hospital data-governance policies.

## 9. Publication-Facing Positioning
The codebase is structured to support:
- Transparent method reporting
- Reviewer-facing ablation/baseline comparisons
- Reproducible manuscript figure and table regeneration

## 10. Citation
If you use this repository in academic work, please cite the associated manuscript (details to be added upon publication).

## 11. Contact
Project lead: **Dr. Li Shuo**

For technical issues, open a GitHub issue in this repository.
