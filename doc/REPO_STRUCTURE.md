# Repository Structure (GitHub-Friendly)

## Top-level layout
- `.vscode/` - local editor config (ignored)
- `scripts/` - ML pipeline scripts (feature extraction, RF, XGBoost, split checks)
- `源代码/` - original deep-learning notebooks and companion Python files
- `深度学习模型图片修复/resnet_experiment/` - retraining and figure/table regeneration scripts
- `doc/` - manuscript and review/rebuttal related documents
- `my_folder/` - raw per-patient CSV data (ignored)
- `湖南大学/` - pre-segmented dataset (ignored)
- `results/` - generated metrics/figures/reports by experiment day (ignored)

## Code entry points
- `scripts/data_pipeline.py`
- `scripts/feature_engineering.py`
- `scripts/train_rf.py`
- `scripts/train_xgboost.py`
- `scripts/verify_patient_split.py`

## Notes
- Directory names are intentionally preserved to avoid path breakage in existing scripts/notebooks.
- `.gitignore` excludes large/sensitive data and generated outputs for safer GitHub publishing.
