# Video Engagement Prediction

## Objective
Predict engagement outcomes for educational videos from metadata and content-derived features.

## Method
- Feature engineering (including aggregate lexical and speaking features).
- Model comparison: logistic regression, random forest, gradient boosting.
- Stacking ensemble with cross-validation.

## Repository Structure
- `notebooks/` experiment notebooks.
- `src/` reusable code modules.
- `results/` metrics and plots.
- `assets/` README figures.
- `models/` saved models (optional).
- `data/` local dataset directory (not versioned).

## Data Access
Use local lecture-feature data under `data/`.

## Run
1. Put `lectures_dataset.csv` under `data/`.
2. Run `notebooks/video_engagement_prediction.ipynb`.
3. Export model-comparison outputs to `results/`.

## Result Artifacts
- LR/RF/GB comparison table
- Stacking vs single-model comparison
- PR/ROC curves

## Validated Baseline Run
- binary label threshold (median of `median_engagement`): `0.0611`
- `accuracy`: `0.7732`
- `f1`: `0.7749`
- `roc_auc`: `0.8561`
- `pr_auc`: `0.8696`
- metrics file: `results/metrics_video_baseline.json`
- model file: `results/artifacts/video_logreg.joblib`
