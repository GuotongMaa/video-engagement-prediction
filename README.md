# Video Engagement Prediction

Model comparison and stacking ensemble for lecture engagement prediction from metadata and transcript features.

## Problem
Predict engagement outcomes of educational videos from structured features.

## Approach
Feature engineering, model comparison, and stacking ensemble evaluation.

## Highlights
- Feature engineering including lexical complexity and speaking intensity
- Model comparison: logistic regression, random forest, gradient boosting
- Stacking ensemble with 5-fold evaluation (CV-aligned)

## Data
Video lecture features (`data/lectures_dataset.csv`).

## Project Structure
- notebooks/ - Main workflow notebooks
- data/ - Datasets (as noted above)
- models/ - Model checkpoints (optional)
- results/ - Metrics, plots, and outputs
- assets/ - Figures for README

## How to Run
- Open `notebooks/video_engagement_prediction.ipynb`
- Run all cells to reproduce metrics

## Status
Metrics reported in the CV are from prior runs; rerun the notebook to reproduce them.

## Results Showcase
- Recommended outputs in `results/`: model comparison chart (LR/RF/GB), stacking vs single-model comparison, and PR curve.
- Add final figures to `assets/` and link them in this section after reruns.
