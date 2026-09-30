# Model Card

## Purpose

Demonstrate a reproducible statistical health-risk pattern from public data as one separate input to a worker-resilience prototype. It is not intended to estimate an individual worker's clinical risk.

## Dataset and features

UCI Machine Learning Repository Heart Disease dataset, processed Cleveland subset, 303 rows and 13 source features. The source target `num` is mapped to 0 versus 1-4. Source reference: https://archive.ics.uci.edu/dataset/45/heart+disease

## Training methodology

The training pipeline compares Logistic Regression, Random Forest, and Gradient Boosting. A fixed seed of 42 and stratified 75/25 holdout split are used. Missing numeric values use training-fold medians; categorical values use training-fold most-frequent imputation and one-hot encoding. The candidate with the highest holdout ROC-AUC is selected. Current generated metrics are in `backend/app/ml/model_metadata.json`.

## Intended usage

Educational hackathon demonstration with fictional profiles and visible limitations. Inputs use the original dataset feature definitions and codes.

## Out-of-scope usage

Diagnosis, treatment, screening decisions, workplace fitness, employment, lending, insurance, eligibility, or deployment on real patients.

## Evaluation and limitations

Metrics are from a single small held-out split and should not be interpreted as clinical validation, generalization, calibration, or worker-population performance. The dataset is historical and geographically limited. It does not represent occupational exposures or daily-wage workers specifically.

## Bias and safety

Performance can vary across populations and measurement settings; this prototype does not conduct subgroup fairness analysis. Predictions are labeled statistical patterns, never diagnoses. Feature perturbation reflects model response and is not causal.