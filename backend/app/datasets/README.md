# UCI Heart Disease Dataset

- **Dataset:** Heart Disease (processed Cleveland subset)
- **Publisher:** UCI Machine Learning Repository
- **Source:** https://archive.ics.uci.edu/dataset/45/heart+disease
- **Observations:** 303 records in the Cleveland subset
- **Features:** 13 clinical/demographic attributes; see `app/ml/train.py` for names
- **Target:** `num`, binarized as 0 (no angiographic disease) versus 1-4 (presence)
- **Missing values:** `ca` and `thal` include `?` values, parsed as missing and imputed inside the training pipeline
- **Preprocessing:** Numeric imputation and standardization for logistic regression; missing-value imputation and one-hot encoding for categorical attributes; stratified train/test split with a fixed seed
- **Limitations:** Small, historical, geographically limited dataset. This is a research demonstration, not a clinical model. Labels and cohort do not represent all workers or populations.
- **Intended use:** Reproducible educational evaluation only. Not for diagnosis, treatment, employment, lending, insurance, or eligibility decisions.

Run `python scripts/download_dataset.py` from `backend/`, then `python -m app.ml.train` to fetch and train. The downloader obtains the official UCI ZIP; no synthetic rows are used for model training.