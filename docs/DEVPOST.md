# SehatRozgar

## Inspiration

Daily-wage and manual workers can face a health event and an income interruption at the same time. The household impact can include both care costs and days without earnings, yet health and financial tools often show only one side.

## What it does

SehatRozgar presents a statistical health-risk pattern from public research data beside deterministic financial exposure, savings runway, and occupational hazard references. Its dashboard turns those separate outputs into scenario comparisons and practical discussion prompts. The demo uses fictional worker profiles only.

## How we built it

The backend is Python and FastAPI. A scikit-learn pipeline compares Logistic Regression, Random Forest, and Gradient Boosting on a fixed stratified holdout. Financial formulas and occupation mappings are deterministic. The React, TypeScript, and Vite frontend calls the backend for assessments, scenarios, metrics, and aggregated synthetic analytics.

## Real-world data

The empirical model uses the UCI Machine Learning Repository Heart Disease dataset, processed Cleveland subset (303 records). Synthetic personas are only for demonstration and are never used as training data.

## Machine learning

Preprocessing is fitted in the training pipeline. The selected candidate is chosen by holdout ROC-AUC and measured metrics are generated during training. A one-feature baseline perturbation view helps show model behavior; it is not a causal explanation. The cohort is small and historical, so this project does not claim clinical validation.

## Financial modeling

The system calculates interruption income loss, event exposure, health-savings use, remaining savings, runway, and emergency-fund gap with deterministic arithmetic. A transparent 0-100 index excludes disease probability and is not a credit score or eligibility tool.

## Featherless AI

When configured, Featherless receives sanitized structured results and returns schema-validated explanations. It cannot calculate scores or modify model outputs. Missing credentials, malformed JSON, timeout, or request failure trigger deterministic local guidance.

## Challenges

The main design challenge was combining health and finance without turning a model indicator into a diagnosis or allowing medical information to influence a credit-like score. We addressed this by separating dimensions and making the underlying calculations visible.

## Accomplishments

We built a locally runnable end-to-end assessment, a real public-data model training path, a connected income-interruption simulator, multilingual explanation modes, and an offline fallback.

## What we learned

A useful explanation layer must preserve the boundaries of the underlying analytics. Showing assumptions and limits matters as much as making the dashboard legible.

## What's next

External validation on appropriate worker-relevant data, calibration and subgroup analysis, reviewed Roman Urdu language quality, and stronger local privacy controls.