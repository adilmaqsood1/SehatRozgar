# Data Card

## Source

UCI Machine Learning Repository, [Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease), processed Cleveland subset.

## Composition

303 empirical records with 13 input attributes and the original `num` target. The target is mapped to a binary label: 0 remains 0; values 1-4 become 1. Synthetic worker profiles are separately stored for dashboard demonstrations and are never mixed into model training.

## Features and preprocessing

Features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, and thal. Question-mark entries are parsed as missing. Imputation and encoding are fitted as part of each training pipeline, after the stratified split.

## Missing data and limitations

The source includes missing values, notably among `ca` and `thal`. The dataset is small, historical, and geographically limited; feature codes and labels have their original clinical-study context. It is not a cohort of manual workers and does not establish medical validity for any population.

## Licensing and ethical considerations

See the current UCI dataset page for terms and citation requirements. Preserve attribution. This prototype uses public research data for educational evaluation only, does not store submitted worker assessments, and should not be used with identifiable or real patient data.