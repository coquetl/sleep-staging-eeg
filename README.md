# Automated sleep staging from EEG

Classifying 30-second epochs of a night's sleep into the five sleep stages, from the raw signals of a Dreem headband.

Dreem Data Challenge 2023 — CentraleSupélec.
Léa Coquet and Nicolas Criscuolo.

## The problem

The headband records five EEG channels at 250 Hz and a three-axis accelerometer at 50 Hz. Each 30-second epoch has to be labelled **Wake, N1, N2, N3 or REM**, the five stages a sleep specialist assigns by hand.

The classes are strongly imbalanced — N2 dominates a normal night, N1 is rare — so the challenge is scored on **macro F1**, which weighs every stage equally.

| | |
|---|---|
| Records | 7 nights |
| Epochs | 6,175 |
| Raw dimensions per epoch | 42,000 |
| Metric | Macro F1 |

## Approach

**Preprocessing.** A Butterworth band-pass filter and several scalers (MinMax, Standard, applied before and after feature extraction) were tested and dropped — none improved validation scores.

**Feature extraction** is where most of the work went. For each epoch:

- Descriptive statistics: mean, standard deviation, skewness, kurtosis, quantiles, interquartile range
- Spectral entropy
- Zero-crossing rate
- Fractal dimensions: Higuchi, Katz, Petrosian
- Hjorth parameters: mobility and complexity
- General acceleration from the accelerometer

Each epoch is split into three sub-segments, with features computed on each one **and on its difference with the previous segment**, to capture dynamics within the epoch. Rolling means over 8 epochs — one window into the past, one into the future — add the temporal context a scorer uses: a sleep stage depends on what came before and after it.

This yields **3,390 features**.

**Feature selection.** Zero-variance features are dropped, then collinearity is reduced by hierarchical clustering at a 0.95 threshold, leaving **2,532 features**. PCA was deliberately avoided to keep the features interpretable.

**Class imbalance.** SMOTE oversamples the minority stages.

**Models.** A `StandardScaler` + estimator pipeline, tuned by grid search with 5-fold cross-validation.

## Results

| Model | Macro F1 | Accuracy |
|---|---|---|
| Logistic regression | 0.71 | 0.87 |
| **XGBoost** (300 trees, lr 0.1) | **0.78** | **0.91** |

Per-class F1 for the XGBoost model:

| Stage | F1 | Support |
|---|---|---|
| Wake | 0.68 | 38 |
| N1 | 0.36 | 25 |
| N2 | 0.94 | 234 |
| N3 | 0.96 | 140 |
| REM | 0.95 | 181 |

## What the errors say

The model is strong on N3, N2 and REM, and weak on **N1** — which is expected rather than disappointing. N1 is the brief transition between wakefulness and N2: it is rare, short, and the stage on which human scorers themselves agree least. The confusion matrix shows most N1 epochs absorbed into N2 and REM.

## What I would do differently

- **Split by subject, not at random.** With seven recordings and a plain `train_test_split`, epochs from the same sleeper land in both train and test, so the reported score is optimistic. A leave-one-record-out split would measure generalisation to a new person.
- **Model the sequence.** The rolling windows are a hand-built proxy for temporal structure. An LSTM or a CRF over the epoch sequence would learn the transitions between stages directly, which is where N1 lives.
- **Watch SMOTE.** Oversampling the minority stages helps the macro F1 but risks overfitting to synthetic N1 epochs.

## Files

- `sleep_staging.ipynb` — the full pipeline, from loading the raw `.npy` records to the submission file
- `report.pdf` — the same work as a written report

## References

1. Aboalayon, Faezipour, Almuhammadi, Moslehpour (2016). *Sleep Stage Classification Using EEG Signal Analysis: A Comprehensive Survey and New Investigation.* Entropy 18(9).
2. He, Cai, Niyogi (2005). *Laplacian Score for Feature Selection.*
3. Esteller, Vachtsevanos, Echauz, Litt (2001). *A Comparison of Waveform Fractal Dimension Algorithms.* IEEE TCAS-I 48(2), 177–183.
4. Chen, Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD, 785–794.
