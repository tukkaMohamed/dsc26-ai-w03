# Tawseel – Baseline Results

## Evaluation setup

- Dataset: `make_parcels(n=4000, seed=0)`
- Split: Chronological split, last 20% used for testing.
- History features: Calculated using strictly earlier parcels.
- Model: Logistic Regression with preprocessing inside a scikit-learn Pipeline.
- Baseline: DummyClassifier with `strategy="prior"`.

## Results

| Model | ROC-AUC | PR-AUC | Brier Score | Base Rate |
|---|---:|---:|---:|---:|
| Dummy baseline | 0.5000 | 0.2200 | 0.1717 | 0.2200 |
| Logistic Regression | 0.6476 | 0.3803 | 0.1632 | 0.2200 |

**PR-AUC lift over dummy:** 0.1603

## Three sentences for Sherif

1. I built a Logistic Regression baseline using only information available when a parcel is created, together with point-in-time customer history.
2. I evaluated it on the last 20% of parcels by time and compared it against a prior-probability dummy baseline.
3. The model achieved a PR-AUC of 0.3803 compared to 0.2200 for the dummy baseline, with a PR-AUC lift of 0.1603 and a lower Brier score.
