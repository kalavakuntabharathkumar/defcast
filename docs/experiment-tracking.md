# Experiment tracking checklist

For each run, record dataset version/hash, train/validation/test split strategy, class distribution, preprocessing version, model/configuration, random seed, precision/recall/F1 and confusion matrix, subgroup results, calibration, latency, and artifact checksum. Keep a final holdout untouched during tuning. For imbalanced defect detection, prioritize defect-class recall alongside false-positive rate and production review costs. The bundled toy data is only for code smoke tests.
