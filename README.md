# Balanced LightGBM vs MLP for image-region labelling (stand-in for SEM line segmentation)

Motivation: SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation (https://arxiv.org/abs/2603.16548) segments metal lines in SEM images of integrated circuits with a fine-tuned SAM2-based model, evaluated by electrostatic-discharge (ESD) error rates. The SEM images and annotations are not public, so this repository runs a small, reproducible proxy: labelling image regions with classical models on a public image-segmentation dataset.

Question: Can a class-balanced, cross-validation-tuned LightGBM classifier beat a standard MLP at labelling image regions on a public image-segmentation dataset, as a small stand-in for the paper's SEM metal-line segmentation?

Dataset: UCI Image Segmentation data (2,310 hand-labelled 3x3 image regions, 19 numeric features, 7 classes), fetched from OpenML dataset 36. This is not the paper's own data (`same_as_paper: false`).

Baseline: a standardised scikit-learn MLPClassifier with one hidden layer, untuned.

Improved model: a LightGBM classifier with balanced class weights, whose num_leaves, learning_rate and min_child_samples are chosen by 5-fold stratified cross-validation on the training split only.

Metric: test-set macro-F1 (main), with accuracy and log-loss as secondary metrics. Both models are run over seeds 42, 43 and 44.

## How to run

```bash
pip install -r requirements.txt
python analysis.py
```

Requires internet access to openml.org. Runs on CPU in well under a minute.

## Results

Results are written to `results.json` (and a chart to `figures/macro_f1.png`).

## Results (latest run)

On the public UCI Image Segmentation dataset (OpenML 36), a standard scikit-learn MLP classifier reached 97.2% macro-F1 on the held-out test set, while my class-balanced LightGBM with CV-tuned hyperparameters reached 98.1%, a small gain from a single 75/25 split.

| Metric | Value |
|---|---|
| baseline_score | 0.972 |
| improved_score | 0.9812 |
| improvement | 0.009194 |
| baseline_score_std | 0.0008418 |
| improved_score_std | 0.00000579 |
| baseline_accuracy | 0.9717 |
| improved_accuracy | 0.981 |
| baseline_logloss | 0.1347 |
| improved_logloss | 0.097 |
| improved_cv_macro_f1 | 0.985 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in dataset: the paper's SEM images and annotations are not public, so this tests only the general question of a balanced gradient-boosted model vs an MLP on small tabular region labels. Only the LightGBM is tuned (8-point grid, 5-fold CV on the training split); the MLP is the untuned standard baseline, so the comparison is not compute-matched in tuning. Single 75/25 split: test estimates carry split-level uncertainty; '_std' is across model seeds only. Selected LightGBM params: {'learning_rate': 0.05, 'min_child_samples': 5, 'num_leaves': 15}. Log-loss is computed on predicted probabilities without calibration. The paper's ESD error rates are not comparable to macro-F1.

## Reference

This project is motivated by the following paper:

> C Gehrmann, J Ricker, S Damm, D Cheng (2026). *SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation*. arXiv preprint arXiv. https://arxiv.org/abs/2603.16548

Paper author: Prof. Dr. Asja Fischer, Ruhr-Universität Bochum (asja.fischer@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
