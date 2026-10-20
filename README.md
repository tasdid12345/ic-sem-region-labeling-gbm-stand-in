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

## Reference

This project is motivated by the following paper:

> C Gehrmann, J Ricker, S Damm, D Cheng (2026). *SAMSEM--A Generic and Scalable Approach for IC Metal Line Segmentation*. arXiv preprint arXiv. https://arxiv.org/abs/2603.16548

Paper author: Prof. Dr. Asja Fischer, Ruhr-Universität Bochum (asja.fischer@rub.de)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
