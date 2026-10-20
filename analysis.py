"""Baseline MLP vs. class-balanced, CV-tuned LightGBM on UCI Image Segmentation (OpenML 36).

Small stand-in for SEM metal-line segmentation: each 3x3 image region is labelled into 7 classes.
"""
import json
import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")  # headless plotting on CI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from lightgbm import LGBMClassifier
from scipy.io import arff
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

SEED = 42
SEEDS = [42, 43, 44]        # both models are randomised -> each is run over these seeds
MAX_ROWS = 200_000          # subsample cap (the dataset itself has only 2,310 rows)
DATA_ID = 36
SOURCE_URL = "https://www.openml.org/d/36"
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def get(url):
    """GET with timeout=60; retry with exponential backoff on 429/5xx and network errors."""
    err = ""
    for attempt in range(5):
        try:
            r = requests.get(url, timeout=60)
        except requests.RequestException as e:
            err = str(e)
        else:
            if r.status_code < 400:
                return r
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} for {url}")  # not retryable
            err = f"HTTP {r.status_code}"
        if attempt < 4:
            log(f"download problem ({err}); retrying in {2 ** attempt}s")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"download failed after retries: {url} ({err})")


def load_data():
    """Fetch OpenML dataset metadata and ARFF file; return features X, string labels y, name."""
    desc = get(f"https://www.openml.org/api/v1/json/data/{DATA_ID}").json()["data_set_description"]
    log(f"OpenML dataset '{desc['name']}' (id {DATA_ID}); downloading ARFF")
    with tempfile.NamedTemporaryFile(suffix=".arff", delete=False) as tmp:
        tmp.write(get(desc["url"]).content)
    data, _ = arff.loadarff(tmp.name)
    os.remove(tmp.name)
    df = pd.DataFrame(data)
    y_raw = df.pop("class").str.decode("utf-8").to_numpy()
    X = df.astype(float).to_numpy()
    return X, y_raw, desc["name"]


def score(model, X, y):
    """Shared evaluation: macro-F1 (main), accuracy and log-loss (secondary)."""
    proba = model.predict_proba(X)
    pred = proba.argmax(axis=1)
    return {"f1": f1_score(y, pred, average="macro"),
            "acc": accuracy_score(y, pred),
            "logloss": log_loss(y, proba, labels=np.arange(proba.shape[1]))}


def summarise(runs):
    """Mean and population std over seeds for each metric."""
    return {k: (float(np.mean([r[k] for r in runs])), float(np.std([r[k] for r in runs])))
            for k in runs[0]}


def main():
    X, y_raw, dname = load_data()
    if len(X) > MAX_ROWS:  # subsample very large data (not triggered for this dataset)
        idx = np.random.RandomState(SEED).choice(len(X), MAX_ROWS, replace=False)
        X, y_raw = X[idx], y_raw[idx]
    y = LabelEncoder().fit_transform(y_raw)
    log(f"{len(X)} rows, {X.shape[1]} features, {len(np.unique(y))} classes")

    # One stratified split with a fixed seed; the test set is only used for final scoring.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    log(f"train={len(X_tr)} test={len(X_te)}")

    # BASELINE: standardised MLP with default-style settings, no tuning
    base_runs = []
    for s in SEEDS:
        m = make_pipeline(StandardScaler(),
                          MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=s))
        m.fit(X_tr, y_tr)
        base_runs.append(score(m, X_te, y_te))
        log(f"baseline MLP seed {s}: test macro-F1={base_runs[-1]['f1']:.4f}")

    # IMPROVED: class-balanced LightGBM; hyper-parameters chosen by 5-fold CV on TRAIN only
    grid = {"num_leaves": [15, 31], "learning_rate": [0.05, 0.1], "min_child_samples": [5, 20]}
    lgbm = LGBMClassifier(n_estimators=300, class_weight="balanced", subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, random_state=SEED, verbose=-1)
    search = GridSearchCV(lgbm, grid, scoring="f1_macro",
                          cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    search.fit(X_tr, y_tr)
    log(f"CV best {search.best_params_} (CV macro-F1={search.best_score_:.4f})")

    imp_runs = []
    for s in SEEDS:
        m = clone(search.estimator).set_params(**search.best_params_, random_state=s)
        m.fit(X_tr, y_tr)
        imp_runs.append(score(m, X_te, y_te))
        log(f"improved LightGBM seed {s}: test macro-F1={imp_runs[-1]['f1']:.4f}")

    b, i = summarise(base_runs), summarise(imp_runs)
    if not all(np.isfinite(v[0]) for v in list(b.values()) + list(i.values())):
        log("non-finite metrics; aborting")
        sys.exit(1)
    bm, im = b["f1"][0], i["f1"][0]

    metrics = {
        "baseline_score": bm, "improved_score": im, "improvement": im - bm,
        "baseline_score_std": b["f1"][1], "improved_score_std": i["f1"][1],
        "baseline_accuracy": b["acc"][0], "improved_accuracy": i["acc"][0],
        "baseline_logloss": b["logloss"][0], "improved_logloss": i["logloss"][0],
        "improved_cv_macro_f1": float(search.best_score_),
    }
    notes = (
        "Stand-in dataset: the paper's SEM images and annotations are not public, so this tests only "
        "the general question of a balanced gradient-boosted model vs an MLP on small tabular region "
        "labels. Only the LightGBM is tuned (8-point grid, 5-fold CV on the training split); the MLP is "
        "the untuned standard baseline, so the comparison is not compute-matched in tuning. "
        "Single 75/25 split: test estimates carry split-level uncertainty; '_std' is across model seeds "
        f"only. Selected LightGBM params: {search.best_params_}. Log-loss is computed on predicted "
        "probabilities without calibration. The paper's ESD error rates are not comparable to macro-F1."
    )
    results = {
        "question": "Can a class-balanced, CV-tuned LightGBM classifier beat a standard MLP at labelling "
                    "image regions on a public image-segmentation dataset, as a stand-in for SEM "
                    "metal-line segmentation?",
        "method": "Single stratified 75/25 split (seed 42); MLP baseline vs class-balanced LightGBM with "
                  "CV-selected hyper-parameters on the training set; test macro-F1 averaged over seeds 42-44.",
        "dataset": {
            "name": f"UCI Image Segmentation (OpenML {DATA_ID}: {dname})",
            "source_url": SOURCE_URL,
            "n_rows": int(len(X)),
            "n_train": int(len(X_tr)),
            "n_test": int(len(X_te)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 (test set)",
        "higher_is_better": True,
        "baseline": "StandardScaler + scikit-learn MLPClassifier (one hidden layer of 100 units), untuned.",
        "improved": "LightGBM with class_weight='balanced'; num_leaves, learning_rate and min_child_samples "
                    "chosen by 5-fold stratified CV on the training split only; refit per seed.",
        "metrics": metrics,
        "notes": notes,
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log("wrote results.json")

    # Optional chart: baseline vs improved macro-F1 (mean +/- std over seeds)
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["MLP (baseline)", "Balanced LightGBM"], [bm, im],
           yerr=[b["f1"][1], i["f1"][1]], capsize=6, color=["grey", "tab:blue"])
    ax.set_ylabel("Test macro-F1 (mean ± std over seeds)")
    ax.set_title("UCI Image Segmentation (stand-in)")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)
    plt.close(fig)
    log("saved figures/macro_f1.png")


if __name__ == "__main__":
    main()
