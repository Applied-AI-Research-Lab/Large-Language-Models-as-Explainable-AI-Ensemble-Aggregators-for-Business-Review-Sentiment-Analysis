#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for classical models on Spanish Amazon (T3.5).
Matches Yelp LR/4_bayesian_tuning_phase.py functionality.

Usage:
    python3 Amazon/2_bayesian_tuning_spanish.py --model lr --n-trials 100
    python3 Amazon/2_bayesian_tuning_spanish.py --model svm --n-trials 100
    python3 Amazon/2_bayesian_tuning_spanish.py --model nb --n-trials 50
"""

import argparse
import json
import pickle
import random
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def load_data(data_dir: Path):
    """Load train and validation data."""
    train = pd.read_csv(data_dir / "train_balanced.csv")
    val = pd.read_csv(data_dir / "validation_balanced.csv")
    
    # Convert stars (1-5) to labels (0-4)
    train['label'] = train['stars'] - 1
    val['label'] = val['stars'] - 1
    
    return train, val


def objective_lr(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Logistic Regression."""
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    
    model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # Evaluate on validation set (calibration done during final training)
    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    return f1_macro


def objective_svm(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for SVM."""
    C = trial.suggest_float("C", 0.01, 10.0, log=True)
    
    model = LinearSVC(C=C, max_iter=5000, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set (calibration done during final training)
    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    return f1_macro


def objective_nb(trial, X_train, y_train, X_val, y_val):
    """Optuna objective for Naive Bayes."""
    alpha = trial.suggest_float("alpha", 1e-10, 1.0, log=True)
    
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_val)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    return f1_macro


def main():
    parser = argparse.ArgumentParser(description="Bayesian tuning for Spanish Amazon")
    parser.add_argument("--model", choices=["lr", "svm", "nb"], required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("Amazon"))
    parser.add_argument("--output-dir", type=Path, default=Path("Amazon/tuning_outputs"))
    parser.add_argument("--n-trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Load data
    print("Loading data...")
    train, val = load_data(args.data_dir)
    
    # Vectorize
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train['text'])
    X_val = vectorizer.transform(val['text'])
    y_train = train['label'].values
    y_val = val['label'].values
    
    # Run Optuna
    print(f"Running Optuna for {args.model.upper()} ({args.n_trials} trials)...")
    
    if args.model == "lr":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_lr(trial, X_train, y_train, X_val, y_val), 
                      n_trials=args.n_trials)
    elif args.model == "svm":
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_svm(trial, X_train, y_train, X_val, y_val), 
                      n_trials=args.n_trials)
    else:  # nb
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective_nb(trial, X_train, y_train, X_val, y_val), 
                      n_trials=args.n_trials)
    
    # Save results
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        "model": args.model,
        "best_params": study.best_params,
        "best_f1_macro": study.best_value,
        "n_trials": args.n_trials,
    }
    
    output_file = args.output_dir / f"{args.model}_best_params.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save vectorizer
    with open(args.output_dir / "vectorizer.pkl", 'wb') as f:
        pickle.dump(vectorizer, f)
    
    print(f"\nBest F1-macro: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")
    print(f"Saved to {output_file}")


if __name__ == "__main__":
    main()
