#!/usr/bin/env python3
"""
Bayesian hyperparameter tuning for Spanish BERT (T3.5).
Follows the same approach as Yelp's BERT tuning.

Usage:
    python3 Amazon/3_bert_bayesian_tuning_spanish.py --n-trials 100
"""

import argparse
import gc
import inspect
import json
import random
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'optuna'. Install with: pip install optuna"
    ) from exc


class TextClassificationDataset(Dataset):
    def __init__(self, encodings: Dict[str, List[int]], labels: List[int]):
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    f1_weighted = f1_score(labels, predictions, average="weighted", zero_division=0)
    accuracy = accuracy_score(labels, predictions)
    return {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }


def load_split(csv_path: Path) -> Tuple[List[str], List[int]]:
    data = pd.read_csv(csv_path)
    data = data.dropna(subset=["text", "stars"])
    data["text"] = data["text"].astype(str)
    data["label"] = data["stars"].astype(int) - 1  # Convert to 0-4
    return data["text"].tolist(), data["label"].tolist()


def build_dataset(
    tokenizer, texts: List[str], labels: List[int], max_length: int
) -> TextClassificationDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return TextClassificationDataset(encodings=encodings, labels=labels)


def get_eval_strategy_key() -> str:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    if "eval_strategy" in parameters:
        return "eval_strategy"
    raise RuntimeError("Unsupported transformers version")


def objective(trial: optuna.Trial, train_texts, train_labels, val_texts, val_labels, tokenizer, args) -> float:
    """Optuna objective for Spanish BERT tuning."""
    
    # Hyperparameters to tune
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    num_train_epochs = trial.suggest_int("num_train_epochs", 2, 5)
    per_device_batch_size = trial.suggest_categorical("per_device_batch_size", [8, 16, 32])
    gradient_accumulation_steps = trial.suggest_categorical("gradient_accumulation_steps", [1, 2, 4])
    max_length = trial.suggest_categorical("max_length", [128, 256, 512])
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler_type", ["linear", "cosine"])
    dropout = trial.suggest_float("dropout", 0.0, 0.2)
    attention_dropout = trial.suggest_float("attention_dropout", 0.0, 0.2)
    
    # Create datasets
    train_dataset = build_dataset(tokenizer, train_texts, train_labels, max_length)
    val_dataset = build_dataset(tokenizer, val_texts, val_labels, max_length)
    
    # Load model with trial-specific dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=5,
        hidden_dropout_prob=dropout,
        attention_probs_dropout_prob=attention_dropout,
    )
    
    # Training arguments
    trial_dir = args.output_dir / f"trial_{trial.number}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    
    training_kwargs = {
        "output_dir": str(trial_dir),
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_batch_size,
        "per_device_eval_batch_size": per_device_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "warmup_ratio": warmup_ratio,
        "lr_scheduler_type": lr_scheduler_type,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "logging_strategy": "epoch",
        "report_to": "none",
        "seed": args.seed,
        "bf16": True,
        "remove_unused_columns": True,
    }
    training_kwargs[get_eval_strategy_key()] = "epoch"
    
    training_args = TrainingArguments(**training_kwargs)
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train
    trainer.train()
    
    # Evaluate
    metrics = trainer.evaluate(eval_dataset=val_dataset)
    f1_macro = metrics.get("eval_f1_macro", 0.0)
    
    # Cleanup
    del model, trainer
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Remove trial directory to save space
    if trial_dir.exists():
        shutil.rmtree(trial_dir)
    
    return f1_macro


def main():
    parser = argparse.ArgumentParser(
        description="Bayesian tuning for Spanish BERT"
    )
    parser.add_argument("--model-name", type=str, 
                       default="dccuchile/bert-base-spanish-wwm-uncased")
    parser.add_argument("--data-dir", type=Path, default=Path("Amazon"))
    parser.add_argument("--output-dir", type=Path, 
                       default=Path("Amazon/tuning_outputs/bert"))
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_texts, train_labels = load_split(args.data_dir / "train_balanced.csv")
    val_texts, val_labels = load_split(args.data_dir / "validation_balanced.csv")
    
    print(f"Train: {len(train_texts)} samples")
    print(f"Validation: {len(val_texts)} samples")
    
    # Load tokenizer
    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=args.startup_trials
        ),
    )
    
    print(f"\nStarting tuning with {args.n_trials} trials...")
    print(f"Startup trials: {args.startup_trials}")
    
    study.optimize(
        lambda trial: objective(trial, train_texts, train_labels, val_texts, val_labels, tokenizer, args),
        n_trials=args.n_trials,
        catch=(Exception,),
    )
    
    # Save results
    best_params = study.best_params
    best_value = study.best_value
    
    results = {
        "best_params": best_params,
        "best_f1_macro": best_value,
        "n_trials": len(study.trials),
        "completed_trials": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    output_file = args.output_dir / "best_hyperparameters.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Best F1-Macro: {best_value:.4f}")
    print(f"Best hyperparameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"\nSaved to: {output_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
