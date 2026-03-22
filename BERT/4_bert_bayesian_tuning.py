# python3 BERT/4_bert_bayesian_tuning.py --pipeline balanced --n-trials 200 --startup-trials 40 --mixed-precision bf16 --objective-mode f1_only --pruner median --pruner-startup-trials 40 --pruner-warmup-steps 4 --pruner-min-trials 10 --enable-early-stopping --early-stopping-patience 1 --early-stopping-threshold 0.0005 --max-epochs 6 --weight-decay-max 0.4 --label-smoothing-max 0.25
# python3 BERT/4_bert_bayesian_tuning.py --pipeline natural --n-trials 200 --startup-trials 40 --mixed-precision bf16 --objective-mode f1_only --pruner median --pruner-startup-trials 40 --pruner-warmup-steps 4 --pruner-min-trials 10 --enable-early-stopping --early-stopping-patience 1 --early-stopping-threshold 0.0005 --max-epochs 6 --weight-decay-max 0.4 --label-smoothing-max 0.25

# With pruning (faster, but may prune promising trials early):
# python3 BERT/4_bert_bayesian_tuning.py --pipeline balanced --n-trials 200 --startup-trials 30 --mixed-precision bf16 --objective-mode f1_only --pruner median --pruner-startup-trials 30 --pruner-warmup-steps 3

# No pruning Max exploration (slower):
# python3 4_bert_bayesian_tuning.py --pipeline balanced --n-trials 200 --startup-trials 40 --mixed-precision bf16 --objective-mode f1_only --pruner none

import argparse
import gc
import inspect
import json
import math
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
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

try:
    import optuna
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'optuna'. Install dependencies with: "
        "pip install optuna transformers datasets scikit-learn pandas numpy"
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


class OptunaPruningCallback(TrainerCallback):
    def __init__(self, trial: optuna.trial.Trial, metric_name: str):
        self.trial = trial
        self.metric_name = metric_name
        self._reported_steps: set[int] = set()

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        metric_value = metrics.get(self.metric_name)
        if metric_value is None:
            return control
        step = int(state.global_step)
        if step in self._reported_steps:
            return control
        self.trial.report(float(metric_value), step=step)
        self._reported_steps.add(step)
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Pruned at step={step} with {self.metric_name}={metric_value}")
        return control


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
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


def stars_to_multiclass_labels(stars: pd.Series) -> pd.Series:
    stars_int = stars.round().astype(int)
    if not stars_int.isin([1, 2, 3, 4, 5]).all():
        invalid = stars_int[~stars_int.isin([1, 2, 3, 4, 5])].unique().tolist()
        raise ValueError(f"Found invalid star ratings outside [1,2,3,4,5]: {invalid[:10]}")
    return stars_int - 1


def load_split(
    csv_path: Path,
    text_col: str,
    star_col: str,
    min_text_chars: int,
) -> Tuple[List[str], List[int]]:
    data = pd.read_csv(csv_path, usecols=[text_col, star_col])
    data = data.dropna(subset=[text_col, star_col])
    data[text_col] = data[text_col].astype(str)
    data = data[data[text_col].str.len() >= min_text_chars]
    data[star_col] = pd.to_numeric(data[star_col], errors="coerce")
    data = data.dropna(subset=[star_col])
    data["label"] = stars_to_multiclass_labels(data[star_col]).astype(int)
    texts = data[text_col].tolist()
    labels = data["label"].tolist()
    return texts, labels


def build_dataset(
    tokenizer,
    texts: List[str],
    labels: List[int],
    max_length: int,
) -> TextClassificationDataset:
    encodings = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=max_length,
    )
    return TextClassificationDataset(encodings=encodings, labels=labels)


def resolve_precision_mode(mode: str) -> Dict[str, bool]:
    if mode == "none":
        return {"fp16": False, "bf16": False}
    if mode == "fp16":
        return {"fp16": True, "bf16": False}
    if mode == "bf16":
        return {"fp16": False, "bf16": True}

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return {"fp16": False, "bf16": True}
    if torch.cuda.is_available():
        return {"fp16": True, "bf16": False}
    return {"fp16": False, "bf16": False}


def get_eval_strategy_key() -> str:
    parameters = inspect.signature(TrainingArguments.__init__).parameters
    if "evaluation_strategy" in parameters:
        return "evaluation_strategy"
    if "eval_strategy" in parameters:
        return "eval_strategy"
    raise RuntimeError(
        "Unsupported transformers TrainingArguments signature: "
        "missing both evaluation_strategy and eval_strategy."
    )


def safe_float(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def summarize_best_reason(
    best_record: Dict[str, Any],
    completed_records: List[Dict[str, Any]],
    objective_name: str,
) -> str:
    if not completed_records:
        return "No completed trials available to explain best hyperparameters."

    sorted_records = sorted(
        completed_records,
        key=lambda record: float(record.get("objective_score", float("-inf"))),
        reverse=True,
    )
    second_best = sorted_records[1] if len(sorted_records) > 1 else None

    best_obj = best_record.get("objective_score")
    best_f1 = best_record.get("metrics", {}).get("eval_f1_macro")
    best_error = best_record.get("val_error")

    if second_best is None:
        return (
            f"This configuration is currently best because it achieved {objective_name}={best_obj:.6f}, "
            f"validation macro-F1={best_f1:.6f}, and validation error={best_error:.6f} in the available search."
        )

    second_obj = second_best.get("objective_score")
    second_f1 = second_best.get("metrics", {}).get("eval_f1_macro")
    delta_obj = float(best_obj) - float(second_obj)
    delta_f1 = float(best_f1) - float(second_f1)

    return (
        f"This configuration is best because it maximized {objective_name} ({best_obj:.6f}), "
        f"improving objective by {delta_obj:.6f} and macro-F1 by {delta_f1:.6f} over the second-best trial. "
        f"It also achieved a low validation error ({best_error:.6f}), indicating stronger generalization on the tuning validation split."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bayesian hyperparameter tuning for BERT multiclass Yelp star classification."
    )
    parser.add_argument(
        "--pipeline",
        type=str,
        choices=["balanced", "natural"],
        default="balanced",
        help="Which explicit pipeline tuning datasets to use by default.",
    )
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=None,
        help="Optional override path for training tuning split CSV.",
    )
    parser.add_argument(
        "--validation-csv",
        type=Path,
        default=None,
        help="Optional override path for validation tuning split CSV.",
    )
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--star-col", type=str, default="stars")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override directory for tuning outputs. Defaults to BERT/tuning_outputs/<pipeline>.",
    )
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--startup-trials", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--max-epochs", type=int, default=6)
    parser.add_argument("--weight-decay-max", type=float, default=0.4)
    parser.add_argument("--label-smoothing-max", type=float, default=0.25)
    parser.add_argument(
        "--mixed-precision",
        type=str,
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--objective-mode",
        type=str,
        choices=["f1_only", "gap_penalized"],
        default="f1_only",
        help="Optimization target: maximize raw validation macro-F1 or apply train-vs-val gap penalty.",
    )
    parser.add_argument(
        "--objective-gap-penalty",
        type=float,
        default=0.10,
        help="Penalty coefficient for positive train-vs-validation macro-F1 gap (used when --objective-mode gap_penalized).",
    )
    parser.add_argument(
        "--pruner",
        type=str,
        choices=["median", "none"],
        default="median",
        help="Optuna trial pruner strategy.",
    )
    parser.add_argument(
        "--pruner-startup-trials",
        type=int,
        default=20,
        help="Minimum completed trials before median pruning starts.",
    )
    parser.add_argument(
        "--pruner-warmup-steps",
        type=int,
        default=2,
        help="Minimum evaluation steps before median pruning can prune a trial.",
    )
    parser.add_argument(
        "--pruner-min-trials",
        type=int,
        default=8,
        help="Minimum number of reported trials required at a step before pruning is considered.",
    )
    parser.add_argument(
        "--enable-early-stopping",
        action="store_true",
        help="Enable early stopping during each trial using validation macro-F1.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=1,
        help="Number of evaluation calls with no improvement before early stopping.",
    )
    parser.add_argument(
        "--early-stopping-threshold",
        type=float,
        default=0.0005,
        help="Minimum macro-F1 improvement to qualify as progress for early stopping.",
    )
    args = parser.parse_args()

    default_train_csv = Path(f"Datasets/train_tune_{args.pipeline}.csv")
    default_validation_csv = Path(f"Datasets/validation_tune_{args.pipeline}.csv")
    train_csv = args.train_csv if args.train_csv is not None else default_train_csv
    validation_csv = (
        args.validation_csv if args.validation_csv is not None else default_validation_csv
    )
    output_dir = args.output_dir if args.output_dir is not None else Path(f"BERT/tuning_outputs/{args.pipeline}")

    if not train_csv.exists():
        raise FileNotFoundError(f"Training tuning split not found: {train_csv}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation tuning split not found: {validation_csv}")

    set_seed(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir = output_dir / "best_model"
    trials_dir = output_dir / "trial_artifacts"
    trials_dir.mkdir(parents=True, exist_ok=True)

    json_log_path = output_dir / "hyperparameter_tuning_log.json"

    train_texts, train_labels = load_split(
        csv_path=train_csv,
        text_col=args.text_col,
        star_col=args.star_col,
        min_text_chars=args.min_text_chars,
    )
    val_texts, val_labels = load_split(
        csv_path=validation_csv,
        text_col=args.text_col,
        star_col=args.star_col,
        min_text_chars=args.min_text_chars,
    )

    num_labels = len(sorted(set(train_labels)))
    if num_labels != 5:
        raise ValueError(
            f"Expected 5 labels for Yelp multiclass setup, but found {num_labels}."
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    precision_flags = resolve_precision_mode(args.mixed_precision)

    trial_records: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] = {}

    seeded_trials = [
        {
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.10,
            "num_train_epochs": 3,
            "per_device_batch_size": 16,
            "gradient_accumulation_steps": 2,
            "max_length": 256,
            "lr_scheduler_type": "linear",
            "dropout": 0.10,
            "attention_dropout": 0.10,
        },
        {
            "learning_rate": 3e-5,
            "weight_decay": 0.05,
            "warmup_ratio": 0.08,
            "num_train_epochs": 3,
            "per_device_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "max_length": 256,
            "lr_scheduler_type": "cosine",
            "dropout": 0.10,
            "attention_dropout": 0.10,
        },
        {
            "learning_rate": 1.5e-5,
            "weight_decay": 0.01,
            "warmup_ratio": 0.06,
            "num_train_epochs": 4,
            "per_device_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "max_length": 320,
            "lr_scheduler_type": "linear",
            "dropout": 0.20,
            "attention_dropout": 0.10,
        },
    ]

    run_metadata: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "run_seed": args.seed,
        "pipeline": args.pipeline,
        "model_name": args.model_name,
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "output_dir": str(output_dir),
        "dataset_sizes": {
            "train": len(train_texts),
            "validation": len(val_texts),
        },
        "num_labels": num_labels,
        "n_trials_requested": args.n_trials,
        "startup_trials": args.startup_trials,
        "timeout_seconds": args.timeout_seconds,
        "mixed_precision": args.mixed_precision,
        "resolved_precision": precision_flags,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "objective": (
            "maximize(eval_f1_macro)"
            if args.objective_mode == "f1_only"
            else "maximize(eval_f1_macro - objective_gap_penalty * max(train_eval_f1_macro - eval_f1_macro, 0))"
        ),
        "objective_mode": args.objective_mode,
        "objective_gap_penalty": args.objective_gap_penalty,
        "pruner": args.pruner,
        "pruner_startup_trials": args.pruner_startup_trials,
        "pruner_warmup_steps": args.pruner_warmup_steps,
        "pruner_min_trials": args.pruner_min_trials,
        "enable_early_stopping": args.enable_early_stopping,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "seeded_promising_trials": seeded_trials,
    }

    search_space: Dict[str, Any] = {
        "learning_rate": "loguniform[1e-6, 8e-5]",
        "weight_decay": f"uniform[0.0, {args.weight_decay_max}]",
        "warmup_ratio": "uniform[0.0, 0.2]",
        "num_train_epochs": f"int[2, {args.max_epochs}]",
        "per_device_batch_size": "categorical[8,16,32]",
        "gradient_accumulation_steps": "categorical[1,2,4]",
        "max_length": "categorical[128,192,256,320]",
        "lr_scheduler_type": "categorical[linear,cosine,polynomial]",
        "dropout": "uniform[0.05, 0.30]",
        "attention_dropout": "uniform[0.05, 0.20]",
        "label_smoothing_factor": f"uniform[0.0, {args.label_smoothing_max}]",
    }

    def write_log(study: optuna.study.Study | None = None) -> None:
        completed = [record for record in trial_records if record.get("status") == "completed"]
        pruned = [record for record in trial_records if record.get("status") == "pruned"]
        failed = [record for record in trial_records if record.get("status") == "failed"]

        best_summary = None
        if best_record:
            best_summary = dict(best_record)
            best_summary["explanation"] = summarize_best_reason(
                best_record=best_record,
                completed_records=completed,
                objective_name="objective_score",
            )

        payload = {
            "run_metadata": run_metadata,
            "search_space": search_space,
            "summary": {
                "total_logged_trials": len(trial_records),
                "completed_trials": len(completed),
                "pruned_trials": len(pruned),
                "failed_trials": len(failed),
            },
            "best_trial": best_summary,
            "trials": trial_records,
            "updated_at": utc_now_iso(),
        }

        if study is not None:
            payload["optuna"] = {
                "study_name": study.study_name,
                "direction": study.direction.name,
                "n_trials_total": len(study.trials),
            }

        with json_log_path.open("w", encoding="utf-8") as log_file:
            json.dump(payload, log_file, indent=2)

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal best_record

        trial_seed = args.seed + trial.number
        set_seed(trial_seed)

        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-6, 8e-5, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 0.0, args.weight_decay_max),
            "warmup_ratio": trial.suggest_float("warmup_ratio", 0.0, 0.2),
            "num_train_epochs": trial.suggest_int("num_train_epochs", 2, args.max_epochs),
            "per_device_batch_size": trial.suggest_categorical("per_device_batch_size", [8, 16, 32]),
            "gradient_accumulation_steps": trial.suggest_categorical(
                "gradient_accumulation_steps", [1, 2, 4]
            ),
            "max_length": trial.suggest_categorical("max_length", [128, 192, 256, 320]),
            "lr_scheduler_type": trial.suggest_categorical(
                "lr_scheduler_type", ["linear", "cosine", "polynomial"]
            ),
            "dropout": trial.suggest_float("dropout", 0.05, 0.30),
            "attention_dropout": trial.suggest_float("attention_dropout", 0.05, 0.20),
            "label_smoothing_factor": trial.suggest_float(
                "label_smoothing_factor", 0.0, args.label_smoothing_max
            ),
        }

        trial_start = time.time()
        trial_start_iso = utc_now_iso()

        trial_output_dir = trials_dir / f"trial_{trial.number:04d}"
        trial_output_dir.mkdir(parents=True, exist_ok=True)

        train_dataset = build_dataset(
            tokenizer=tokenizer,
            texts=train_texts,
            labels=train_labels,
            max_length=params["max_length"],
        )
        val_dataset = build_dataset(
            tokenizer=tokenizer,
            texts=val_texts,
            labels=val_labels,
            max_length=params["max_length"],
        )

        model = None
        trainer = None

        try:
            model = AutoModelForSequenceClassification.from_pretrained(
                args.model_name,
                num_labels=num_labels,
                hidden_dropout_prob=params["dropout"],
                attention_probs_dropout_prob=params["attention_dropout"],
            )

            steps_per_epoch = math.ceil(
                len(train_labels)
                / (params["per_device_batch_size"] * params["gradient_accumulation_steps"])
            )
            total_training_steps = max(1, int(steps_per_epoch * params["num_train_epochs"]))
            warmup_steps = int(total_training_steps * params["warmup_ratio"])

            training_kwargs = {
                "output_dir": str(trial_output_dir),
                "learning_rate": params["learning_rate"],
                "per_device_train_batch_size": params["per_device_batch_size"],
                "per_device_eval_batch_size": params["per_device_batch_size"],
                "gradient_accumulation_steps": params["gradient_accumulation_steps"],
                "num_train_epochs": params["num_train_epochs"],
                "weight_decay": params["weight_decay"],
                "warmup_steps": warmup_steps,
                "lr_scheduler_type": params["lr_scheduler_type"],
                "label_smoothing_factor": params["label_smoothing_factor"],
                "save_strategy": "epoch",
                "save_total_limit": 1,
                "logging_strategy": "steps",
                "logging_steps": args.logging_steps,
                "report_to": "none",
                "seed": trial_seed,
                "dataloader_num_workers": args.num_workers,
                "fp16": precision_flags["fp16"],
                "bf16": precision_flags["bf16"],
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_f1_macro",
                "greater_is_better": True,
                "remove_unused_columns": True,
            }
            training_kwargs[get_eval_strategy_key()] = "epoch"
            training_args = TrainingArguments(**training_kwargs)

            trainer_callbacks = [OptunaPruningCallback(trial=trial, metric_name="eval_f1_macro")]
            if args.enable_early_stopping:
                trainer_callbacks.append(
                    EarlyStoppingCallback(
                        early_stopping_patience=args.early_stopping_patience,
                        early_stopping_threshold=args.early_stopping_threshold,
                    )
                )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=trainer_callbacks,
            )

            trainer.train()
            val_metrics = trainer.evaluate(eval_dataset=val_dataset)
            train_eval_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train_eval")

            eval_f1_macro = float(val_metrics.get("eval_f1_macro", 0.0))
            train_eval_f1_macro = float(train_eval_metrics.get("train_eval_f1_macro", 0.0))
            generalization_gap = train_eval_f1_macro - eval_f1_macro
            if args.objective_mode == "f1_only":
                objective_score = eval_f1_macro
            else:
                objective_score = eval_f1_macro - args.objective_gap_penalty * max(generalization_gap, 0.0)

            trial_end = time.time()
            record = {
                "trial_number": trial.number,
                "status": "completed",
                "start_time_utc": trial_start_iso,
                "end_time_utc": utc_now_iso(),
                "duration_seconds": float(trial_end - trial_start),
                "seed": trial_seed,
                "params": params,
                "effective_batch_size": int(
                    params["per_device_batch_size"] * params["gradient_accumulation_steps"]
                ),
                "metrics": {
                    **{k: safe_float(v) for k, v in val_metrics.items()},
                    **{k: safe_float(v) for k, v in train_eval_metrics.items()},
                },
                "val_error": float(1.0 - eval_f1_macro),
                "generalization_gap_f1_macro": float(generalization_gap),
                "objective_score": float(objective_score),
                "gpu_max_memory_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2))
                if torch.cuda.is_available()
                else None,
            }
            trial_records.append(record)

            previous_best = best_record.get("objective_score", float("-inf")) if best_record else float("-inf")
            if objective_score > previous_best:
                if best_model_dir.exists():
                    shutil.rmtree(best_model_dir)
                best_model_dir.mkdir(parents=True, exist_ok=True)

                trainer.save_model(str(best_model_dir))
                tokenizer.save_pretrained(str(best_model_dir))

                best_record = {
                    "trial_number": trial.number,
                    "status": "completed",
                    "objective_score": float(objective_score),
                    "val_error": float(1.0 - eval_f1_macro),
                    "metrics": {
                        "eval_f1_macro": float(eval_f1_macro),
                        "eval_accuracy": safe_float(val_metrics.get("eval_accuracy")),
                        "eval_f1_weighted": safe_float(val_metrics.get("eval_f1_weighted")),
                        "eval_precision_macro": safe_float(val_metrics.get("eval_precision_macro")),
                        "eval_recall_macro": safe_float(val_metrics.get("eval_recall_macro")),
                        "train_eval_f1_macro": float(train_eval_f1_macro),
                        "generalization_gap_f1_macro": float(generalization_gap),
                    },
                    "params": params,
                    "saved_model_dir": str(best_model_dir),
                    "updated_at": utc_now_iso(),
                }

                with (best_model_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as best_file:
                    json.dump(best_record, best_file, indent=2)

            write_log()
            return float(objective_score)

        except optuna.TrialPruned as pruned_error:
            trial_end = time.time()
            trial_records.append(
                {
                    "trial_number": trial.number,
                    "status": "pruned",
                    "start_time_utc": trial_start_iso,
                    "end_time_utc": utc_now_iso(),
                    "duration_seconds": float(trial_end - trial_start),
                    "seed": trial_seed,
                    "params": params,
                    "reason": str(pruned_error),
                }
            )
            write_log()
            raise

        except Exception as error:
            trial_end = time.time()
            trial_records.append(
                {
                    "trial_number": trial.number,
                    "status": "failed",
                    "start_time_utc": trial_start_iso,
                    "end_time_utc": utc_now_iso(),
                    "duration_seconds": float(trial_end - trial_start),
                    "seed": trial_seed,
                    "params": params,
                    "error": str(error),
                }
            )
            write_log()
            raise

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            del trainer
            del model

    sampler = optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials)
    if args.pruner == "none":
        pruner = optuna.pruners.NopPruner()
    else:
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=args.pruner_startup_trials,
            n_warmup_steps=args.pruner_warmup_steps,
            n_min_trials=args.pruner_min_trials,
        )
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)

    for seeded in seeded_trials:
        study.enqueue_trial(seeded)

    write_log(study=study)

    optimize_kwargs: Dict[str, Any] = {
        "func": objective,
        "n_trials": args.n_trials,
        "catch": (RuntimeError, ValueError),
    }
    if args.timeout_seconds > 0:
        optimize_kwargs["timeout"] = args.timeout_seconds

    study.optimize(**optimize_kwargs)

    write_log(study=study)

    final_summary = {
        "finished_at": utc_now_iso(),
        "best_trial_number": best_record.get("trial_number") if best_record else None,
        "best_objective_score": best_record.get("objective_score") if best_record else None,
        "best_val_error": best_record.get("val_error") if best_record else None,
        "best_model_dir": str(best_model_dir) if best_record else None,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_trials_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "n_trials_failed": len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
    }

    with (output_dir / "final_summary.json").open("w", encoding="utf-8") as summary_file:
        json.dump(final_summary, summary_file, indent=2)

    if best_record:
        explanation = summarize_best_reason(
            best_record=best_record,
            completed_records=[
                record for record in trial_records if record.get("status") == "completed"
            ],
            objective_name="objective_score",
        )
        explanation_payload = {
            "best_trial_number": best_record["trial_number"],
            "best_params": best_record["params"],
            "best_metrics": best_record["metrics"],
            "best_val_error": best_record["val_error"],
            "objective_score": best_record["objective_score"],
            "explanation": explanation,
        }
        with (output_dir / "best_hyperparameters_explanation.json").open(
            "w", encoding="utf-8"
        ) as explanation_file:
            json.dump(explanation_payload, explanation_file, indent=2)

    print(f"Tuning log: {json_log_path}")
    print(f"Final summary: {output_dir / 'final_summary.json'}")
    if best_record:
        print(f"Best model saved at: {best_model_dir}")
        print(f"Best hyperparameters explanation: {output_dir / 'best_hyperparameters_explanation.json'}")


if __name__ == "__main__":
    main()
