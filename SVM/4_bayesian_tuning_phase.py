# python3 SVM/4_bayesian_tuning_phase.py --pipeline balanced --n-trials 250 --startup-trials 25 --c-max 5.0 --class-weight-mode search --output-dir SVM/tuning_outputs/balanced_wider_250
# python3 SVM/4_bayesian_tuning_phase.py --pipeline natural --n-trials 250 --startup-trials 25 --c-max 5.0 --class-weight-mode search --output-dir SVM/tuning_outputs/natural_wider_250
import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def stars_to_labels(stars: pd.Series) -> pd.Series:
    stars_int = stars.round().astype(int)
    if not stars_int.isin([1, 2, 3, 4, 5]).all():
        invalid = stars_int[~stars_int.isin([1, 2, 3, 4, 5])].unique().tolist()
        raise ValueError(f"Invalid star values: {invalid[:10]}")
    return stars_int - 1


def load_split(csv_path: Path, text_col: str, star_col: str, min_text_chars: int) -> Tuple[List[str], List[int]]:
    data = pd.read_csv(csv_path, usecols=[text_col, star_col])
    data = data.dropna(subset=[text_col, star_col])
    data[text_col] = data[text_col].astype(str)
    data = data[data[text_col].str.len() >= min_text_chars]
    data[star_col] = pd.to_numeric(data[star_col], errors="coerce")
    data = data.dropna(subset=[star_col])
    data["label"] = stars_to_labels(data[star_col]).astype(int)
    return data[text_col].tolist(), data["label"].tolist()


def compute_metrics(y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def build_model(params: Dict[str, Any]) -> Pipeline:
    base_svm = LinearSVC(
        C=float(params["C"]),
        class_weight=params["class_weight"],
        max_iter=5000,
        random_state=int(params["seed"]),
    )
    calibrated = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, int(params["ngram_max"])),
                    max_features=int(params["max_features"]),
                    min_df=int(params["min_df"]),
                    max_df=float(params["max_df"]),
                    sublinear_tf=bool(params["sublinear_tf"]),
                    strip_accents="unicode",
                ),
            ),
            ("clf", calibrated),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Bayesian tuning for TF-IDF + Linear SVM.")
    parser.add_argument("--pipeline", choices=["balanced", "natural"], required=True)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--star-col", type=str, default="stars")
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--startup-trials", type=int, default=10)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-min", type=float, default=1e-3)
    parser.add_argument("--c-max", type=float, default=3.0)
    parser.add_argument("--objective-gap-penalty", type=float, default=0.2)
    parser.add_argument(
        "--class-weight-mode",
        choices=["balanced_only", "none_only", "search"],
        default="balanced_only",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.c_min <= 0 or args.c_max <= 0:
        raise ValueError("--c-min and --c-max must be positive.")
    if args.c_min >= args.c_max:
        raise ValueError("--c-min must be smaller than --c-max.")

    set_seed(args.seed)

    train_csv = args.train_csv or Path(f"Datasets/train_tune_{args.pipeline}.csv")
    validation_csv = args.validation_csv or Path(f"Datasets/validation_tune_{args.pipeline}.csv")
    output_dir = args.output_dir or Path(f"SVM/tuning_outputs/{args.pipeline}")
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    if not train_csv.exists():
        raise FileNotFoundError(f"Training split not found: {train_csv}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Validation split not found: {validation_csv}")

    train_texts, train_labels = load_split(train_csv, args.text_col, args.star_col, args.min_text_chars)
    val_texts, val_labels = load_split(validation_csv, args.text_col, args.star_col, args.min_text_chars)

    trial_records: List[Dict[str, Any]] = []
    best_record: Dict[str, Any] = {}

    def objective(trial: optuna.trial.Trial) -> float:
        nonlocal best_record
        trial_start = time.time()

        if args.class_weight_mode == "balanced_only":
            class_weight = "balanced"
        elif args.class_weight_mode == "none_only":
            class_weight = None
        else:
            class_weight = trial.suggest_categorical("class_weight", [None, "balanced"])

        params = {
            "ngram_max": trial.suggest_categorical("ngram_max", [1, 2]),
            "max_features": trial.suggest_categorical("max_features", [30000, 50000, 100000]),
            "min_df": trial.suggest_categorical("min_df", [2, 5, 10]),
            "max_df": trial.suggest_categorical("max_df", [0.90, 0.95, 1.0]),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False]),
            "C": trial.suggest_float("C", args.c_min, args.c_max, log=True),
            "class_weight": class_weight,
            "seed": args.seed + trial.number,
        }

        try:
            model = build_model(params)
            model.fit(train_texts, train_labels)
            val_pred = model.predict(val_texts)
            train_pred = model.predict(train_texts)

            val_metrics = compute_metrics(val_labels, val_pred)
            train_metrics = compute_metrics(train_labels, train_pred)
            gap = train_metrics["f1_macro"] - val_metrics["f1_macro"]
            objective_score = val_metrics["f1_macro"] - args.objective_gap_penalty * max(gap, 0.0)

            record = {
                "trial_number": trial.number,
                "status": "completed",
                "duration_seconds": float(time.time() - trial_start),
                "params": params,
                "metrics": {
                    **{f"eval_{k}": v for k, v in val_metrics.items()},
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                },
                "generalization_gap_f1_macro": float(gap),
                "objective_score": float(objective_score),
                "val_error": float(1.0 - val_metrics["f1_macro"]),
            }
            trial_records.append(record)

            if (not best_record) or (objective_score > best_record["objective_score"]):
                best_record = {
                    "trial_number": trial.number,
                    "params": params,
                    "metrics": {
                        "eval_f1_macro": val_metrics["f1_macro"],
                        "eval_accuracy": val_metrics["accuracy"],
                        "eval_f1_weighted": val_metrics["f1_weighted"],
                        "eval_precision_macro": val_metrics["precision_macro"],
                        "eval_recall_macro": val_metrics["recall_macro"],
                        "train_f1_macro": train_metrics["f1_macro"],
                        "generalization_gap_f1_macro": float(gap),
                    },
                    "objective_score": float(objective_score),
                    "val_error": float(1.0 - val_metrics["f1_macro"]),
                }
                with (best_model_dir / "best_hyperparameters.json").open("w", encoding="utf-8") as f:
                    json.dump(best_record, f, indent=2)

            return float(objective_score)

        except Exception as exc:
            trial_records.append(
                {
                    "trial_number": trial.number,
                    "status": "failed",
                    "duration_seconds": float(time.time() - trial_start),
                    "params": params,
                    "error": str(exc),
                }
            )
            raise

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, n_startup_trials=args.startup_trials),
    )
    optimize_kwargs: Dict[str, Any] = {"func": objective, "n_trials": args.n_trials, "catch": (Exception,)}
    if args.timeout_seconds > 0:
        optimize_kwargs["timeout"] = args.timeout_seconds
    study.optimize(**optimize_kwargs)

    payload = {
        "run_metadata": {
            "pipeline": args.pipeline,
            "train_csv": str(train_csv),
            "validation_csv": str(validation_csv),
            "output_dir": str(output_dir),
            "n_trials_requested": args.n_trials,
            "startup_trials": args.startup_trials,
            "seed": args.seed,
            "c_min": args.c_min,
            "c_max": args.c_max,
            "objective_gap_penalty": args.objective_gap_penalty,
            "class_weight_mode": args.class_weight_mode,
        },
        "summary": {
            "total_trials": len(trial_records),
            "completed_trials": len([r for r in trial_records if r.get("status") == "completed"]),
            "failed_trials": len([r for r in trial_records if r.get("status") == "failed"]),
        },
        "best_trial": best_record,
        "trials": trial_records,
    }

    with (output_dir / "hyperparameter_tuning_log.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    with (output_dir / "final_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "best_trial_number": best_record.get("trial_number"),
                "best_objective_score": best_record.get("objective_score"),
                "best_val_error": best_record.get("val_error"),
                "best_hyperparameters_path": str(best_model_dir / "best_hyperparameters.json"),
            },
            f,
            indent=2,
        )

    print(f"Saved tuning artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
