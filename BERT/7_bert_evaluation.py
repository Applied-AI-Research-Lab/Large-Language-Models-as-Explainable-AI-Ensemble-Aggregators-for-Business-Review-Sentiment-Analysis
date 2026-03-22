# python3 BERT/7_bert_evaluation.py --predictions-dir Predictions --tuning-base-dir BERT/tuning_outputs --output-dir BERT/evaluation_outputs

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    precision_recall_fscore_support,
)


EXPECTED_PREDICTION_FILES = [
    "preds_balanced_on_balanced.csv",
    "preds_natural_on_natural.csv",
    "preds_balanced_on_natural.csv",
    "preds_natural_on_balanced.csv",
]


def parse_run_name(filename: str) -> Tuple[str, str]:
    name = filename.replace(".csv", "")
    parts = name.split("_")
    if len(parts) >= 5 and parts[0] == "preds" and parts[2] == "on":
        return parts[1], parts[3]
    return "unknown", "unknown"


def safe_num(value) -> float:
    if value is None:
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_prediction_file(path: Path) -> Tuple[Dict, pd.DataFrame]:
    df = pd.read_csv(path)
    required_cols = {"stars", "pred_stars"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")

    working = df.copy()
    working["stars"] = pd.to_numeric(working["stars"], errors="coerce")
    working["pred_stars"] = pd.to_numeric(working["pred_stars"], errors="coerce")
    working = working.dropna(subset=["stars", "pred_stars"])
    working["stars"] = working["stars"].round().astype(int)
    working["pred_stars"] = working["pred_stars"].round().astype(int)
    working = working[working["stars"].between(1, 5) & working["pred_stars"].between(1, 5)]

    y_true = working["stars"].tolist()
    y_pred = working["pred_stars"].tolist()

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(float(np.mean((np.array(y_true) - np.array(y_pred)) ** 2)))
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    labels = [1, 2, 3, 4, 5]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    row_sums = cm.sum(axis=1)
    per_class_recall = [
        float(cm[idx, idx] / row_sums[idx]) if row_sums[idx] > 0 else 0.0
        for idx in range(len(labels))
    ]

    model_pipeline, test_pipeline = parse_run_name(path.name)
    summary = {
        "file": path.name,
        "model_pipeline": model_pipeline,
        "test_pipeline": test_pipeline,
        "n_samples": int(len(working)),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(rmse),
        "recall_star_1": per_class_recall[0],
        "recall_star_2": per_class_recall[1],
        "recall_star_3": per_class_recall[2],
        "recall_star_4": per_class_recall[3],
        "recall_star_5": per_class_recall[4],
    }

    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in labels], columns=[f"pred_{i}" for i in labels])
    return summary, cm_df


def latex_wrap_table(df: pd.DataFrame, caption: str, label: str, float_fmt: str = "%.4f") -> str:
    return df.to_latex(index=False, caption=caption, label=label, float_format=float_fmt)


def parse_tuning_log(log_path: Path) -> Tuple[pd.DataFrame, Dict]:
    with log_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    trials = payload.get("trials", [])
    run_meta = payload.get("run_metadata", {})
    pipeline = run_meta.get("pipeline", log_path.parent.name)

    rows: List[Dict] = []
    for trial in trials:
        params = trial.get("params", {})
        metrics = trial.get("metrics", {})
        rows.append(
            {
                "pipeline": pipeline,
                "trial_number": trial.get("trial_number"),
                "status": trial.get("status"),
                "duration_seconds": safe_num(trial.get("duration_seconds")),
                "objective_score": safe_num(trial.get("objective_score")),
                "val_error": safe_num(trial.get("val_error")),
                "eval_f1_macro": safe_num(metrics.get("eval_f1_macro")),
                "eval_accuracy": safe_num(metrics.get("eval_accuracy")),
                "eval_f1_weighted": safe_num(metrics.get("eval_f1_weighted")),
                "train_eval_f1_macro": safe_num(metrics.get("train_eval_f1_macro")),
                "generalization_gap_f1_macro": safe_num(trial.get("generalization_gap_f1_macro")),
                "learning_rate": safe_num(params.get("learning_rate")),
                "weight_decay": safe_num(params.get("weight_decay")),
                "warmup_ratio": safe_num(params.get("warmup_ratio")),
                "num_train_epochs": safe_num(params.get("num_train_epochs")),
                "per_device_batch_size": safe_num(params.get("per_device_batch_size")),
                "gradient_accumulation_steps": safe_num(params.get("gradient_accumulation_steps")),
                "max_length": safe_num(params.get("max_length")),
                "lr_scheduler_type": params.get("lr_scheduler_type"),
                "dropout": safe_num(params.get("dropout")),
                "attention_dropout": safe_num(params.get("attention_dropout")),
            }
        )

    trial_df = pd.DataFrame(rows)
    return trial_df, payload


def create_tuning_summary(trial_df: pd.DataFrame, payload: Dict) -> Dict:
    run_meta = payload.get("run_metadata", {})
    best_trial = payload.get("best_trial", {}) or {}
    pipeline = run_meta.get("pipeline", "unknown")

    completed_df = trial_df[trial_df["status"] == "completed"] if not trial_df.empty else pd.DataFrame()
    total_duration = float(trial_df["duration_seconds"].fillna(0).sum()) if not trial_df.empty else 0.0

    return {
        "pipeline": pipeline,
        "n_trials_logged": int(len(trial_df)),
        "n_completed": int((trial_df["status"] == "completed").sum()) if not trial_df.empty else 0,
        "n_pruned": int((trial_df["status"] == "pruned").sum()) if not trial_df.empty else 0,
        "n_failed": int((trial_df["status"] == "failed").sum()) if not trial_df.empty else 0,
        "total_time_hours": total_duration / 3600.0,
        "mean_trial_minutes": (float(trial_df["duration_seconds"].mean()) / 60.0) if not trial_df.empty else 0.0,
        "best_trial_number": best_trial.get("trial_number"),
        "best_objective_score": safe_num(best_trial.get("objective_score")),
        "best_eval_f1_macro": safe_num((best_trial.get("metrics") or {}).get("eval_f1_macro")),
        "best_eval_accuracy": safe_num((best_trial.get("metrics") or {}).get("eval_accuracy")),
        "best_val_error": safe_num(best_trial.get("val_error")),
        "train_csv": run_meta.get("train_csv"),
        "validation_csv": run_meta.get("validation_csv"),
        "model_name": run_meta.get("model_name"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate BERT prediction outputs and Bayesian tuning logs, then export LaTeX tables."
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("Predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("BERT/evaluation_outputs"))
    parser.add_argument("--tuning-base-dir", type=Path, default=Path("BERT/tuning_outputs"))
    parser.add_argument("--top-k-trials", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    prediction_summaries: List[Dict] = []
    confusion_files: List[Path] = []

    for filename in EXPECTED_PREDICTION_FILES:
        pred_path = args.predictions_dir / filename
        if not pred_path.exists():
            print(f"Warning: prediction file not found, skipping: {pred_path}")
            continue

        summary, cm_df = evaluate_prediction_file(pred_path)
        prediction_summaries.append(summary)
        cm_path = args.output_dir / f"confusion_{filename.replace('.csv', '.tex')}"
        cm_latex = cm_df.reset_index().rename(columns={"index": "true_label"}).to_latex(index=False)
        cm_path.write_text(cm_latex, encoding="utf-8")
        confusion_files.append(cm_path)

    if prediction_summaries:
        pred_df = pd.DataFrame(prediction_summaries)
        pred_metrics_cols = [
            "file",
            "model_pipeline",
            "test_pipeline",
            "n_samples",
            "accuracy",
            "f1_macro",
            "f1_weighted",
            "precision_macro",
            "recall_macro",
            "qwk",
            "mae",
            "rmse",
        ]
        pred_recall_cols = [
            "file",
            "recall_star_1",
            "recall_star_2",
            "recall_star_3",
            "recall_star_4",
            "recall_star_5",
        ]

        metrics_table = pred_df[pred_metrics_cols].sort_values(["test_pipeline", "model_pipeline"])
        recalls_table = pred_df[pred_recall_cols].sort_values("file")

        (args.output_dir / "predictions_metrics.tex").write_text(
            latex_wrap_table(
                metrics_table,
                caption="Prediction performance across in-domain and cross-pipeline evaluation files.",
                label="tab:prediction_metrics",
            ),
            encoding="utf-8",
        )
        (args.output_dir / "predictions_per_class_recall.tex").write_text(
            latex_wrap_table(
                recalls_table,
                caption="Per-class recall by prediction file.",
                label="tab:prediction_per_class_recall",
            ),
            encoding="utf-8",
        )
        pred_df.to_csv(args.output_dir / "predictions_metrics.csv", index=False)
    else:
        print("Warning: no prediction files were evaluated.")

    tuning_logs = [
        args.tuning_base_dir / "balanced" / "hyperparameter_tuning_log.json",
        args.tuning_base_dir / "natural" / "hyperparameter_tuning_log.json",
    ]

    all_trial_rows: List[pd.DataFrame] = []
    tuning_summaries: List[Dict] = []
    best_rows: List[Dict] = []
    top_trials_tables: List[pd.DataFrame] = []

    for log_path in tuning_logs:
        if not log_path.exists():
            print(f"Warning: tuning log not found, skipping: {log_path}")
            continue

        trial_df, payload = parse_tuning_log(log_path)
        all_trial_rows.append(trial_df)
        tuning_summaries.append(create_tuning_summary(trial_df, payload))

        best_trial = payload.get("best_trial", {}) or {}
        params = best_trial.get("params", {}) or {}
        metrics = best_trial.get("metrics", {}) or {}
        pipeline = (payload.get("run_metadata") or {}).get("pipeline", log_path.parent.name)

        best_rows.append(
            {
                "pipeline": pipeline,
                "trial_number": best_trial.get("trial_number"),
                "objective_score": safe_num(best_trial.get("objective_score")),
                "eval_f1_macro": safe_num(metrics.get("eval_f1_macro")),
                "eval_accuracy": safe_num(metrics.get("eval_accuracy")),
                "val_error": safe_num(best_trial.get("val_error")),
                "learning_rate": safe_num(params.get("learning_rate")),
                "weight_decay": safe_num(params.get("weight_decay")),
                "warmup_ratio": safe_num(params.get("warmup_ratio")),
                "num_train_epochs": safe_num(params.get("num_train_epochs")),
                "per_device_batch_size": safe_num(params.get("per_device_batch_size")),
                "gradient_accumulation_steps": safe_num(params.get("gradient_accumulation_steps")),
                "max_length": safe_num(params.get("max_length")),
                "lr_scheduler_type": params.get("lr_scheduler_type"),
                "dropout": safe_num(params.get("dropout")),
                "attention_dropout": safe_num(params.get("attention_dropout")),
            }
        )

        completed_trials = trial_df[trial_df["status"] == "completed"].copy()
        if not completed_trials.empty:
            completed_trials = completed_trials.sort_values("objective_score", ascending=False)
            completed_trials.insert(1, "rank", range(1, len(completed_trials) + 1))
            top_trials_tables.append(
                completed_trials[
                    [
                        "pipeline",
                        "rank",
                        "trial_number",
                        "objective_score",
                        "eval_f1_macro",
                        "eval_accuracy",
                        "val_error",
                        "generalization_gap_f1_macro",
                        "duration_seconds",
                        "learning_rate",
                        "weight_decay",
                        "warmup_ratio",
                        "num_train_epochs",
                        "per_device_batch_size",
                        "gradient_accumulation_steps",
                        "max_length",
                        "lr_scheduler_type",
                        "dropout",
                        "attention_dropout",
                    ]
                ].head(args.top_k_trials)
            )

    if tuning_summaries:
        tuning_summary_df = pd.DataFrame(tuning_summaries).sort_values("pipeline")
        tuning_best_df = pd.DataFrame(best_rows).sort_values("pipeline")

        (args.output_dir / "tuning_summary.tex").write_text(
            latex_wrap_table(
                tuning_summary_df,
                caption="Bayesian tuning run-level summary per pipeline.",
                label="tab:tuning_summary",
            ),
            encoding="utf-8",
        )
        (args.output_dir / "tuning_best_hyperparameters.tex").write_text(
            latex_wrap_table(
                tuning_best_df,
                caption="Best hyperparameters and validation metrics per pipeline.",
                label="tab:tuning_best_hyperparameters",
            ),
            encoding="utf-8",
        )

        tuning_summary_df.to_csv(args.output_dir / "tuning_summary.csv", index=False)
        tuning_best_df.to_csv(args.output_dir / "tuning_best_hyperparameters.csv", index=False)

        if top_trials_tables:
            top_trials_df = pd.concat(top_trials_tables, axis=0, ignore_index=True)
            (args.output_dir / "tuning_top_trials.tex").write_text(
                latex_wrap_table(
                    top_trials_df,
                    caption=f"Top-{args.top_k_trials} completed trials per pipeline.",
                    label="tab:tuning_top_trials",
                ),
                encoding="utf-8",
            )
            top_trials_df.to_csv(args.output_dir / "tuning_top_trials.csv", index=False)

        if all_trial_rows:
            all_trials_df = pd.concat(all_trial_rows, axis=0, ignore_index=True)
            all_trials_df.to_csv(args.output_dir / "tuning_all_trials.csv", index=False)
    else:
        print("Warning: no tuning logs were evaluated.")

    manifest = {
        "prediction_expected_files": EXPECTED_PREDICTION_FILES,
        "confusion_tables": [str(path) for path in confusion_files],
        "output_dir": str(args.output_dir),
    }
    with (args.output_dir / "evaluation_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Evaluation artifacts saved in: {args.output_dir}")


if __name__ == "__main__":
    main()