# python3 SVM/7_evaluation_phase.py --predictions-dir SVM/predictions --tuning-base-dir SVM/tuning_outputs --output-dir SVM/evaluation_outputs
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

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


EXPECTED_FILES = [
    "preds_svm_balanced_on_balanced.csv",
    "preds_svm_natural_on_natural.csv",
    "preds_svm_balanced_on_natural.csv",
    "preds_svm_natural_on_balanced.csv",
]


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_prediction_file(path: Path) -> Tuple[Dict, pd.DataFrame]:
    df = pd.read_csv(path)
    if "stars" not in df.columns or "pred_stars" not in df.columns:
        raise ValueError(f"Missing stars/pred_stars in {path}")

    data = df.copy()
    data["stars"] = pd.to_numeric(data["stars"], errors="coerce")
    data["pred_stars"] = pd.to_numeric(data["pred_stars"], errors="coerce")
    data = data.dropna(subset=["stars", "pred_stars"])
    data["stars"] = data["stars"].round().astype(int)
    data["pred_stars"] = data["pred_stars"].round().astype(int)
    data = data[data["stars"].between(1, 5) & data["pred_stars"].between(1, 5)]

    y_true = data["stars"].tolist()
    y_pred = data["pred_stars"].tolist()

    precision, recall, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic")

    cm = confusion_matrix(y_true, y_pred, labels=[1, 2, 3, 4, 5])
    row_sums = cm.sum(axis=1)
    per_class_recall = [
        float(cm[idx, idx] / row_sums[idx]) if row_sums[idx] > 0 else 0.0
        for idx in range(5)
    ]

    summary = {
        "file": path.name,
        "n_samples": int(len(data)),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(np.sqrt(np.mean((np.array(y_true) - np.array(y_pred)) ** 2))),
        "recall_star_1": per_class_recall[0],
        "recall_star_2": per_class_recall[1],
        "recall_star_3": per_class_recall[2],
        "recall_star_4": per_class_recall[3],
        "recall_star_5": per_class_recall[4],
    }
    cm_df = pd.DataFrame(cm, index=[f"true_{i}" for i in [1, 2, 3, 4, 5]], columns=[f"pred_{i}" for i in [1, 2, 3, 4, 5]])
    return summary, cm_df


def parse_tuning_log(path: Path) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    pipeline = (payload.get("run_metadata") or {}).get("pipeline", path.parent.name)

    rows = []
    for trial in payload.get("trials", []):
        params = trial.get("params", {})
        metrics = trial.get("metrics", {})
        rows.append(
            {
                "pipeline": pipeline,
                "trial_number": trial.get("trial_number"),
                "status": trial.get("status"),
                "duration_seconds": safe_float(trial.get("duration_seconds")),
                "objective_score": safe_float(trial.get("objective_score")),
                "eval_f1_macro": safe_float(metrics.get("eval_f1_macro")),
                "eval_accuracy": safe_float(metrics.get("eval_accuracy")),
                "val_error": safe_float(trial.get("val_error")),
                "generalization_gap_f1_macro": safe_float(trial.get("generalization_gap_f1_macro")),
                "ngram_max": params.get("ngram_max"),
                "max_features": params.get("max_features"),
                "min_df": params.get("min_df"),
                "max_df": params.get("max_df"),
                "sublinear_tf": params.get("sublinear_tf"),
                "C": params.get("C"),
                "class_weight": params.get("class_weight"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation phase for SVM model family.")
    parser.add_argument("--predictions-dir", type=Path, default=Path("SVM/predictions"))
    parser.add_argument("--tuning-base-dir", type=Path, default=Path("SVM/tuning_outputs"))
    parser.add_argument("--output-dir", type=Path, default=Path("SVM/evaluation_outputs"))
    parser.add_argument("--top-k-trials", type=int, default=10)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    pred_rows: List[Dict] = []
    for filename in EXPECTED_FILES:
        file_path = args.predictions_dir / filename
        if not file_path.exists():
            print(f"Warning: missing prediction file {file_path}")
            continue
        summary, cm_df = evaluate_prediction_file(file_path)
        pred_rows.append(summary)
        (args.output_dir / f"confusion_{filename.replace('.csv', '.tex')}").write_text(
            cm_df.reset_index().rename(columns={"index": "true_label"}).to_latex(index=False),
            encoding="utf-8",
        )

    if pred_rows:
        pred_df = pd.DataFrame(pred_rows)
        pred_df.to_csv(args.output_dir / "predictions_metrics.csv", index=False)
        pred_df.to_latex(
            args.output_dir / "predictions_metrics.tex",
            index=False,
            caption="SVM prediction performance across evaluation files.",
            label="tab:svm_prediction_metrics",
            float_format="%.4f",
        )

    tuning_tables: List[pd.DataFrame] = []
    summary_rows: List[Dict] = []
    for pipeline in ["balanced", "natural"]:
        log_path = args.tuning_base_dir / pipeline / "hyperparameter_tuning_log.json"
        if not log_path.exists():
            print(f"Warning: missing tuning log {log_path}")
            continue

        df = parse_tuning_log(log_path)
        if df.empty:
            continue
        tuning_tables.append(df)

        completed = df[df["status"] == "completed"]
        best = completed.sort_values("objective_score", ascending=False).head(1)
        if not best.empty:
            row = best.iloc[0].to_dict()
            row["n_trials"] = int(len(df))
            row["n_completed"] = int((df["status"] == "completed").sum())
            row["n_failed"] = int((df["status"] == "failed").sum())
            summary_rows.append(row)

        top = completed.sort_values("objective_score", ascending=False).head(args.top_k_trials)
        if not top.empty:
            top.to_csv(args.output_dir / f"tuning_top_trials_{pipeline}.csv", index=False)
            top.to_latex(
                args.output_dir / f"tuning_top_trials_{pipeline}.tex",
                index=False,
                caption=f"SVM top-{args.top_k_trials} tuning trials ({pipeline}).",
                label=f"tab:svm_top_trials_{pipeline}",
                float_format="%.4f",
            )

    if tuning_tables:
        all_trials = pd.concat(tuning_tables, ignore_index=True)
        all_trials.to_csv(args.output_dir / "tuning_all_trials.csv", index=False)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(args.output_dir / "tuning_summary.csv", index=False)
        summary_df.to_latex(
            args.output_dir / "tuning_summary.tex",
            index=False,
            caption="SVM best tuning configuration summary per pipeline.",
            label="tab:svm_tuning_summary",
            float_format="%.4f",
        )

    print(f"Saved evaluation outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
