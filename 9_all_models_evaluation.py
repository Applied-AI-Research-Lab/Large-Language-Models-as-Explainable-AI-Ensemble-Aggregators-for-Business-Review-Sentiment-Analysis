# cd /local/users/k.roumeliotis/Projects/YELP

# # Evaluate all available rows in all_models_predictions.csv
# python3 9_all_models_evaluation.py

# # Evaluate only one pipeline
# python3 9_all_models_evaluation.py --pipeline balanced
# python3 9_all_models_evaluation.py --pipeline natural
# Default split outputs:
# - Datasets/evaluation_outputs/balanced
# - Datasets/evaluation_outputs/natural

import argparse
import json
import math
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


MODEL_PREFIXES = ["bert", "lr", "svm", "nb"]
STAR_LABELS = [1, 2, 3, 4, 5]


def validate_columns(df: pd.DataFrame, model_prefixes: List[str]) -> None:
    required = {"stars", "evaluation_pipeline"}
    for prefix in model_prefixes:
        required.add(f"{prefix}_pred_stars")

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def sanitize_star_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=columns)
    for col in columns:
        out[col] = out[col].round().astype(int)
        out = out[out[col].between(1, 5)]
    return out


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def evaluate_single_model(data: pd.DataFrame, model_prefix: str) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    pred_col = f"{model_prefix}_pred_stars"

    y_true = data["stars"].tolist()
    y_pred = data[pred_col].tolist()

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

    cm = confusion_matrix(y_true, y_pred, labels=STAR_LABELS)

    class_precision, class_recall, class_f1, class_support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=STAR_LABELS,
        average=None,
        zero_division=0,
    )

    summary = {
        "model": model_prefix.upper(),
        "n_samples": int(len(data)),
        "accuracy": float(accuracy),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "qwk": float(qwk),
        "mae": float(mae),
        "rmse": float(rmse),
    }

    confidence_col = f"{model_prefix}_pred_confidence"
    if confidence_col in data.columns:
        conf = pd.to_numeric(data[confidence_col], errors="coerce")
        summary["mean_confidence"] = safe_float(conf.mean())
        summary["median_confidence"] = safe_float(conf.median())

    class_rows = []
    for idx, label in enumerate(STAR_LABELS):
        class_rows.append(
            {
                "model": model_prefix.upper(),
                "star": int(label),
                "precision": float(class_precision[idx]),
                "recall": float(class_recall[idx]),
                "f1": float(class_f1[idx]),
                "support": int(class_support[idx]),
            }
        )

    class_df = pd.DataFrame(class_rows)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{star}" for star in STAR_LABELS],
        columns=[f"pred_{star}" for star in STAR_LABELS],
    )

    return summary, class_df, cm_df


def evaluate_split(data: pd.DataFrame, split_name: str, output_dir: Path) -> Dict[str, pd.DataFrame]:
    summaries: List[Dict] = []
    all_class_rows: List[pd.DataFrame] = []

    confusion_dir = output_dir / "confusion_matrices" / split_name
    confusion_dir.mkdir(parents=True, exist_ok=True)

    for prefix in MODEL_PREFIXES:
        summary, class_df, cm_df = evaluate_single_model(data, prefix)
        summary["split"] = split_name
        summaries.append(summary)
        class_df.insert(0, "split", split_name)
        all_class_rows.append(class_df)

        cm_path = confusion_dir / f"confusion_{prefix}.csv"
        cm_df.to_csv(cm_path)

    summary_df = pd.DataFrame(summaries).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    summary_df.insert(0, "rank_by_f1_macro", np.arange(1, len(summary_df) + 1))

    per_class_df = pd.concat(all_class_rows, ignore_index=True)

    summary_df.to_csv(output_dir / f"metrics_{split_name}.csv", index=False)
    per_class_df.to_csv(output_dir / f"per_class_metrics_{split_name}.csv", index=False)

    return {
        "summary": summary_df,
        "per_class": per_class_df,
    }


def compute_model_agreement(data: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for i in range(len(MODEL_PREFIXES)):
        for j in range(i + 1, len(MODEL_PREFIXES)):
            a = MODEL_PREFIXES[i]
            b = MODEL_PREFIXES[j]
            a_col = f"{a}_pred_stars"
            b_col = f"{b}_pred_stars"

            agree_rate = float((data[a_col] == data[b_col]).mean())
            kappa = float(cohen_kappa_score(data[a_col], data[b_col], weights="quadratic"))

            rows.append(
                {
                    "model_a": a.upper(),
                    "model_b": b.upper(),
                    "agreement_rate": agree_rate,
                    "quadratic_kappa": kappa,
                }
            )

    return pd.DataFrame(rows).sort_values(["agreement_rate", "quadratic_kappa"], ascending=False)


def build_json_summary(
    overall_df: pd.DataFrame,
    by_split: Dict[str, pd.DataFrame],
    agreement_overall: pd.DataFrame,
    agreement_by_split: Dict[str, pd.DataFrame],
) -> Dict:
    payload: Dict = {
        "overall_ranking_by_f1_macro": overall_df[
            ["rank_by_f1_macro", "model", "f1_macro", "accuracy", "qwk", "mae", "rmse"]
        ].to_dict(orient="records"),
        "best_model_overall": overall_df.iloc[0][
            ["model", "f1_macro", "accuracy", "qwk", "mae", "rmse"]
        ].to_dict(),
        "split_rankings": {},
        "agreement": {
            "overall": agreement_overall.to_dict(orient="records"),
            "by_split": {},
        },
    }

    for split_name, split_df in by_split.items():
        payload["split_rankings"][split_name] = split_df[
            ["rank_by_f1_macro", "model", "f1_macro", "accuracy", "qwk", "mae", "rmse"]
        ].to_dict(orient="records")

    for split_name, split_agreement in agreement_by_split.items():
        payload["agreement"]["by_split"][split_name] = split_agreement.to_dict(orient="records")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Comprehensive evaluator for Datasets/all_models_predictions.csv. "
            "Computes per-model metrics, per-class metrics, confusion matrices, and inter-model agreement."
        )
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help=(
            "Input predictions CSV. If omitted and --pipeline is set, defaults to "
            "Datasets/all_models_predictions_<pipeline>.csv; otherwise Datasets/all_models_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output folder. If omitted and --pipeline is set, defaults to "
            "Datasets/evaluation_outputs/<pipeline>; otherwise Datasets/evaluation_outputs"
        ),
    )
    parser.add_argument(
        "--pipeline",
        choices=["balanced", "natural"],
        default=None,
        help="Optional filter to evaluate only one pipeline.",
    )
    args = parser.parse_args()

    if args.input_csv is None:
        if args.pipeline is None:
            args.input_csv = Path("Datasets/all_models_predictions.csv")
        else:
            args.input_csv = Path(f"Datasets/all_models_predictions_{args.pipeline}.csv")

    if args.output_dir is None:
        if args.pipeline is None:
            args.output_dir = Path("Datasets/evaluation_outputs")
        else:
            args.output_dir = Path(f"Datasets/evaluation_outputs/{args.pipeline}")

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input predictions CSV not found: {args.input_csv}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = pd.read_csv(args.input_csv)
    validate_columns(raw_df, MODEL_PREFIXES)

    if args.pipeline is not None:
        raw_df = raw_df[raw_df["evaluation_pipeline"].astype(str).str.lower() == args.pipeline].copy()

    model_star_cols = [f"{prefix}_pred_stars" for prefix in MODEL_PREFIXES]
    working = sanitize_star_columns(raw_df, ["stars", *model_star_cols])

    if working.empty:
        raise ValueError("No valid rows available after cleaning stars and prediction columns.")

    splits_present = sorted(working["evaluation_pipeline"].astype(str).str.lower().unique().tolist())

    print(f"Rows evaluated: {len(working)}")
    print(f"Pipelines present: {splits_present}")

    overall_results = evaluate_split(working, "overall", args.output_dir)
    overall_summary_df = overall_results["summary"]

    by_split_summary: Dict[str, pd.DataFrame] = {}
    by_split_agreement: Dict[str, pd.DataFrame] = {}

    for split_name in splits_present:
        split_df = working[working["evaluation_pipeline"].astype(str).str.lower() == split_name]
        if split_df.empty:
            continue
        split_results = evaluate_split(split_df, split_name, args.output_dir)
        by_split_summary[split_name] = split_results["summary"]

        split_agreement = compute_model_agreement(split_df)
        by_split_agreement[split_name] = split_agreement
        split_agreement.to_csv(args.output_dir / f"model_agreement_{split_name}.csv", index=False)

    agreement_overall = compute_model_agreement(working)
    agreement_overall.to_csv(args.output_dir / "model_agreement_overall.csv", index=False)

    summary_payload = build_json_summary(
        overall_df=overall_summary_df,
        by_split=by_split_summary,
        agreement_overall=agreement_overall,
        agreement_by_split=by_split_agreement,
    )

    with (args.output_dir / "evaluation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2)

    best_row = overall_summary_df.iloc[0]
    report_lines = [
        "All Models Evaluation Report",
        "============================",
        f"Input CSV: {args.input_csv}",
        f"Rows evaluated: {len(working)}",
        f"Pipelines present: {', '.join(splits_present)}",
        "",
        "Best model overall (ranked by f1_macro):",
        f"- Model: {best_row['model']}",
        f"- f1_macro: {best_row['f1_macro']:.6f}",
        f"- accuracy: {best_row['accuracy']:.6f}",
        f"- qwk: {best_row['qwk']:.6f}",
        f"- mae: {best_row['mae']:.6f}",
        f"- rmse: {best_row['rmse']:.6f}",
        "",
        "Generated files:",
        "- metrics_overall.csv",
        "- per_class_metrics_overall.csv",
        "- metrics_balanced.csv / metrics_natural.csv (if present)",
        "- per_class_metrics_balanced.csv / per_class_metrics_natural.csv (if present)",
        "- model_agreement_overall.csv",
        "- model_agreement_balanced.csv / model_agreement_natural.csv (if present)",
        "- confusion_matrices/<split>/confusion_<model>.csv",
        "- evaluation_summary.json",
    ]

    (args.output_dir / "evaluation_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(f"Saved evaluation outputs in: {args.output_dir}")


if __name__ == "__main__":
    main()
