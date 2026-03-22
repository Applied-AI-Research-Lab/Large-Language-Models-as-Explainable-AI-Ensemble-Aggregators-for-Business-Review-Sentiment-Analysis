# python3 3_create_explicit_pipelines_datasets.py --input-csv yelp_academic_dataset_review.csv --output-dir Datasets
import argparse
import json
import re
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.model_selection import train_test_split


STAR_VALUES = [1, 2, 3, 4, 5]


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def class_counts(df: pd.DataFrame, star_col: str) -> Dict[str, int]:
    counts = df[star_col].value_counts().sort_index().to_dict()
    return {str(int(k)): int(v) for k, v in counts.items()}


def class_percentages(df: pd.DataFrame, star_col: str) -> Dict[str, float]:
    counts = df[star_col].value_counts().sort_index()
    total = float(len(df))
    if total == 0:
        return {str(star): 0.0 for star in STAR_VALUES}
    percentages = {str(int(star)): float(100.0 * counts.get(star, 0) / total) for star in STAR_VALUES}
    return percentages


def assert_divisible_by_five(value: int, name: str) -> None:
    if value % 5 != 0:
        raise ValueError(f"{name} must be divisible by 5 for perfectly balanced sampling. Got {value}.")


def load_and_clean(
    input_csv: Path,
    text_col: str,
    star_col: str,
    review_id_col: str,
    min_text_chars: int,
) -> pd.DataFrame:
    use_cols: List[str] = [text_col, star_col]
    if review_id_col:
        use_cols.append(review_id_col)

    df = pd.read_csv(input_csv, usecols=use_cols)
    df = df.dropna(subset=[text_col, star_col])
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.len() >= min_text_chars]

    df[star_col] = pd.to_numeric(df[star_col], errors="coerce")
    df = df.dropna(subset=[star_col])
    df[star_col] = df[star_col].round().astype(int)
    df = df[df[star_col].isin(STAR_VALUES)]

    if review_id_col in df.columns:
        df = df.drop_duplicates(subset=[review_id_col], keep="first")

    df["_normalized_text"] = df[text_col].map(normalize_text)
    df = df.drop_duplicates(subset=["_normalized_text"], keep="first")
    return df


def sample_balanced(df: pd.DataFrame, star_col: str, total_size: int, seed: int) -> pd.DataFrame:
    assert_divisible_by_five(total_size, "Balanced sampling size")
    per_class = total_size // len(STAR_VALUES)
    if per_class <= 0:
        raise ValueError("Balanced size is too small. It must allow at least 1 sample per class.")

    parts = []
    for star in STAR_VALUES:
        star_df = df[df[star_col] == star]
        if len(star_df) < per_class:
            raise ValueError(
                f"Not enough samples for star {star}: requested {per_class}, available {len(star_df)}"
            )
        parts.append(star_df.sample(n=per_class, random_state=seed))

    sampled = pd.concat(parts, axis=0)
    sampled = sampled.sample(frac=1.0, random_state=seed)
    return sampled


def write_latex_report(
    output_path: Path,
    report: Dict,
    class_percentages_report: Dict[str, Dict[str, float]],
) -> None:
    dataset_order = [
        "train_balanced",
        "validation_balanced",
        "train_tune_balanced",
        "validation_tune_balanced",
        "test_balanced",
        "train_natural",
        "validation_natural",
        "train_tune_natural",
        "validation_tune_natural",
        "test_natural",
    ]

    lines: List[str] = []
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Class distributions per dataset split (counts and percentages).}")
    lines.append("\\label{tab:class_distributions}")
    lines.append("\\begin{tabular}{lrrrrrr}")
    lines.append("\\hline")
    lines.append("Split & Size & 1-star & 2-star & 3-star & 4-star & 5-star \\\\")
    lines.append("\\hline")

    for split_name in dataset_order:
        size = report["sizes"][split_name]
        counts = report["class_counts"][split_name]
        pcts = class_percentages_report[split_name]
        row = (
            f"{split_name.replace('_', '\\_')} & {size} & "
            f"{counts.get('1', 0)} ({pcts.get('1', 0.0):.2f}\\%) & "
            f"{counts.get('2', 0)} ({pcts.get('2', 0.0):.2f}\\%) & "
            f"{counts.get('3', 0)} ({pcts.get('3', 0.0):.2f}\\%) & "
            f"{counts.get('4', 0)} ({pcts.get('4', 0.0):.2f}\\%) & "
            f"{counts.get('5', 0)} ({pcts.get('5', 0.0):.2f}\\%) \\\\"
        )
        lines.append(row)

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    lines.append("")

    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Leakage checks across held-out test sets and fine-tuning splits.}")
    lines.append("\\label{tab:leakage_checks}")
    lines.append("\\begin{tabular}{lr}")
    lines.append("\\hline")
    lines.append("Check & Overlap Count \\\\")
    lines.append("\\hline")

    for key, value in report["leakage_report"].items():
        lines.append(f"{key.replace('_', '\\_')} & {value} \\\\")

    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create explicit dataset files for two pipelines with strict hidden test sets: "
            "Pipeline A (balanced) and Pipeline B (natural)."
        )
    )
    parser.add_argument("--input-csv", type=Path, default=Path("yelp_academic_dataset_review.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("Datasets"))
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--star-col", type=str, default="stars")
    parser.add_argument("--review-id-col", type=str, default="review_id")
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pipeline-total-size",
        type=int,
        default=50000,
        help="Total samples per pipeline (train + validation + test).",
    )

    parser.add_argument(
        "--test-balanced-size",
        type=int,
        default=10000,
        help="Total size for balanced test set (equal per star).",
    )
    parser.add_argument(
        "--test-natural-size",
        type=int,
        default=10000,
        help="Total size for natural-skew test set.",
    )
    parser.add_argument(
        "--balanced-val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for balanced train/validation split.",
    )

    parser.add_argument(
        "--natural-val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio for natural train/validation split.",
    )
    parser.add_argument(
        "--tune-balanced-train-size",
        type=int,
        default=5000,
        help="Balanced tuning-train subset size sampled from train_balanced.",
    )
    parser.add_argument(
        "--tune-balanced-validation-size",
        type=int,
        default=1000,
        help="Balanced tuning-validation subset size sampled from validation_balanced.",
    )
    parser.add_argument(
        "--tune-natural-train-size",
        type=int,
        default=5000,
        help="Natural-skew tuning-train subset size sampled from train_natural.",
    )
    parser.add_argument(
        "--tune-natural-validation-size",
        type=int,
        default=1000,
        help="Natural-skew tuning-validation subset size sampled from validation_natural.",
    )

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pipeline_total_size <= 0:
        raise ValueError("pipeline-total-size must be positive.")
    if args.test_balanced_size >= args.pipeline_total_size:
        raise ValueError("test-balanced-size must be smaller than pipeline-total-size.")
    if args.test_natural_size >= args.pipeline_total_size:
        raise ValueError("test-natural-size must be smaller than pipeline-total-size.")

    balanced_train_pool_size = args.pipeline_total_size - args.test_balanced_size
    natural_train_pool_size = args.pipeline_total_size - args.test_natural_size

    assert_divisible_by_five(args.test_balanced_size, "test-balanced-size")
    assert_divisible_by_five(balanced_train_pool_size, "balanced train+validation pool size")
    assert_divisible_by_five(args.tune_balanced_train_size, "tune-balanced-train-size")
    assert_divisible_by_five(
        args.tune_balanced_validation_size,
        "tune-balanced-validation-size",
    )

    df = load_and_clean(
        input_csv=args.input_csv,
        text_col=args.text_col,
        star_col=args.star_col,
        review_id_col=args.review_id_col,
        min_text_chars=args.min_text_chars,
    )

    test_balanced_df = sample_balanced(
        df=df,
        star_col=args.star_col,
        total_size=args.test_balanced_size,
        seed=args.seed,
    )

    remaining_after_balanced_test = df.drop(index=test_balanced_df.index)
    if len(remaining_after_balanced_test) < args.test_natural_size:
        raise ValueError("Not enough rows to build natural test set after selecting balanced test set.")

    test_natural_df = remaining_after_balanced_test.sample(
        n=args.test_natural_size,
        random_state=args.seed,
    )

    holdout_indices = set(test_balanced_df.index) | set(test_natural_df.index)
    train_pool_df = df.drop(index=list(holdout_indices))

    balanced_pool_df = sample_balanced(
        df=train_pool_df,
        star_col=args.star_col,
        total_size=balanced_train_pool_size,
        seed=args.seed,
    )
    train_balanced_df, validation_balanced_df = train_test_split(
        balanced_pool_df,
        test_size=args.balanced_val_ratio,
        random_state=args.seed,
        stratify=balanced_pool_df[args.star_col],
    )

    if len(train_pool_df) < natural_train_pool_size:
        raise ValueError("Not enough rows to build natural train/validation pool.")

    natural_pool_df = train_pool_df.sample(n=natural_train_pool_size, random_state=args.seed)
    train_natural_df, validation_natural_df = train_test_split(
        natural_pool_df,
        test_size=args.natural_val_ratio,
        random_state=args.seed,
        stratify=natural_pool_df[args.star_col],
    )

    if args.tune_balanced_train_size > len(train_balanced_df):
        raise ValueError("tune-balanced-train-size exceeds train_balanced size.")
    if args.tune_balanced_validation_size > len(validation_balanced_df):
        raise ValueError("tune-balanced-validation-size exceeds validation_balanced size.")
    if args.tune_natural_train_size > len(train_natural_df):
        raise ValueError("tune-natural-train-size exceeds train_natural size.")
    if args.tune_natural_validation_size > len(validation_natural_df):
        raise ValueError("tune-natural-validation-size exceeds validation_natural size.")

    train_tune_balanced_df = sample_balanced(
        df=train_balanced_df,
        star_col=args.star_col,
        total_size=args.tune_balanced_train_size,
        seed=args.seed,
    )
    validation_tune_balanced_df = sample_balanced(
        df=validation_balanced_df,
        star_col=args.star_col,
        total_size=args.tune_balanced_validation_size,
        seed=args.seed,
    )
    train_tune_natural_df = train_natural_df.sample(
        n=args.tune_natural_train_size,
        random_state=args.seed,
    )
    validation_tune_natural_df = validation_natural_df.sample(
        n=args.tune_natural_validation_size,
        random_state=args.seed,
    )

    export_cols = [args.text_col, args.star_col]
    if args.review_id_col in df.columns:
        export_cols = [args.review_id_col] + export_cols

    train_balanced_df[export_cols].to_csv(args.output_dir / "train_balanced.csv", index=False)
    validation_balanced_df[export_cols].to_csv(
        args.output_dir / "validation_balanced.csv", index=False
    )
    test_balanced_df[export_cols].to_csv(args.output_dir / "test_balanced.csv", index=False)

    train_natural_df[export_cols].to_csv(args.output_dir / "train_natural.csv", index=False)
    validation_natural_df[export_cols].to_csv(
        args.output_dir / "validation_natural.csv", index=False
    )
    test_natural_df[export_cols].to_csv(args.output_dir / "test_natural.csv", index=False)
    train_tune_balanced_df[export_cols].to_csv(
        args.output_dir / "train_tune_balanced.csv", index=False
    )
    validation_tune_balanced_df[export_cols].to_csv(
        args.output_dir / "validation_tune_balanced.csv", index=False
    )
    train_tune_natural_df[export_cols].to_csv(
        args.output_dir / "train_tune_natural.csv", index=False
    )
    validation_tune_natural_df[export_cols].to_csv(
        args.output_dir / "validation_tune_natural.csv", index=False
    )

    leakage_report = {
        "test_overlap_balanced_natural": int(len(set(test_balanced_df.index) & set(test_natural_df.index))),
        "train_balanced_vs_test_union": int(
            len(set(train_balanced_df.index) & holdout_indices)
        ),
        "validation_balanced_vs_test_union": int(
            len(set(validation_balanced_df.index) & holdout_indices)
        ),
        "train_natural_vs_test_union": int(len(set(train_natural_df.index) & holdout_indices)),
        "validation_natural_vs_test_union": int(
            len(set(validation_natural_df.index) & holdout_indices)
        ),
        "train_tune_balanced_vs_test_union": int(
            len(set(train_tune_balanced_df.index) & holdout_indices)
        ),
        "validation_tune_balanced_vs_test_union": int(
            len(set(validation_tune_balanced_df.index) & holdout_indices)
        ),
        "train_tune_natural_vs_test_union": int(
            len(set(train_tune_natural_df.index) & holdout_indices)
        ),
        "validation_tune_natural_vs_test_union": int(
            len(set(validation_tune_natural_df.index) & holdout_indices)
        ),
    }

    report = {
        "input_csv": str(args.input_csv),
        "seed": args.seed,
        "pipeline_total_size": args.pipeline_total_size,
        "balanced_train_plus_validation_size": balanced_train_pool_size,
        "natural_train_plus_validation_size": natural_train_pool_size,
        "sizes": {
            "train_balanced": int(len(train_balanced_df)),
            "validation_balanced": int(len(validation_balanced_df)),
            "test_balanced": int(len(test_balanced_df)),
            "train_natural": int(len(train_natural_df)),
            "validation_natural": int(len(validation_natural_df)),
            "test_natural": int(len(test_natural_df)),
            "train_tune_balanced": int(len(train_tune_balanced_df)),
            "validation_tune_balanced": int(len(validation_tune_balanced_df)),
            "train_tune_natural": int(len(train_tune_natural_df)),
            "validation_tune_natural": int(len(validation_tune_natural_df)),
        },
        "class_counts": {
            "train_balanced": class_counts(train_balanced_df, args.star_col),
            "validation_balanced": class_counts(validation_balanced_df, args.star_col),
            "test_balanced": class_counts(test_balanced_df, args.star_col),
            "train_natural": class_counts(train_natural_df, args.star_col),
            "validation_natural": class_counts(validation_natural_df, args.star_col),
            "test_natural": class_counts(test_natural_df, args.star_col),
            "train_tune_balanced": class_counts(train_tune_balanced_df, args.star_col),
            "validation_tune_balanced": class_counts(
                validation_tune_balanced_df, args.star_col
            ),
            "train_tune_natural": class_counts(train_tune_natural_df, args.star_col),
            "validation_tune_natural": class_counts(
                validation_tune_natural_df, args.star_col
            ),
        },
        "leakage_report": leakage_report,
    }

    class_percentages_report = {
        "train_balanced": class_percentages(train_balanced_df, args.star_col),
        "validation_balanced": class_percentages(validation_balanced_df, args.star_col),
        "test_balanced": class_percentages(test_balanced_df, args.star_col),
        "train_natural": class_percentages(train_natural_df, args.star_col),
        "validation_natural": class_percentages(validation_natural_df, args.star_col),
        "test_natural": class_percentages(test_natural_df, args.star_col),
        "train_tune_balanced": class_percentages(train_tune_balanced_df, args.star_col),
        "validation_tune_balanced": class_percentages(
            validation_tune_balanced_df, args.star_col
        ),
        "train_tune_natural": class_percentages(train_tune_natural_df, args.star_col),
        "validation_tune_natural": class_percentages(
            validation_tune_natural_df, args.star_col
        ),
    }
    report["class_percentages"] = class_percentages_report

    with (args.output_dir / "split_report_explicit_pipelines.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    latex_path = args.output_dir / "split_report_explicit_pipelines.tex"
    write_latex_report(latex_path, report, class_percentages_report)

    print(f"Saved datasets in: {args.output_dir}")
    print(f"Saved report: {args.output_dir / 'split_report_explicit_pipelines.json'}")
    print(f"Saved LaTeX tables: {latex_path}")


if __name__ == "__main__":
    main()