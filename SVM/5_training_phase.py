# python3 SVM/5_training_phase.py --pipeline balanced
# python3 SVM/5_training_phase.py --pipeline natural
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


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
        random_state=int(params.get("seed", 42)),
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
    parser = argparse.ArgumentParser(description="Training phase for TF-IDF + calibrated Linear SVM.")
    parser.add_argument("--pipeline", choices=["balanced", "natural"], required=True)
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--validation-csv", type=Path, default=None)
    parser.add_argument("--tuning-base-dir", type=Path, default=Path("SVM/tuning_outputs"))
    parser.add_argument("--output-base-dir", type=Path, default=Path("SVM/trained_models"))
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--star-col", type=str, default="stars")
    parser.add_argument("--min-text-chars", type=int, default=20)
    args = parser.parse_args()

    train_csv = args.train_csv or Path(f"Datasets/train_{args.pipeline}.csv")
    validation_csv = args.validation_csv or Path(f"Datasets/validation_{args.pipeline}.csv")
    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Missing validation split: {validation_csv}")

    best_hparams_path = args.tuning_base_dir / args.pipeline / "best_model" / "best_hyperparameters.json"
    if not best_hparams_path.exists():
        raise FileNotFoundError(f"Best hyperparameters not found: {best_hparams_path}")

    with best_hparams_path.open("r", encoding="utf-8") as f:
        best_payload = json.load(f)
    best_params = best_payload.get("params")
    if not isinstance(best_params, dict):
        raise ValueError(f"Invalid params in {best_hparams_path}")

    train_texts, train_labels = load_split(train_csv, args.text_col, args.star_col, args.min_text_chars)
    val_texts, val_labels = load_split(validation_csv, args.text_col, args.star_col, args.min_text_chars)

    model = build_model(best_params)
    model.fit(train_texts, train_labels)
    val_pred = model.predict(val_texts)
    val_metrics = compute_metrics(val_labels, val_pred)

    output_dir = args.output_base_dir / args.pipeline
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.joblib"
    joblib.dump({"estimator": model, "label_offset": 1, "model_family": "SVM"}, model_path)

    summary = {
        "pipeline": args.pipeline,
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "best_hyperparameters_source": str(best_hparams_path),
        "best_hyperparameters": best_params,
        "validation_metrics": val_metrics,
        "saved_model": str(model_path),
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved trained model: {model_path}")


if __name__ == "__main__":
    main()
