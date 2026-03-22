# cd /local/users/k.roumeliotis/Projects/YELP
# python3 -m pip install torch transformers scikit-learn joblib pandas numpy
# python3 8_all_models_predictions.py --pipeline balanced --overwrite-output
# python3 8_all_models_predictions.py --pipeline natural
# Default outputs:
# - Datasets/all_models_predictions_balanced.csv
# - Datasets/all_models_predictions_natural.csv

import argparse
import inspect
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no GPU is available.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_autocast_dtype(mixed_precision: str) -> Optional[torch.dtype]:
    if mixed_precision == "none":
        return None
    if mixed_precision == "fp16":
        return torch.float16
    if mixed_precision == "bf16":
        return torch.bfloat16

    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return None


def infer_bert_max_length(model_dir: Path, fallback: int = 256) -> int:
    summary_path = model_dir / "fine_tuning_summary.json"
    if not summary_path.exists():
        return fallback

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        value = payload.get("best_hyperparameters", {}).get("max_length")
        if value is None:
            return fallback
        return int(value)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return fallback


def predict_bert(
    texts: List[str],
    model_dir: Path,
    batch_size: int,
    max_length: int,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> Dict[str, List]:
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
    model.to(device)
    model.eval()

    all_pred_labels: List[int] = []
    all_pred_stars: List[int] = []
    all_confidences: List[float] = []
    all_probabilities: List[List[float]] = []

    for start in range(0, len(texts), batch_size):
        end = start + batch_size
        batch_texts = texts[start:end]
        encoded = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.inference_mode():
            if device.type == "cuda" and autocast_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                    logits = model(**encoded).logits
            else:
                logits = model(**encoded).logits

        probabilities = torch.softmax(logits, dim=-1)
        pred_labels = torch.argmax(probabilities, dim=-1)
        confidences = torch.max(probabilities, dim=-1).values

        all_pred_labels.extend(pred_labels.cpu().tolist())
        all_pred_stars.extend((pred_labels + 1).cpu().tolist())
        all_confidences.extend(confidences.cpu().tolist())
        all_probabilities.extend(probabilities.cpu().tolist())

    result: Dict[str, List] = {
        "pred_label": all_pred_labels,
        "pred_stars": all_pred_stars,
        "pred_confidence": all_confidences,
        "pred_probabilities": all_probabilities,
    }
    return result


def load_classical_estimator(model_path: Path):
    bundle = joblib.load(model_path)
    return bundle["estimator"] if isinstance(bundle, dict) and "estimator" in bundle else bundle


def predict_classical(
    estimator,
    texts: List[str],
) -> Dict[str, List]:
    pred_labels = estimator.predict(texts)
    pred_stars = (np.asarray(pred_labels) + 1).tolist()

    result: Dict[str, List] = {
        "pred_label": np.asarray(pred_labels).tolist(),
        "pred_stars": pred_stars,
    }

    if hasattr(estimator, "predict_proba"):
        probs = estimator.predict_proba(texts)
        result["pred_confidence"] = probs.max(axis=1).tolist()
        result["pred_probabilities"] = probs.tolist()
    else:
        result["pred_confidence"] = [float("nan")] * len(texts)
        one_hot = np.zeros((len(pred_labels), 5), dtype=float)
        pred_labels_np = np.asarray(pred_labels, dtype=int)
        pred_labels_np = np.clip(pred_labels_np, 0, 4)
        one_hot[np.arange(len(pred_labels_np)), pred_labels_np] = 1.0
        result["pred_probabilities"] = one_hot.tolist()

    return result


def find_best_classical_model(
    family_name: str,
    family_dir: Path,
    pipeline: str,
) -> Tuple[Path, Dict[str, float | str]]:
    if not family_dir.exists():
        raise FileNotFoundError(f"{family_name} selection directory not found: {family_dir}")

    candidates: List[Tuple[float, float, Path, Dict[str, float | str]]] = []

    for setup_dir in sorted(path for path in family_dir.iterdir() if path.is_dir()):
        report_path = setup_dir / "final_selection_report.json"
        best_model_path = setup_dir / "best_model.joblib"

        if not report_path.exists() or not best_model_path.exists():
            continue

        with report_path.open("r", encoding="utf-8") as f:
            report = json.load(f)

        if str(report.get("pipeline", "")).strip().lower() != pipeline:
            continue

        best_candidate = report.get("best_candidate", {})
        f1_macro = float(best_candidate.get("test_f1_macro", float("-inf")))
        accuracy = float(best_candidate.get("test_accuracy", float("-inf")))

        info: Dict[str, float | str] = {
            "setup_folder": str(report.get("setup_folder", setup_dir.name)),
            "test_f1_macro": f1_macro,
            "test_accuracy": accuracy,
        }
        candidates.append((f1_macro, accuracy, best_model_path, info))

    if not candidates:
        raise FileNotFoundError(
            f"No valid {family_name} model for pipeline '{pipeline}' found in {family_dir}"
        )

    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    _, _, best_path, best_info = candidates[0]
    return best_path, best_info


def write_probability_columns(df: pd.DataFrame, prefix: str, probabilities: List[List[float]]) -> None:
    if not probabilities:
        return
    n_classes = len(probabilities[0])
    for idx in range(n_classes):
        df[f"{prefix}_prob_star_{idx + 1}"] = [row[idx] for row in probabilities]


def parse_model_weights(raw_value: str) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for part in raw_value.split(","):
        token = part.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(
                "Invalid --voting-weights format. Expected comma-separated values like: "
                "bert=0.5,lr=0.15,svm=0.15,nb=0.2"
            )
        name, value = token.split("=", 1)
        key = name.strip().lower()
        if key not in {"bert", "lr", "svm", "nb"}:
            raise ValueError(f"Unknown model in --voting-weights: {key}")
        parsed[key] = float(value)

    for required in ["bert", "lr", "svm", "nb"]:
        if required not in parsed:
            raise ValueError(f"Missing weight for model '{required}' in --voting-weights")
    return parsed


def normalize_probability_rows(prob_matrix: np.ndarray) -> np.ndarray:
    row_sums = prob_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums <= 0.0] = 1.0
    return prob_matrix / row_sums


def probabilities_to_predictions(prob_matrix: np.ndarray) -> Dict[str, List[float]]:
    normalized = normalize_probability_rows(prob_matrix)
    pred_label = np.argmax(normalized, axis=1)
    pred_conf = np.max(normalized, axis=1)
    return {
        "pred_label": pred_label.tolist(),
        "pred_stars": (pred_label + 1).tolist(),
        "pred_confidence": pred_conf.tolist(),
        "pred_probabilities": normalized.tolist(),
    }


def compute_soft_voting(
    bert_probs: np.ndarray,
    lr_probs: np.ndarray,
    svm_probs: np.ndarray,
    nb_probs: np.ndarray,
) -> Dict[str, List[float]]:
    avg_probs = (bert_probs + lr_probs + svm_probs + nb_probs) / 4.0
    return probabilities_to_predictions(avg_probs)


def compute_weighted_voting(
    bert_probs: np.ndarray,
    lr_probs: np.ndarray,
    svm_probs: np.ndarray,
    nb_probs: np.ndarray,
    bert_conf: np.ndarray,
    lr_conf: np.ndarray,
    svm_conf: np.ndarray,
    nb_conf: np.ndarray,
    model_weights: Dict[str, float],
) -> Dict[str, List[float]]:
    stacked_probs = np.stack([bert_probs, lr_probs, svm_probs, nb_probs], axis=0)
    base_weights = np.array(
        [
            model_weights["bert"],
            model_weights["lr"],
            model_weights["svm"],
            model_weights["nb"],
        ],
        dtype=float,
    )
    conf_matrix = np.vstack([bert_conf, lr_conf, svm_conf, nb_conf]).astype(float)
    conf_matrix = np.nan_to_num(conf_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    # Weighted soft-vote: global model reliability weights scaled by per-row confidence.
    dynamic_weights = base_weights[:, None] * np.clip(conf_matrix, 1e-6, 1.0)
    denom = dynamic_weights.sum(axis=0, keepdims=True)
    denom[denom <= 0.0] = 1.0
    normalized_weights = dynamic_weights / denom

    weighted_probs = np.sum(stacked_probs * normalized_weights[:, :, None], axis=0)
    return probabilities_to_predictions(weighted_probs)


def valid_stacking_labels(stars_series: pd.Series) -> Optional[np.ndarray]:
    y = pd.to_numeric(stars_series, errors="coerce")
    if y.isna().any():
        return None
    y_int = y.round().astype(int)
    if not y_int.between(1, 5).all():
        return None
    return (y_int.to_numpy(dtype=int) - 1)


def compute_stacking_predictions(
    bert_probs: np.ndarray,
    lr_probs: np.ndarray,
    svm_probs: np.ndarray,
    nb_probs: np.ndarray,
    bert_conf: np.ndarray,
    lr_conf: np.ndarray,
    svm_conf: np.ndarray,
    nb_conf: np.ndarray,
    stars_series: Optional[pd.Series],
    mode: str,
    n_splits: int,
    random_state: int,
    weighted_vote_fallback: Dict[str, List[float]],
) -> Tuple[Dict[str, List[float]], str]:
    if mode == "none":
        return weighted_vote_fallback, "fallback_weighted_mode_none"

    if stars_series is None:
        return weighted_vote_fallback, "fallback_weighted_no_labels"

    y = valid_stacking_labels(stars_series)
    if y is None:
        return weighted_vote_fallback, "fallback_weighted_invalid_labels"

    class_counts = np.bincount(y, minlength=5)
    min_class_count = int(class_counts[class_counts > 0].min()) if np.any(class_counts > 0) else 0
    effective_splits = min(n_splits, min_class_count)
    if effective_splits < 2:
        return weighted_vote_fallback, "fallback_weighted_insufficient_class_samples"

    features = np.hstack(
        [
            bert_probs,
            lr_probs,
            svm_probs,
            nb_probs,
            bert_conf.reshape(-1, 1),
            lr_conf.reshape(-1, 1),
            svm_conf.reshape(-1, 1),
            nb_conf.reshape(-1, 1),
        ]
    )

    oof_probs = np.zeros((features.shape[0], 5), dtype=float)
    skf = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

    lr_init_params = inspect.signature(LogisticRegression.__init__).parameters
    meta_kwargs: Dict[str, object] = {"max_iter": 2000}
    if "class_weight" in lr_init_params:
        meta_kwargs["class_weight"] = "balanced"
    if "random_state" in lr_init_params:
        meta_kwargs["random_state"] = random_state

    for train_idx, valid_idx in skf.split(features, y):
        model = LogisticRegression(**meta_kwargs)
        model.fit(features[train_idx], y[train_idx])
        fold_probs = model.predict_proba(features[valid_idx])

        aligned_probs = np.zeros((len(valid_idx), 5), dtype=float)
        for local_idx, cls in enumerate(model.classes_):
            aligned_probs[:, int(cls)] = fold_probs[:, local_idx]
        oof_probs[valid_idx] = aligned_probs

    return probabilities_to_predictions(oof_probs), f"cv_logreg_{effective_splits}fold"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run BERT/LR/SVM/NB predictions for one pipeline (balanced or natural) "
            "and append/update Datasets/all_models_predictions.csv."
        )
    )
    parser.add_argument("--pipeline", type=str, choices=["balanced", "natural"], required=True)
    parser.add_argument("--datasets-dir", type=Path, default=Path("Datasets"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help=(
            "Output CSV path. If omitted, defaults to "
            "Datasets/all_models_predictions_<pipeline>.csv"
        ),
    )
    parser.add_argument("--text-col", type=str, default="text")

    parser.add_argument("--bert-model-base-dir", type=Path, default=Path("BERT/fine_tuned_models"))
    parser.add_argument("--lr-selection-dir", type=Path, default=Path("LR/final_model_selection"))
    parser.add_argument("--svm-selection-dir", type=Path, default=Path("SVM/final_model_selection"))
    parser.add_argument("--nb-selection-dir", type=Path, default=Path("NB/final_model_selection"))

    parser.add_argument("--bert-batch-size", type=int, default=64)
    parser.add_argument("--bert-max-length", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--mixed-precision", choices=["auto", "none", "fp16", "bf16"], default="auto")

    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument(
        "--voting-weights",
        type=str,
        default="bert=0.50,lr=0.15,svm=0.15,nb=0.20",
        help="Global model weights for weighted voting, e.g. bert=0.5,lr=0.15,svm=0.15,nb=0.2",
    )
    parser.add_argument(
        "--stacking-mode",
        choices=["auto", "none"],
        default="auto",
        help="auto: fit CV logistic stacking when labels are available, otherwise fallback to weighted voting.",
    )
    parser.add_argument("--stacking-cv-folds", type=int, default=5)
    parser.add_argument("--stacking-random-state", type=int, default=42)
    parser.add_argument("--overwrite-output", action="store_true")
    parser.add_argument(
        "--replace-pipeline-rows",
        action="store_true",
        help="If output exists, remove old rows for this pipeline before appending new predictions.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Optional debug limit: predict only the first N rows of the selected test set.",
    )
    args = parser.parse_args()

    model_weights = parse_model_weights(args.voting_weights)

    if args.output_csv is None:
        args.output_csv = args.datasets_dir / f"all_models_predictions_{args.pipeline}.csv"

    input_csv = args.datasets_dir / f"test_{args.pipeline}.csv"
    if not input_csv.exists():
        raise FileNotFoundError(f"Test CSV not found: {input_csv}")

    bert_model_dir = args.bert_model_base_dir / args.pipeline
    if not bert_model_dir.exists():
        raise FileNotFoundError(f"BERT model directory not found: {bert_model_dir}")

    lr_model_path, lr_info = find_best_classical_model(
        family_name="LR",
        family_dir=args.lr_selection_dir,
        pipeline=args.pipeline,
    )
    svm_model_path, svm_info = find_best_classical_model(
        family_name="SVM",
        family_dir=args.svm_selection_dir,
        pipeline=args.pipeline,
    )
    nb_model_path, nb_info = find_best_classical_model(
        family_name="NB",
        family_dir=args.nb_selection_dir,
        pipeline=args.pipeline,
    )

    print(f"Pipeline: {args.pipeline}")
    print(f"Input CSV: {input_csv}")
    print(f"Selected LR model: {lr_model_path} (setup={lr_info['setup_folder']}, f1={lr_info['test_f1_macro']:.6f})")
    print(f"Selected SVM model: {svm_model_path} (setup={svm_info['setup_folder']}, f1={svm_info['test_f1_macro']:.6f})")
    print(f"Selected NB model: {nb_model_path} (setup={nb_info['setup_folder']}, f1={nb_info['test_f1_macro']:.6f})")

    df = pd.read_csv(input_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in input CSV: {input_csv}")

    if args.max_rows > 0:
        df = df.head(args.max_rows).copy()

    # Keep a stable row id for reproducibility across appended pipeline runs.
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", np.arange(len(df), dtype=int))
    df.insert(1, "evaluation_pipeline", args.pipeline)

    texts = df[args.text_col].astype(str).tolist()

    device = resolve_device(args.device)
    autocast_dtype = resolve_autocast_dtype(args.mixed_precision)
    bert_max_length = args.bert_max_length if args.bert_max_length > 0 else infer_bert_max_length(bert_model_dir)

    print(f"BERT model directory: {bert_model_dir}")
    print(f"BERT max_length used: {bert_max_length}")

    bert_preds = predict_bert(
        texts=texts,
        model_dir=bert_model_dir,
        batch_size=args.bert_batch_size,
        max_length=bert_max_length,
        device=device,
        autocast_dtype=autocast_dtype,
    )
    df["bert_pred_label"] = bert_preds["pred_label"]
    df["bert_pred_stars"] = bert_preds["pred_stars"]
    df["bert_pred_confidence"] = bert_preds["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "bert", bert_preds.get("pred_probabilities", []))

    lr_estimator = load_classical_estimator(lr_model_path)
    lr_preds = predict_classical(lr_estimator, texts)
    df["lr_pred_label"] = lr_preds["pred_label"]
    df["lr_pred_stars"] = lr_preds["pred_stars"]
    df["lr_pred_confidence"] = lr_preds["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "lr", lr_preds.get("pred_probabilities", []))

    svm_estimator = load_classical_estimator(svm_model_path)
    svm_preds = predict_classical(svm_estimator, texts)
    df["svm_pred_label"] = svm_preds["pred_label"]
    df["svm_pred_stars"] = svm_preds["pred_stars"]
    df["svm_pred_confidence"] = svm_preds["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "svm", svm_preds.get("pred_probabilities", []))

    nb_estimator = load_classical_estimator(nb_model_path)
    nb_preds = predict_classical(nb_estimator, texts)
    df["nb_pred_label"] = nb_preds["pred_label"]
    df["nb_pred_stars"] = nb_preds["pred_stars"]
    df["nb_pred_confidence"] = nb_preds["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "nb", nb_preds.get("pred_probabilities", []))

    bert_probs = np.asarray(bert_preds["pred_probabilities"], dtype=float)
    lr_probs = np.asarray(lr_preds["pred_probabilities"], dtype=float)
    svm_probs = np.asarray(svm_preds["pred_probabilities"], dtype=float)
    nb_probs = np.asarray(nb_preds["pred_probabilities"], dtype=float)

    bert_conf = np.asarray(bert_preds["pred_confidence"], dtype=float)
    lr_conf = np.asarray(lr_preds["pred_confidence"], dtype=float)
    svm_conf = np.asarray(svm_preds["pred_confidence"], dtype=float)
    nb_conf = np.asarray(nb_preds["pred_confidence"], dtype=float)

    soft_vote = compute_soft_voting(
        bert_probs=bert_probs,
        lr_probs=lr_probs,
        svm_probs=svm_probs,
        nb_probs=nb_probs,
    )
    df["soft_vote_pred_label"] = soft_vote["pred_label"]
    df["soft_vote_pred_stars"] = soft_vote["pred_stars"]
    df["soft_vote_pred_confidence"] = soft_vote["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "soft_vote", soft_vote["pred_probabilities"])

    weighted_vote = compute_weighted_voting(
        bert_probs=bert_probs,
        lr_probs=lr_probs,
        svm_probs=svm_probs,
        nb_probs=nb_probs,
        bert_conf=bert_conf,
        lr_conf=lr_conf,
        svm_conf=svm_conf,
        nb_conf=nb_conf,
        model_weights=model_weights,
    )
    df["weighted_vote_pred_label"] = weighted_vote["pred_label"]
    df["weighted_vote_pred_stars"] = weighted_vote["pred_stars"]
    df["weighted_vote_pred_confidence"] = weighted_vote["pred_confidence"]
    if args.save_probabilities:
        write_probability_columns(df, "weighted_vote", weighted_vote["pred_probabilities"])

    stacking_preds, stacking_mode_used = compute_stacking_predictions(
        bert_probs=bert_probs,
        lr_probs=lr_probs,
        svm_probs=svm_probs,
        nb_probs=nb_probs,
        bert_conf=bert_conf,
        lr_conf=lr_conf,
        svm_conf=svm_conf,
        nb_conf=nb_conf,
        stars_series=df["stars"] if "stars" in df.columns else None,
        mode=args.stacking_mode,
        n_splits=args.stacking_cv_folds,
        random_state=args.stacking_random_state,
        weighted_vote_fallback=weighted_vote,
    )
    df["stacking_pred_label"] = stacking_preds["pred_label"]
    df["stacking_pred_stars"] = stacking_preds["pred_stars"]
    df["stacking_pred_confidence"] = stacking_preds["pred_confidence"]
    df["stacking_mode_used"] = stacking_mode_used
    if args.save_probabilities:
        write_probability_columns(df, "stacking", stacking_preds["pred_probabilities"])

    df["bert_model_dir"] = str(bert_model_dir)
    df["lr_model_path"] = str(lr_model_path)
    df["svm_model_path"] = str(svm_model_path)
    df["nb_model_path"] = str(nb_model_path)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.overwrite_output or not args.output_csv.exists():
        output_df = df
    else:
        existing = pd.read_csv(args.output_csv)
        if args.replace_pipeline_rows and "evaluation_pipeline" in existing.columns:
            existing = existing[existing["evaluation_pipeline"] != args.pipeline]
        output_df = pd.concat([existing, df], ignore_index=True)

    output_df.to_csv(args.output_csv, index=False)

    print(f"Saved predictions to: {args.output_csv}")
    print(f"Rows added for pipeline '{args.pipeline}': {len(df)}")
    print(f"Total rows in output: {len(output_df)}")
    print(f"Stacking mode used: {stacking_mode_used}")


if __name__ == "__main__":
    main()
