# python3 LR/6_prediction_phase.py \
#   --model-path LR/final_model_selection/balanced_wider_250/best_model.joblib \
#   --input-csv Datasets/test_balanced.csv \
#   --output-csv LR/predictions/preds_lr_balanced_on_balanced.csv --overwrite

# python3 LR/6_prediction_phase.py \
#   --model-path LR/final_model_selection/natural_wider_250/best_model.joblib \
#   --input-csv Datasets/test_natural.csv \
#   --output-csv LR/predictions/preds_lr_natural_on_natural.csv --overwrite

import argparse
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Prediction phase for TF-IDF + Logistic Regression.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.model_path.exists():
        raise FileNotFoundError(f"Model not found: {args.model_path}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {args.output_csv}. Use --overwrite.")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() and args.overwrite:
        args.output_csv.unlink()

    bundle = joblib.load(args.model_path)
    model = bundle["estimator"] if isinstance(bundle, dict) and "estimator" in bundle else bundle

    wrote_header = False
    total_rows = 0
    chunk_idx = 0

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize):
        chunk_idx += 1
        if args.text_col not in chunk.columns:
            raise ValueError(f"Text column '{args.text_col}' not found in input CSV")

        texts: List[str] = chunk[args.text_col].astype(str).tolist()
        pred_labels = model.predict(texts)
        chunk["pred_label"] = pred_labels
        chunk["pred_stars"] = np.array(pred_labels) + 1

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(texts)
            chunk["pred_confidence"] = probs.max(axis=1)
            if args.save_probabilities:
                for idx in range(probs.shape[1]):
                    chunk[f"prob_star_{idx + 1}"] = probs[:, idx]
        else:
            chunk["pred_confidence"] = np.nan

        chunk.to_csv(args.output_csv, mode="a", header=not wrote_header, index=False)
        wrote_header = True
        total_rows += len(chunk)

        print(f"Processed chunk {chunk_idx}: +{len(chunk)} rows (total: {total_rows})")

    print(f"Prediction file saved: {args.output_csv}")


if __name__ == "__main__":
    main()
