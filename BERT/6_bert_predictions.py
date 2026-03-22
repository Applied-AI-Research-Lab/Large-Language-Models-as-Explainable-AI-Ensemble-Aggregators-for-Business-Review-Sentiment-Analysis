# python3 BERT/6_bert_predictions.py --model-dir BERT/fine_tuned_models/balanced --input-csv Datasets/test_balanced.csv --output-csv BERT/Predictions/preds_balanced_on_balanced.csv --mixed-precision bf16 --save-probabilities
# python3 BERT/6_bert_predictions.py --model-dir BERT/fine_tuned_models/natural --input-csv Datasets/test_natural.csv --output-csv BERT/Predictions/preds_natural_on_natural.csv --mixed-precision bf16 --save-probabilities

# Cross-pipeline evaluation
# python3 BERT/6_bert_predictions.py --model-dir BERT/fine_tuned_models/balanced --input-csv Datasets/test_natural.csv --output-csv BERT/Predictions/preds_balanced_on_natural.csv --mixed-precision bf16 --save-probabilities
# python3 BERT/6_bert_predictions.py --model-dir BERT/fine_tuned_models/natural --input-csv Datasets/test_balanced.csv --output-csv BERT/Predictions/preds_natural_on_balanced.csv --mixed-precision bf16 --save-probabilities

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import torch
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


def predict_chunk(
    model,
    tokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
    autocast_dtype: Optional[torch.dtype],
) -> Dict[str, List]:
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

    return {
        "pred_label": all_pred_labels,
        "pred_stars": all_pred_stars,
        "pred_confidence": all_confidences,
        "pred_probabilities": all_probabilities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run BERT predictions on a CSV and save outputs incrementally."
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Path to fine-tuned model directory (e.g., BERT/fine_tuned_models/balanced).",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="CSV to predict on (e.g., Datasets/test_balanced.csv).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Output CSV path for predictions.",
    )
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--chunksize", type=int, default=5000)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Inference device.",
    )
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--save-probabilities",
        action="store_true",
        help="Save class probability columns (prob_star_1 ... prob_star_5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")

    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output file already exists: {args.output_csv}. Use --overwrite to replace it."
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    if args.output_csv.exists() and args.overwrite:
        args.output_csv.unlink()

    device = resolve_device(args.device)
    autocast_dtype = resolve_autocast_dtype(args.mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(str(args.model_dir))
    model = AutoModelForSequenceClassification.from_pretrained(str(args.model_dir))
    model.to(device)
    model.eval()

    total_rows = 0
    chunk_id = 0
    wrote_header = False

    for chunk in pd.read_csv(args.input_csv, chunksize=args.chunksize):
        chunk_id += 1
        if args.text_col not in chunk.columns:
            raise ValueError(f"Text column '{args.text_col}' not found in input CSV.")

        chunk[args.text_col] = chunk[args.text_col].astype(str)
        texts = chunk[args.text_col].tolist()

        predictions = predict_chunk(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device,
            autocast_dtype=autocast_dtype,
        )

        chunk["pred_label"] = predictions["pred_label"]
        chunk["pred_stars"] = predictions["pred_stars"]
        chunk["pred_confidence"] = predictions["pred_confidence"]

        if args.save_probabilities:
            probs = predictions["pred_probabilities"]
            expected_labels = model.config.num_labels
            for label_idx in range(expected_labels):
                col = f"prob_star_{label_idx + 1}"
                chunk[col] = [row[label_idx] for row in probs]

        chunk.to_csv(
            args.output_csv,
            mode="a",
            header=not wrote_header,
            index=False,
        )
        wrote_header = True

        total_rows += len(chunk)
        print(
            f"Processed chunk {chunk_id}: +{len(chunk)} rows (total written: {total_rows}) -> {args.output_csv}"
        )

    print("Prediction completed.")
    print(f"Model used: {args.model_dir}")
    print(f"Input CSV: {args.input_csv}")
    print(f"Output CSV: {args.output_csv}")
    print(f"Total rows written: {total_rows}")


if __name__ == "__main__":
    main()