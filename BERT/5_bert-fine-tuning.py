# python3 BERT/5_bert-fine-tuning.py --pipeline balanced --mixed-precision bf16
# python3 BERT/5_bert-fine-tuning.py --pipeline natural --mixed-precision bf16

import argparse
import inspect
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stars_to_labels(stars: pd.Series) -> pd.Series:
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
    data["label"] = stars_to_labels(data[star_col]).astype(int)
    return data[text_col].tolist(), data["label"].tolist()


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
    return TextClassificationDataset(encodings, labels)


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


def get_trainer_processing_key() -> str | None:
    parameters = inspect.signature(Trainer.__init__).parameters
    if "processing_class" in parameters:
        return "processing_class"
    if "tokenizer" in parameters:
        return "tokenizer"
    return None


def find_best_hyperparams_file(pipeline: str, tuning_base_dir: Path) -> Path:
    candidate_paths = [
        tuning_base_dir / pipeline / "best_model" / "best_hyperparameters.json",
        tuning_base_dir / "best_model" / "best_hyperparameters.json",
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find best hyperparameters JSON. Expected one of: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune BERT using best Bayesian-tuned hyperparameters for a selected pipeline."
    )
    parser.add_argument("--pipeline", choices=["balanced", "natural"], required=True)
    parser.add_argument("--datasets-dir", type=Path, default=Path("Datasets"))
    parser.add_argument("--tuning-base-dir", type=Path, default=Path("BERT/tuning_outputs"))
    parser.add_argument("--output-base-dir", type=Path, default=Path("BERT/fine_tuned_models"))
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--star-col", type=str, default="stars")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-text-chars", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument(
        "--mixed-precision",
        choices=["auto", "none", "fp16", "bf16"],
        default="auto",
    )
    parser.add_argument(
        "--evaluate-on-test",
        action="store_true",
        help="Evaluate final model on test_<pipeline>.csv (read-only evaluation).",
    )
    args = parser.parse_args()

    set_seed(args.seed)

    train_csv = args.datasets_dir / f"train_{args.pipeline}.csv"
    validation_csv = args.datasets_dir / f"validation_{args.pipeline}.csv"
    test_csv = args.datasets_dir / f"test_{args.pipeline}.csv"

    if not train_csv.exists():
        raise FileNotFoundError(f"Missing train split: {train_csv}")
    if not validation_csv.exists():
        raise FileNotFoundError(f"Missing validation split: {validation_csv}")
    if args.evaluate_on_test and not test_csv.exists():
        raise FileNotFoundError(f"Missing test split: {test_csv}")

    best_hparams_path = find_best_hyperparams_file(
        pipeline=args.pipeline,
        tuning_base_dir=args.tuning_base_dir,
    )
    with best_hparams_path.open("r", encoding="utf-8") as f:
        best_payload = json.load(f)

    best_params = best_payload.get("params")
    if not isinstance(best_params, dict):
        raise ValueError(f"Invalid best hyperparameters format in: {best_hparams_path}")

    required = [
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
    missing = [key for key in required if key not in best_params]
    if missing:
        raise ValueError(f"Missing required hyperparameters in tuning file: {missing}")

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
        raise ValueError(f"Expected 5 labels, found {num_labels} in training split.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    train_dataset = build_dataset(
        tokenizer=tokenizer,
        texts=train_texts,
        labels=train_labels,
        max_length=int(best_params["max_length"]),
    )
    val_dataset = build_dataset(
        tokenizer=tokenizer,
        texts=val_texts,
        labels=val_labels,
        max_length=int(best_params["max_length"]),
    )

    test_dataset = None
    if args.evaluate_on_test:
        test_texts, test_labels = load_split(
            csv_path=test_csv,
            text_col=args.text_col,
            star_col=args.star_col,
            min_text_chars=args.min_text_chars,
        )
        test_dataset = build_dataset(
            tokenizer=tokenizer,
            texts=test_texts,
            labels=test_labels,
            max_length=int(best_params["max_length"]),
        )

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        hidden_dropout_prob=float(best_params["dropout"]),
        attention_probs_dropout_prob=float(best_params["attention_dropout"]),
    )

    precision_flags = resolve_precision_mode(args.mixed_precision)
    run_output_dir = args.output_base_dir / args.pipeline
    run_output_dir.mkdir(parents=True, exist_ok=True)

    training_kwargs = {
        "output_dir": str(run_output_dir),
        "learning_rate": float(best_params["learning_rate"]),
        "per_device_train_batch_size": int(best_params["per_device_batch_size"]),
        "per_device_eval_batch_size": int(best_params["per_device_batch_size"]),
        "gradient_accumulation_steps": int(best_params["gradient_accumulation_steps"]),
        "num_train_epochs": int(best_params["num_train_epochs"]),
        "weight_decay": float(best_params["weight_decay"]),
        "warmup_ratio": float(best_params["warmup_ratio"]),
        "lr_scheduler_type": str(best_params["lr_scheduler_type"]),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "greater_is_better": True,
        "save_total_limit": 1,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "report_to": "none",
        "seed": args.seed,
        "dataloader_num_workers": args.num_workers,
        "fp16": precision_flags["fp16"],
        "bf16": precision_flags["bf16"],
        "remove_unused_columns": True,
    }
    training_kwargs[get_eval_strategy_key()] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "compute_metrics": compute_metrics,
    }
    processing_key = get_trainer_processing_key()
    if processing_key is not None:
        trainer_kwargs[processing_key] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    trainer.train()
    val_metrics = trainer.evaluate(eval_dataset=val_dataset)

    summary = {
        "pipeline": args.pipeline,
        "model_name": args.model_name,
        "train_csv": str(train_csv),
        "validation_csv": str(validation_csv),
        "best_hyperparameters_source": str(best_hparams_path),
        "best_hyperparameters": best_params,
        "validation_metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in val_metrics.items()},
        "output_model_dir": str(run_output_dir),
    }

    if test_dataset is not None:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        summary["test_csv"] = str(test_csv)
        summary["test_metrics"] = {
            k: float(v) if isinstance(v, (int, float)) else v for k, v in test_metrics.items()
        }

    trainer.save_model(str(run_output_dir))
    tokenizer.save_pretrained(str(run_output_dir))

    with (run_output_dir / "fine_tuning_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Pipeline: {args.pipeline}")
    print(f"Loaded best hyperparameters from: {best_hparams_path}")
    print(f"Saved fine-tuned model to: {run_output_dir}")
    print(f"Saved summary: {run_output_dir / 'fine_tuning_summary.json'}")


if __name__ == "__main__":
    main()