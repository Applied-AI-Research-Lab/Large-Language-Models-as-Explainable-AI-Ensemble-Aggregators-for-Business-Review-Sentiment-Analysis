# Check setup-llama.md

import argparse
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

STAR_PATTERN = re.compile(r"\b([1-5])\b")


def resolve_dtype(dtype_name: str) -> Optional[torch.dtype]:
    if dtype_name == "auto":
        return None
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def parse_star_from_text(text: str) -> Optional[int]:
    match = STAR_PATTERN.search(text)
    if match is None:
        return None
    return int(match.group(1))


def build_messages(review_text: str) -> list[dict[str, str]]:
    prompt = (
        "You are an expert Yelp rating classifier. "
        "Given the review text, predict the star rating from 1 to 5. "
        "Return ONLY one digit: 1, 2, 3, 4, or 5.\n\n"
        f"Review:\n{review_text}"
    )
    return [{"role": "user", "content": prompt}]


def predict_one(
    model,
    tokenizer,
    review_text: str,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    fallback_star: int,
) -> Tuple[int, str, float]:
    messages = build_messages(review_text)
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    output = model.generate(
        input_ids=inputs,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        do_sample=True,
        temperature=temperature,
        min_p=min_p,
        return_dict_in_generate=True,
        output_scores=True,
    )

    generated = output.sequences[0, inputs.shape[-1] :]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()

    star = parse_star_from_text(decoded)
    if star is None:
        star = fallback_star

    # Confidence proxy: probability of the first generated token.
    confidence = float("nan")
    if output.scores:
        first_scores = output.scores[0][0]
        first_token_id = int(generated[0].item()) if generated.numel() > 0 else None
        if first_token_id is not None:
            probs = torch.softmax(first_scores, dim=-1)
            confidence = float(probs[first_token_id].item())

    return star, decoded, confidence


def run_file(
    model,
    tokenizer,
    input_csv: Path,
    output_csv: Path,
    text_col: str,
    max_new_tokens: int,
    temperature: float,
    min_p: float,
    fallback_star: int,
    overwrite: bool,
    log_every: int,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if output_csv.exists() and not overwrite:
        raise FileExistsError(f"Output CSV already exists: {output_csv}. Use --overwrite.")

    df = pd.read_csv(input_csv)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in {input_csv}")

    pred_stars = []
    pred_labels = []
    pred_texts = []
    pred_confidences = []

    total = len(df)
    for idx, text in enumerate(df[text_col].astype(str).tolist(), start=1):
        star, raw_text, conf = predict_one(
            model=model,
            tokenizer=tokenizer,
            review_text=text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            min_p=min_p,
            fallback_star=fallback_star,
        )
        pred_stars.append(star)
        pred_labels.append(star - 1)
        pred_texts.append(raw_text)
        pred_confidences.append(conf)

        if log_every > 0 and (idx % log_every == 0 or idx == total):
            print(f"[{output_csv.name}] processed {idx}/{total}")

    df["llama_pred_label"] = pred_labels
    df["llama_pred_stars"] = pred_stars
    df["llama_pred_confidence"] = pred_confidences
    df["llama_pred_text"] = pred_texts

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"Saved predictions: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LLaMA (Unsloth) star predictions for all_models_predictions CSV files."
    )
    parser.add_argument("--mode", choices=["balanced", "natural", "both"], default="both")
    parser.add_argument(
        "--balanced-input",
        type=Path,
        default=Path("Datasets/all_models_predictions_balanced.csv"),
    )
    parser.add_argument(
        "--natural-input",
        type=Path,
        default=Path("Datasets/all_models_predictions_natural.csv"),
    )
    parser.add_argument(
        "--balanced-output",
        type=Path,
        default=Path("Datasets/all_models_predictions_balanced_llama.csv"),
    )
    parser.add_argument(
        "--natural-output",
        type=Path,
        default=Path("Datasets/all_models_predictions_natural_llama.csv"),
    )
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument("--fallback-star", type=int, default=3)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.fallback_star < 1 or args.fallback_star > 5:
        raise ValueError("--fallback-star must be between 1 and 5")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.min_p < 0 or args.min_p > 1:
        raise ValueError("--min-p must be in [0, 1]")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this Unsloth LLaMA inference script.")

    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'unsloth'. Install it in your environment before running this script."
        ) from exc

    dtype = resolve_dtype(args.dtype)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)

    if args.mode in {"balanced", "both"}:
        run_file(
            model=model,
            tokenizer=tokenizer,
            input_csv=args.balanced_input,
            output_csv=args.balanced_output,
            text_col=args.text_col,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            fallback_star=args.fallback_star,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )

    if args.mode in {"natural", "both"}:
        run_file(
            model=model,
            tokenizer=tokenizer,
            input_csv=args.natural_input,
            output_csv=args.natural_output,
            text_col=args.text_col,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            min_p=args.min_p,
            fallback_star=args.fallback_star,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
