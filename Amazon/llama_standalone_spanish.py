#!/usr/bin/env python3
"""
LLaMA standalone predictions for Spanish Amazon Reviews using Unsloth.
Zero-shot classification with Spanish prompts.

Usage:
    python3 Amazon/llama_standalone_spanish.py --model unsloth/Llama-3.3-70B-Instruct
"""

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import torch


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


def create_messages(text: str) -> list[dict[str, str]]:
    """Create Spanish prompt messages for rating prediction."""
    prompt = (
        "Analiza esta reseña de Amazon y predice su calificación de estrellas (1-5).\n\n"
        f'Reseña: "{text}"\n\n'
        "Responde ÚNICAMENTE con un JSON en este formato exacto:\n"
        '{"rating": <int 1-5>, "reasoning": <string>}\n\n'
        "No escribas nada más que el JSON."
    )
    return [{"role": "user", "content": prompt}]


def parse_rating(text: str) -> tuple[Optional[int], Optional[str]]:
    """Extract rating and reasoning from LLM response."""
    text = text.strip()
    
    # Try to parse as JSON
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            rating = payload.get("rating")
            reasoning = payload.get("reasoning")
            
            if isinstance(rating, int) and 1 <= rating <= 5:
                return rating, reasoning
            elif isinstance(rating, str) and rating.strip().isdigit():
                parsed = int(rating.strip())
                if 1 <= parsed <= 5:
                    return parsed, reasoning
        except json.JSONDecodeError:
            pass
    
    # Fallback: look for single digit 1-5
    match = re.search(r'\b([1-5])\b', text)
    if match:
        return int(match.group(1)), None
    
    return None, None


def predict_one(
    model,
    tokenizer,
    review_text: str,
    max_new_tokens: int = 60,
    do_sample: bool = False,
    temperature: float = 0.2,
    min_p: float = 0.1,
) -> tuple[Optional[int], Optional[str], str]:
    """Generate prediction for a single review."""
    messages = create_messages(review_text)
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")
    
    generate_kwargs = {
        "input_ids": inputs,
        "max_new_tokens": max_new_tokens,
        "use_cache": True,
    }
    
    if do_sample:
        generate_kwargs.update({
            "do_sample": True,
            "temperature": temperature,
            "min_p": min_p,
        })
    
    output = model.generate(**generate_kwargs)
    generated = output[0, inputs.shape[-1]:]
    decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    rating, reasoning = parse_rating(decoded)
    return rating, reasoning, decoded


def run_predictions(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    output_file: Path,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    min_p: float,
    resume: bool,
    write_every: int,
    log_every: int,
) -> None:
    """Run predictions on test set."""
    
    start_idx = 0
    if output_file.exists() and resume:
        existing = pd.read_csv(output_file)
        start_idx = len(existing)
        print(f"Resuming from index {start_idx}")
        results = existing.to_dict('records')
    else:
        results = []
    
    print(f"\nGenerating predictions...")
    print(f"Total samples: {len(test_df)}")
    print(f"Starting from: {start_idx}")
    
    for idx in range(start_idx, len(test_df)):
        if idx % log_every == 0:
            print(f"  Processing {idx}/{len(test_df)}...")
        
        row = test_df.iloc[idx]
        text = row['text']
        true_label = row['stars']
        
        rating, reasoning, raw_response = predict_one(
            model=model,
            tokenizer=tokenizer,
            review_text=text,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            min_p=min_p,
        )
        
        results.append({
            'review_id': row['review_id'],
            'true_label': true_label - 1,  # Convert to 0-4
            'predicted_label': rating - 1 if rating else None,
            'predicted_rating': rating,
            'reasoning': reasoning,
            'raw_response': raw_response,
        })
        
        # Save intermediate results
        if (idx + 1) % write_every == 0 or idx == len(test_df) - 1:
            pd.DataFrame(results).to_csv(output_file, index=False)
    
    # Final save
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Calculate accuracy
    valid_preds = results_df[results_df['predicted_label'].notna()]
    if len(valid_preds) > 0:
        accuracy = (valid_preds['true_label'] == valid_preds['predicted_label']).mean()
        print(f"\n✓ Accuracy: {accuracy:.4f} ({len(valid_preds)}/{len(results_df)} valid)")
    
    print(f"✓ Predictions saved to {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLaMA standalone predictions for Spanish Amazon Reviews"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.3-70B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Amazon"),
        help="Directory containing test_balanced.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("Amazon/llama_predictions"),
        help="Output directory for predictions",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float16", "bfloat16", "float32"],
        default="auto",
        help="Data type for model",
    )
    parser.add_argument(
        "--load-in-4bit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load model in 4-bit quantization",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=60,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--do-sample",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampling for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.1,
        help="Minimum p for sampling",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from existing output file",
    )
    parser.add_argument(
        "--write-every",
        type=int,
        default=50,
        help="Save progress every N samples",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=100,
        help="Log progress every N samples",
    )
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")
    
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'unsloth'. Install it in your environment."
        ) from exc
    
    # Load data
    print("Loading test data...")
    test_file = args.data_dir / "test_balanced.csv"
    if not test_file.exists():
        raise FileNotFoundError(f"Test file not found: {test_file}")
    test_df = pd.read_csv(test_file)
    
    # Setup output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_name_safe = args.model.replace("/", "_").replace(":", "_")
    output_file = args.output_dir / f"llama_standalone_spanish_{model_name_safe}.csv"
    
    # Load model
    print(f"\nLoading model: {args.model}")
    dtype = resolve_dtype(args.dtype)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    
    print("Model loaded. Starting predictions...\n")
    
    # Run predictions
    run_predictions(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        output_file=output_file,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        min_p=args.min_p,
        resume=args.resume,
        write_every=args.write_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
