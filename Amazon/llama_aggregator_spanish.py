#!/usr/bin/env python3
"""
LLaMA aggregator for Spanish Amazon Reviews using Unsloth.
Aggregates predictions from base models (LR, SVM, NB, BERT) with reasoning.

Usage:
    python3 Amazon/llama_aggregator_spanish.py --model unsloth/Llama-3.3-70B-Instruct
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

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


def safe_float(value: object) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def build_messages(
    review_text: str,
    bert_star: object,
    bert_conf: object,
    lr_star: object,
    lr_conf: object,
    svm_star: object,
    svm_conf: object,
    nb_star: object,
    nb_conf: object,
    reasoning_sentence_instruction: str,
) -> list[dict[str, str]]:
    """Build Spanish prompt messages for aggregator."""
    # Convert None values to "N/A" for safe formatting
    def fmt(val):
        return "N/A" if val is None else str(val)
    
    prompt = (
        "Eres un experto en análisis de reseñas de Amazon. "
        "Se te proporciona una reseña y las predicciones de 4 modelos con sus confianzas: BERT, LR, SVM, NB. "
        "Usa tanto el sentimiento del texto como las predicciones de los modelos para elegir una calificación final de 1-5 estrellas.\n\n"
        "Reglas:\n"
        "1) Devuelve ÚNICAMENTE JSON, sin markdown, sin texto adicional.\n"
        '2) Esquema JSON: {"final_star": <int 1-5>, "reasoning": <string>}\n'
        f"3) El razonamiento debe ser {reasoning_sentence_instruction} y mencionar acuerdo/desacuerdo entre modelos.\n\n"
        f"Texto de la reseña:\n{review_text}\n\n"
        "Predicciones de modelos:\n"
        f"- BERT: estrella={fmt(bert_star)}, confianza={fmt(bert_conf)}\n"
        f"- LR: estrella={fmt(lr_star)}, confianza={fmt(lr_conf)}\n"
        f"- SVM: estrella={fmt(svm_star)}, confianza={fmt(svm_conf)}\n"
        f"- NB: estrella={fmt(nb_star)}, confianza={fmt(nb_conf)}\n"
    )
    return [{"role": "user", "content": prompt}]


JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
STAR_DIGIT_RE = re.compile(r"\b([1-5])\b")


def parse_llama_json(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse JSON response from LLaMA."""
    candidate = text.strip()
    
    # Extract JSON block
    match = JSON_BLOCK_RE.search(candidate)
    if match is not None:
        candidate = match.group(0)
    
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None, None
    
    star = payload.get("final_star")
    reasoning = payload.get("reasoning")
    
    star_int: Optional[int] = None
    if isinstance(star, int) and 1 <= star <= 5:
        star_int = star
    elif isinstance(star, str) and star.strip().isdigit():
        parsed = int(star.strip())
        if 1 <= parsed <= 5:
            star_int = parsed
    
    reasoning_str: Optional[str] = None
    if isinstance(reasoning, str):
        cleaned = " ".join(reasoning.split())
        if cleaned:
            reasoning_str = cleaned
    
    return star_int, reasoning_str


def fallback_star_from_models(
    bert_star: object,
    bert_conf: object,
    lr_star: object,
    lr_conf: object,
    svm_star: object,
    svm_conf: object,
    nb_star: object,
    nb_conf: object,
    default_star: int,
) -> int:
    """Calculate weighted fallback star from model predictions."""
    weighted_votes = []
    for star, conf in (
        (bert_star, bert_conf),
        (lr_star, lr_conf),
        (svm_star, svm_conf),
        (nb_star, nb_conf),
    ):
        try:
            star_i = int(star)
        except (TypeError, ValueError):
            continue
        if star_i < 1 or star_i > 5:
            continue
        conf_f = safe_float(conf)
        weighted_votes.append((star_i, conf_f if conf_f is not None else 0.0))
    
    if not weighted_votes:
        return default_star
    
    totals = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
    for star_i, conf_f in weighted_votes:
        totals[star_i] += conf_f
    
    best = max(totals.items(), key=lambda kv: kv[1])[0]
    return int(best)


def predict_one(
    model,
    tokenizer,
    review_text: str,
    bert_star: object,
    bert_conf: object,
    lr_star: object,
    lr_conf: object,
    svm_star: object,
    svm_conf: object,
    nb_star: object,
    nb_conf: object,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    min_p: float,
    fallback_star: int,
    max_json_retries: int,
    reasoning_sentence_instruction: str,
) -> Tuple[int, str, str]:
    """Generate aggregated prediction for a single review."""
    messages = build_messages(
        review_text=review_text,
        bert_star=bert_star,
        bert_conf=bert_conf,
        lr_star=lr_star,
        lr_conf=lr_conf,
        svm_star=svm_star,
        svm_conf=svm_conf,
        nb_star=nb_star,
        nb_conf=nb_conf,
        reasoning_sentence_instruction=reasoning_sentence_instruction,
    )
    
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
    
    decoded = ""
    star: Optional[int] = None
    reasoning: Optional[str] = None
    
    total_attempts = max_json_retries + 1
    for _attempt in range(total_attempts):
        output = model.generate(**generate_kwargs)
        generated = output[0, inputs.shape[-1]:]
        decoded = tokenizer.decode(generated, skip_special_tokens=True).strip()
        star, reasoning = parse_llama_json(decoded)
        if star is not None and reasoning is not None:
            break
    
    if star is None:
        star = fallback_star_from_models(
            bert_star=bert_star,
            bert_conf=bert_conf,
            lr_star=lr_star,
            lr_conf=lr_conf,
            svm_star=svm_star,
            svm_conf=svm_conf,
            nb_star=nb_star,
            nb_conf=nb_conf,
            default_star=fallback_star,
        )
    
    if reasoning is None:
        match = STAR_DIGIT_RE.search(decoded)
        extracted = match.group(1) if match else "N/A"
        reasoning = (
            f"Fallback usado porque el parseo JSON falló. "
            f"El texto del modelo contenía token de estrella: {extracted}."
        )
    
    return int(star), reasoning, decoded


def run_aggregation(
    model,
    tokenizer,
    input_csv: Path,
    output_csv: Path,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    min_p: float,
    fallback_star: int,
    max_json_retries: int,
    reasoning_sentence_instruction: str,
    resume: bool,
    write_every: int,
    log_every: int,
) -> None:
    """Run aggregation on predictions file."""
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    df = pd.read_csv(input_csv)
    
    # Required columns
    required_cols = [
        "text", "true_label",
        "bert_pred_stars", "bert_pred_confidence",
        "lr_pred_stars", "lr_pred_confidence",
        "svm_pred_stars", "svm_pred_confidence",
        "nb_pred_stars", "nb_pred_confidence",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Setup output columns
    output_columns = list(df.columns) + [
        "llama_agg_pred_label",
        "llama_agg_pred_stars",
        "llama_agg_reasoning",
        "llama_agg_raw_response",
    ]
    
    start_idx = 0
    has_output = output_csv.exists()
    
    if has_output and resume:
        existing_cols = list(pd.read_csv(output_csv, nrows=0).columns)
        required_out = ["llama_agg_pred_label", "llama_agg_pred_stars", "llama_agg_reasoning", "llama_agg_raw_response"]
        missing_out = [c for c in required_out if c not in existing_cols]
        if missing_out:
            raise ValueError(f"Cannot resume: output file missing columns: {missing_out}")
        
        output_columns = existing_cols
        row_count_probe_col = existing_cols[0]
        start_idx = len(pd.read_csv(output_csv, usecols=[row_count_probe_col]))
        
        if start_idx >= len(df):
            print(f"[{output_csv.name}] already complete ({start_idx}/{len(df)} rows).")
            return
        
        print(f"[{output_csv.name}] resuming from row {start_idx} of {len(df)}")
    
    total = len(df)
    pending_rows: list[dict] = []
    
    for pos in range(start_idx, total):
        row_dict = df.iloc[pos].to_dict()
        
        final_star, final_reason, raw_text = predict_one(
            model=model,
            tokenizer=tokenizer,
            review_text=str(row_dict["text"]),
            bert_star=row_dict["bert_pred_stars"],
            bert_conf=row_dict["bert_pred_confidence"],
            lr_star=row_dict["lr_pred_stars"],
            lr_conf=row_dict["lr_pred_confidence"],
            svm_star=row_dict["svm_pred_stars"],
            svm_conf=row_dict["svm_pred_confidence"],
            nb_star=row_dict["nb_pred_stars"],
            nb_conf=row_dict["nb_pred_confidence"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            min_p=min_p,
            fallback_star=fallback_star,
            max_json_retries=max_json_retries,
            reasoning_sentence_instruction=reasoning_sentence_instruction,
        )
        
        out_row = dict(row_dict)
        out_row["llama_agg_pred_label"] = final_star - 1  # Convert to 0-4
        out_row["llama_agg_pred_stars"] = final_star
        out_row["llama_agg_reasoning"] = final_reason
        out_row["llama_agg_raw_response"] = raw_text
        pending_rows.append(out_row)
        
        should_flush = (len(pending_rows) >= write_every) or (pos == total - 1)
        if should_flush:
            chunk_df = pd.DataFrame(pending_rows).reindex(columns=output_columns)
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            chunk_df.to_csv(
                output_csv,
                mode="a" if has_output else "w",
                header=not has_output,
                index=False,
            )
            has_output = True
            pending_rows.clear()
        
        done = pos + 1
        if log_every > 0 and (done % log_every == 0 or done == total):
            print(f"[{output_csv.name}] processed {done}/{total}")
    
    print(f"Saved aggregated predictions: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate base model predictions using LLaMA reasoning for Spanish Amazon"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unsloth/Llama-3.3-70B-Instruct",
        help="Model name or path",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("Amazon/all_models_predictions_spanish.csv"),
        help="Input CSV with base model predictions",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("Amazon/all_models_predictions_spanish_llama_reasoned.csv"),
        help="Output CSV path",
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
        default=120,
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
        "--fallback-star",
        type=int,
        default=3,
        help="Fallback star rating if parsing fails",
    )
    parser.add_argument(
        "--max-json-retries",
        type=int,
        default=2,
        help="Maximum retries for JSON parsing",
    )
    parser.add_argument(
        "--reasoning-sentences",
        type=str,
        default="2-3 oraciones",
        help="Instruction for reasoning length",
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
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output file",
    )
    args = parser.parse_args()
    
    if args.fallback_star < 1 or args.fallback_star > 5:
        raise ValueError("--fallback-star must be between 1 and 5")
    if args.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be > 0")
    if args.temperature < 0:
        raise ValueError("--temperature must be >= 0")
    if args.min_p < 0 or args.min_p > 1:
        raise ValueError("--min-p must be in [0, 1]")
    if args.max_json_retries < 0:
        raise ValueError("--max-json-retries must be >= 0")
    if args.write_every <= 0:
        raise ValueError("--write-every must be > 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this script.")
    
    try:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency 'unsloth'. Install it in your environment."
        ) from exc
    
    # Load model
    print(f"Loading model: {args.model}")
    dtype = resolve_dtype(args.dtype)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model,
        max_seq_length=args.max_seq_length,
        dtype=dtype,
        load_in_4bit=args.load_in_4bit,
    )
    tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")
    FastLanguageModel.for_inference(model)
    
    print("Model loaded. Starting aggregation...\n")
    
    # Run aggregation
    run_aggregation(
        model=model,
        tokenizer=tokenizer,
        input_csv=args.input_csv,
        output_csv=args.output_csv,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        min_p=args.min_p,
        fallback_star=args.fallback_star,
        max_json_retries=args.max_json_retries,
        reasoning_sentence_instruction=args.reasoning_sentences,
        resume=args.resume,
        write_every=args.write_every,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load predictions
    print("Loading base model predictions...")
    preds_df = pd.read_csv(args.predictions_file)
    
    # Load test data for texts
    test_df = pd.read_csv(args.data_dir / "test_balanced.csv")
    
    # Merge to get texts
    merged = preds_df.merge(test_df[['review_id', 'text']], on='review_id', how='left')
    
    # Check for existing predictions
    output_file = args.output_dir / f"llama_aggregator_spanish_{args.model.replace(':', '_')}.csv"
    if output_file.exists():
        existing = pd.read_csv(output_file)
        start_idx = len(existing)
        print(f"Resuming from index {start_idx}")
    else:
        existing = None
        start_idx = 0
    
    results = []
    if existing is not None:
        results = existing.to_dict('records')
    
    print(f"\nGenerating aggregator predictions with {args.model}...")
    print(f"Total samples: {len(merged)}")
    print(f"Starting from: {start_idx}")
    
    for idx in range(start_idx, len(merged)):
        if idx % args.batch_size == 0:
            print(f"  Processing {idx}/{len(merged)}...")
            if results:
                pd.DataFrame(results).to_csv(output_file, index=False)
        
        row = merged.iloc[idx]
        text = row['text']
        true_label = row['true_label']
        
        # Get base predictions (convert from 0-4 to 1-5 for prompt)
        base_preds = {}
        if 'lr_pred' in row:
            base_preds['LR'] = int(row['lr_pred']) if pd.notna(row['lr_pred']) else None
        if 'svm_pred' in row:
            base_preds['SVM'] = int(row['svm_pred']) if pd.notna(row['svm_pred']) else None
        if 'nb_pred' in row:
            base_preds['NB'] = int(row['nb_pred']) if pd.notna(row['nb_pred']) else None
        if 'bert_pred' in row:
            base_preds['BERT'] = int(row['bert_pred']) if pd.notna(row['bert_pred']) else None
        
        prompt = create_aggregator_prompt(text, base_preds)
        response = query_ollama(prompt, args.model)
        pred = extract_rating(response)
        
        results.append({
            'review_id': row['review_id'],
            'true_label': true_label,
            'predicted_label': pred - 1 if pred else None,  # Convert to 0-4
            'raw_response': response,
        })
    
    # Save final results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    
    # Calculate accuracy
    valid_preds = results_df[results_df['predicted_label'].notna()]
    if len(valid_preds) > 0:
        accuracy = (valid_preds['true_label'] == valid_preds['predicted_label']).mean()
        print(f"\n✓ Aggregator Accuracy: {accuracy:.4f} ({len(valid_preds)}/{len(results_df)} valid)")
    
    print(f"✓ Predictions saved to {output_file}")


if __name__ == "__main__":
    main()
