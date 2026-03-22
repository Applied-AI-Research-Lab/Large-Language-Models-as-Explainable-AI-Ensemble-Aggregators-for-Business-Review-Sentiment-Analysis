# Check setup-llama.md

import argparse
import json
import math
import re
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd
import torch

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)
STAR_DIGIT_RE = re.compile(r"\b([1-5])\b")


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
    prompt = (
        "You are an expert Yelp rating adjudicator. "
        "You are given one review text and 4 model predictions with confidences: BERT, LR, SVM, NB. "
        "Use both text sentiment and model evidence to choose a final star in [1,2,3,4,5].\n\n"
        "Rules:\n"
        "1) Return STRICT JSON only, no markdown, no extra text.\n"
        "2) JSON schema: {\"final_star\": <int 1-5>, \"reasoning\": <string>}\n"
        f"3) reasoning must be {reasoning_sentence_instruction} and mention model agreement/disagreement.\n\n"
        f"Review text:\n{review_text}\n\n"
        "Model predictions:\n"
        f"- BERT: star={bert_star}, confidence={bert_conf}\n"
        f"- LR: star={lr_star}, confidence={lr_conf}\n"
        f"- SVM: star={svm_star}, confidence={svm_conf}\n"
        f"- NB: star={nb_star}, confidence={nb_conf}\n"
    )
    return [{"role": "user", "content": prompt}]


def parse_llama_json(text: str) -> Tuple[Optional[int], Optional[str]]:
    candidate = text.strip()

    # Handle fenced code blocks or extra chatter by extracting the first JSON object.
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
        generate_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "min_p": min_p,
            }
        )

    decoded = ""
    star: Optional[int] = None
    reasoning: Optional[str] = None

    total_attempts = max_json_retries + 1
    for _attempt in range(total_attempts):
        output = model.generate(**generate_kwargs)
        generated = output[0, inputs.shape[-1] :]
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
        # Best effort fallback if model returns non-JSON text.
        match = STAR_DIGIT_RE.search(decoded)
        extracted = match.group(1) if match else "N/A"
        reasoning = (
            "Fallback used because JSON parse failed. "
            f"Model text contained star token: {extracted}."
        )

    return int(star), reasoning, decoded


def run_file(
    model,
    tokenizer,
    input_csv: Path,
    output_csv: Path,
    text_col: str,
    bert_star_col: str,
    bert_conf_col: str,
    lr_star_col: str,
    lr_conf_col: str,
    svm_star_col: str,
    svm_conf_col: str,
    nb_star_col: str,
    nb_conf_col: str,
    output_star_col: str,
    output_reason_col: str,
    output_raw_col: str,
    output_label_col: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float,
    min_p: float,
    fallback_star: int,
    max_json_retries: int,
    reasoning_sentence_instruction: str,
    resume: bool,
    write_every: int,
    overwrite: bool,
    log_every: int,
) -> None:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    if overwrite and resume:
        raise ValueError("Cannot use --overwrite and --resume together.")

    df = pd.read_csv(input_csv)

    required_cols = [
        text_col,
        bert_star_col,
        bert_conf_col,
        lr_star_col,
        lr_conf_col,
        svm_star_col,
        svm_conf_col,
        nb_star_col,
        nb_conf_col,
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {missing}")

    # Default behavior: if output already exists, continue from it unless overwrite is requested.
    effective_resume = resume or (output_csv.exists() and not overwrite)

    if output_csv.exists() and overwrite:
        output_csv.unlink()

    output_columns = list(df.columns) + [
        output_label_col,
        output_star_col,
        output_reason_col,
        output_raw_col,
    ]

    start_idx = 0
    if output_csv.exists() and effective_resume:
        existing_cols = list(pd.read_csv(output_csv, nrows=0).columns)
        required_out = [output_label_col, output_star_col, output_reason_col, output_raw_col]
        missing_out = [c for c in required_out if c not in existing_cols]
        if missing_out:
            raise ValueError(
                f"Cannot resume: output file is missing expected columns: {missing_out}"
            )

        output_columns = existing_cols
        row_count_probe_col = existing_cols[0]
        start_idx = len(pd.read_csv(output_csv, usecols=[row_count_probe_col]))

        if start_idx > len(df):
            raise ValueError(
                f"Cannot resume: output has {start_idx} rows but input has only {len(df)} rows."
            )

        if start_idx == len(df):
            print(f"[{output_csv.name}] already complete ({start_idx}/{len(df)} rows).")
            return

        print(f"[{output_csv.name}] resuming from row {start_idx} of {len(df)}")

    total = len(df)
    pending_rows: list[dict] = []
    has_output = output_csv.exists()

    for pos in range(start_idx, total):
        row_dict = df.iloc[pos].to_dict()
        final_star, final_reason, raw_text = predict_one(
            model=model,
            tokenizer=tokenizer,
            review_text=str(row_dict[text_col]),
            bert_star=row_dict[bert_star_col],
            bert_conf=row_dict[bert_conf_col],
            lr_star=row_dict[lr_star_col],
            lr_conf=row_dict[lr_conf_col],
            svm_star=row_dict[svm_star_col],
            svm_conf=row_dict[svm_conf_col],
            nb_star=row_dict[nb_star_col],
            nb_conf=row_dict[nb_conf_col],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            min_p=min_p,
            fallback_star=fallback_star,
            max_json_retries=max_json_retries,
            reasoning_sentence_instruction=reasoning_sentence_instruction,
        )

        out_row = dict(row_dict)
        out_row[output_label_col] = final_star - 1
        out_row[output_star_col] = final_star
        out_row[output_reason_col] = final_reason
        out_row[output_raw_col] = raw_text
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
        description=(
            "Aggregate BERT/LR/SVM/NB predictions using LLaMA 3.3 reasoning and save final star + reasoning columns."
        )
    )
    parser.add_argument("--mode", choices=["balanced", "natural", "both"], default="both")

    parser.add_argument(
        "--balanced-input",
        type=Path,
        default=Path("Datasets/all_models_predictions_balanced_llama.csv"),
    )
    parser.add_argument(
        "--natural-input",
        type=Path,
        default=Path("Datasets/all_models_predictions_natural_llama.csv"),
    )
    parser.add_argument(
        "--balanced-output",
        type=Path,
        default=Path("Datasets/all_models_predictions_balanced_llama_reasoned.csv"),
    )
    parser.add_argument(
        "--natural-output",
        type=Path,
        default=Path("Datasets/all_models_predictions_natural_llama_reasoned.csv"),
    )

    parser.add_argument("--text-col", type=str, default="text")

    parser.add_argument("--bert-star-col", type=str, default="bert_pred_stars")
    parser.add_argument("--bert-conf-col", type=str, default="bert_pred_confidence")
    parser.add_argument("--lr-star-col", type=str, default="lr_pred_stars")
    parser.add_argument("--lr-conf-col", type=str, default="lr_pred_confidence")
    parser.add_argument("--svm-star-col", type=str, default="svm_pred_stars")
    parser.add_argument("--svm-conf-col", type=str, default="svm_pred_confidence")
    parser.add_argument("--nb-star-col", type=str, default="nb_pred_stars")
    parser.add_argument("--nb-conf-col", type=str, default="nb_pred_confidence")

    parser.add_argument("--output-label-col", type=str, default="llama_agg_pred_label")
    parser.add_argument("--output-star-col", type=str, default="llama_agg_pred_stars")
    parser.add_argument("--output-reason-col", type=str, default="llama_agg_reasoning")
    parser.add_argument("--output-raw-col", type=str, default="llama_agg_raw_response")

    parser.add_argument("--model-name", type=str, default="unsloth/Llama-3.3-70B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--min-p", type=float, default=0.1)
    parser.add_argument("--fallback-star", type=int, default=3)
    parser.add_argument("--max-json-retries", type=int, default=2)
    parser.add_argument(
        "--reasoning-sentences",
        type=str,
        default="2-3 sentences",
        help="Instruction text for reasoning length (example: '2-3 sentences').",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--write-every", type=int, default=50)

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
    if args.max_json_retries < 0:
        raise ValueError("--max-json-retries must be >= 0")
    if args.write_every <= 0:
        raise ValueError("--write-every must be > 0")
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
            bert_star_col=args.bert_star_col,
            bert_conf_col=args.bert_conf_col,
            lr_star_col=args.lr_star_col,
            lr_conf_col=args.lr_conf_col,
            svm_star_col=args.svm_star_col,
            svm_conf_col=args.svm_conf_col,
            nb_star_col=args.nb_star_col,
            nb_conf_col=args.nb_conf_col,
            output_star_col=args.output_star_col,
            output_reason_col=args.output_reason_col,
            output_raw_col=args.output_raw_col,
            output_label_col=args.output_label_col,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            min_p=args.min_p,
            fallback_star=args.fallback_star,
            max_json_retries=args.max_json_retries,
            reasoning_sentence_instruction=args.reasoning_sentences,
            resume=args.resume,
            write_every=args.write_every,
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
            bert_star_col=args.bert_star_col,
            bert_conf_col=args.bert_conf_col,
            lr_star_col=args.lr_star_col,
            lr_conf_col=args.lr_conf_col,
            svm_star_col=args.svm_star_col,
            svm_conf_col=args.svm_conf_col,
            nb_star_col=args.nb_star_col,
            nb_conf_col=args.nb_conf_col,
            output_star_col=args.output_star_col,
            output_reason_col=args.output_reason_col,
            output_raw_col=args.output_raw_col,
            output_label_col=args.output_label_col,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample,
            temperature=args.temperature,
            min_p=args.min_p,
            fallback_star=args.fallback_star,
            max_json_retries=args.max_json_retries,
            reasoning_sentence_instruction=args.reasoning_sentences,
            resume=args.resume,
            write_every=args.write_every,
            overwrite=args.overwrite,
            log_every=args.log_every,
        )


if __name__ == "__main__":
    main()
