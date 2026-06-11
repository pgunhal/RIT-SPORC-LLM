from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from prompts import SYSTEM_PROMPT, build_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # CHANGE HERE:
    # - input/output paths
    # - model name
    # - generation length
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    import torch

    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return getattr(torch, dtype_name)


def build_generation_kwargs(tokenizer, max_new_tokens: int) -> dict[str, Any]:
    # CHANGE HERE:
    # - Set do_sample=True for stochastic generation.
    # - Tune temperature/top_p for sampling behavior.
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "pad_token_id": tokenizer.eos_token_id,
    }


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=args.hf_device_map,
        torch_dtype=resolve_dtype(args.hf_torch_dtype),
    )

    input_rows = load_jsonl(args.input)
    output_rows = []
    generation_kwargs = build_generation_kwargs(tokenizer, args.max_new_tokens)

    for row in input_rows:
        # CHANGE HERE:
        # - Remap these keys if your input JSONL uses different field names.
        user_prompt = build_user_prompt(
            document=row["document"],
            question=row.get("question", "Does this document contain PII? Cite the supporting evidence."),
            instruction=row.get("instruction"),
            few_shot_examples=[],
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer(rendered, return_tensors="pt").to(model.device)
        generated = model.generate(**model_inputs, **generation_kwargs)
        new_tokens = generated[0][model_inputs["input_ids"].shape[1] :]
        response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        output_rows.append(
            {
                # CHANGE HERE:
                # - Add parsed labels, confidence, evidence spans, or gold labels here.
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "model_name_or_path": args.model_name_or_path,
                "prompt_type": "zero_shot",
                "response_text": response_text,
                "input": row,
            }
        )

    save_jsonl(args.output, output_rows)
    print(f"saved {len(output_rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
