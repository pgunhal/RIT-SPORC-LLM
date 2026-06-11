from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prompts import SYSTEM_PROMPT, FewShotExample, build_user_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # CHANGE HERE:
    # - This script compares multiple prompt settings on the same input rows.
    parser.add_argument("--input", required=True)
    parser.add_argument("--few-shot-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name-or-path", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--hf-device-map", default="auto")
    parser.add_argument("--hf-torch-dtype", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    return parser.parse_args()


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: str) -> list[dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
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
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
        "top_k": None,
        "pad_token_id": tokenizer.eos_token_id,
    }


def load_few_shot_pool(path: str) -> list[FewShotExample]:
    data = load_json(path)
    return [FewShotExample(**row) for row in data["examples"]]


def select_examples(
    pool: list[FewShotExample],
    target_dataset: str,
    shot_count: int,
    incorrect_count: int,
) -> list[FewShotExample]:
    # CHANGE HERE:
    # - Replace this with same-dataset, random, balanced, or task-specific sampling if needed.
    cross_dataset = [row for row in pool if row.dataset != target_dataset]
    incorrect = [row for row in cross_dataset if row.is_intentionally_incorrect]
    correct = [row for row in cross_dataset if not row.is_intentionally_incorrect]
    picked = incorrect[:incorrect_count]
    picked.extend(correct[: max(shot_count - len(picked), 0)])
    return picked[:shot_count]


def run_variant(
    tokenizer,
    model,
    input_rows: list[dict[str, Any]],
    few_shot_pool: list[FewShotExample],
    shot_count: int,
    incorrect_count: int,
    max_new_tokens: int,
    model_name_or_path: str,
) -> list[dict[str, Any]]:
    output_rows = []
    generation_kwargs = build_generation_kwargs(tokenizer, max_new_tokens)
    for row in input_rows:
        examples = select_examples(
            pool=few_shot_pool,
            target_dataset=row.get("dataset", ""),
            shot_count=shot_count,
            incorrect_count=incorrect_count,
        )
        user_prompt = build_user_prompt(
            document=row["document"],
            question=row.get("question", "Does this document contain PII? Cite the supporting evidence."),
            instruction=row.get("instruction"),
            few_shot_examples=examples,
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
                # - Add metadata useful for your analysis.
                "id": row.get("id"),
                "dataset": row.get("dataset"),
                "model_name_or_path": model_name_or_path,
                "shot_count": shot_count,
                "incorrect_demo_count": incorrect_count,
                "selected_examples": [asdict(example) for example in examples],
                "response_text": response_text,
                "input": row,
            }
        )
    return output_rows


def main() -> None:
    args = parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    input_rows = load_jsonl(args.input)
    few_shot_pool = load_few_shot_pool(args.few_shot_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map=args.hf_device_map,
        torch_dtype=resolve_dtype(args.hf_torch_dtype),
    )

    # CHANGE HERE:
    # - Replace these with the prompt conditions you want to compare.
    variants = [
        {"shot_count": 0, "incorrect_count": 0},
        {"shot_count": 2, "incorrect_count": 0},
        {"shot_count": 4, "incorrect_count": 1},
        {"shot_count": 6, "incorrect_count": 2},
    ]

    output_dir = Path(args.output_dir)
    summary = []
    for variant in variants:
        rows = run_variant(
            tokenizer=tokenizer,
            model=model,
            input_rows=input_rows,
            few_shot_pool=few_shot_pool,
            shot_count=variant["shot_count"],
            incorrect_count=variant["incorrect_count"],
            max_new_tokens=args.max_new_tokens,
            model_name_or_path=args.model_name_or_path,
        )
        output_path = output_dir / (
            f"shots_{variant['shot_count']}_incorrect_{variant['incorrect_count']}.jsonl"
        )
        save_jsonl(output_path, rows)
        summary.append({"output_path": str(output_path), **variant})

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"saved prompt sweep outputs to {output_dir}")


if __name__ == "__main__":
    main()
