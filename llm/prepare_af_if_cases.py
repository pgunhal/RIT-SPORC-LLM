from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--document-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--doc-id", default="doc-1")
    parser.add_argument("--dataset", default="custom")
    parser.add_argument(
        "--question",
        default="Does this document contain PII? Cite the supporting evidence.",
    )
    parser.add_argument(
        "--instruction",
        default="Focus only on the specified evidence type.",
    )
    parser.add_argument("--model-evidence-span", default="")
    parser.add_argument("--gold-evidence-span", default="")
    parser.add_argument("--instruction-evidence-span", default="")
    parser.add_argument("--if-instruction", default="")
    return parser.parse_args()


def read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def remove_first(text: str, span: str) -> str:
    if not span:
        return text
    return text.replace(span, "", 1)


def write_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")


def make_row(
    doc_id: str,
    dataset: str,
    document: str,
    instruction: str,
    question: str,
    variant: str,
) -> dict:
    return {
        "id": f"{doc_id}-{variant}",
        "dataset": dataset,
        "document": document,
        "instruction": instruction,
        "question": question,
        "variant": variant,
    }


def main() -> None:
    args = parse_args()
    document = read_text(args.document_file)
    output_dir = Path(args.output_dir)

    rows: list[tuple[str, dict]] = []

    rows.append(
        (
            "baseline.jsonl",
            make_row(
                doc_id=args.doc_id,
                dataset=args.dataset,
                document=document,
                instruction=args.instruction,
                question=args.question,
                variant="baseline",
            ),
        )
    )

    if args.model_evidence_span:
        rows.append(
            (
                "af_remove_model_evidence.jsonl",
                make_row(
                    doc_id=args.doc_id,
                    dataset=args.dataset,
                    document=remove_first(document, args.model_evidence_span),
                    instruction=args.instruction,
                    question=args.question,
                    variant="af_remove_model_evidence",
                ),
            )
        )

    if args.gold_evidence_span:
        rows.append(
            (
                "af_remove_gold_evidence.jsonl",
                make_row(
                    doc_id=args.doc_id,
                    dataset=args.dataset,
                    document=remove_first(document, args.gold_evidence_span),
                    instruction=args.instruction,
                    question=args.question,
                    variant="af_remove_gold_evidence",
                ),
            )
        )

    if args.if_instruction:
        rows.append(
            (
                "if_baseline.jsonl",
                make_row(
                    doc_id=args.doc_id,
                    dataset=args.dataset,
                    document=document,
                    instruction=args.if_instruction,
                    question=args.question,
                    variant="if_baseline",
                ),
            )
        )

    if args.if_instruction and args.instruction_evidence_span:
        rows.append(
            (
                "if_remove_instruction_span.jsonl",
                make_row(
                    doc_id=args.doc_id,
                    dataset=args.dataset,
                    document=remove_first(document, args.instruction_evidence_span),
                    instruction=args.if_instruction,
                    question=args.question,
                    variant="if_remove_instruction_span",
                ),
            )
        )

    for filename, row in rows:
        write_jsonl(output_dir / filename, row)

    print(f"wrote {len(rows)} AF/IF case files to {output_dir}")


if __name__ == "__main__":
    main()
