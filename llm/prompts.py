from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


# CHANGE HERE:
# - This is the main task definition.
# - Rewrite this when switching tasks, for example:
#   - corruption prediction
#   - confidence scoring
#   - evidence extraction
#   - binary or multiclass classification
# - Keep the output format stable if you want easy downstream parsing.
SYSTEM_PROMPT = """You are a careful research assistant for evidence-grounded classification.
Answer the user's question using only the document text provided.
Treat the following as positive evidence of PII or sensitive identifiers when present in the document: names, email addresses, phone numbers, physical addresses, usernames tied to a person, account numbers, tax IDs, social security numbers, API keys, passwords, access tokens, and other credentials.
If the instruction narrows attention to a specific type of evidence, follow that instruction and ignore unrelated evidence.
State a binary answer first, then a confidence from 0 to 100, then an exact short evidence quote copied from the document.
Do not invent evidence that is not in the document."""


@dataclass(frozen=True)
class FewShotExample:
    dataset: str
    document: str
    question: str
    answer: str
    confidence: int
    evidence: str
    is_intentionally_incorrect: bool = False


def format_demo(example: FewShotExample) -> str:
    note = ""
    if example.is_intentionally_incorrect:
        note = " [demonstration intentionally incorrect for robustness testing]"
    return (
        f"Dataset: {example.dataset}\n"
        f"Document:\n{example.document}\n\n"
        f"Question: {example.question}\n"
        f"Assistant answer:{note}\n"
        f"Answer: {example.answer}\n"
        f"Confidence: {example.confidence}\n"
        f"Evidence: {example.evidence}\n"
    )


def build_user_prompt(
    document: str,
    question: str,
    instruction: str | None,
    few_shot_examples: Iterable[FewShotExample],
) -> str:
    # CHANGE HERE:
    # - This function controls the final task wording.
    # - Change the rules and output schema below for different tasks.
    sections = []
    shots = list(few_shot_examples)

    if shots:
        sections.append(
            "Here are prior labeled examples. Some may be intentionally incorrect for robustness testing."
        )
        for idx, shot in enumerate(shots, start=1):
            sections.append(f"Example {idx}\n{format_demo(shot)}")

    if instruction:
        sections.append(f"Instruction constraint:\n{instruction}")

    sections.append(
        "Now answer this instance.\n"
        f"Document:\n{document}\n\n"
        f"Question: {question}\n\n"
        "Use this exact output format:\n"
        "Answer: <YES or NO>\n"
        "Confidence: <0-100 integer>\n"
        "Evidence: <exact quote from the document or NONE>\n\n"
        "Rules:\n"
        "- If the document contains the instructed PII or sensitive identifier type, answer YES.\n"
        "- If the instruction says to focus only on one type, ignore other possible PII types.\n"
        "- Evidence must be copied verbatim from the document.\n"
        "- If you answer NO, use Evidence: NONE."
    )
    return "\n\n".join(sections)
