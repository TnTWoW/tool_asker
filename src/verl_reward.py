from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, Set


_SECTION_PATTERN = re.compile(
    r"\[(DECISION|GAP_TYPE|MISSING_SLOTS|QUESTION)\]\s*(.*?)(?=\n\[[A-Z_]+\]|\Z)",
    re.DOTALL,
)


def _normalize_text(value: Any) -> str:
    return str(value or "").strip()


def _normalize_label(value: Any) -> str:
    return _normalize_text(value).lower()


def _parse_slots(value: Any) -> Set[str]:
    text = _normalize_label(value)
    if not text or text in {"none", "null", "n/a"}:
        return set()

    parts = re.split(r"[\n,]", text)
    normalized = set()
    for part in parts:
        item = part.strip().strip("-").strip("*").strip()
        if item and item not in {"none", "null", "n/a"}:
            normalized.add(item)
    return normalized


def _parse_response(solution_str: str) -> Dict[str, str]:
    parsed: Dict[str, str] = {}
    for key, value in _SECTION_PATTERN.findall(solution_str):
        parsed[key.lower()] = value.strip()
    return parsed


def _load_ground_truth(ground_truth: Any) -> Dict[str, Any]:
    if isinstance(ground_truth, dict):
        return ground_truth
    if isinstance(ground_truth, str):
        text = ground_truth.strip()
        if not text:
            return {}
        return json.loads(text)
    raise TypeError(f"Unsupported ground_truth type: {type(ground_truth)!r}")


def _f1_score(predicted: Set[str], gold: Set[str]) -> float:
    if not predicted and not gold:
        return 1.0
    if not predicted or not gold:
        return 0.0

    tp = len(predicted & gold)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(gold) if gold else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _all_gold_slots_collected(predicted: Set[str], gold: Set[str]) -> bool:
    return gold.issubset(predicted)


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Dict[str, Any] | None = None,
) -> Dict[str, float]:
    del data_source
    del extra_info

    gold = _load_ground_truth(ground_truth)
    parsed = _parse_response(solution_str)

    predicted_decision = _normalize_label(parsed.get("decision"))
    predicted_gap_type = _normalize_label(parsed.get("gap_type"))
    predicted_slots = _parse_slots(parsed.get("missing_slots"))
    predicted_question = _normalize_text(parsed.get("question"))

    gold_decision = _normalize_label(gold.get("decision"))
    gold_gap_type = _normalize_label(gold.get("gap_type"))
    gold_slots = _parse_slots(gold.get("missing_slots", []))

    decision_correct = 1.0 if predicted_decision == gold_decision else 0.0
    gap_type_correct = 1.0 if predicted_gap_type == gold_gap_type else 0.0
    slot_f1 = _f1_score(predicted_slots, gold_slots)
    hallucination = 1.0 if predicted_slots - gold_slots else 0.0

    asked_question = predicted_decision == "ask"
    question_present = 1.0 if (asked_question and predicted_question) else 0.0
    num_questions = 1.0 if asked_question else 0.0

    if gold_decision == "ask":
        success = 1.0 if asked_question and _all_gold_slots_collected(predicted_slots, gold_slots) else 0.0
    else:
        success = 1.0 if predicted_decision == gold_decision else 0.0

    score = (
        1.0 * success
        + 0.5 * decision_correct
        + 0.3 * gap_type_correct
        + 0.5 * slot_f1
        + 0.1 * question_present
        - 0.2 * num_questions
        - 1.0 * hallucination
    )

    return {
        "score": float(score),
        "success": float(success),
        "decision_correct": float(decision_correct),
        "gap_type_correct": float(gap_type_correct),
        "slot_f1": float(slot_f1),
        "question_present": float(question_present),
        "hallucination": float(hallucination),
    }
