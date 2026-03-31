from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set


def safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def decision_accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return safe_div(correct, len(gold))


def gap_type_accuracy(gold: Sequence[str], pred: Sequence[str]) -> float:
    correct = sum(1 for g, p in zip(gold, pred) if g == p)
    return safe_div(correct, len(gold))


def set_f1(gold_sets: Sequence[Set[str]], pred_sets: Sequence[Set[str]]) -> float:
    precision_sum = 0.0
    recall_sum = 0.0
    count = len(gold_sets)
    for gold, pred in zip(gold_sets, pred_sets):
        tp = len(gold & pred)
        precision = safe_div(tp, len(pred))
        recall = safe_div(tp, len(gold))
        precision_sum += precision
        recall_sum += recall
    precision_avg = safe_div(precision_sum, count)
    recall_avg = safe_div(recall_sum, count)
    if precision_avg + recall_avg == 0:
        return 0.0
    return 2 * precision_avg * recall_avg / (precision_avg + recall_avg)


def question_sufficiency_rate(gold_slots: Sequence[Set[str]], predicted_questions: Sequence[str]) -> float:
    sufficient = 0
    for slots, question in zip(gold_slots, predicted_questions):
        q = question.lower()
        if all(slot.lower() in q for slot in slots):
            sufficient += 1
    return safe_div(sufficient, len(gold_slots))


def hallucination_rate(
    should_ask_flags: Sequence[bool],
    predicted_decisions: Sequence[str],
    predicted_missing_slots: Sequence[Set[str]],
    gold_missing_slots: Sequence[Set[str]],
) -> float:
    total = 0
    hallucinations = 0
    for should_ask, decision, pred_slots, gold_slots in zip(
        should_ask_flags,
        predicted_decisions,
        predicted_missing_slots,
        gold_missing_slots,
    ):
        if not should_ask:
            continue
        total += 1
        if decision != "ask":
            hallucinations += 1
            continue
        extra_slots = pred_slots - gold_slots
        if extra_slots:
            hallucinations += 1
    return safe_div(hallucinations, total)


@dataclass
class MetricBundle:
    decision_accuracy: float
    gap_type_accuracy: float
    missing_slot_f1: float
    question_sufficiency_rate: float
    hallucination_rate: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "decision_accuracy": round(self.decision_accuracy, 6),
            "gap_type_accuracy": round(self.gap_type_accuracy, 6),
            "missing_slot_f1": round(self.missing_slot_f1, 6),
            "question_sufficiency_rate": round(self.question_sufficiency_rate, 6),
            "hallucination_rate": round(self.hallucination_rate, 6),
        }
