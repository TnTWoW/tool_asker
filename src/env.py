from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from src.reward import compute_reward


@dataclass
class EnvStepResult:
    done: bool
    reward: float
    user_reply: Dict[str, str]
    success: bool


def simulate_user_reply(gold_missing_slots: Set[str], predicted_slots: Set[str], gold_values: Dict[str, str]) -> Dict[str, str]:
    reply: Dict[str, str] = {}
    for slot in predicted_slots:
        if slot in gold_missing_slots and slot in gold_values:
            reply[slot] = str(gold_values[slot])
    return reply


def single_turn_step(
    predicted_decision: str,
    predicted_slots: Set[str],
    gold_decision: str,
    gold_missing_slots: Set[str],
    gold_values: Dict[str, str],
) -> EnvStepResult:
    should_ask = gold_decision == "ask"
    if predicted_decision == "ask":
        reply = simulate_user_reply(gold_missing_slots, predicted_slots, gold_values)
    else:
        reply = {}

    success = (not should_ask and predicted_decision == "act") or (
        should_ask and predicted_decision == "ask" and gold_missing_slots.issubset(set(reply.keys()))
    )
    decision_correct = 1.0 if predicted_decision == gold_decision else 0.0
    slot_tp = len(gold_missing_slots & predicted_slots)
    slot_precision = slot_tp / len(predicted_slots) if predicted_slots else 0.0
    slot_recall = slot_tp / len(gold_missing_slots) if gold_missing_slots else 0.0
    slot_f1 = (
        2 * slot_precision * slot_recall / (slot_precision + slot_recall)
        if (slot_precision + slot_recall) > 0
        else 0.0
    )
    hallucination = 1.0 if len(predicted_slots - gold_missing_slots) > 0 else 0.0
    reward = compute_reward(
        success=1.0 if success else 0.0,
        decision_correct=decision_correct,
        slot_f1=slot_f1,
        num_questions=1.0 if predicted_decision == "ask" else 0.0,
        hallucination=hallucination,
    )
    return EnvStepResult(done=True, reward=reward, user_reply=reply, success=success)
