from __future__ import annotations


def compute_reward(
    success: float,
    decision_correct: float,
    slot_f1: float,
    num_questions: float,
    hallucination: float,
) -> float:
    return (
        1.0 * success
        + 0.5 * decision_correct
        + 0.5 * slot_f1
        - 0.2 * num_questions
        - 1.0 * hallucination
    )
