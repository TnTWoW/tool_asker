from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SoftThreeHeadOutput:
    decision: str
    gap_type: str
    missing_slots: List[str]
    question: str


def parse_soft_three_head(text: str) -> SoftThreeHeadOutput:
    sections = {"DECISION": "", "GAP_TYPE": "", "MISSING_SLOTS": "", "QUESTION": ""}
    active = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "[DECISION]":
            active = "DECISION"
            continue
        if line == "[GAP_TYPE]":
            active = "GAP_TYPE"
            continue
        if line == "[MISSING_SLOTS]":
            active = "MISSING_SLOTS"
            continue
        if line == "[QUESTION]":
            active = "QUESTION"
            continue
        if active and line:
            if sections[active]:
                sections[active] += " " + line
            else:
                sections[active] = line

    slots = [s.strip() for s in sections["MISSING_SLOTS"].split(",") if s.strip() and s.strip().lower() != "none"]
    return SoftThreeHeadOutput(
        decision=sections["DECISION"].lower(),
        gap_type=sections["GAP_TYPE"],
        missing_slots=slots,
        question=sections["QUESTION"],
    )
