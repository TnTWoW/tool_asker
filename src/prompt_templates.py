from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List

from src.data_schema import GapSample


def render_instruction(gap: GapSample) -> str:
    spec: Dict[str, Any] = {
        "tool_name": gap.available_tool_spec.tool_name,
        "tool_description": gap.available_tool_spec.tool_description,
        "required_params": gap.available_tool_spec.required_params,
        "known_arguments": gap.available_tool_spec.known_arguments,
        "api_base_url": gap.available_tool_spec.api_base_url,
        "auth_type": gap.available_tool_spec.auth_type,
    }
    lines: List[str] = [
        "You are an assistant that must decide whether to ask clarifying questions before calling a tool.",
        f"User query: {gap.user_query}",
        f"Available tool spec: {json.dumps(spec, ensure_ascii=False)}",
    ]
    if gap.candidate_tools:
        lines.append(f"Candidate tools: {json.dumps(gap.candidate_tools, ensure_ascii=False)}")
    lines.append("Output format must contain [DECISION], [GAP_TYPE], [MISSING_SLOTS], [QUESTION].")
    return "\n".join(lines)


def render_soft_three_head_target(
    decision: str,
    gap_type: str,
    missing_slots: Iterable[str],
    question: str,
) -> str:
    slot_text = ", ".join(list(missing_slots)) if missing_slots else "none"
    return (
        "[DECISION]\n"
        f"{decision}\n\n"
        "[GAP_TYPE]\n"
        f"{gap_type}\n\n"
        "[MISSING_SLOTS]\n"
        f"{slot_text}\n\n"
        "[QUESTION]\n"
        f"{question}"
    )
