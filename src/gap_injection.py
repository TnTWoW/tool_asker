from __future__ import annotations

import copy
import random
from typing import Dict, List, Sequence

from src.data_schema import APIGenSample, AvailableToolSpec, Decision, GapSample, GapType, ToolCall


def _build_tool_spec(base: APIGenSample, known_arguments: Dict[str, object]) -> AvailableToolSpec:
    return AvailableToolSpec(
        tool_name=base.tool_name,
        tool_description=base.tool_description,
        required_params=base.required_params,
        known_arguments=known_arguments,
        api_base_url=base.api_base_url,
        auth_type=base.auth_type,
    )


def inject_missing_required_parameter(base: APIGenSample, gap_id: str, rng: random.Random) -> GapSample:
    missing_count = min(len(base.required_params), 2)
    missing_count = 1 if missing_count == 1 else rng.randint(1, missing_count)
    missing_slots = rng.sample(base.required_params, missing_count)
    known_arguments = {k: v for k, v in base.arguments.items() if k not in set(missing_slots)}

    question = f"Please provide the following required parameter(s): {', '.join(missing_slots)}."
    after_reply = {k: base.arguments[k] for k in missing_slots}

    sample = GapSample(
        id=gap_id,
        base_sample_id=base.id,
        user_query=base.user_query,
        available_tool_spec=_build_tool_spec(base, known_arguments),
        gap_type=GapType.MISSING_REQUIRED_PARAMETER,
        gold_missing_slots=missing_slots,
        gold_decision=Decision.ASK,
        gold_question=question,
        gold_after_user_reply=after_reply,
        gold_final_call=ToolCall(tool_name=base.tool_name, arguments=copy.deepcopy(base.arguments)),
    )
    sample.validate()
    return sample


def inject_missing_api_endpoint(base: APIGenSample, gap_id: str) -> GapSample:
    known_arguments = copy.deepcopy(base.arguments)
    tool_spec = _build_tool_spec(base, known_arguments)
    tool_spec.api_base_url = None

    sample = GapSample(
        id=gap_id,
        base_sample_id=base.id,
        user_query=base.user_query,
        available_tool_spec=tool_spec,
        gap_type=GapType.MISSING_API_ENDPOINT,
        gold_missing_slots=["api_base_url"],
        gold_decision=Decision.ASK,
        gold_question="Please provide the API base URL or API documentation URL.",
        gold_after_user_reply={"api_base_url": base.api_base_url or "https://example.com"},
        gold_final_call=ToolCall(tool_name=base.tool_name, arguments=copy.deepcopy(base.arguments)),
    )
    sample.validate()
    return sample


def inject_missing_auth(base: APIGenSample, gap_id: str) -> GapSample:
    known_arguments = copy.deepcopy(base.arguments)
    tool_spec = _build_tool_spec(base, known_arguments)
    tool_spec.auth_type = None

    sample = GapSample(
        id=gap_id,
        base_sample_id=base.id,
        user_query=base.user_query,
        available_tool_spec=tool_spec,
        gap_type=GapType.MISSING_AUTH,
        gold_missing_slots=["auth_type"],
        gold_decision=Decision.ASK,
        gold_question="What authentication method is required (for example API key, bearer token, OAuth)?",
        gold_after_user_reply={"auth_type": base.auth_type or "api_key"},
        gold_final_call=ToolCall(tool_name=base.tool_name, arguments=copy.deepcopy(base.arguments)),
    )
    sample.validate()
    return sample


def inject_ambiguous_tool_choice(
    base: APIGenSample,
    distractor: APIGenSample,
    gap_id: str,
) -> GapSample:
    known_arguments = copy.deepcopy(base.arguments)
    tool_spec = _build_tool_spec(base, known_arguments)
    tool_spec.tool_name = "redacted_tool_name"

    candidate_tools: List[Dict[str, str]] = [
        {"tool_description": base.tool_description, "signature_hint": _signature_hint(base)},
        {"tool_description": distractor.tool_description, "signature_hint": _signature_hint(distractor)},
    ]

    sample = GapSample(
        id=gap_id,
        base_sample_id=base.id,
        user_query=base.user_query,
        available_tool_spec=tool_spec,
        gap_type=GapType.AMBIGUOUS_TOOL_CHOICE,
        gold_missing_slots=["tool_choice_disambiguation"],
        gold_decision=Decision.ASK,
        gold_question="To choose the right tool, can you clarify which entity you mean (for example order id vs shipment tracking id)?",
        gold_after_user_reply={"tool_choice_disambiguation": base.tool_name},
        gold_final_call=ToolCall(tool_name=base.tool_name, arguments=copy.deepcopy(base.arguments)),
        candidate_tools=candidate_tools,
    )
    sample.validate()
    return sample


def _signature_hint(sample: APIGenSample) -> str:
    ordered = sample.required_params + [p for p in sample.optional_params if p not in sample.required_params]
    return f"{sample.tool_name}({', '.join(ordered)})"


def pick_distractor(pool: Sequence[APIGenSample], source: APIGenSample, rng: random.Random) -> APIGenSample:
    candidates = [x for x in pool if x.tool_name != source.tool_name]
    if not candidates:
        return source
    return rng.choice(candidates)
