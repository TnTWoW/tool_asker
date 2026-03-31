from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


class GapType(str, Enum):
    MISSING_REQUIRED_PARAMETER = "missing_required_parameter"
    MISSING_API_ENDPOINT = "missing_api_endpoint"
    MISSING_AUTH = "missing_auth"
    AMBIGUOUS_TOOL_CHOICE = "ambiguous_tool_choice"


class Decision(str, Enum):
    ASK = "ask"
    INFER = "infer"
    ACT = "act"


def _is_scalar(value: Any) -> bool:
    return isinstance(value, (str, int, float, bool)) or value is None


def _validate_simple_arguments(arguments: Dict[str, Any]) -> None:
    if not isinstance(arguments, dict):
        raise ValueError("arguments must be a dictionary")
    for key, value in arguments.items():
        if isinstance(value, list):
            if not all(_is_scalar(v) for v in value):
                raise ValueError(f"argument '{key}' must be scalar or list of scalars")
            continue
        if not _is_scalar(value):
            raise ValueError(f"argument '{key}' must be scalar or list of scalars")


@dataclass
class ToolCall:
    tool_name: str
    arguments: Dict[str, Any]

    def validate(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name is required")
        _validate_simple_arguments(self.arguments)


@dataclass
class APIGenSample:
    id: str
    user_query: str
    tool_name: str
    tool_description: str
    arguments: Dict[str, Any]
    required_params: List[str]
    optional_params: List[str]
    api_base_url: Optional[str]
    auth_type: Optional[str]
    ground_truth_call: ToolCall

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id is required")
        if not self.user_query:
            raise ValueError("user_query is required")
        if not self.tool_name:
            raise ValueError("tool_name is required")
        if not self.tool_description:
            raise ValueError("tool_description is required")
        if not self.required_params:
            raise ValueError("required_params cannot be empty")
        _validate_simple_arguments(self.arguments)
        if not set(self.required_params).issubset(set(self.arguments.keys())):
            raise ValueError("all required_params must exist in arguments")
        self.ground_truth_call.validate()

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "APIGenSample":
        call = ToolCall(**obj["ground_truth_call"])
        instance = cls(
            id=obj["id"],
            user_query=obj["user_query"],
            tool_name=obj["tool_name"],
            tool_description=obj["tool_description"],
            arguments=obj["arguments"],
            required_params=obj.get("required_params", []),
            optional_params=obj.get("optional_params", []),
            api_base_url=obj.get("api_base_url"),
            auth_type=obj.get("auth_type"),
            ground_truth_call=call,
        )
        instance.validate()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AvailableToolSpec:
    tool_name: str
    tool_description: str
    required_params: List[str]
    known_arguments: Dict[str, Any]
    api_base_url: Optional[str]
    auth_type: Optional[str]

    def validate(self) -> None:
        if not self.tool_description:
            raise ValueError("tool_description is required")
        _validate_simple_arguments(self.known_arguments)


@dataclass
class GapSample:
    id: str
    base_sample_id: str
    user_query: str
    available_tool_spec: AvailableToolSpec
    gap_type: GapType
    gold_missing_slots: List[str]
    gold_decision: Decision
    gold_question: str
    gold_after_user_reply: Dict[str, Any]
    gold_final_call: ToolCall
    candidate_tools: Optional[List[Dict[str, str]]] = None

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id is required")
        if not self.base_sample_id:
            raise ValueError("base_sample_id is required")
        self.available_tool_spec.validate()
        self.gold_final_call.validate()
        _validate_simple_arguments(self.gold_after_user_reply)
        if self.gold_decision not in set(Decision):
            raise ValueError("gold_decision must be ask/infer/act")
        if not self.gold_question and self.gold_decision == Decision.ASK:
            raise ValueError("gold_question is required when gold_decision is ask")

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "GapSample":
        instance = cls(
            id=obj["id"],
            base_sample_id=obj["base_sample_id"],
            user_query=obj["user_query"],
            available_tool_spec=AvailableToolSpec(**obj["available_tool_spec"]),
            gap_type=GapType(obj["gap_type"]),
            gold_missing_slots=obj.get("gold_missing_slots", []),
            gold_decision=Decision(obj["gold_decision"]),
            gold_question=obj.get("gold_question", ""),
            gold_after_user_reply=obj.get("gold_after_user_reply", {}),
            gold_final_call=ToolCall(**obj["gold_final_call"]),
            candidate_tools=obj.get("candidate_tools"),
        )
        instance.validate()
        return instance

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gap_type"] = self.gap_type.value
        payload["gold_decision"] = self.gold_decision.value
        return payload


@dataclass
class SFTExample:
    id: str
    instruction: str
    target: str
    gap_type: GapType
    source_gap_id: str

    def validate(self) -> None:
        if not self.id:
            raise ValueError("id is required")
        if not self.instruction:
            raise ValueError("instruction is required")
        if not self.target:
            raise ValueError("target is required")

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gap_type"] = self.gap_type.value
        return payload


@dataclass
class EvalCase:
    id: str
    source_gap_id: str
    user_query: str
    available_tool_spec: Dict[str, Any]
    gold_decision: Decision
    gold_gap_type: GapType
    gold_missing_slots: List[str]
    gold_question: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["gold_decision"] = self.gold_decision.value
        payload["gold_gap_type"] = self.gold_gap_type.value
        return payload


def dump_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows
