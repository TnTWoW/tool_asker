from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import APIGenSample, ToolCall, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize raw APIGen rows to canonical schema.")
    parser.add_argument("--input", type=Path, default=Path("data/raw/apigen/apigen_raw.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/apigen_base_5k.jsonl"))
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def maybe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if (text.startswith("{") and text.endswith("}")) or (text.startswith("[") and text.endswith("]")):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    return value


def extract_user_query(row: Dict[str, Any]) -> Optional[str]:
    for key in ("user_query", "query", "instruction", "prompt", "input"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    messages = row.get("messages")
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user" and isinstance(msg.get("content"), str):
                return msg["content"].strip()

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if isinstance(turn, dict) and str(turn.get("from", "")).lower() == "human":
                value = turn.get("value")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def extract_tool_spec(row: Dict[str, Any], preferred_tool_name: Optional[str] = None) -> Tuple[Optional[str], Optional[str], List[str], List[str]]:
    tools = row.get("tools") or row.get("functions")
    if isinstance(tools, str):
        tools = maybe_json_load(tools)
    if not isinstance(tools, list) or not tools:
        return None, None, [], []

    selected = None
    if preferred_tool_name:
        for tool in tools:
            if isinstance(tool, dict) and (tool.get("name") == preferred_tool_name or tool.get("tool_name") == preferred_tool_name):
                selected = tool
                break
    if selected is None:
        selected = tools[0]

    if not isinstance(selected, dict):
        return None, None, [], []

    tool_name = selected.get("name") or selected.get("tool_name")
    tool_desc = selected.get("description") or selected.get("tool_description")

    required: List[str] = []
    optional: List[str] = []
    params = selected.get("parameters") or selected.get("schema") or {}
    if isinstance(params, str):
        params = maybe_json_load(params)
    if isinstance(params, dict):
        required = [str(x) for x in (params.get("required") or []) if x]
        properties = params.get("properties") or {}
        if isinstance(properties, dict):
            optional = [k for k in properties.keys() if k not in set(required)]

    return str(tool_name) if tool_name else None, str(tool_desc) if tool_desc else None, required, optional


def extract_tool_call(row: Dict[str, Any], fallback_tool_name: Optional[str]) -> Tuple[Optional[str], Dict[str, Any]]:
    call = row.get("tool_call") or row.get("function_call")
    if isinstance(call, str):
        call = maybe_json_load(call)
    if isinstance(call, dict):
        name = call.get("name") or call.get("tool_name") or fallback_tool_name
        args = call.get("arguments") or call.get("args") or {}
        args = maybe_json_load(args)
        if isinstance(args, dict):
            return str(name) if name else None, args

    calls = row.get("tool_calls")
    if isinstance(calls, str):
        calls = maybe_json_load(calls)
    if isinstance(calls, list) and calls:
        c0 = calls[0]
        if isinstance(c0, dict):
            func = c0.get("function", c0)
            if isinstance(func, str):
                func = maybe_json_load(func)
            if isinstance(func, dict):
                name = func.get("name") or fallback_tool_name
                args = func.get("arguments") or func.get("args") or {}
                args = maybe_json_load(args)
                if isinstance(args, dict):
                    return str(name) if name else None, args

    conversations = row.get("conversations")
    if isinstance(conversations, list):
        for turn in conversations:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("from", "")).strip().lower()
            if role != "function_call":
                continue
            payload = maybe_json_load(turn.get("value"))
            if isinstance(payload, dict):
                name = payload.get("name") or payload.get("tool_name") or fallback_tool_name
                args = payload.get("arguments") or payload.get("args") or {}
                args = maybe_json_load(args)
                if isinstance(args, dict):
                    return str(name) if name else None, args

    args = row.get("arguments") or row.get("args")
    args = maybe_json_load(args)
    if isinstance(args, dict):
        return fallback_tool_name, args
    return fallback_tool_name, {}


def extract_api_auth(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    api_base_url = row.get("api_base_url") or row.get("base_url") or row.get("endpoint")
    auth_type = row.get("auth_type") or row.get("auth") or row.get("auth_method")
    if isinstance(api_base_url, str) and api_base_url.strip():
        api_base_url = api_base_url.strip()
    else:
        api_base_url = None
    if isinstance(auth_type, str) and auth_type.strip():
        auth_type = auth_type.strip()
    else:
        auth_type = None
    return api_base_url, auth_type


def clean_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    cleaned: Dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            cleaned[str(key)] = value
        elif isinstance(value, list) and all(isinstance(v, (str, int, float, bool)) or v is None for v in value):
            cleaned[str(key)] = value
    return cleaned


def slugify(text: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_]+", "_", text.strip().lower())
    return token.strip("_") or "tool"


def normalize_rows(raw_rows: List[Dict[str, Any]], limit: int) -> List[APIGenSample]:
    normalized: List[APIGenSample] = []
    for raw_idx, row in enumerate(raw_rows):
        user_query = extract_user_query(row)
        if not user_query:
            continue

        call_tool_name, arguments = extract_tool_call(row, None)
        tool_name, tool_desc, required, optional = extract_tool_spec(row, preferred_tool_name=call_tool_name)
        tool_name = call_tool_name or tool_name
        if not tool_name or not tool_desc:
            continue
        if not required:
            continue

        arguments = clean_arguments(arguments)
        if not arguments:
            continue
        if not set(required).issubset(set(arguments.keys())):
            continue

        api_base_url, auth_type = extract_api_auth(row)
        sample_id = f"apigen_{len(normalized) + 1:06d}_{slugify(tool_name)}"
        sample = APIGenSample(
            id=sample_id,
            user_query=user_query,
            tool_name=tool_name,
            tool_description=tool_desc,
            arguments=arguments,
            required_params=required,
            optional_params=optional,
            api_base_url=api_base_url,
            auth_type=auth_type,
            ground_truth_call=ToolCall(tool_name=tool_name, arguments=arguments),
        )
        try:
            sample.validate()
        except ValueError:
            continue
        normalized.append(sample)
        if limit and len(normalized) >= limit:
            break
    return normalized


def main() -> None:
    args = parse_args()
    raw_rows = load_jsonl(args.input)
    normalized = normalize_rows(raw_rows, args.limit)
    dump_jsonl(args.output, [x.to_dict() for x in normalized])

    required_param_counter = Counter(len(x.required_params) for x in normalized)
    print(f"[normalize] input rows={len(raw_rows)}")
    print(f"[normalize] output rows={len(normalized)} -> {args.output}")
    print("[normalize] required_params distribution:")
    for key in sorted(required_param_counter):
        print(f"  {key}: {required_param_counter[key]}")


if __name__ == "__main__":
    main()
