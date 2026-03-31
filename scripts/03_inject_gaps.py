from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import APIGenSample, dump_jsonl, load_jsonl
from src.gap_injection import (
    inject_ambiguous_tool_choice,
    inject_missing_api_endpoint,
    inject_missing_auth,
    inject_missing_required_parameter,
    pick_distractor,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inject tool gaps into canonical APIGen samples.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/apigen_base_5k.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--per-gap-target", type=int, default=1000)
    return parser.parse_args()


def load_base_samples(path: Path) -> List[APIGenSample]:
    rows = load_jsonl(path)
    samples: List[APIGenSample] = []
    for row in rows:
        try:
            samples.append(APIGenSample.from_dict(row))
        except Exception:
            continue
    return samples


def build_gap_samples(base_samples: List[APIGenSample], per_gap_target: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    if not base_samples:
        return []

    rows: List[Dict] = []
    counters = {
        "missing_required_parameter": 0,
        "missing_api_endpoint": 0,
        "missing_auth": 0,
        "ambiguous_tool_choice": 0,
    }
    base_index = 0
    while min(counters.values()) < per_gap_target:
        base = base_samples[base_index % len(base_samples)]
        base_index += 1

        if counters["missing_required_parameter"] < per_gap_target:
            gid = f"gap_{len(rows) + 1:06d}"
            rows.append(inject_missing_required_parameter(base, gid, rng).to_dict())
            counters["missing_required_parameter"] += 1
        if counters["missing_api_endpoint"] < per_gap_target:
            gid = f"gap_{len(rows) + 1:06d}"
            rows.append(inject_missing_api_endpoint(base, gid).to_dict())
            counters["missing_api_endpoint"] += 1
        if counters["missing_auth"] < per_gap_target:
            gid = f"gap_{len(rows) + 1:06d}"
            rows.append(inject_missing_auth(base, gid).to_dict())
            counters["missing_auth"] += 1
        if counters["ambiguous_tool_choice"] < per_gap_target:
            gid = f"gap_{len(rows) + 1:06d}"
            distractor = pick_distractor(base_samples, base, rng)
            rows.append(inject_ambiguous_tool_choice(base, distractor, gid).to_dict())
            counters["ambiguous_tool_choice"] += 1
    return rows


def main() -> None:
    args = parse_args()
    base_samples = load_base_samples(args.input)
    rows = build_gap_samples(base_samples, args.per_gap_target, args.seed)
    dump_jsonl(args.output, rows)

    c = Counter(row["gap_type"] for row in rows)
    print(f"[inject] base_samples={len(base_samples)}")
    print(f"[inject] output rows={len(rows)} -> {args.output}")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")


if __name__ == "__main__":
    main()
