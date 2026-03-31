from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import GapSample, dump_jsonl, load_jsonl
from src.env import single_turn_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based user simulator for ITGE RL environment.")
    parser.add_argument("--gaps", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--predictions", type=Path, default=None, help="JSONL: id, decision, missing_slots.")
    parser.add_argument("--output", type=Path, default=Path("data/processed/simulator_results.jsonl"))
    return parser.parse_args()


def load_predictions(path: Path) -> Dict[str, dict]:
    mapping: Dict[str, dict] = {}
    for row in load_jsonl(path):
        if "id" in row:
            mapping[str(row["id"])] = row
    return mapping


def parse_slots(value) -> Set[str]:
    if isinstance(value, list):
        return set(str(x) for x in value if x)
    if isinstance(value, str):
        return set(x.strip() for x in value.split(",") if x.strip())
    return set()


def main() -> None:
    args = parse_args()
    gap_rows = [GapSample.from_dict(x) for x in load_jsonl(args.gaps)]
    predictions = load_predictions(args.predictions) if args.predictions else {}

    results: List[dict] = []
    success_count = 0
    for gap in gap_rows:
        pred = predictions.get(gap.id, {})
        decision = str(pred.get("decision", gap.gold_decision.value))
        slots = parse_slots(pred.get("missing_slots", gap.gold_missing_slots))
        step = single_turn_step(
            predicted_decision=decision,
            predicted_slots=slots,
            gold_decision=gap.gold_decision.value,
            gold_missing_slots=set(gap.gold_missing_slots),
            gold_values={k: str(v) for k, v in gap.gold_after_user_reply.items()},
        )
        success_count += 1 if step.success else 0
        results.append(
            {
                "id": gap.id,
                "predicted_decision": decision,
                "predicted_slots": sorted(list(slots)),
                "user_reply": step.user_reply,
                "reward": step.reward,
                "success": step.success,
            }
        )

    dump_jsonl(args.output, results)
    success_rate = success_count / len(results) if results else 0.0
    avg_reward = sum(x["reward"] for x in results) / len(results) if results else 0.0
    print(f"[sim] rows={len(results)} success_rate={success_rate:.4f} avg_reward={avg_reward:.4f}")
    print(f"[sim] output -> {args.output}")


if __name__ == "__main__":
    main()
