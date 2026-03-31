from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.metrics import (
    MetricBundle,
    decision_accuracy,
    gap_type_accuracy,
    hallucination_rate,
    question_sufficiency_rate,
    set_f1,
)
from src.model_heads import parse_soft_three_head
from src.data_schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SFT predictions.")
    parser.add_argument("--gold", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--pred", type=Path, required=True, help="JSONL with id + either fields or model_output.")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def load_gold(path: Path) -> Dict[str, dict]:
    return {row["id"]: row for row in load_jsonl(path) if "id" in row}


def load_predictions(path: Path) -> Dict[str, dict]:
    parsed: Dict[str, dict] = {}
    for row in load_jsonl(path):
        if "id" not in row:
            continue
        if "model_output" in row and isinstance(row["model_output"], str):
            p = parse_soft_three_head(row["model_output"])
            parsed[row["id"]] = {
                "decision": p.decision,
                "gap_type": p.gap_type,
                "missing_slots": p.missing_slots,
                "question": p.question,
            }
        else:
            parsed[row["id"]] = {
                "decision": str(row.get("decision", "")).lower(),
                "gap_type": str(row.get("gap_type", "")),
                "missing_slots": row.get("missing_slots", []),
                "question": str(row.get("question", "")),
            }
    return parsed


def ensure_slots(value) -> Set[str]:
    if isinstance(value, list):
        return set(str(x) for x in value if str(x).strip())
    if isinstance(value, str):
        return set(x.strip() for x in value.split(",") if x.strip())
    return set()


def compute_bundle(gold_rows: List[dict], pred_rows: List[dict]) -> MetricBundle:
    gold_decision = [g["gold_decision"] for g in gold_rows]
    pred_decision = [p["decision"] for p in pred_rows]
    gold_gap = [g["gap_type"] for g in gold_rows]
    pred_gap = [p["gap_type"] for p in pred_rows]
    gold_slots = [set(g.get("gold_missing_slots", [])) for g in gold_rows]
    pred_slots = [ensure_slots(p.get("missing_slots", [])) for p in pred_rows]
    pred_questions = [p.get("question", "") for p in pred_rows]
    should_ask = [g["gold_decision"] == "ask" for g in gold_rows]

    return MetricBundle(
        decision_accuracy=decision_accuracy(gold_decision, pred_decision),
        gap_type_accuracy=gap_type_accuracy(gold_gap, pred_gap),
        missing_slot_f1=set_f1(gold_slots, pred_slots),
        question_sufficiency_rate=question_sufficiency_rate(gold_slots, pred_questions),
        hallucination_rate=hallucination_rate(should_ask, pred_decision, pred_slots, gold_slots),
    )


def main() -> None:
    args = parse_args()
    gold = load_gold(args.gold)
    pred = load_predictions(args.pred)

    gold_rows: List[dict] = []
    pred_rows: List[dict] = []
    for sid, gold_row in gold.items():
        if sid not in pred:
            continue
        gold_rows.append(gold_row)
        pred_rows.append(pred[sid])

    overall = compute_bundle(gold_rows, pred_rows).to_dict()
    by_gap: Dict[str, dict] = {}
    grouped_gold: Dict[str, List[dict]] = defaultdict(list)
    grouped_pred: Dict[str, List[dict]] = defaultdict(list)
    for g, p in zip(gold_rows, pred_rows):
        grouped_gold[g["gap_type"]].append(g)
        grouped_pred[g["gap_type"]].append(p)
    for gap_type in grouped_gold:
        by_gap[gap_type] = compute_bundle(grouped_gold[gap_type], grouped_pred[gap_type]).to_dict()

    result = {"num_scored": len(gold_rows), "overall": overall, "by_gap_type": by_gap}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
