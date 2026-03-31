from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import Decision, EvalCase, GapSample, GapType, dump_jsonl, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build evaluation data from gap samples.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--eval-out", type=Path, default=Path("data/processed/eval_cases.jsonl"))
    parser.add_argument("--rl-out", type=Path, default=Path("data/processed/rl_env_cases.jsonl"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)

    eval_cases: List[dict] = []
    for idx, row in enumerate(rows, start=1):
        try:
            gap = GapSample.from_dict(row)
        except Exception:
            continue
        case = EvalCase(
            id=f"eval_{idx:06d}",
            source_gap_id=gap.id,
            user_query=gap.user_query,
            available_tool_spec=gap.available_tool_spec.__dict__,
            gold_decision=Decision(gap.gold_decision.value),
            gold_gap_type=GapType(gap.gap_type.value),
            gold_missing_slots=gap.gold_missing_slots,
            gold_question=gap.gold_question,
        )
        eval_cases.append(case.to_dict())

    dump_jsonl(args.eval_out, eval_cases)
    dump_jsonl(args.rl_out, eval_cases)
    print(f"[eval] wrote {len(eval_cases)} eval rows -> {args.eval_out}")
    print(f"[eval] wrote {len(eval_cases)} rl rows -> {args.rl_out}")


if __name__ == "__main__":
    main()
