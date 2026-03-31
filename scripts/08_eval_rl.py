from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate RL rollout results.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/simulator_results.jsonl"))
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input)
    if not rows:
        result = {"num_rows": 0, "success_rate": 0.0, "avg_reward": 0.0}
    else:
        success_rate = sum(1 for row in rows if row.get("success")) / len(rows)
        rewards: List[float] = [float(row.get("reward", 0.0)) for row in rows]
        result = {
            "num_rows": len(rows),
            "success_rate": round(success_rate, 6),
            "avg_reward": round(sum(rewards) / len(rewards), 6),
            "reward_std": round(statistics.pstdev(rewards), 6),
        }

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
