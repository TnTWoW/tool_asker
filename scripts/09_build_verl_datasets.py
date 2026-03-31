from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import GapSample, load_jsonl
from src.prompt_templates import render_instruction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local JSONL datasets into VERL parquet datasets.")
    parser.add_argument("--sft-train", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--sft-val", type=Path, default=Path("data/processed/sft_dev.jsonl"))
    parser.add_argument("--sft-test", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--gap-input", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--output-root", type=Path, default=Path("data/verl"))
    parser.add_argument("--rl-train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _write_parquet(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_parquet(path, index=False)


def _build_sft_rows(rows: Sequence[Dict]) -> List[Dict]:
    verl_rows: List[Dict] = []
    for row in rows:
        verl_rows.append(
            {
                "prompt": row["instruction"],
                "response": row["target"],
                "source_gap_id": row.get("source_gap_id"),
                "gap_type": row.get("gap_type"),
            }
        )
    return verl_rows


def _stratified_split(rows: Sequence[GapSample], train_ratio: float, seed: int) -> Tuple[List[GapSample], List[GapSample]]:
    grouped: Dict[str, List[GapSample]] = defaultdict(list)
    for row in rows:
        grouped[row.gap_type.value].append(row)

    rng = random.Random(seed)
    train_rows: List[GapSample] = []
    val_rows: List[GapSample] = []
    for bucket in grouped.values():
        rng.shuffle(bucket)
        split_at = int(len(bucket) * train_ratio)
        train_rows.extend(bucket[:split_at])
        val_rows.extend(bucket[split_at:])

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def _build_rl_row(gap: GapSample, index: int) -> Dict:
    prompt = render_instruction(gap)
    ground_truth = {
        "decision": gap.gold_decision.value,
        "gap_type": gap.gap_type.value,
        "missing_slots": gap.gold_missing_slots,
        "question": gap.gold_question,
    }
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "data_source": "tool_gap_elicitation",
        "ability": gap.gap_type.value,
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(ground_truth, ensure_ascii=False),
        },
        "extra_info": {
            "index": index,
            "source_gap_id": gap.id,
            "user_query": gap.user_query,
            "gold_question": gap.gold_question,
        },
    }


def main() -> None:
    args = parse_args()

    sft_train_rows = _build_sft_rows(load_jsonl(args.sft_train))
    sft_val_rows = _build_sft_rows(load_jsonl(args.sft_val))
    sft_test_rows = _build_sft_rows(load_jsonl(args.sft_test))

    _write_parquet(args.output_root / "sft" / "train.parquet", sft_train_rows)
    _write_parquet(args.output_root / "sft" / "val.parquet", sft_val_rows)
    _write_parquet(args.output_root / "sft" / "test.parquet", sft_test_rows)

    gap_rows = [GapSample.from_dict(row) for row in load_jsonl(args.gap_input)]
    rl_train, rl_val = _stratified_split(gap_rows, train_ratio=args.rl_train_ratio, seed=args.seed)

    rl_train_rows = [_build_rl_row(gap, index=i) for i, gap in enumerate(rl_train)]
    rl_val_rows = [_build_rl_row(gap, index=i) for i, gap in enumerate(rl_val)]

    _write_parquet(args.output_root / "rl" / "train.parquet", rl_train_rows)
    _write_parquet(args.output_root / "rl" / "val.parquet", rl_val_rows)

    print(f"[verl] sft train -> {args.output_root / 'sft' / 'train.parquet'} ({len(sft_train_rows)})")
    print(f"[verl] sft val   -> {args.output_root / 'sft' / 'val.parquet'} ({len(sft_val_rows)})")
    print(f"[verl] sft test  -> {args.output_root / 'sft' / 'test.parquet'} ({len(sft_test_rows)})")
    print(f"[verl] rl train  -> {args.output_root / 'rl' / 'train.parquet'} ({len(rl_train_rows)})")
    print(f"[verl] rl val    -> {args.output_root / 'rl' / 'val.parquet'} ({len(rl_val_rows)})")


if __name__ == "__main__":
    main()
