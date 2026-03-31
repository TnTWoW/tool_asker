from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_schema import GapSample, GapType, SFTExample, dump_jsonl, load_jsonl
from src.prompt_templates import render_instruction, render_soft_three_head_target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SFT train/dev/test data from gap samples.")
    parser.add_argument("--input", type=Path, default=Path("data/processed/gap_samples.jsonl"))
    parser.add_argument("--train-out", type=Path, default=Path("data/processed/sft_train.jsonl"))
    parser.add_argument("--dev-out", type=Path, default=Path("data/processed/sft_dev.jsonl"))
    parser.add_argument("--test-out", type=Path, default=Path("data/processed/sft_test.jsonl"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--dev-ratio", type=float, default=0.1)
    return parser.parse_args()


def stratified_split(rows: Sequence[SFTExample], train_ratio: float, dev_ratio: float, seed: int) -> Tuple[List[SFTExample], List[SFTExample], List[SFTExample]]:
    by_gap: Dict[str, List[SFTExample]] = defaultdict(list)
    for row in rows:
        by_gap[row.gap_type.value].append(row)

    rng = random.Random(seed)
    train, dev, test = [], [], []
    for _, bucket in by_gap.items():
        rng.shuffle(bucket)
        n = len(bucket)
        n_train = int(n * train_ratio)
        n_dev = int(n * dev_ratio)
        train.extend(bucket[:n_train])
        dev.extend(bucket[n_train:n_train + n_dev])
        test.extend(bucket[n_train + n_dev :])
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    return train, dev, test


def main() -> None:
    args = parse_args()
    raw_rows = load_jsonl(args.input)

    sft_rows: List[SFTExample] = []
    for idx, row in enumerate(raw_rows, start=1):
        try:
            gap = GapSample.from_dict(row)
        except Exception:
            continue
        instruction = render_instruction(gap)
        target = render_soft_three_head_target(
            decision=gap.gold_decision.value,
            gap_type=gap.gap_type.value,
            missing_slots=gap.gold_missing_slots,
            question=gap.gold_question,
        )
        sft_rows.append(
            SFTExample(
                id=f"sft_{idx:06d}",
                instruction=instruction,
                target=target,
                gap_type=GapType(gap.gap_type.value),
                source_gap_id=gap.id,
            )
        )

    train, dev, test = stratified_split(sft_rows, args.train_ratio, args.dev_ratio, args.seed)
    dump_jsonl(args.train_out, [x.to_dict() for x in train])
    dump_jsonl(args.dev_out, [x.to_dict() for x in dev])
    dump_jsonl(args.test_out, [x.to_dict() for x in test])

    print(f"[sft] total={len(sft_rows)}")
    print(f"[sft] train={len(train)} -> {args.train_out}")
    print(f"[sft] dev={len(dev)} -> {args.dev_out}")
    print(f"[sft] test={len(test)} -> {args.test_out}")


if __name__ == "__main__":
    main()
