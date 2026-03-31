from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable


def dump_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download APIGen/xLAM function-calling data.")
    parser.add_argument("--dataset-id", default="Salesforce/xlam-function-calling-60k")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--output", type=Path, default=Path("data/raw/apigen/apigen_raw.jsonl"))
    parser.add_argument("--input-jsonl", type=Path, default=None, help="Use existing local JSONL instead of downloading.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.input_jsonl:
        shutil.copyfile(args.input_jsonl, args.output)
        print(f"[download] copied local file: {args.input_jsonl} -> {args.output}")
        return

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("`datasets` is required. Install dependencies from requirements.txt first.") from exc

    print(f"[download] loading dataset={args.dataset_id}, split={args.split}")
    ds = load_dataset(args.dataset_id, split=args.split)
    if args.limit and args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    rows = []
    for idx, row in enumerate(ds):
        payload = dict(row)
        payload["_source_index"] = idx
        rows.append(payload)

    dump_jsonl(args.output, rows)
    print(f"[download] wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
