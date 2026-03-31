from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training entry (scaffold).")
    parser.add_argument("--config", type=Path, default=Path("configs/sft.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print("[train_sft] Loaded config:")
    print(config)
    print("[train_sft] Scaffold only. Plug your trainer here (transformers/trl/peft).")


if __name__ == "__main__":
    main()
