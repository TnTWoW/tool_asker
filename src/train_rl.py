from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="RL training entry (scaffold).")
    parser.add_argument("--config", type=Path, default=Path("configs/rl.yaml"))
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    print("[train_rl] Loaded config:")
    print(config)
    print("[train_rl] Scaffold only. Plug your ranking/group-RL loop here.")


if __name__ == "__main__":
    main()
