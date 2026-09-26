
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

from src.config import Config
from src.pipeline import train_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train-dir", required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    summary = train_pipeline(args.train_dir, args.artifacts, cfg)

    print("Training complete.")
    for k, v in summary.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
