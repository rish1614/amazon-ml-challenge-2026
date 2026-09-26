
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

from src.pipeline import evaluate_saved_validation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--train-dir", default="dataset/train")
    args = p.parse_args()

    result = evaluate_saved_validation(args.artifacts, args.train_dir)
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
