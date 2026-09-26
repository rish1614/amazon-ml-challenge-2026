
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution"))

from src.config import Config
from src.pipeline import run_inference


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", required=True)
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--config", default="configs/baseline.yaml")
    args = p.parse_args()

    cfg = Config.from_yaml(args.config)
    result = run_inference(args.test_dir, args.artifacts, args.output_dir, cfg)

    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
