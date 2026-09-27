
#!/usr/bin/env python3
"""
Thin wrapper for the official challenge validator.

Put the supplied official validator at:
    utils/validate_submission.py

Then this script forwards arguments to it.
"""

from __future__ import annotations

from pathlib import Path
import runpy
import sys


def main():
    official = Path(__file__).resolve().parents[1] / "utils" / "validate_submission.py"
    if not official.exists():
        raise SystemExit(
            "Official validator not found. Copy the challenge-supplied "
            "utils/validate_submission.py into this repository first."
        )
    sys.argv[0] = str(official)
    runpy.run_path(str(official), run_name="__main__")


if __name__ == "__main__":
    main()
