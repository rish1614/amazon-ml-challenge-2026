
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="output")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--package-path", default="amazon_ml_challenge_2026_submission.zip")
    p.add_argument("--documentation", default="Documentation_template.md")
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    staging = root / ".submission_staging"
    if staging.exists():
        shutil.rmtree(staging)

    app = staging / "code" / "business_entity_resolution"
    app.mkdir(parents=True)

    shutil.copytree(
        root / "code" / "business_entity_resolution" / "src",
        app / "src",
    )
    shutil.copy2(root / "code" / "business_entity_resolution" / "README.md", app / "README.md")
    shutil.copy2(root / "code" / "business_entity_resolution" / "requirements.txt", app / "requirements.txt")

    out = staging / "output"
    out.mkdir()
    shutil.copy2(root / args.output_dir / "matching_results.tsv", out / "matching_results.tsv")
    shutil.copy2(root / args.output_dir / "candidate_pairs.tsv", out / "candidate_pairs.tsv")

    doc = root / args.documentation
    if doc.exists():
        shutil.copy2(doc, staging / "Documentation_template.md")
    else:
        (staging / "Documentation_template.md").write_text(
            "# Documentation\n\nFill this with your final methodology.\n"
        )

    package_path = root / args.package_path
    if package_path.exists():
        package_path.unlink()

    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in staging.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(staging))

    shutil.rmtree(staging)
    print(f"Created: {package_path}")


if __name__ == "__main__":
    main()
