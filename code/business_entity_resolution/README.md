
# Amazon ML Challenge 2026 — Reference Entity Resolution Pipeline

This repository is a reproducible reference implementation for the Business Entity Resolution challenge.

## Chosen approach

Hybrid, precision-oriented pipeline:

1. Multi-key blocking / candidate generation
2. Candidate compression with cheap fuzzy signals
3. Pairwise feature engineering
4. XGBoost binary classifier
5. Validation-tuned source-specific thresholds
6. Final `matching_results.tsv` and `candidate_pairs.tsv`

The design follows the challenge constraints:
- only the supplied data is used;
- no external business lookup, geocoding, or data enrichment;
- Source 1 is the reference entity list;
- Source 2/3 records are candidate targets;
- final matches must be a subset of final candidates;
- test-time country labels are treated as open-set strings.

## Important limitation

The challenge dataset itself was not provided in this conversation, so the pipeline could not be scored on the real hidden/public dataset here. You must run the EDA and validation on your actual challenge data before treating any configuration as final.

## Repository layout

```text
code/business_entity_resolution/src/
    config.py
    io.py
    normalize.py
    indexing.py
    blocking.py
    features.py
    metrics.py
    model.py
    pipeline.py

scripts/
    train.py
    evaluate.py
    infer.py
    validate_submission.py
    package_submission.py

configs/
    baseline.yaml

experiments/
    experiments.csv
```

## Expected local dataset layout

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

Do NOT commit the challenge dataset to Git.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py \
  --train-dir dataset/train \
  --artifacts artifacts \
  --config configs/baseline.yaml

python scripts/evaluate.py \
  --artifacts artifacts

python scripts/infer.py \
  --test-dir dataset/test \
  --artifacts artifacts \
  --output-dir output

python scripts/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test

python scripts/package_submission.py \
  --output-dir output \
  --artifacts artifacts \
  --package-path amazon_ml_challenge_2026_submission.zip
```

For the official validator, use the exact supplied `utils/validate_submission.py` from the challenge package.
