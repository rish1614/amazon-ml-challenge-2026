
# Methodology Notes

## Core approach

Hybrid blocking + supervised pair classification.

## Blocking

Candidate generation uses multiple independent keys:
- country + exact normalized name
- country + name prefix
- country + rare name token
- country + rare address token
- country + postal-like token
- country + house-number token

The union of these blocks is then compressed using cheap fuzzy signals.

## Matching model

XGBoost binary classifier over pairwise lexical/address/country features.

## Decision

Thresholds are tuned on a held-out Source-1 validation split for macro F_0.5, with separate thresholds for S2 and S3.

## Fair play

No external data enrichment is used.
