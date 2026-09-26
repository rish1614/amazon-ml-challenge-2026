
install:
	python3 -m pip install -r requirements.txt

train:
	python3 scripts/train.py --train-dir dataset/train --artifacts artifacts --config configs/baseline.yaml

evaluate:
	python3 scripts/evaluate.py --artifacts artifacts --train-dir dataset/train

infer:
	python3 scripts/infer.py --test-dir dataset/test --artifacts artifacts --output-dir output --config configs/baseline.yaml

validate:
	python3 scripts/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test

package:
	python3 scripts/package_submission.py --output-dir output --artifacts artifacts --package-path amazon_ml_challenge_2026_submission.zip
