
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Config:
    random_seed: int = 42
    validation_fraction: float = 0.20

    name_prefix_len: int = 5
    min_token_len: int = 3
    max_token_freq: int = 50
    max_block_size: int = 250
    max_candidates_per_s1: int = 80

    compression_enabled: bool = True
    cheap_name_weight: float = 0.60
    cheap_address_weight: float = 0.40
    keep_if_cheap_score_at_least: float = 0.50

    negatives_per_positive: int = 4
    singleton_negatives: int = 12
    min_training_candidates_per_s1: int = 4

    n_estimators: int = 600
    learning_rate: float = 0.05
    max_depth: int = 7
    min_child_weight: int = 3
    subsample: float = 0.85
    colsample_bytree: float = 0.90
    reg_lambda: float = 3.0
    reg_alpha: float = 0.1
    gamma: float = 0.0

    threshold_min: float = 0.30
    threshold_max: float = 0.99
    threshold_step: float = 0.01

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        b = raw.get("blocking", {})
        c = raw.get("candidate_compression", {})
        t = raw.get("training", {})
        th = raw.get("thresholds", {})
        return cls(
            random_seed=raw.get("random_seed", cls.random_seed),
            validation_fraction=raw.get("validation_fraction", cls.validation_fraction),
            name_prefix_len=b.get("name_prefix_len", cls.name_prefix_len),
            min_token_len=b.get("min_token_len", cls.min_token_len),
            max_token_freq=b.get("max_token_freq", cls.max_token_freq),
            max_block_size=b.get("max_block_size", cls.max_block_size),
            max_candidates_per_s1=b.get("max_candidates_per_s1", cls.max_candidates_per_s1),
            compression_enabled=c.get("enabled", cls.compression_enabled),
            cheap_name_weight=c.get("cheap_name_weight", cls.cheap_name_weight),
            cheap_address_weight=c.get("cheap_address_weight", cls.cheap_address_weight),
            keep_if_cheap_score_at_least=c.get(
                "keep_if_cheap_score_at_least", cls.keep_if_cheap_score_at_least
            ),
            negatives_per_positive=t.get(
                "negatives_per_positive", cls.negatives_per_positive
            ),
            singleton_negatives=t.get("singleton_negatives", cls.singleton_negatives),
            min_training_candidates_per_s1=t.get(
                "min_training_candidates_per_s1", cls.min_training_candidates_per_s1
            ),
            n_estimators=t.get("n_estimators", cls.n_estimators),
            learning_rate=t.get("learning_rate", cls.learning_rate),
            max_depth=t.get("max_depth", cls.max_depth),
            min_child_weight=t.get("min_child_weight", cls.min_child_weight),
            subsample=t.get("subsample", cls.subsample),
            colsample_bytree=t.get("colsample_bytree", cls.colsample_bytree),
            reg_lambda=t.get("reg_lambda", cls.reg_lambda),
            reg_alpha=t.get("reg_alpha", cls.reg_alpha),
            gamma=t.get("gamma", cls.gamma),
            threshold_min=th.get("min_value", cls.threshold_min),
            threshold_max=th.get("max_value", cls.threshold_max),
            threshold_step=th.get("step", cls.threshold_step),
        )
