"""Typed access to the YAML experiment configs (``configs/experiments/*.yaml``).

The training and generation scripts read one config file; this module loads it
into dataclasses so field names are checked in one place instead of scattered
``cfg["training"]["lr"]`` lookups. Unknown YAML keys are ignored so configs can
carry documentation the code does not consume.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from .data import repo_root


def _only_known(cls, raw: dict[str, Any]) -> dict[str, Any]:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in raw.items() if k in known}


@dataclass
class DataConfig:
    selection: str
    holdouts: str
    source: str = "greek"
    max_len: int = 192
    max_ratio: float = 2.0         # 0 disables the ratio filter (vref sources)
    pairing: str = "one-to-many"   # or "many-to-many"
    k: int = 4                     # many-to-many: sources sampled per target verse
    pivot: str = "spa"             # NLLB eval: source language for held-out generation
    vref_encoding: str = "struct"  # source == "vref": struct | vtok | text
    # Repo-relative path to a committed manifest checksum; when set, training
    # asserts the train pair set matches it (spec-vref.md, verification #2).
    expected_train_manifest: str | None = None


@dataclass
class TokenizerConfig:
    type: str = "bpe"
    vocab_size: int = 32000


@dataclass
class ModelConfig:
    arch: str = "marian"
    pretrained: str | None = None   # HF model id for fine-tune archs (e.g. NLLB)
    encoder_layers: int = 6
    decoder_layers: int = 6
    d_model: int = 1024
    encoder_attention_heads: int = 16
    decoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    decoder_ffn_dim: int = 4096
    dropout: float = 0.1
    label_smoothing: float = 0.1


@dataclass
class TrainingConfig:
    optimizer: str = "adamw"
    lr: float = 5.0e-4
    warmup_steps: int = 4000
    lr_scheduler: str = "inverse_sqrt"
    max_tokens_per_batch: int = 16000
    gradient_accumulation: int = 2
    max_grad_norm: float = 1.0
    bf16: bool = True
    max_steps: int = 100000
    early_stopping_patience: int = 5
    eval_every_steps: int = 2000
    seed: int = 13
    gradient_checkpointing: bool = False
    # Batch is sized by sentence count on the smaller dev GPU; the training
    # script derives it from max_tokens_per_batch unless this is set.
    per_device_batch_size: int | None = None


@dataclass
class InferenceConfig:
    beam: int = 5
    length_penalty: float = 1.0
    max_length: int = 192


@dataclass
class ProbeConfig:
    """Held-out probe evaluation during training (spec-vref.md, "Training").

    Present in a config's ``probe:`` section => probes run every
    ``every_steps``, drive early stopping on macro chrF3 and select the best
    checkpoint; the loss-based EarlyStoppingCallback is then disabled.
    """

    every_steps: int = 1000
    verses_per_language: int = 250
    min_gain: float = 1.0          # macro chrF3 must gain this much ...
    patience_steps: int = 20000    # ... within this many steps, else stop
    batch_size: int = 64
    seed: int = 13
    early_stop: bool = True         # False => run to max_steps, probe throughout
    # Also probe SEEN (trained) verses of the holdout languages, to watch
    # memorisation separately from held-out transfer (spec-vref.md). 0 disables.
    seen_verses_per_language: int = 250


@dataclass
class ExperimentConfig:
    name: str
    phase: str
    data: DataConfig
    tokenizer: TokenizerConfig
    model: ModelConfig
    training: TrainingConfig
    inference: InferenceConfig
    probe: ProbeConfig | None = None
    oversample_holdouts: int = 1
    path: Path | None = None

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            name=raw["name"],
            phase=raw.get("phase", ""),
            data=DataConfig(**_only_known(DataConfig, raw["data"])),
            tokenizer=TokenizerConfig(**_only_known(TokenizerConfig, raw.get("tokenizer", {}))),
            model=ModelConfig(**_only_known(ModelConfig, raw.get("model", {}))),
            training=TrainingConfig(**_only_known(TrainingConfig, raw.get("training", {}))),
            inference=InferenceConfig(**_only_known(InferenceConfig, raw.get("inference", {}))),
            probe=(
                ProbeConfig(**_only_known(ProbeConfig, raw["probe"]))
                if "probe" in raw else None
            ),
            oversample_holdouts=int(raw.get("oversample_holdouts", 1)),
            path=path,
        )

    def resolve(self, relative: str) -> Path:
        """Resolve a repo-relative config path (e.g. the selection CSV)."""
        return repo_root() / relative
